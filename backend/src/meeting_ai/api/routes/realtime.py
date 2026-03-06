"""
实时流式录音 WebSocket 端点

协议:
  Client → Server:
    JSON: { "type": "preload_models", "config": { "asr_engine": "funasr" } }
    JSON: { "type": "unload_models" }
    JSON: { "type": "start_recording", "config": { ... } }
    Binary: PCM Int16 LE 16kHz mono (每 200-300ms 发送一次)
    JSON: { "type": "stop_recording" }

  Server → Client:
    { "type": "connected" }
    { "type": "models_ready", "engine": "funasr", "load_time": 2.3 }
    { "type": "models_unloaded" }
    { "type": "recording_started", "session_id": "..." }
    { "type": "partial", "text": "...", "is_final": false, "segment_id": 0,
      "start_time": 0.0, "end_time": 0.6 }
    { "type": "recording_stopped", "duration": 10.5, "segment_count": 8 }
    { "type": "post_progress", "step": "diarization", "progress": 0.4,
      "overall_progress": 0.2, "message": "说话人分离中..." }
    { "type": "final_result", "result": {...}, "history_id": "..." }
    { "type": "error", "message": "...", "recoverable": true }

架构:
  采用 producer-consumer 模式，将 WebSocket 接收与 ASR 处理分离：

  Producer (主循环):
    - 持续接收 WebSocket 消息（极快，永不阻塞）
    - Binary PCM → 写入 WAV + 放入 asyncio.Queue
    - JSON 控制消息 → 直接处理

  Consumer (_audio_processor 后台任务):
    - 从 Queue 取出 PCM chunk
    - 如果 Queue 中有多个堆积的 chunk，自动合并批量处理（追赶延迟）
    - 调用 ASR 引擎 feed_chunk()（在线程池中执行）
    - 发送识别结果给前端

  这样避免了旧架构中 "receive → process → receive" 串行瓶颈导致的
  WebSocket 消息丢失和文字严重滞后。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...config import get_settings
from ...logger import get_logger
from ...models import StreamingSegment
from ...services.streaming_asr import (
    StreamingVADProcessor,
    get_streaming_asr_engine,
    reset_streaming_asr_engine,
)
from ...utils.wav_writer import IncrementalWavWriter

logger = get_logger("api.routes.realtime")

router = APIRouter()

# Audio queue 最大容量（防止内存爆炸）
# 200ms/chunk × 2000 = 400s 音频缓冲，远超正常需求
_AUDIO_QUEUE_MAX = 2000


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------

async def send_json(
    ws: WebSocket,
    data: dict,
    lock: asyncio.Lock | None = None,
) -> bool:
    """安全发送 JSON 消息，返回是否成功。
    传入 lock 时串行化发送，防止并发 task 同时写 WebSocket 帧。
    """
    try:
        if lock:
            async with lock:
                await ws.send_json(data)
        else:
            await ws.send_json(data)
        return True
    except Exception:
        return False


async def send_error(ws: WebSocket, message: str, recoverable: bool = True) -> None:
    await send_json(ws, {
        "type": "error",
        "message": message,
        "recoverable": recoverable,
    })


# ---------------------------------------------------------------------------
# Recording modes
# ---------------------------------------------------------------------------

RECORDING_MODES = ("streaming", "segment", "hybrid")


# ---------------------------------------------------------------------------
# Temp WAV helper (segment / hybrid modes)
# ---------------------------------------------------------------------------

def _write_temp_wav(pcm_int16: np.ndarray, sample_rate: int = 16000) -> str:
    """将 PCM int16 数据写为临时 WAV 文件，返回路径。调用者负责 os.unlink。"""
    import tempfile
    import wave

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return tmp.name


async def _segment_asr_transcribe(
    audio_segment: dict,
    asr_engine: Any,
    segment_id: int,
) -> StreamingSegment:
    """用文件 ASR 引擎转写一个完整语音段，返回 StreamingSegment"""
    import os

    wav_path = _write_temp_wav(audio_segment["audio"])
    try:
        loop = asyncio.get_event_loop()
        # VAD 已切段，所有引擎都跳过内部 VAD 防止二次过滤丢字
        from functools import partial as _partial
        from ...services.asr import FasterWhisperEngine, FireRedASREngine
        if isinstance(asr_engine, FasterWhisperEngine):
            result = await loop.run_in_executor(
                None, _partial(asr_engine.transcribe, wav_path, vad_filter=False)
            )
        else:
            result = await loop.run_in_executor(
                None, _partial(asr_engine.transcribe, wav_path, skip_vad=True)
            )

        # Whisper 对短片段经常不带标点，用 ct-punc 恢复
        # 注意：FireRedASR 引擎内部已调用 ct-punc，这里不重复加标点
        text = result.full_text.strip()
        punc_applied = False
        if isinstance(asr_engine, FasterWhisperEngine) and text:
            from ...services.asr import _add_punctuation
            new_text = await loop.run_in_executor(None, _add_punctuation, text)
            if new_text != text:
                punc_applied = True
                text = new_text

        # 提取 char_timestamps（如果有），偏移到全局录音时间
        # 注意：ct-punc 加标点后 char_timestamps 字符不匹配，对齐时会丢标点
        # 此时放弃 char_timestamps，让对齐用中点匹配（短片段精度足够）
        char_ts = None
        seg_offset = audio_segment["start"]
        if not punc_applied and result.char_timestamps and len(result.char_timestamps) > 0:
            from ...models import CharTimestamp
            merged_ts: list[CharTimestamp] = []
            for ts_list in result.char_timestamps:
                for ct in ts_list:
                    merged_ts.append(CharTimestamp(
                        char=ct.char,
                        start=ct.start + seg_offset,
                        end=ct.end + seg_offset,
                    ))
            if merged_ts:
                char_ts = merged_ts

        seg = StreamingSegment(
            id=segment_id,
            start=audio_segment["start"],
            end=audio_segment["end"],
            text=text,
            temp_speaker="SPEAKER_00",
            char_timestamps=char_ts,
        )
        return seg
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


async def _do_segment_transcribe(
    ws: WebSocket,
    seg_audio: dict,
    asr_engine: Any,
    cur_id: int,
    streaming_segments: list,
    sem: asyncio.Semaphore,
    send_lock: asyncio.Lock,
) -> None:
    """并发安全的段转写任务。
    - sem：限制同时占用 GPU 的任务数（防止 OOM），VAD 循环不阻塞
    - send_lock：串行化 WebSocket 发送，防止并发帧交叉
    """
    async with sem:
        try:
            result = await _segment_asr_transcribe(seg_audio, asr_engine, cur_id)
            streaming_segments.append(result)
            await send_json(ws, {
                "type": "partial",
                "segment_id": cur_id,
                "text": result.text,
                "is_final": True,
                "start_time": result.start,
                "end_time": result.end,
            }, lock=send_lock)
        except Exception as e:
            logger.error(f"段级 ASR 转写错误: {e}")
            await send_json(ws, {
                "type": "partial",
                "segment_id": cur_id,
                "text": f"[转写失败: {e}]",
                "is_final": True,
                "start_time": seg_audio["start"],
                "end_time": seg_audio["end"],
            }, lock=send_lock)


# ---------------------------------------------------------------------------
# Audio processor (consumer task) — streaming mode (original)
# ---------------------------------------------------------------------------

async def _audio_processor(
    ws: WebSocket,
    audio_queue: asyncio.Queue,
    asr_engine: Any,
    session: Any,
    streaming_segments: list[StreamingSegment],
) -> None:
    """
    后台任务：从 audio_queue 中读取 PCM chunk，批量喂给 ASR 引擎。

    当 ASR 处理速度跟不上音频输入时，自动将多个排队的 chunk
    合并成一个大 chunk 一次性处理，减少 generate() 调用次数，加快追赶。
    """
    loop = asyncio.get_event_loop()
    total_chunks = 0
    total_batches = 0

    while True:
        # 等待第一个 chunk（阻塞直到有数据）
        chunk = await audio_queue.get()
        if chunk is None:
            # 收到停止信号
            break

        # 尝试从队列中取出更多已排队的 chunk（非阻塞），合并处理
        batched = bytearray(chunk)
        batch_count = 1
        stop_after_batch = False

        while not audio_queue.empty():
            try:
                next_chunk = audio_queue.get_nowait()
                if next_chunk is None:
                    stop_after_batch = True
                    break
                batched.extend(next_chunk)
                batch_count += 1
            except asyncio.QueueEmpty:
                break

        total_chunks += batch_count
        total_batches += 1

        if batch_count > 1:
            logger.debug(
                f"ASR 批量处理: {batch_count} chunks 合并 "
                f"({len(batched)} bytes, queue残余={audio_queue.qsize()})"
            )

        # 在线程池中运行 ASR 推理
        try:
            asr_results = await loop.run_in_executor(
                None,
                asr_engine.feed_chunk,
                session,
                bytes(batched),
                False,
            )
        except Exception as e:
            logger.error(f"ASR 推理错误: {e}")
            await send_error(ws, f"ASR 推理错误: {e}")
            if stop_after_batch:
                break
            continue

        # 发送识别结果给前端
        for seg, is_seg_final in asr_results:
            if is_seg_final:
                streaming_segments.append(seg)
            await send_json(ws, {
                "type": "partial",
                "text": seg.text,
                "is_final": is_seg_final,
                "segment_id": seg.id,
                "start_time": seg.start,
                "end_time": seg.end,
                "is_speech": session.is_speech,
            })

        if stop_after_batch:
            break

    logger.info(
        f"ASR processor 结束: 共接收 {total_chunks} chunks, "
        f"处理 {total_batches} 批次 "
        f"(平均 {total_chunks / max(total_batches, 1):.1f} chunks/batch)"
    )


# ---------------------------------------------------------------------------
# Segment-mode consumer
# ---------------------------------------------------------------------------

async def _segment_processor(
    ws: WebSocket,
    audio_queue: asyncio.Queue,
    vad: StreamingVADProcessor,
    asr_engine: Any,
    streaming_segments: list[StreamingSegment],
) -> None:
    """段级模式 consumer: VAD 切段 → 并发文件 ASR 转写

    VAD 循环与 ASR 转写完全解耦：
    - VAD 主循环持续消费音频 queue，不等待 GPU 转写
    - 每个完成的语音段立即 create_task 提交转写（GPU 任务异步执行）
    - sem=1 保证 GPU 串行使用（防 OOM），队列可堆积等待
    - send_lock 防止并发 task 同时写 WebSocket 帧
    """
    segment_id = 0
    was_speaking = False
    placeholder_sent = False
    transcribe_tasks: list[asyncio.Task] = []
    sem = asyncio.Semaphore(1)        # GPU 同时转写任务数上限
    send_lock = asyncio.Lock()        # WebSocket 并发写保护

    while True:
        chunk = await audio_queue.get()
        if chunk is None:
            break

        # 批量合并排队的 chunk
        batched = bytearray(chunk)
        stop_after_batch = False
        while not audio_queue.empty():
            try:
                next_chunk = audio_queue.get_nowait()
                if next_chunk is None:
                    stop_after_batch = True
                    break
                batched.extend(next_chunk)
            except asyncio.QueueEmpty:
                break

        completed = vad.feed_chunk(bytes(batched))

        # ── A. 语音起始 → 立即发送占位符 ──
        if vad.is_speaking and not was_speaking and not placeholder_sent:
            segment_id += 1
            placeholder_sent = True
            await send_json(ws, {
                "type": "partial",
                "segment_id": segment_id,
                "text": "语音检测中…",
                "is_final": False,
                "is_placeholder": True,
                "start_time": vad._speech_start,
                "end_time": vad._speech_start,
            })

        # ── B. 完成的语音段 → create_task，立即返回继续循环 ──
        for seg_audio in completed:
            if placeholder_sent:
                cur_id = segment_id
                placeholder_sent = False
            else:
                segment_id += 1
                cur_id = segment_id

            await send_json(ws, {
                "type": "partial",
                "segment_id": cur_id,
                "text": "转写中…",
                "is_final": False,
                "is_placeholder": True,
                "start_time": seg_audio["start"],
                "end_time": seg_audio["end"],
            })

            task = asyncio.create_task(
                _do_segment_transcribe(
                    ws, seg_audio, asr_engine, cur_id,
                    streaming_segments, sem, send_lock,
                )
            )
            transcribe_tasks.append(task)

        # ── C. 同 batch 内前段结束+新段开始 → 补发占位符 ──
        if completed and vad.is_speaking and not placeholder_sent:
            segment_id += 1
            placeholder_sent = True
            await send_json(ws, {
                "type": "partial",
                "segment_id": segment_id,
                "text": "语音检测中…",
                "is_final": False,
                "is_placeholder": True,
                "start_time": vad._speech_start,
                "end_time": vad._speech_start,
            })

        was_speaking = vad.is_speaking
        await send_json(ws, {"type": "vad_status", "is_speech": vad.is_speaking})

        if stop_after_batch:
            break

    # 刷出 VAD 缓冲区（剩余未完成的语音段也并发提交）
    remaining = vad.flush()
    for seg_audio in remaining:
        segment_id += 1
        await send_json(ws, {
            "type": "partial",
            "segment_id": segment_id,
            "text": "转写中…",
            "is_final": False,
            "is_placeholder": True,
            "start_time": seg_audio["start"],
            "end_time": seg_audio["end"],
        })
        task = asyncio.create_task(
            _do_segment_transcribe(
                ws, seg_audio, asr_engine, segment_id,
                streaming_segments, sem, send_lock,
            )
        )
        transcribe_tasks.append(task)

    # 等待所有转写任务完成后再结束（后处理依赖 streaming_segments 完整）
    if transcribe_tasks:
        await asyncio.gather(*transcribe_tasks, return_exceptions=True)

    logger.info(f"Segment processor 结束: {segment_id} 段")


# ---------------------------------------------------------------------------
# Hybrid-mode consumer
# ---------------------------------------------------------------------------

async def _hybrid_processor(
    ws: WebSocket,
    audio_queue: asyncio.Queue,
    streaming_engine: Any,
    session: Any,
    vad: StreamingVADProcessor,
    asr_engine: Any,
    streaming_segments: list[StreamingSegment],
) -> None:
    """
    混合模式-段级 consumer:
    1. 流式引擎实时出字（前端即时显示）
    2. 流式引擎完成一段 (is_final=True) → 用文件 ASR 升级替换该段
    3. 流式段未完成时不升级，等待完成

    音频缓冲：保留全部 PCM，流式段完成后按时间范围提取音频给文件 ASR。
    VAD：仅用于前端 speech 状态指示，不触发升级。
    """
    loop = asyncio.get_event_loop()
    upgrade_tasks: list[asyncio.Task] = []
    audio_buffer = bytearray()  # 累积全部 PCM (16kHz int16 mono)
    upgraded_count = 0

    while True:
        chunk = await audio_queue.get()
        if chunk is None:
            break

        # 批量合并
        batched = bytearray(chunk)
        stop_after_batch = False
        while not audio_queue.empty():
            try:
                next_chunk = audio_queue.get_nowait()
                if next_chunk is None:
                    stop_after_batch = True
                    break
                batched.extend(next_chunk)
            except asyncio.QueueEmpty:
                break

        pcm_bytes = bytes(batched)
        audio_buffer.extend(pcm_bytes)

        # 1. 流式引擎：实时出字
        try:
            asr_results = await loop.run_in_executor(
                None, streaming_engine.feed_chunk, session, pcm_bytes, False,
            )
        except Exception as e:
            logger.error(f"流式 ASR 推理错误: {e}")
            asr_results = []

        for seg, is_seg_final in asr_results:
            if is_seg_final:
                streaming_segments.append(seg)
                # 流式段完成 → 提取音频 → 启动文件 ASR 升级
                seg_audio = _extract_audio_segment(
                    audio_buffer, seg.start, seg.end,
                )
                if seg_audio is not None:
                    upgraded_count += 1
                    task = asyncio.create_task(
                        _hybrid_upgrade_segment(
                            seg_audio, asr_engine, ws,
                            seg.id, streaming_segments,
                        )
                    )
                    upgrade_tasks.append(task)

            await send_json(ws, {
                "type": "partial",
                "text": seg.text,
                "is_final": is_seg_final,
                "segment_id": seg.id,
                "start_time": seg.start,
                "end_time": seg.end,
                "is_speech": session.is_speech,
            })

        # 2. VAD 并行：仅用于前端 speech 状态指示
        vad.feed_chunk(pcm_bytes)
        await send_json(ws, {
            "type": "vad_status",
            "is_speech": vad.is_speaking,
        })

        if stop_after_batch:
            break

    # 等待所有升级任务完成
    pending = [t for t in upgrade_tasks if not t.done()]
    if pending:
        await send_json(ws, {
            "type": "upgrade_pending",
            "count": len(pending),
            "message": f"正在完成剩余 {len(pending)} 段升级...",
        })
        await asyncio.gather(*pending, return_exceptions=True)
        await send_json(ws, {
            "type": "upgrade_done",
            "message": "所有段升级完成",
        })

    logger.info(
        f"Hybrid-段级 processor 结束: {len(streaming_segments)} 段, "
        f"{upgraded_count} 段已升级"
    )


def _extract_audio_segment(
    audio_buffer: bytearray,
    start: float,
    end: float,
    sample_rate: int = 16000,
) -> dict | None:
    """从 PCM 缓冲区中按时间范围提取音频段"""
    bytes_per_sample = 2  # int16
    start_byte = int(start * sample_rate) * bytes_per_sample
    end_byte = int(end * sample_rate) * bytes_per_sample

    # 边界检查
    start_byte = max(0, start_byte)
    end_byte = min(len(audio_buffer), end_byte)

    if end_byte <= start_byte:
        return None

    audio_bytes = bytes(audio_buffer[start_byte:end_byte])
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    if len(audio_array) < sample_rate * 0.1:  # 最少 0.1s
        return None

    return {"start": start, "end": end, "audio": audio_array}


async def _hybrid_upgrade_segment(
    seg_audio: dict,
    asr_engine: Any,
    ws: WebSocket,
    target_id: int,
    streaming_segments: list[StreamingSegment],
) -> None:
    """混合模式-段级：用文件 ASR 升级一个已完成的流式段，原地替换"""
    try:
        logger.info(
            f"段级升级开始: id={target_id}, "
            f"时间={seg_audio['start']:.1f}-{seg_audio['end']:.1f}s, "
            f"引擎={type(asr_engine).__name__}"
        )
        result = await _segment_asr_transcribe(seg_audio, asr_engine, target_id)

        # 原地替换对应的流式段
        for i, s in enumerate(streaming_segments):
            if s.id == target_id:
                streaming_segments[i] = result
                break

        logger.info(
            f"段级升级完成: id={target_id}, "
            f"文本={result.text[:30]}..."
        )
        await send_json(ws, {
            "type": "partial",
            "segment_id": target_id,
            "text": result.text,
            "is_final": True,
            "is_upgrade": True,
            "start_time": result.start,
            "end_time": result.end,
        })
    except Exception as e:
        logger.error(f"混合模式升级失败 (id={target_id}): {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Model load/unload helpers
# ---------------------------------------------------------------------------

async def _load_models(ws: WebSocket, engine_type: str | None = None) -> Any:
    """
    加载 ASR 模型（由用户手动触发）。

    Returns:
        已加载的 asr_engine 实例，或 None（加载失败时）
    """
    t0 = time.time()
    try:
        engine = get_streaming_asr_engine(engine_type)
        if engine.is_loaded():
            elapsed = time.time() - t0
            await send_json(ws, {
                "type": "models_ready",
                "engine": engine_type or "default",
                "load_time": round(elapsed, 2),
            })
            return engine

        await send_json(ws, {
            "type": "status",
            "message": "正在加载 ASR 模型...",
        })

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine.load)

        elapsed = time.time() - t0
        logger.info(f"模型加载完成: {elapsed:.1f}s")
        await send_json(ws, {
            "type": "models_ready",
            "engine": engine_type or "default",
            "load_time": round(elapsed, 2),
        })
        return engine

    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        await send_error(ws, f"模型加载失败: {e}", recoverable=True)
        return None


async def _unload_models(ws: WebSocket, engine_type: str | None = None) -> None:
    """卸载 ASR 模型，释放 GPU 显存。"""
    try:
        engine = get_streaming_asr_engine(engine_type)
        if engine.is_loaded():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, engine.unload)
            logger.info("ASR 模型已卸载")

        await send_json(ws, {"type": "models_unloaded"})

    except Exception as e:
        logger.error(f"模型卸载失败: {e}", exc_info=True)
        await send_error(ws, f"模型卸载失败: {e}", recoverable=True)


async def _load_sentence_models(
    ws: WebSocket, model_name: str | None = None,
) -> tuple[Any, "StreamingVADProcessor | None"]:
    """加载段级模式所需的 VAD + 文件 ASR 引擎"""
    t0 = time.time()
    loop = asyncio.get_event_loop()
    try:
        await send_json(ws, {"type": "status", "message": "正在加载 VAD 模型..."})
        vad = StreamingVADProcessor()
        await loop.run_in_executor(None, vad.load)

        await send_json(ws, {"type": "status", "message": "正在加载文件 ASR 模型..."})
        from ...services.asr import get_asr_engine
        engine = await loop.run_in_executor(None, get_asr_engine, model_name)

        # 预加载辅助模型（避免首段转写时懒加载延迟）
        await send_json(ws, {"type": "status", "message": "正在加载辅助模型..."})
        from ...services.asr import (
            _get_vad_model, _get_punc_model, _get_fa_model,
            FasterWhisperEngine, FireRedASREngine,
        )
        # Silero VAD（所有引擎内部 _transcribe_with_vad 使用）
        await loop.run_in_executor(None, _get_vad_model)
        # Whisper / FireRedASR 短片段无标点 → 需要 ct-punc
        if isinstance(engine, (FasterWhisperEngine, FireRedASREngine)):
            await loop.run_in_executor(None, _get_punc_model)
        # FireRedASR 额外需要 Whisper 强制对齐
        if isinstance(engine, FireRedASREngine):
            await loop.run_in_executor(None, _get_fa_model)

        elapsed = time.time() - t0
        logger.info(f"段级模型加载完成: {elapsed:.1f}s (ASR={model_name})")
        await send_json(ws, {
            "type": "models_ready",
            "engine": model_name or "default",
            "load_time": round(elapsed, 2),
        })
        return engine, vad

    except Exception as e:
        logger.error(f"段级模型加载失败: {e}", exc_info=True)
        await send_error(ws, f"模型加载失败: {e}", recoverable=True)
        return None, None


async def _unload_sentence_models(ws: WebSocket) -> None:
    """卸载段级模式的 VAD + 文件 ASR 引擎 + 辅助模型"""
    try:
        loop = asyncio.get_event_loop()
        from ...services.asr import get_asr_engine as _get_engine
        # 卸载文件 ASR 单例
        from ...services import asr as _asr_mod
        if _asr_mod._engine is not None:
            await loop.run_in_executor(None, _asr_mod._engine.unload)
            _asr_mod._engine = None
            _asr_mod._engine_model = None
            logger.info("文件 ASR 引擎已卸载")

        # 卸载辅助模型
        from ...services.asr import unload_vad_model, unload_punc_model, unload_fa_model
        await loop.run_in_executor(None, unload_vad_model)
        await loop.run_in_executor(None, unload_punc_model)
        await loop.run_in_executor(None, unload_fa_model)

        await send_json(ws, {"type": "models_unloaded"})

    except Exception as e:
        logger.error(f"段级模型卸载失败: {e}", exc_info=True)
        await send_error(ws, f"模型卸载失败: {e}", recoverable=True)


async def _load_hybrid_models(
    ws: WebSocket,
    engine_type: str | None = None,
    model_name: str | None = None,
    hybrid_upgrade: str = "segment",
) -> tuple[Any, Any, "StreamingVADProcessor | None"]:
    """加载混合模式所需的全部模型：流式引擎 + VAD + 文件 ASR

    Returns:
        (streaming_engine, file_asr_engine, vad_processor)
    """
    t0 = time.time()
    loop = asyncio.get_event_loop()
    try:
        # 1. 流式引擎
        await send_json(ws, {"type": "status", "message": "正在加载流式引擎..."})
        streaming = get_streaming_asr_engine(engine_type)
        if not streaming.is_loaded():
            await loop.run_in_executor(None, streaming.load)

        file_engine = None
        vad = None

        # 2. full 模式只需流式引擎；段级还需 VAD + 文件 ASR
        if hybrid_upgrade != "full":
            await send_json(ws, {"type": "status", "message": "正在加载 VAD 模型..."})
            vad = StreamingVADProcessor()
            await loop.run_in_executor(None, vad.load)

            await send_json(ws, {"type": "status", "message": "正在加载文件 ASR 模型..."})
            from ...services.asr import get_asr_engine
            file_engine = await loop.run_in_executor(None, get_asr_engine, model_name)

            # 预加载辅助模型
            await send_json(ws, {"type": "status", "message": "正在加载辅助模型..."})
            from ...services.asr import (
                _get_vad_model, _get_punc_model, _get_fa_model,
                FasterWhisperEngine, FireRedASREngine,
            )
            await loop.run_in_executor(None, _get_vad_model)
            # Whisper / FireRedASR 短片段无标点 → 需要 ct-punc
            if isinstance(file_engine, (FasterWhisperEngine, FireRedASREngine)):
                await loop.run_in_executor(None, _get_punc_model)
                await loop.run_in_executor(None, _get_fa_model)

        elapsed = time.time() - t0
        logger.info(f"混合模型加载完成: {elapsed:.1f}s (streaming={engine_type}, file_asr={model_name}, upgrade={hybrid_upgrade})")
        await send_json(ws, {
            "type": "models_ready",
            "engine": engine_type or "default",
            "load_time": round(elapsed, 2),
        })
        return streaming, file_engine, vad

    except Exception as e:
        logger.error(f"混合模型加载失败: {e}", exc_info=True)
        await send_error(ws, f"模型加载失败: {e}", recoverable=True)
        return None, None, None


async def _unload_hybrid_models(ws: WebSocket, engine_type: str | None = None) -> None:
    """卸载混合模式的全部模型：流式引擎 + VAD + 文件 ASR"""
    loop = asyncio.get_event_loop()
    try:
        # 卸载流式引擎
        engine = get_streaming_asr_engine(engine_type)
        if engine.is_loaded():
            await loop.run_in_executor(None, engine.unload)
            logger.info("流式引擎已卸载")
    except Exception as e:
        logger.error(f"流式引擎卸载失败: {e}")

    try:
        # 卸载文件 ASR + VAD + 辅助模型
        from ...services import asr as _asr_mod
        if _asr_mod._engine is not None:
            await loop.run_in_executor(None, _asr_mod._engine.unload)
            _asr_mod._engine = None
            _asr_mod._engine_model = None
            logger.info("文件 ASR 引擎已卸载")

        from ...services.asr import unload_vad_model, unload_punc_model, unload_fa_model
        await loop.run_in_executor(None, unload_vad_model)
        await loop.run_in_executor(None, unload_punc_model)
        await loop.run_in_executor(None, unload_fa_model)
    except Exception as e:
        logger.error(f"混合模型卸载失败: {e}", exc_info=True)

    await send_json(ws, {"type": "models_unloaded"})


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/api/ws/realtime")
async def realtime_websocket(ws: WebSocket):
    """实时流式录音 WebSocket 端点"""
    await ws.accept()
    logger.info("WebSocket 连接已建立")

    await send_json(ws, {"type": "connected"})

    # 会话状态
    session = None
    wav_writer: IncrementalWavWriter | None = None
    asr_engine = None
    streaming_segments: list[StreamingSegment] = []
    recording = False
    config: dict = {}
    audio_queue: asyncio.Queue | None = None
    processor_task: asyncio.Task | None = None
    load_task: asyncio.Task | None = None
    chunks_received = 0
    recording_mode = "streaming"  # streaming / segment / hybrid
    vad_processor: StreamingVADProcessor | None = None
    file_asr_engine: Any = None  # 文件 ASR 引擎（段级/混合模式）
    denoiser = None  # StreamingDenoiser 实例（实时降噪）
    denoised_wav_writer: IncrementalWavWriter | None = None

    try:
        while True:
            message = await ws.receive()

            # 检查 WebSocket 断连消息
            if message.get("type") == "websocket.disconnect":
                logger.info("收到 WebSocket 断连消息")
                break

            # Binary frame: PCM 音频数据
            if "bytes" in message and message["bytes"] is not None:
                if not recording or audio_queue is None:
                    continue

                pcm_data: bytes = message["bytes"]
                chunks_received += 1

                # 写入原始 WAV 文件（极快，不阻塞）
                if wav_writer:
                    wav_writer.write_chunk(pcm_data)

                # 降噪处理 + 降噪 WAV 写入
                if denoiser is not None:
                    pcm_for_asr = denoiser.process_chunk(pcm_data)
                    if denoised_wav_writer is not None:
                        denoised_wav_writer.write_chunk(pcm_for_asr)
                else:
                    pcm_for_asr = pcm_data

                # 放入队列，由 processor 后台任务处理
                try:
                    audio_queue.put_nowait(pcm_for_asr)
                except asyncio.QueueFull:
                    logger.warning(
                        f"Audio queue 已满 ({_AUDIO_QUEUE_MAX}), 丢弃 chunk"
                    )

            # Text frame: JSON 控制消息
            elif "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await send_error(ws, "无效的 JSON 消息")
                    continue

                msg_type = data.get("type", "")

                # ── preload_models ──
                if msg_type == "preload_models":
                    preload_config = data.get("config", {})
                    preload_mode = preload_config.get("mode", "streaming")
                    # 取消正在进行的加载任务
                    if load_task and not load_task.done():
                        load_task.cancel()
                        try:
                            await load_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    if preload_mode == "segment":
                        model_name = preload_config.get("sentence_asr_model")

                        async def _load_and_store():
                            nonlocal file_asr_engine, vad_processor
                            eng, vad = await _load_sentence_models(ws, model_name)
                            file_asr_engine = eng
                            vad_processor = vad

                        load_task = asyncio.create_task(_load_and_store())
                    elif preload_mode == "hybrid":
                        h_engine_type = preload_config.get("asr_engine")
                        h_model_name = preload_config.get("sentence_asr_model")
                        h_upgrade = preload_config.get("hybrid_upgrade", "segment")

                        async def _load_hybrid_and_store():
                            nonlocal asr_engine, file_asr_engine, vad_processor
                            s_eng, f_eng, vad = await _load_hybrid_models(
                                ws, h_engine_type, h_model_name, h_upgrade,
                            )
                            if s_eng is not None:
                                asr_engine = s_eng
                            if f_eng is not None:
                                file_asr_engine = f_eng
                            if vad is not None:
                                vad_processor = vad

                        load_task = asyncio.create_task(_load_hybrid_and_store())
                    else:
                        engine_type = preload_config.get("asr_engine")
                        load_task = asyncio.create_task(
                            _load_models(ws, engine_type)
                        )

                # ── unload_models ──
                elif msg_type == "unload_models":
                    # 取消正在进行的加载任务
                    if load_task and not load_task.done():
                        load_task.cancel()
                        try:
                            await load_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        load_task = None
                    unload_config = data.get("config", {})
                    unload_mode = unload_config.get("mode", "streaming")
                    if unload_mode == "segment":
                        await _unload_sentence_models(ws)
                        file_asr_engine = None
                        vad_processor = None
                    elif unload_mode == "hybrid":
                        engine_type = unload_config.get("asr_engine")
                        await _unload_hybrid_models(ws, engine_type)
                        asr_engine = None
                        file_asr_engine = None
                        vad_processor = None
                    else:
                        engine_type = unload_config.get("asr_engine")
                        await _unload_models(ws, engine_type)
                        asr_engine = None

                # ── start_recording ──
                elif msg_type == "start_recording":
                    if recording:
                        await send_error(ws, "已在录音中")
                        continue

                    config = data.get("config", {})
                    recording_mode = config.get("mode", "streaming")
                    if recording_mode not in RECORDING_MODES:
                        recording_mode = "streaming"
                    sentence_asr_model = config.get("sentence_asr_model")

                    logger.info(
                        f"开始录音, mode={recording_mode}, config: {config}"
                    )

                    try:
                        engine_type = config.get("asr_engine", None)
                        loop = asyncio.get_event_loop()

                        # ── 模式初始化 ──
                        if recording_mode == "streaming":
                            # 现有逻辑：加载流式 ASR 引擎
                            if load_task and not load_task.done():
                                logger.info("等待模型加载完成...")
                                await send_json(ws, {
                                    "type": "status",
                                    "message": "等待模型加载完成...",
                                })
                                loaded = await load_task
                                if loaded is not None:
                                    asr_engine = loaded
                            elif load_task and load_task.done():
                                try:
                                    loaded = load_task.result()
                                    if loaded is not None:
                                        asr_engine = loaded
                                except Exception:
                                    pass

                            if asr_engine is None or not asr_engine.is_loaded():
                                asr_engine = get_streaming_asr_engine(engine_type)
                                if not asr_engine.is_loaded():
                                    await send_json(ws, {
                                        "type": "status",
                                        "message": "正在加载 ASR 模型...",
                                    })
                                    await loop.run_in_executor(
                                        None, asr_engine.load
                                    )

                            session = asr_engine.create_session()

                        elif recording_mode == "segment":
                            # 段级模式：复用预加载的引擎，否则按需加载
                            if vad_processor is None:
                                await send_json(ws, {
                                    "type": "status",
                                    "message": "正在加载 VAD 模型...",
                                })
                                vad_processor = StreamingVADProcessor()
                                await loop.run_in_executor(None, vad_processor.load)

                            if file_asr_engine is None:
                                await send_json(ws, {
                                    "type": "status",
                                    "message": "正在加载文件 ASR 模型...",
                                })
                                from ...services.asr import get_asr_engine
                                file_asr_engine = await loop.run_in_executor(
                                    None, get_asr_engine, sentence_asr_model
                                )

                            # 创建 dummy session for session_id
                            session_id = f"segment_{int(time.time())}"
                            from ...services.streaming_asr import StreamingSession
                            session = StreamingSession(session_id=session_id)

                        elif recording_mode == "hybrid":
                            # 混合模式：流式引擎 + VAD + 文件 ASR
                            hybrid_upgrade = config.get("hybrid_upgrade", "segment")

                            # 等待预加载任务完成（如有）
                            if load_task and not load_task.done():
                                await send_json(ws, {
                                    "type": "status",
                                    "message": "等待模型加载完成...",
                                })
                                await load_task

                            # 流式引擎：若未预加载则现场加载
                            if asr_engine is None or not asr_engine.is_loaded():
                                await send_json(ws, {
                                    "type": "status",
                                    "message": "正在加载流式引擎...",
                                })
                                asr_engine = get_streaming_asr_engine(engine_type)
                                if not asr_engine.is_loaded():
                                    await loop.run_in_executor(
                                        None, asr_engine.load
                                    )

                            session = asr_engine.create_session()

                            # full 模式只需流式引擎；段级还需 VAD + 文件 ASR
                            if hybrid_upgrade != "full":
                                if vad_processor is None:
                                    await send_json(ws, {
                                        "type": "status",
                                        "message": "正在加载 VAD 模型...",
                                    })
                                    vad_processor = StreamingVADProcessor()
                                    await loop.run_in_executor(None, vad_processor.load)

                                if file_asr_engine is None:
                                    await send_json(ws, {
                                        "type": "status",
                                        "message": "正在加载文件 ASR 模型...",
                                    })
                                    from ...services.asr import get_asr_engine
                                    file_asr_engine = await loop.run_in_executor(
                                        None, get_asr_engine, sentence_asr_model
                                    )

                        # ── 创建 WAV 文件 ──
                        settings = get_settings()
                        meeting_name = config.get("meeting_name", "realtime")
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_dir = settings.paths.output_dir / f"{meeting_name}_{timestamp}"
                        output_dir.mkdir(parents=True, exist_ok=True)

                        wav_path = output_dir / "recording.wav"
                        wav_writer = IncrementalWavWriter(
                            wav_path,
                            sample_rate=settings.streaming.sample_rate,
                        )

                        # ── 实时降噪初始化 ──
                        enable_denoise = config.get("enable_denoise", False)
                        if enable_denoise:
                            try:
                                from ...utils.enhance import (
                                    StreamingDenoiser,
                                    load_streaming_denoiser,
                                )
                                await loop.run_in_executor(
                                    None, load_streaming_denoiser
                                )
                                denoiser = StreamingDenoiser(
                                    input_sr=settings.streaming.sample_rate,
                                )
                                denoised_wav_writer = IncrementalWavWriter(
                                    output_dir / "audio_enhanced.wav",
                                    sample_rate=settings.streaming.sample_rate,
                                )
                                logger.info("实时降噪已启用 (DeepFilterNet3)")
                            except Exception as e:
                                logger.warning(f"降噪初始化失败: {e}")
                                denoiser = None
                                denoised_wav_writer = None

                        streaming_segments = []
                        chunks_received = 0
                        recording = True

                        # ── 启动 processor 后台任务 ──
                        audio_queue = asyncio.Queue(maxsize=_AUDIO_QUEUE_MAX)

                        if recording_mode == "streaming":
                            processor_task = asyncio.create_task(
                                _audio_processor(
                                    ws, audio_queue,
                                    asr_engine, session,
                                    streaming_segments,
                                )
                            )
                        elif recording_mode == "segment":
                            processor_task = asyncio.create_task(
                                _segment_processor(
                                    ws, audio_queue,
                                    vad_processor, file_asr_engine,
                                    streaming_segments,
                                )
                            )
                        elif recording_mode == "hybrid":
                            hybrid_upgrade = config.get("hybrid_upgrade", "segment")
                            if hybrid_upgrade == "full":
                                # 整体模式：只跑流式 ASR（纯预览）
                                processor_task = asyncio.create_task(
                                    _audio_processor(
                                        ws, audio_queue,
                                        asr_engine, session,
                                        streaming_segments,
                                    )
                                )
                            else:
                                # 段级：流式引擎完成一段后用文件 ASR 升级
                                processor_task = asyncio.create_task(
                                    _hybrid_processor(
                                        ws, audio_queue,
                                        asr_engine, session,
                                        vad_processor, file_asr_engine,
                                        streaming_segments,
                                    )
                                )

                        await send_json(ws, {
                            "type": "recording_started",
                            "session_id": session.session_id,
                            "mode": recording_mode,
                        })
                        logger.info(
                            f"录音已开始: {session.session_id} "
                            f"(mode={recording_mode})"
                        )

                    except Exception as e:
                        logger.error(f"启动录音失败: {e}", exc_info=True)
                        await send_error(ws, f"启动录音失败: {e}", recoverable=False)

                # ── stop_recording ──
                elif msg_type == "stop_recording":
                    if not recording or session is None:
                        await send_error(ws, "当前未在录音")
                        continue

                    logger.info(
                        f"停止录音... (mode={recording_mode}, "
                        f"共接收 {chunks_received} 个 PCM chunks)"
                    )
                    recording = False

                    # 发送停止信号给 processor，等待处理完队列中剩余数据
                    if audio_queue is not None:
                        await audio_queue.put(None)  # sentinel
                    if processor_task is not None:
                        await processor_task
                        processor_task = None
                    audio_queue = None

                    # 流式模式：发送 is_final=True 获取最后的识别结果
                    # 混合模式不需要：VAD flush + 文件 ASR 升级已覆盖全部内容
                    # 流式引擎的残余缓冲会产生重复文本和零时长片段
                    if recording_mode == "streaming":
                        if asr_engine and session:
                            try:
                                loop = asyncio.get_event_loop()
                                final_results = await loop.run_in_executor(
                                    None,
                                    asr_engine.feed_chunk,
                                    session,
                                    b"",
                                    True,
                                )
                                for seg, is_seg_final in final_results:
                                    if is_seg_final:
                                        streaming_segments.append(seg)
                                    await send_json(ws, {
                                        "type": "partial",
                                        "text": seg.text,
                                        "is_final": True,
                                        "segment_id": seg.id,
                                        "start_time": seg.start,
                                        "end_time": seg.end,
                                    })
                            except Exception as e:
                                logger.error(f"最终 ASR 推理错误: {e}")

                    # 关闭 WAV 文件
                    duration = 0.0
                    wav_path = None
                    if wav_writer:
                        wav_path = wav_writer.finalize()
                        duration = wav_writer.duration_seconds

                    # 关闭降噪 WAV + 重置降噪器
                    if denoised_wav_writer is not None:
                        denoised_wav_writer.finalize()
                        denoised_wav_writer = None
                    if denoiser is not None:
                        denoiser.reset()
                        denoiser = None

                    await send_json(ws, {
                        "type": "recording_stopped",
                        "duration": duration,
                        "segment_count": len(streaming_segments),
                    })
                    logger.info(
                        f"录音已停止: {duration:.1f}s, "
                        f"{len(streaming_segments)} segments"
                    )

                    # 收集需要在后处理中卸载的引擎
                    engines_to_unload = []
                    if asr_engine:
                        engines_to_unload.append(asr_engine)
                    if file_asr_engine:
                        engines_to_unload.append(file_asr_engine)
                    if vad_processor:
                        engines_to_unload.append(vad_processor)

                    # 录音过短（< 3s）：删除输出目录，不保存历史
                    if duration < 3.0:
                        import shutil
                        output_dir = wav_path.parent if wav_path else None
                        if output_dir and output_dir.exists():
                            try:
                                shutil.rmtree(output_dir)
                                logger.info(f"录音过短 ({duration:.1f}s < 3s)，已删除: {output_dir.name}")
                            except Exception as e:
                                logger.warning(f"删除短录音目录失败: {e}")
                        loop = asyncio.get_event_loop()
                        for eng in engines_to_unload:
                            try:
                                await loop.run_in_executor(None, eng.unload)
                            except Exception:
                                pass
                        await send_json(ws, {"type": "models_unloaded"})
                        await send_json(ws, {
                            "type": "final_result",
                            "result": None,
                            "history_id": None,
                            "message": f"录音过短 ({duration:.1f}s)，已自动丢弃",
                        })

                    # 启动后处理
                    elif wav_path and wav_path.exists() and streaming_segments:
                        asyncio.create_task(
                            _post_process_recording(
                                ws,
                                wav_path,
                                streaming_segments,
                                config,
                                engines_to_unload,
                            )
                        )
                    else:
                        # 没有录到有效内容（>= 3s 但无片段），保留目录让用户决定
                        loop = asyncio.get_event_loop()
                        for eng in engines_to_unload:
                            try:
                                await loop.run_in_executor(None, eng.unload)
                            except Exception:
                                pass
                        await send_json(ws, {"type": "models_unloaded"})
                        await send_json(ws, {
                            "type": "final_result",
                            "result": None,
                            "history_id": wav_path.parent.name if wav_path else None,
                            "message": "未检测到语音内容",
                        })

                    # 重置模式状态
                    vad_processor = None
                    file_asr_engine = None

                # ── unknown ──
                else:
                    await send_error(ws, f"未知消息类型: {msg_type}")

    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}", exc_info=True)
    finally:
        # 清理 load 任务
        if load_task is not None and not load_task.done():
            load_task.cancel()
            try:
                await load_task
            except (asyncio.CancelledError, Exception):
                pass
        # 清理 processor 任务
        if processor_task is not None and not processor_task.done():
            processor_task.cancel()
            try:
                await processor_task
            except (asyncio.CancelledError, Exception):
                pass
        # 清理 VAD processor
        if vad_processor is not None:
            try:
                vad_processor.unload()
            except Exception:
                pass
        # 清理降噪 WAV writer
        if denoised_wav_writer is not None:
            try:
                denoised_wav_writer.finalize()
            except Exception:
                pass
        # 清理 WAV writer
        if wav_writer and not wav_writer._closed:
            try:
                wav_writer.finalize()
            except Exception:
                wav_writer.close()
        recording = False


# ---------------------------------------------------------------------------
# 后处理管线
# ---------------------------------------------------------------------------

async def _post_process_recording(
    ws: WebSocket,
    wav_path: Path,
    streaming_segments: list[StreamingSegment],
    config: dict,
    engines_to_unload: list[Any],
) -> None:
    """
    录音结束后执行后处理管线:
    1. 卸载所有 ASR/VAD 引擎（释放显存）
    2. 说话人分离 (pyannote)
    3. 对齐流式片段与说话人
    4. LLM 管线（校正 + 命名 + 总结）
    5. 保存结果
    """
    loop = asyncio.get_event_loop()
    output_dir = wav_path.parent
    enable_naming = config.get("enable_naming", True)
    enable_correction = config.get("enable_correction", True)
    enable_summary = config.get("enable_summary", True)
    diarization_model = config.get("diarization_model", None)
    gender_model = config.get("gender_model", None)
    llm_model = config.get("llm_model", None)

    try:
        # ── 配置 LLM 模型（与 process.py 一致） ──
        settings = get_settings()
        if llm_model and llm_model != "disabled":
            settings.llm.enabled = True
            llm_model_name = llm_model
            if not llm_model_name.endswith(".gguf"):
                llm_model_name = f"{llm_model_name}.gguf"
            if not llm_model_name.startswith("llm/") and not llm_model_name.startswith("llm\\"):
                llm_model_name = f"llm/{llm_model_name}"
            settings.llm.model_path = Path(llm_model_name)
            from ...services.llm import reset_llm
            reset_llm()
            logger.info(f"LLM 模型已切换: {llm_model}")
        elif llm_model == "disabled":
            settings.llm.enabled = False

        # ── Step 0: 卸载所有 ASR/VAD 引擎，释放 GPU 显存 ──
        await send_json(ws, {
            "type": "post_progress",
            "step": "unload_asr",
            "progress": 0,
            "overall_progress": 0.0,
            "message": "释放 ASR 引擎显存...",
        })
        for eng in engines_to_unload:
            try:
                await loop.run_in_executor(None, eng.unload)
            except Exception as e:
                logger.warning(f"引擎卸载失败: {e}")
        logger.info(f"已卸载 {len(engines_to_unload)} 个引擎")

        # 通知前端模型已卸载
        await send_json(ws, {"type": "models_unloaded"})

        # ── Step 1: 说话人分离 (40%) ──
        from ...models import CharTimestamp, Segment, TranscriptResult
        from ...services.alignment import (
            align_transcript_with_speakers,
            fix_unknown_speakers,
            merge_adjacent_segments,
        )

        # 估算音频时长
        duration = wav_path.stat().st_size / (16000 * 2)  # 16kHz 16bit mono

        # -- 构建基础片段（无论后续步骤是否成功，至少有这些） --
        valid_segments = [
            seg for seg in streaming_segments
            if seg.end > seg.start and seg.text.strip()
        ]
        if len(valid_segments) < len(streaming_segments):
            dropped = len(streaming_segments) - len(valid_segments)
            logger.warning(f"过滤 {dropped} 个无效片段（零时长或空文本）")

        # 回退用基础 Segment（万一说话人分离/对齐失败时使用）
        fallback_segments = [
            Segment(
                id=i, start=seg.start, end=seg.end,
                text=seg.text, speaker="SPEAKER_00",
            )
            for i, seg in enumerate(valid_segments)
        ]

        merged = fallback_segments  # 默认值，后续步骤成功时覆盖
        diar_result = None
        speakers_info: dict = {}
        summary_text = ""

        # -- 说话人分离 --
        try:
            await send_json(ws, {
                "type": "post_progress",
                "step": "diarization",
                "progress": 0,
                "overall_progress": 0.05,
                "message": "说话人分离中...",
            })

            from ...services.diarization import get_diarization_service
            diar_service = get_diarization_service(diarization_model)

            diar_result = await loop.run_in_executor(
                None, diar_service.diarize, str(wav_path)
            )
            logger.info(f"说话人分离完成: {diar_result.speaker_count} 位说话人")

            await send_json(ws, {
                "type": "post_progress",
                "step": "diarization",
                "progress": 1.0,
                "overall_progress": 0.4,
                "message": f"说话人分离完成: {diar_result.speaker_count} 位说话人",
            })
        except Exception as e:
            logger.error(f"说话人分离失败（使用回退）: {e}", exc_info=True)
            await send_json(ws, {
                "type": "post_progress",
                "step": "diarization",
                "progress": 1.0,
                "overall_progress": 0.4,
                "message": f"说话人分离失败: {e}",
            })

        # ── Step 2: 对齐流式片段与说话人 (50%) ──
        if diar_result is not None:
            try:
                await send_json(ws, {
                    "type": "post_progress",
                    "step": "alignment",
                    "progress": 0,
                    "overall_progress": 0.45,
                    "message": "对齐文本与说话人...",
                })

                hybrid_upgrade = config.get("hybrid_upgrade", "segment")
                sentence_asr_model = config.get("sentence_asr_model")

                if hybrid_upgrade == "full" and sentence_asr_model:
                    # ── 整体重新转写（替代流式片段） ──
                    await send_json(ws, {
                        "type": "post_progress",
                        "step": "retranscribe",
                        "progress": 0,
                        "overall_progress": 0.42,
                        "message": "整体重新转写中...",
                    })
                    from ...services.asr import get_asr_engine
                    file_engine = await loop.run_in_executor(
                        None, get_asr_engine, sentence_asr_model
                    )
                    transcript_result = await loop.run_in_executor(
                        None, file_engine.transcribe, str(wav_path)
                    )
                    await loop.run_in_executor(None, file_engine.unload)
                    logger.info(
                        f"整体重新转写完成: {len(transcript_result.segments)} 段, "
                        f"字级时间戳={'有' if transcript_result.char_timestamps else '无'}"
                    )
                else:
                    # 段级：将流式片段转换为标准 Segment + 提取字级时间戳
                    asr_segments = []
                    all_char_ts: list[list[CharTimestamp]] = []
                    has_any_ts = False
                    for seg in valid_segments:
                        asr_segments.append(Segment(
                            id=seg.id,
                            start=seg.start,
                            end=seg.end,
                            text=seg.text,
                            speaker=None,
                        ))
                        if seg.char_timestamps:
                            all_char_ts.append(seg.char_timestamps)
                            has_any_ts = True
                        else:
                            all_char_ts.append([])

                    transcript_result = TranscriptResult(
                        language="zh",
                        language_probability=0.95,
                        duration=duration,
                        segments=asr_segments,
                        char_timestamps=all_char_ts if has_any_ts else None,
                    )

                    logger.info(
                        f"流式片段 → TranscriptResult: {len(asr_segments)} 段, "
                        f"字级时间戳={'有' if has_any_ts else '无'}"
                    )

                aligned = await loop.run_in_executor(
                    None,
                    align_transcript_with_speakers,
                    transcript_result,
                    diar_result,
                )
                fixed = fix_unknown_speakers(aligned)
                merged = merge_adjacent_segments(fixed, max_gap=0.3)

                await send_json(ws, {
                    "type": "post_progress",
                    "step": "alignment",
                    "progress": 1.0,
                    "overall_progress": 0.5,
                    "message": f"对齐完成: {len(merged)} 个片段",
                })
            except Exception as e:
                logger.error(f"对齐失败（使用原始片段）: {e}", exc_info=True)
                # merged 保持 fallback_segments

        # ── Step 3: LLM 校正 (60%) ──
        if enable_correction:
            try:
                await send_json(ws, {
                    "type": "post_progress",
                    "step": "correction",
                    "progress": 0,
                    "overall_progress": 0.55,
                    "message": "校正转写文本...",
                })

                from ...services.correction import correct_segments
                merged = await loop.run_in_executor(None, correct_segments, merged)

                await send_json(ws, {
                    "type": "post_progress",
                    "step": "correction",
                    "progress": 1.0,
                    "overall_progress": 0.65,
                    "message": "文本校正完成",
                })
            except Exception as e:
                logger.error(f"校正失败（跳过）: {e}", exc_info=True)

        # ── Step 4a: 性别检测（始终运行，作为命名兜底） ──
        gender_map = {}
        if diar_result is not None:
            try:
                await send_json(ws, {
                    "type": "post_progress",
                    "step": "naming",
                    "progress": 0,
                    "overall_progress": 0.7,
                    "message": "检测说话人性别...",
                })

                from ...services.gender import detect_all_genders
                from functools import partial
                gender_map = await loop.run_in_executor(
                    None, partial(detect_all_genders, str(wav_path), diar_result.segments, engine_name=gender_model)
                )
            except Exception as e:
                logger.warning(f"性别检测失败（回退空性别信息）: {e}")

        # ── Step 4b: 智能命名（仅在启用时运行 LLM 推断） ──
        if enable_naming and diar_result is not None:
            try:
                await send_json(ws, {
                    "type": "post_progress",
                    "step": "naming",
                    "progress": 0.5,
                    "overall_progress": 0.75,
                    "message": "智能命名中...",
                })

                from ...services.naming import get_naming_service
                naming_service = get_naming_service()
                speakers_info = await loop.run_in_executor(
                    None, naming_service.name_speakers, merged, gender_map
                )

                await send_json(ws, {
                    "type": "post_progress",
                    "step": "naming",
                    "progress": 1.0,
                    "overall_progress": 0.85,
                    "message": "命名完成",
                })
            except Exception as e:
                logger.error(f"智能命名失败（跳过）: {e}", exc_info=True)

        # ── Step 4c: 性别兜底命名（智能命名未启用或失败时） ──
        if not speakers_info and diar_result is not None:
            from ...services.gender import Gender
            from ...models import SpeakerInfo
            speaker_ids = sorted(set(seg.speaker for seg in merged if seg.speaker))
            gender_counters = {"male": 0, "female": 0, "unknown": 0}
            for spk_id in speaker_ids:
                total_dur = sum(seg.duration for seg in merged if seg.speaker == spk_id)
                seg_count = sum(1 for seg in merged if seg.speaker == spk_id)
                gender, _ = gender_map.get(spk_id, (Gender.UNKNOWN, 0.0))
                gender_counters[gender.value] = gender_counters.get(gender.value, 0) + 1
                count = gender_counters[gender.value]
                if gender == Gender.MALE:
                    display_name = f"男性{count:02d}"
                elif gender == Gender.FEMALE:
                    display_name = f"女性{count:02d}"
                else:
                    display_name = f"说话人{count:02d}"
                speakers_info[spk_id] = SpeakerInfo(
                    id=spk_id,
                    display_name=display_name,
                    gender=gender,
                    total_duration=total_dur,
                    segment_count=seg_count,
                )

        # ── Step 5: 会议总结 (95%) ──
        if enable_summary:
            try:
                await send_json(ws, {
                    "type": "post_progress",
                    "step": "summary",
                    "progress": 0,
                    "overall_progress": 0.88,
                    "message": "生成会议总结...",
                })

                from ...services.summary import summarize_meeting, format_summary_markdown
                summary_result = await loop.run_in_executor(
                    None, summarize_meeting, merged, speakers_info, duration
                )
                if summary_result is not None:
                    summary_text = format_summary_markdown(
                        summary_result, speakers_info, duration
                    )

                await send_json(ws, {
                    "type": "post_progress",
                    "step": "summary",
                    "progress": 1.0,
                    "overall_progress": 0.95,
                    "message": "总结完成",
                })
            except Exception as e:
                logger.error(f"总结失败（跳过）: {e}", exc_info=True)

        # ── Step 6: 保存结果（始终执行） ──
        await send_json(ws, {
            "type": "post_progress",
            "step": "saving",
            "progress": 0,
            "overall_progress": 0.97,
            "message": "保存结果...",
        })

        import json

        segments_data = [s.model_dump() for s in merged]
        speakers_data = {k: v.model_dump() for k, v in speakers_info.items()} if speakers_info else {}

        result_data = {
            "segments": segments_data,
            "speakers": speakers_data,
            "speaker_count": len(speakers_info) if speakers_info else 0,
            "summary": summary_text,
            "duration": duration,
        }

        # 保存 JSON
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 保存 TXT
        txt_path = output_dir / "result.txt"
        txt_lines = []
        for seg in segments_data:
            sid = seg.get("speaker", "UNKNOWN")
            name = speakers_data.get(sid, {}).get("display_name", sid)
            txt_lines.append(f"[{_fmt_time(seg.get('start', 0))}] {name}: {seg.get('text', '')}")
        txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

        # 保存总结
        if summary_text:
            summary_path = output_dir / "summary.md"
            summary_path.write_text(summary_text, encoding="utf-8")

        history_id = output_dir.name
        logger.info(f"结果已保存: {output_dir}")

        # ── 完成 ──
        await send_json(ws, {
            "type": "post_progress",
            "step": "done",
            "progress": 1.0,
            "overall_progress": 1.0,
            "message": "处理完成",
        })

        await send_json(ws, {
            "type": "final_result",
            "result": result_data,
            "history_id": history_id,
        })

    except Exception as e:
        logger.error(f"后处理致命错误: {e}", exc_info=True)
        await send_error(ws, f"后处理错误: {e}", recoverable=False)


def _fmt_time(seconds: float) -> str:
    """格式化时间 MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
