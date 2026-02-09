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
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...config import get_settings
from ...logger import get_logger
from ...models import StreamingSegment
from ...services.streaming_asr import (
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

async def send_json(ws: WebSocket, data: dict) -> bool:
    """安全发送 JSON 消息，返回是否成功"""
    try:
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
# Audio processor (consumer task)
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

                # 写入 WAV 文件（极快，不阻塞）
                if wav_writer:
                    wav_writer.write_chunk(pcm_data)

                # 放入队列，由 processor 后台任务处理
                try:
                    audio_queue.put_nowait(pcm_data)
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
                    engine_type = data.get("config", {}).get("asr_engine")
                    # 取消正在进行的加载任务
                    if load_task and not load_task.done():
                        load_task.cancel()
                        try:
                            await load_task
                        except (asyncio.CancelledError, Exception):
                            pass
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
                    engine_type = data.get("config", {}).get("asr_engine")
                    await _unload_models(ws, engine_type)
                    asr_engine = None

                # ── start_recording ──
                elif msg_type == "start_recording":
                    if recording:
                        await send_error(ws, "已在录音中")
                        continue

                    config = data.get("config", {})
                    logger.info(f"开始录音, config: {config}")

                    try:
                        engine_type = config.get("asr_engine", None)

                        # 等待加载任务完成（如果还在进行中）
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

                        # 如果没有已加载的引擎，尝试加载
                        if asr_engine is None or not asr_engine.is_loaded():
                            asr_engine = get_streaming_asr_engine(engine_type)
                            if not asr_engine.is_loaded():
                                await send_json(ws, {
                                    "type": "status",
                                    "message": "正在加载 ASR 模型...",
                                })
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, asr_engine.load)

                        session = asr_engine.create_session()

                        # 创建 WAV 文件
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

                        streaming_segments = []
                        chunks_received = 0
                        recording = True

                        # 启动 ASR processor 后台任务
                        audio_queue = asyncio.Queue(maxsize=_AUDIO_QUEUE_MAX)
                        processor_task = asyncio.create_task(
                            _audio_processor(
                                ws,
                                audio_queue,
                                asr_engine,
                                session,
                                streaming_segments,
                            )
                        )

                        await send_json(ws, {
                            "type": "recording_started",
                            "session_id": session.session_id,
                        })
                        logger.info(f"录音已开始: {session.session_id}")

                    except Exception as e:
                        logger.error(f"启动录音失败: {e}")
                        await send_error(ws, f"启动录音失败: {e}", recoverable=False)

                # ── stop_recording ──
                elif msg_type == "stop_recording":
                    if not recording or session is None:
                        await send_error(ws, "当前未在录音")
                        continue

                    logger.info(
                        f"停止录音... (共接收 {chunks_received} 个 PCM chunks)"
                    )
                    recording = False

                    # 发送停止信号给 processor，等待处理完队列中剩余数据
                    if audio_queue is not None:
                        await audio_queue.put(None)  # sentinel
                    if processor_task is not None:
                        await processor_task
                        processor_task = None
                    audio_queue = None

                    # 发送 is_final=True 获取最后的识别结果
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

                    await send_json(ws, {
                        "type": "recording_stopped",
                        "duration": duration,
                        "segment_count": len(streaming_segments),
                    })
                    logger.info(
                        f"录音已停止: {duration:.1f}s, "
                        f"{len(streaming_segments)} segments"
                    )

                    # 启动后处理
                    if wav_path and wav_path.exists() and streaming_segments:
                        asyncio.create_task(
                            _post_process_recording(
                                ws,
                                wav_path,
                                streaming_segments,
                                config,
                                asr_engine,
                            )
                        )
                    else:
                        # 没有录到有效内容
                        await send_json(ws, {
                            "type": "final_result",
                            "result": None,
                            "history_id": None,
                            "message": "未检测到语音内容",
                        })

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
    asr_engine: Any,
) -> None:
    """
    录音结束后执行后处理管线:
    1. 卸载流式 ASR 引擎（释放显存）
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

    try:
        # ── Step 0: 卸载 ASR 引擎，释放 GPU 显存 ──
        await send_json(ws, {
            "type": "post_progress",
            "step": "unload_asr",
            "progress": 0,
            "overall_progress": 0.0,
            "message": "释放 ASR 引擎显存...",
        })
        if asr_engine:
            await loop.run_in_executor(None, asr_engine.unload)
        logger.info("ASR 引擎已卸载")

        # 通知前端模型已卸载
        await send_json(ws, {"type": "models_unloaded"})

        # ── Step 1: 说话人分离 (40%) ──
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

        # ── Step 2: 对齐流式片段与说话人 (50%) ──
        await send_json(ws, {
            "type": "post_progress",
            "step": "alignment",
            "progress": 0,
            "overall_progress": 0.45,
            "message": "对齐文本与说话人...",
        })

        from ...models import Segment, TranscriptResult
        from ...services.alignment import (
            align_transcript_with_speakers,
            fix_unknown_speakers,
            merge_adjacent_segments,
        )

        # 将流式片段转换为标准 Segment
        asr_segments = [
            Segment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=None,
            )
            for seg in streaming_segments
        ]

        # 估算音频时长
        duration = wav_path.stat().st_size / (16000 * 2)  # 16kHz 16bit mono

        transcript_result = TranscriptResult(
            language="zh",
            language_probability=0.95,
            duration=duration,
            segments=asr_segments,
        )

        aligned = await loop.run_in_executor(
            None,
            align_transcript_with_speakers,
            transcript_result,
            diar_result,
        )
        fixed = fix_unknown_speakers(aligned)
        # 实时录音：保留 ASR 引擎的语义分段，只合并极短间隔（<50ms）的同说话人片段
        # 文件处理用默认 max_gap=0.3s（见 process.py）
        merged = merge_adjacent_segments(fixed, max_gap=0.05)

        await send_json(ws, {
            "type": "post_progress",
            "step": "alignment",
            "progress": 1.0,
            "overall_progress": 0.5,
            "message": f"对齐完成: {len(merged)} 个片段",
        })

        # ── Step 3: LLM 校正 (60%) ──
        if enable_correction:
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

        # ── Step 4: 性别检测 + 命名 (80%) ──
        speakers_info = {}
        if enable_naming:
            await send_json(ws, {
                "type": "post_progress",
                "step": "naming",
                "progress": 0,
                "overall_progress": 0.7,
                "message": "检测说话人性别...",
            })

            from ...services.gender import detect_all_genders
            from functools import partial
            # 注意: detect_all_genders 需要 list[Segment]，不是 DiarizationResult
            gender_map = await loop.run_in_executor(
                None, partial(detect_all_genders, str(wav_path), diar_result.segments, engine_name=gender_model)
            )

            await send_json(ws, {
                "type": "post_progress",
                "step": "naming",
                "progress": 0.5,
                "overall_progress": 0.75,
                "message": "智能命名中...",
            })

            from ...services.naming import get_naming_service
            naming_service = get_naming_service()
            # name_speakers 返回 dict[str, SpeakerInfo]，不是 tuple
            speakers_info = await loop.run_in_executor(
                None, naming_service.name_speakers, merged, gender_map
            )

            # 注意: 不要修改 seg.speaker（保持原始 SPEAKER_XX ID），
            # 前端通过 speakers dict 查找 display_name，与 process.py 保持一致。

            await send_json(ws, {
                "type": "post_progress",
                "step": "naming",
                "progress": 1.0,
                "overall_progress": 0.85,
                "message": "命名完成",
            })

        # ── Step 5: 会议总结 (95%) ──
        summary_text = ""
        if enable_summary:
            await send_json(ws, {
                "type": "post_progress",
                "step": "summary",
                "progress": 0,
                "overall_progress": 0.88,
                "message": "生成会议总结...",
            })

            from ...services.summary import summarize_meeting, format_summary_markdown
            # summarize_meeting 返回 MeetingSummary 对象，用 format_summary_markdown 转 Markdown
            summary_result = await loop.run_in_executor(
                None, summarize_meeting, merged, speakers_info, duration
            )
            if summary_result is not None:
                summary_text = format_summary_markdown(
                    summary_result, speakers_info, duration
                )
            else:
                summary_text = ""

            await send_json(ws, {
                "type": "post_progress",
                "step": "summary",
                "progress": 1.0,
                "overall_progress": 0.95,
                "message": "总结完成",
            })

        # ── Step 6: 保存结果 ──
        await send_json(ws, {
            "type": "post_progress",
            "step": "saving",
            "progress": 0,
            "overall_progress": 0.97,
            "message": "保存结果...",
        })

        import json

        # 构建结果（segment.speaker 保持原始 ID，speaker_name 用于显示）
        result_data = {
            "segments": [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker or "SPEAKER_00",
                    "speaker_name": (
                        speakers_info[seg.speaker].display_name
                        if speakers_info and seg.speaker and seg.speaker in speakers_info
                        else seg.speaker or "SPEAKER_00"
                    ),
                }
                for seg in merged
            ],
            "speakers": {
                sid: {
                    "display_name": info.display_name,
                    "gender": info.gender.value if hasattr(info.gender, "value") else str(info.gender),
                    "kind": info.kind.value if hasattr(info.kind, "value") else str(info.kind),
                }
                for sid, info in speakers_info.items()
            } if speakers_info else {},
            "summary": summary_text,
            "duration": duration,
        }

        # 保存 JSON
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 保存 TXT（使用 display_name）
        txt_path = output_dir / "result.txt"
        txt_lines = []
        for seg in merged:
            sid = seg.speaker or "SPEAKER_00"
            display = speakers_info[sid].display_name if speakers_info and sid in speakers_info else sid
            txt_lines.append(f"[{_fmt_time(seg.start)}] {display}: {seg.text}")
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

    except WebSocketDisconnect:
        logger.warning("后处理中 WebSocket 断连")
    except Exception as e:
        logger.error(f"后处理错误: {e}", exc_info=True)
        await send_error(ws, f"后处理错误: {e}", recoverable=False)


def _fmt_time(seconds: float) -> str:
    """格式化时间 MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
