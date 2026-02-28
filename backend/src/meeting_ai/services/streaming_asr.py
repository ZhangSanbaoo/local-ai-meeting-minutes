"""
流式 ASR 引擎

提供抽象基类和具体实现，支持运行时切换 ASR 引擎。
已实现: FunASR Paraformer, sherpa-onnx Paraformer

每个 feed_chunk() 返回 list[tuple[StreamingSegment, bool]]，
其中 bool=True 表示该片段已完成（句子结束），bool=False 表示正在更新中（部分识别）。
前端应根据 segment_id 更新已有片段或添加新片段。

使用方式:
    engine = get_streaming_asr_engine()
    session = engine.create_session()

    for pcm_chunk in audio_stream:
        results = engine.feed_chunk(session, pcm_chunk, is_final=False)
        for seg, is_seg_final in results:
            print(f"[{'FINAL' if is_seg_final else 'partial'}] {seg.text}")

    # 最后一个 chunk
    final_results = engine.feed_chunk(session, b"", is_final=True)
    engine.unload()  # 释放显存
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from ..logger import get_logger
from ..models import CharTimestamp, StreamingSegment

logger = get_logger("services.streaming_asr")

# Return type alias: (segment, is_segment_final)
ASRResult = list[tuple[StreamingSegment, bool]]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class StreamingSession:
    """一个流式 ASR 会话的状态"""
    session_id: str
    cache: dict = field(default_factory=dict)
    segment_counter: int = 0
    total_samples: int = 0
    created_at: float = field(default_factory=time.time)
    # 当前正在累积的文本（未完成的片段）
    current_text: str = ""
    current_segment_start: float = 0.0
    # 停顿检测（时间基准，兼容 producer-consumer 批量处理）
    last_text_time: float = 0.0
    # fsmn-vad 独立状态
    vad_cache: dict = field(default_factory=dict)
    is_speech: bool = False
    vad_speech_start: float = -1.0  # VAD 检测到的语音起始时间 (秒)
    # 当前片段累积的字级时间戳 [(char, start, end), ...]
    current_char_timestamps: list = field(default_factory=list)
    # 前缀锁定（减少文字闪烁）
    locked_text: str = ""           # 已锁定的稳定前缀
    pending_text: str = ""          # 上次待确认的扩展部分
    pending_stable_count: int = 0   # 连续稳定次数
    STABILITY_THRESHOLD: int = 3    # 连续 N 次相同才锁定


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class StreamingASREngine(ABC):
    """流式 ASR 引擎抽象基类"""

    @abstractmethod
    def load(self) -> None:
        """加载模型到内存/GPU"""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        ...

    @abstractmethod
    def create_session(self) -> StreamingSession:
        """创建一个新的流式识别会话"""
        ...

    @abstractmethod
    def feed_chunk(
        self,
        session: StreamingSession,
        pcm_int16: bytes,
        is_final: bool = False,
    ) -> ASRResult:
        """
        喂入一段 PCM 音频数据

        Args:
            session: 会话状态
            pcm_int16: PCM Int16 LE 单声道 16kHz 原始音频
            is_final: 是否为最后一个 chunk

        Returns:
            list of (StreamingSegment, is_segment_final):
                is_segment_final=True  → 句子完成，可存入最终结果
                is_segment_final=False → 部分识别，前端应更新同一 segment_id
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """卸载模型，释放 GPU 显存"""
        ...


# ---------------------------------------------------------------------------
# StreamingVADProcessor — 独立段级 VAD（段级/混合模式使用）
# ---------------------------------------------------------------------------

class StreamingVADProcessor:
    """
    独立的流式 VAD 处理器，用于段级/混合模式。

    分段策略（业界标准 — Google Cloud Speech V2、AssemblyAI、Deepgram）：
    - 主策略: fsmn-vad speech_end（静默 ≥ 500-800ms）→ 一句话结束
    - 兜底: 连续说话 > MAX_SEGMENT_DURATION(8s) → 强制切段
    - 过滤: 语音段 < MIN_SEGMENT_DURATION(0.3s) → 丢弃
    """

    MAX_SEGMENT_DURATION = 8.0   # 最大段时长（秒）
    MIN_SEGMENT_DURATION = 0.3   # 最小段时长（秒）
    SAMPLE_RATE = 16000

    def __init__(self) -> None:
        self._vad_model: Any | None = None
        self._vad_cache: dict = {}
        self._is_speaking: bool = False
        self._speech_start: float = 0.0
        self._audio_buffer: list[np.ndarray] = []
        self._total_samples: int = 0
        self._chunk_appended: bool = False  # 当前 chunk 是否已加入 buffer
        self._settings = get_settings().streaming

    def load(self) -> None:
        """加载 fsmn-vad 模型"""
        if self._vad_model is not None:
            return

        from funasr import AutoModel

        settings = self._settings
        device = settings.device
        if device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        models_dir = get_settings().paths.models_dir
        vad_dir = settings.funasr_vad_dir
        if not vad_dir.is_absolute():
            rel = str(vad_dir).replace("\\", "/")
            if rel.startswith("models/"):
                rel = rel[len("models/"):]
            vad_dir = (models_dir / rel).resolve()

        if not vad_dir.exists():
            raise FileNotFoundError(f"VAD 模型不存在: {vad_dir}")

        self._vad_model = AutoModel(
            model=str(vad_dir),
            device=device,
            disable_update=True,
        )
        logger.info(f"StreamingVADProcessor: fsmn-vad 加载完成 ({vad_dir})")

    def is_loaded(self) -> bool:
        return self._vad_model is not None

    def unload(self) -> None:
        """释放模型"""
        if self._vad_model is not None:
            del self._vad_model
            self._vad_model = None
        self._vad_cache = {}
        self._audio_buffer = []
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("StreamingVADProcessor: VAD 模型已卸载")

    # 边界 padding：在 speech_start 前和 speech_end 后各保留一小段，
    # 避免 ASR 丢掉边界处的字。30ms = 480 samples @ 16kHz
    BOUNDARY_PAD_SAMPLES = 480

    def feed_chunk(self, pcm_int16: bytes) -> list[dict]:
        """
        喂入 PCM 音频块，返回完成的语音段列表。

        Args:
            pcm_int16: PCM Int16 LE 单声道 16kHz 原始音频 bytes

        Returns:
            [{"start": float, "end": float, "audio": np.ndarray(int16)}, ...]
            空列表表示无完成段
        """
        if self._vad_model is None:
            raise RuntimeError("VAD 未加载，请先调用 load()")

        audio_array = np.frombuffer(pcm_int16, dtype=np.int16)
        chunk_samples = len(audio_array)
        chunk_start_sample = self._total_samples
        chunk_start_time = chunk_start_sample / self.SAMPLE_RATE
        self._total_samples += chunk_samples
        chunk_end_time = self._total_samples / self.SAMPLE_RATE

        completed: list[dict] = []

        # 1. 调用 fsmn-vad 检测
        try:
            vad_result = self._vad_model.generate(
                input=audio_array,
                cache=self._vad_cache,
                is_final=False,
                chunk_size=200,
                max_end_silence_time=self._settings.funasr_vad_silence_ms,
            )
            # 解析 VAD 输出: [{"value": [[start_ms, end_ms], ...]}]
            # fsmn-vad 流式模式下时间戳是绝对毫秒（从首个 chunk 起算）
            segments = vad_result[0].get("value", []) if vad_result else []
        except Exception as e:
            logger.debug(f"VAD 推理错误: {e}")
            segments = []

        # 2. 解析 VAD 事件，精确切割音频边界
        for seg in segments:
            if len(seg) < 2:
                continue

            # speech_start: seg[0] != -1
            if seg[0] != -1 and not self._is_speaking:
                self._is_speaking = True
                self._speech_start = seg[0] / 1000.0
                # 精确切割：只保留 speech_start 之后的样本（含 padding）
                speech_start_sample = int(self._speech_start * self.SAMPLE_RATE)
                cut_sample = max(0, speech_start_sample - self.BOUNDARY_PAD_SAMPLES)
                rel = cut_sample - chunk_start_sample
                if rel > 0 and rel < chunk_samples:
                    self._audio_buffer = [audio_array[rel:].copy()]
                else:
                    # speech_start 在 chunk 开头或之前，保留整个 chunk
                    self._audio_buffer = [audio_array.copy()]
                self._chunk_appended = True  # 标记当前 chunk 已加入 buffer

            # speech_end: seg[1] != -1
            if seg[1] != -1 and self._is_speaking:
                speech_end_time = seg[1] / 1000.0
                duration = speech_end_time - self._speech_start
                if duration >= self.MIN_SEGMENT_DURATION:
                    # 精确切割：只保留到 speech_end 的样本（含 padding）
                    speech_end_sample = int(speech_end_time * self.SAMPLE_RATE)
                    cut_sample = min(
                        self._total_samples,
                        speech_end_sample + self.BOUNDARY_PAD_SAMPLES,
                    )
                    rel = cut_sample - chunk_start_sample
                    if not self._chunk_appended:
                        # 当前 chunk 还没加入（speech_start 和 end 在同一 chunk）
                        if 0 < rel < chunk_samples:
                            self._audio_buffer.append(audio_array[:rel].copy())
                        else:
                            self._audio_buffer.append(audio_array.copy())
                    else:
                        # 当前 chunk 已在 speech_start 时加入，需替换尾部
                        # 只有当 end 在 chunk 中间时才裁剪
                        if 0 < rel < chunk_samples and self._audio_buffer:
                            # 替换最后一块为裁剪版
                            last = self._audio_buffer[-1]
                            # last 可能是从 speech_start 裁剪过的，长度与整 chunk 不同
                            # 用 rel 相对于 chunk 头部计算需要保留多少
                            keep = min(len(last), rel)
                            self._audio_buffer[-1] = last[:keep]

                    audio = np.concatenate(self._audio_buffer)
                    completed.append({
                        "start": self._speech_start,
                        "end": speech_end_time,
                        "audio": audio,
                    })
                    self._audio_buffer = []
                else:
                    logger.debug(
                        f"VAD 丢弃短段: {duration:.2f}s < {self.MIN_SEGMENT_DURATION}s"
                    )
                    self._audio_buffer = []
                self._is_speaking = False
                self._chunk_appended = False

        # 3. 累积音频（speech 正在进行中，且当前 chunk 尚未加入 buffer）
        if self._is_speaking and not self._chunk_appended:
            self._audio_buffer.append(audio_array.copy())
        # 重置标记供下个 chunk 使用
        self._chunk_appended = False

        # 4. 最大时长兜底
        if self._is_speaking:
            duration = chunk_end_time - self._speech_start
            if duration >= self.MAX_SEGMENT_DURATION:
                audio = np.concatenate(self._audio_buffer)
                completed.append({
                    "start": self._speech_start,
                    "end": chunk_end_time,
                    "audio": audio,
                })
                logger.debug(
                    f"VAD 强制切段: {duration:.1f}s >= {self.MAX_SEGMENT_DURATION}s"
                )
                self._speech_start = chunk_end_time
                self._audio_buffer = []

        return completed

    def flush(self) -> list[dict]:
        """录音结束时刷出缓冲区中剩余的语音段"""
        if not self._is_speaking or not self._audio_buffer:
            return []

        audio = np.concatenate(self._audio_buffer)
        end_time = self._total_samples / self.SAMPLE_RATE
        duration = end_time - self._speech_start

        self._is_speaking = False
        self._audio_buffer = []

        if duration >= self.MIN_SEGMENT_DURATION:
            return [{"start": self._speech_start, "end": end_time, "audio": audio}]
        return []

    def reset(self) -> None:
        """重置状态（新录音前调用）"""
        self._vad_cache = {}
        self._is_speaking = False
        self._speech_start = 0.0
        self._audio_buffer = []
        self._total_samples = 0
        self._chunk_appended = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def current_time(self) -> float:
        return self._total_samples / self.SAMPLE_RATE


# ---------------------------------------------------------------------------
# FunASR Paraformer 流式实现
# ---------------------------------------------------------------------------

class FunASREngine(StreamingASREngine):
    """
    FunASR Paraformer 流式 ASR 引擎

    使用 paraformer-zh-streaming 模型进行实时中文语音识别。
    文本在同一个 segment 中累积，直到 is_final 时统一加标点并确认。
    """

    def __init__(self) -> None:
        self._asr_model: Any | None = None
        self._punc_model: Any | None = None
        self._vad_model: Any | None = None
        self._settings = get_settings().streaming
        self._session_counter = 0

    def load(self) -> None:
        if self._asr_model is not None:
            return

        from funasr import AutoModel

        settings = self._settings
        device = settings.device

        # 自动选择设备
        if device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                logger.warning("CUDA 不可用，流式 ASR 将使用 CPU（延迟可能较高）")

        # 解析模型路径（相对路径基于 models_dir）
        models_dir = get_settings().paths.models_dir

        def _resolve(p: Path) -> Path:
            if p.is_absolute():
                return p
            rel = str(p).replace("\\", "/")
            if rel.startswith("models/"):
                rel = rel[len("models/"):]
            return (models_dir / rel).resolve()

        asr_dir = _resolve(settings.funasr_model_dir)
        punc_dir = _resolve(settings.funasr_punc_dir)
        vad_dir = _resolve(settings.funasr_vad_dir)

        logger.info(f"加载 FunASR 流式模型 (device={device})")
        logger.info(f"  ASR: {asr_dir}")
        logger.info(f"  Punc: {punc_dir}")
        logger.info(f"  VAD: {vad_dir}")

        if not asr_dir.exists():
            raise FileNotFoundError(
                f"FunASR 模型不存在: {asr_dir}\n"
                f"请运行: python backend/scripts/download_funasr_models.py"
            )

        self._asr_model = AutoModel(
            model=str(asr_dir),
            device=device,
            disable_update=True,
        )
        logger.info("FunASR 流式 ASR 模型加载完成")

        # 标点恢复模型（可选，用于 final 结果优化）
        if punc_dir.exists():
            try:
                self._punc_model = AutoModel(
                    model=str(punc_dir),
                    device=device,
                    disable_update=True,
                )
                logger.info("FunASR 标点恢复模型加载完成")
            except Exception as e:
                logger.warning(f"标点恢复模型加载失败（不影响 ASR）: {e}")
                self._punc_model = None
        else:
            logger.warning(f"标点恢复模型不存在: {punc_dir}")
            self._punc_model = None

        # VAD 模型（可选，用于流式语音端点检测）
        if vad_dir.exists():
            try:
                self._vad_model = AutoModel(
                    model=str(vad_dir),
                    device=device,
                    disable_update=True,
                )
                logger.info("FunASR VAD 模型加载完成")
            except Exception as e:
                logger.warning(f"VAD 模型加载失败（将使用时间基准 fallback）: {e}")
                self._vad_model = None
        else:
            logger.warning(f"VAD 模型不存在: {vad_dir}（将使用时间基准 fallback）")
            self._vad_model = None

    def is_loaded(self) -> bool:
        return self._asr_model is not None

    def create_session(self) -> StreamingSession:
        if not self.is_loaded():
            self.load()

        self._session_counter += 1
        session_id = f"funasr_{self._session_counter}_{int(time.time())}"
        logger.info(f"创建流式会话: {session_id}")
        return StreamingSession(session_id=session_id)

    def feed_chunk(
        self,
        session: StreamingSession,
        pcm_int16: bytes,
        is_final: bool = False,
    ) -> ASRResult:
        if self._asr_model is None:
            raise RuntimeError("ASR 引擎未加载，请先调用 load()")

        settings = self._settings
        sample_rate = settings.sample_rate

        # 将 bytes 转为 numpy int16 数组
        if pcm_int16:
            audio_array = np.frombuffer(pcm_int16, dtype=np.int16)
        else:
            audio_array = np.zeros(0, dtype=np.int16)

        # 更新总样本数（用于计算时间戳）
        chunk_samples = len(audio_array)
        session.total_samples += chunk_samples
        chunk_end_time = session.total_samples / sample_rate

        # 调用 FunASR 流式推理
        result = self._asr_model.generate(
            input=audio_array,
            cache=session.cache,
            is_final=is_final,
            chunk_size=settings.funasr_chunk_size,
            encoder_chunk_look_back=settings.funasr_encoder_chunk_look_back,
            decoder_chunk_look_back=settings.funasr_decoder_chunk_look_back,
        )

        results: ASRResult = []

        # ── 1. 累积 ASR 文本 + 提取时间戳 ──
        new_text = ""
        new_char_ts: list[CharTimestamp] = []
        if result:
            for r in result:
                text = r.get("text", "").strip()
                if text:
                    new_text += text
                # 提取 FunASR 流式 timestamp 字段: [[start_ms, end_ms], ...]
                timestamps = r.get("timestamp", [])
                if timestamps and text:
                    chars = list(text)
                    for i, ts_pair in enumerate(timestamps):
                        if i < len(chars) and len(ts_pair) >= 2:
                            new_char_ts.append(CharTimestamp(
                                char=chars[i],
                                start=ts_pair[0] / 1000.0,
                                end=ts_pair[1] / 1000.0,
                            ))

        if new_text:
            if not session.current_text:
                # 优先级: VAD 语音起始 > 首个字级时间戳 > chunk 起始时间
                if session.vad_speech_start >= 0:
                    session.current_segment_start = session.vad_speech_start
                elif new_char_ts:
                    session.current_segment_start = max(0.0, new_char_ts[0].start)
                else:
                    session.current_segment_start = chunk_end_time - (chunk_samples / sample_rate)
                session.current_segment_start = max(0.0, session.current_segment_start)
            session.current_text += new_text
            session.last_text_time = chunk_end_time
            if new_char_ts:
                session.current_char_timestamps.extend(new_char_ts)

        # ── 2. 运行 VAD（与 ASR 独立 cache） ──
        vad_speech_end = False
        vad_end_time = 0.0

        if self._vad_model is not None and len(audio_array) > 0:
            try:
                vad_result = self._vad_model.generate(
                    input=audio_array,
                    cache=session.vad_cache,
                    is_final=is_final,
                    chunk_size=200,
                    max_end_silence_time=self._settings.funasr_vad_silence_ms,
                )
                # 解析 VAD 输出: [{"value": [[start_ms, end_ms], ...]}]
                segments = vad_result[0].get("value", []) if vad_result else []
                for seg in segments:
                    if len(seg) >= 2:
                        if seg[0] != -1:
                            session.is_speech = True
                            session.vad_speech_start = seg[0] / 1000.0
                        if seg[1] != -1:
                            session.is_speech = False
                            vad_speech_end = True
                            vad_end_time = seg[1] / 1000.0  # ms → seconds
            except Exception as e:
                logger.debug(f"VAD 推理错误（不影响 ASR）: {e}")

        # ── 3. 分段决策 ──
        # 时间基准 fallback（当 VAD 不可用或未触发时的保底机制）
        FALLBACK_SILENCE_SECONDS = 3.0
        silence_duration = (
            (chunk_end_time - session.last_text_time)
            if (session.current_text and session.last_text_time > 0)
            else 0.0
        )

        if is_final:
            # 录音结束：强制输出所有累积文本
            if session.current_text:
                final_text = self._add_punctuation(session.current_text)
                char_ts = session.current_char_timestamps or None
                segment = StreamingSegment(
                    id=session.segment_counter,
                    start=session.current_segment_start,
                    end=chunk_end_time,
                    text=final_text,
                    temp_speaker="SPEAKER_00",
                    char_timestamps=char_ts,
                )
                results.append((segment, True))
                session.segment_counter += 1
                self._reset_segment_state(session)
        elif vad_speech_end and session.current_text:
            # VAD 检测到语音结束 → 确认当前片段
            segment_end = min(vad_end_time, chunk_end_time) if vad_end_time > 0 else session.last_text_time
            final_text = self._add_punctuation(session.current_text)
            char_ts = session.current_char_timestamps or None

            # P1-3: 短回应合并（"嗯"等 <=2字 且 <0.5s → 合并到上一段）
            seg_duration = segment_end - session.current_segment_start
            if (
                len(final_text.strip()) <= 2
                and seg_duration < 0.5
                and results  # 本轮有前一段
            ):
                prev_seg, prev_final = results[-1]
                if prev_final:
                    # 合并到上一段
                    merged_seg = StreamingSegment(
                        id=prev_seg.id,
                        start=prev_seg.start,
                        end=segment_end,
                        text=prev_seg.text + final_text,
                        temp_speaker=prev_seg.temp_speaker,
                        char_timestamps=(
                            (prev_seg.char_timestamps or []) + (char_ts or [])
                        ) or None,
                    )
                    results[-1] = (merged_seg, True)
                    self._reset_segment_state(session)
                    logger.debug(f"短回应合并: '{final_text}' → segment {prev_seg.id}")
                else:
                    self._emit_final_segment(
                        session, results, segment_end, final_text, char_ts,
                    )
            else:
                self._emit_final_segment(
                    session, results, segment_end, final_text, char_ts,
                )
            logger.debug(
                f"VAD 分段: segment {session.segment_counter - 1}, "
                f"vad_end={vad_end_time:.1f}s, {final_text[:30]}..."
            )
        elif silence_duration >= FALLBACK_SILENCE_SECONDS:
            # Fallback: VAD 未触发但文字已静默很久
            segment_end = session.last_text_time
            final_text = self._add_punctuation(session.current_text)
            char_ts = session.current_char_timestamps or None
            segment = StreamingSegment(
                id=session.segment_counter,
                start=session.current_segment_start,
                end=segment_end,
                text=final_text,
                temp_speaker="SPEAKER_00",
                char_timestamps=char_ts,
            )
            results.append((segment, True))
            session.segment_counter += 1
            self._reset_segment_state(session)
            logger.debug(
                f"Fallback 分段: segment {segment.id}, "
                f"silence={silence_duration:.1f}s, {segment.text[:30]}..."
            )
        elif session.current_text:
            # 发送部分更新（同一 segment_id，前端更新显示）
            # P2: 前缀锁定 — 稳定文本不闪烁
            display_text = self._apply_prefix_locking(session, session.current_text)
            segment = StreamingSegment(
                id=session.segment_counter,
                start=session.current_segment_start,
                end=chunk_end_time,
                text=display_text,
                temp_speaker="SPEAKER_00",
            )
            results.append((segment, False))

        return results

    @staticmethod
    def _reset_segment_state(session: StreamingSession) -> None:
        """重置片段累积状态"""
        session.current_text = ""
        session.last_text_time = 0.0
        session.current_char_timestamps = []
        session.vad_speech_start = -1.0
        session.locked_text = ""
        session.pending_text = ""
        session.pending_stable_count = 0

    @staticmethod
    def _emit_final_segment(
        session: StreamingSession,
        results: ASRResult,
        end_time: float,
        text: str,
        char_ts: list[CharTimestamp] | None,
    ) -> None:
        """创建并追加一个最终片段，重置片段状态"""
        segment = StreamingSegment(
            id=session.segment_counter,
            start=session.current_segment_start,
            end=end_time,
            text=text,
            temp_speaker="SPEAKER_00",
            char_timestamps=char_ts,
        )
        results.append((segment, True))
        session.segment_counter += 1
        FunASREngine._reset_segment_state(session)

    @staticmethod
    def _apply_prefix_locking(session: StreamingSession, full_text: str) -> str:
        """
        前缀锁定：减少流式部分结果的文字闪烁。

        已锁定的文本前缀不会改变，只有尾部可变。
        连续 N 次扩展部分相同 → 锁定为新前缀。
        """
        locked = session.locked_text

        if full_text.startswith(locked):
            # 正常扩展：新文本以已锁定前缀开头
            extension = full_text[len(locked):]
            if extension == session.pending_text and extension:
                session.pending_stable_count += 1
                if session.pending_stable_count >= session.STABILITY_THRESHOLD:
                    # 扩展部分稳定，锁定
                    session.locked_text = full_text
                    session.pending_text = ""
                    session.pending_stable_count = 0
            else:
                session.pending_text = extension
                session.pending_stable_count = 1 if extension else 0
            return full_text
        else:
            # ASR 回退修改了已锁定文本 — 保留锁定前缀 + 新文本后缀
            # 找到最长公共前缀
            common_len = 0
            for i in range(min(len(locked), len(full_text))):
                if locked[i] == full_text[i]:
                    common_len = i + 1
                else:
                    break
            # 保守处理：用 ASR 的最新完整文本（回退可能是更准确的修正）
            session.locked_text = full_text[:common_len]
            session.pending_text = full_text[common_len:]
            session.pending_stable_count = 0
            return full_text

    def _add_punctuation(self, text: str) -> str:
        """对文本添加标点恢复（如果标点模型可用）"""
        if not text or self._punc_model is None:
            return text
        try:
            punc_result = self._punc_model.generate(input=text)
            if punc_result and len(punc_result) > 0:
                return punc_result[0].get("text", text) or text
        except Exception as e:
            logger.debug(f"标点恢复失败: {e}")
        return text

    def unload(self) -> None:
        """卸载模型，释放 GPU 显存给后续的 pyannote"""
        if self._asr_model is not None:
            logger.info("卸载 FunASR 流式模型...")
            del self._asr_model
            self._asr_model = None

        if self._punc_model is not None:
            del self._punc_model
            self._punc_model = None

        if self._vad_model is not None:
            del self._vad_model
            self._vad_model = None

        # 释放 GPU 缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU 显存已释放")
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# sherpa-onnx Paraformer 流式实现
# ---------------------------------------------------------------------------

class SherpaOnnxEngine(StreamingASREngine):
    """
    sherpa-onnx 流式 ASR 引擎

    使用 ONNX Runtime 推理，不依赖 PyTorch。
    支持 paraformer-trilingual（中/粤/英）。
    文本在 endpoint 之间持续累积，endpoint 触发时确认为完整句子。
    """

    def __init__(self) -> None:
        self._recognizer: Any | None = None
        self._settings = get_settings().streaming
        self._session_counter = 0

    def _resolve_model_dir(self) -> Path:
        """解析 sherpa-onnx 模型目录"""
        return _resolve_relative_model_path(self._settings.sherpa_model_dir)

    def load(self) -> None:
        if self._recognizer is not None:
            return

        import sherpa_onnx

        model_dir = self._resolve_model_dir()
        logger.info(f"加载 sherpa-onnx 流式模型: {model_dir}")

        if not model_dir.exists():
            raise FileNotFoundError(
                f"sherpa-onnx 模型不存在: {model_dir}\n"
                f"请下载模型到该目录。"
            )

        # 查找模型文件
        tokens = model_dir / "tokens.txt"
        encoder = model_dir / "encoder.int8.onnx"
        decoder = model_dir / "decoder.int8.onnx"

        # 如果 int8 不存在，尝试 fp32
        if not encoder.exists():
            encoder = model_dir / "encoder.onnx"
        if not decoder.exists():
            decoder = model_dir / "decoder.onnx"

        for f in [tokens, encoder, decoder]:
            if not f.exists():
                raise FileNotFoundError(f"模型文件不存在: {f}")

        # 选择 provider
        device = self._settings.device
        provider = "cpu"
        if device == "auto":
            try:
                import onnxruntime
                if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                    provider = "cuda"
            except ImportError:
                pass
        elif device.startswith("cuda"):
            provider = "cuda"

        logger.info(f"  tokens: {tokens}")
        logger.info(f"  encoder: {encoder}")
        logger.info(f"  decoder: {decoder}")
        logger.info(f"  provider: {provider}")

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            num_threads=4,
            sample_rate=self._settings.sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=0.8,   # P1: 会议场景说话人切换更快
            rule3_min_utterance_length=20.0,
            decoding_method="greedy_search",
            provider=provider,
        )
        logger.info("sherpa-onnx 流式 ASR 模型加载完成")

    def is_loaded(self) -> bool:
        return self._recognizer is not None

    def create_session(self) -> StreamingSession:
        if not self.is_loaded():
            self.load()

        self._session_counter += 1
        session_id = f"sherpa_{self._session_counter}_{int(time.time())}"

        # 创建 OnlineStream 保存在 cache 中
        stream = self._recognizer.create_stream()
        session = StreamingSession(session_id=session_id)
        session.cache["stream"] = stream

        logger.info(f"创建流式会话: {session_id}")
        return session

    def feed_chunk(
        self,
        session: StreamingSession,
        pcm_int16: bytes,
        is_final: bool = False,
    ) -> ASRResult:
        if self._recognizer is None:
            raise RuntimeError("ASR 引擎未加载，请先调用 load()")

        stream = session.cache["stream"]
        sample_rate = self._settings.sample_rate

        # 将 bytes 转为 float32 数组 (sherpa-onnx 需要 float32)
        if pcm_int16:
            audio_int16 = np.frombuffer(pcm_int16, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
        else:
            audio_float = np.zeros(0, dtype=np.float32)

        chunk_samples = len(audio_float)
        session.total_samples += chunk_samples
        chunk_end_time = session.total_samples / sample_rate

        # 喂入音频
        if len(audio_float) > 0:
            stream.accept_waveform(sample_rate, audio_float.tolist())

        if is_final:
            stream.input_finished()

        results: ASRResult = []

        # 解码
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        # 获取当前累积结果（sherpa-onnx 的 get_result 返回 reset 以来的完整文本）
        result = self._recognizer.get_result(stream)
        text = result.text.strip() if hasattr(result, "text") else str(result).strip()

        # P0-2: 提取 sherpa-onnx 字级时间戳
        char_ts = self._extract_sherpa_timestamps(result, text)

        if is_final:
            # 录音结束：返回当前文本作为最终片段
            if text:
                segment = StreamingSegment(
                    id=session.segment_counter,
                    start=session.current_segment_start,
                    end=chunk_end_time,
                    text=text,
                    temp_speaker="SPEAKER_00",
                    char_timestamps=char_ts,
                )
                results.append((segment, True))
                session.segment_counter += 1
                self._reset_session_segment(session)
        elif self._recognizer.is_endpoint(stream):
            # 端点检测：当前句子完成
            if text:
                seg_duration = chunk_end_time - session.current_segment_start
                # P1-3: 短回应合并
                if (
                    len(text.strip()) <= 2
                    and seg_duration < 0.5
                    and results
                ):
                    prev_seg, prev_final = results[-1]
                    if prev_final:
                        merged_seg = StreamingSegment(
                            id=prev_seg.id,
                            start=prev_seg.start,
                            end=chunk_end_time,
                            text=prev_seg.text + text,
                            temp_speaker=prev_seg.temp_speaker,
                            char_timestamps=(
                                (prev_seg.char_timestamps or []) + (char_ts or [])
                            ) or None,
                        )
                        results[-1] = (merged_seg, True)
                        logger.debug(f"短回应合并: '{text}' → segment {prev_seg.id}")
                    else:
                        segment = StreamingSegment(
                            id=session.segment_counter,
                            start=session.current_segment_start,
                            end=chunk_end_time,
                            text=text,
                            temp_speaker="SPEAKER_00",
                            char_timestamps=char_ts,
                        )
                        results.append((segment, True))
                        session.segment_counter += 1
                else:
                    segment = StreamingSegment(
                        id=session.segment_counter,
                        start=session.current_segment_start,
                        end=chunk_end_time,
                        text=text,
                        temp_speaker="SPEAKER_00",
                        char_timestamps=char_ts,
                    )
                    results.append((segment, True))
                    session.segment_counter += 1
            # 重置流，开始新句子
            self._recognizer.reset(stream)
            self._reset_session_segment(session)
            session.current_segment_start = chunk_end_time
        elif text:
            # 部分识别：更新当前片段（同一 segment_id）
            if not session.current_text:
                # 优先用首个字级时间戳，回退到 chunk 起始时间
                if char_ts:
                    session.current_segment_start = max(0.0, char_ts[0].start)
                else:
                    session.current_segment_start = chunk_end_time - (chunk_samples / sample_rate)
            session.current_text = text  # 记录用于跟踪
            # P2: 前缀锁定
            display_text = self._apply_prefix_locking(session, text)
            segment = StreamingSegment(
                id=session.segment_counter,
                start=session.current_segment_start,
                end=chunk_end_time,
                text=display_text,
                temp_speaker="SPEAKER_00",
            )
            results.append((segment, False))

        return results

    @staticmethod
    def _extract_sherpa_timestamps(
        result: Any, text: str,
    ) -> list[CharTimestamp] | None:
        """从 sherpa-onnx result 中提取字级时间戳"""
        timestamps = getattr(result, "timestamps", None)
        tokens = getattr(result, "tokens", None)
        if not timestamps or not tokens:
            return None

        char_ts: list[CharTimestamp] = []
        # sherpa-onnx tokens 是 subword tokens，需要合并为字级
        # timestamps[i] 是 tokens[i] 的开始时间（秒）
        for i, (token, ts) in enumerate(zip(tokens, timestamps)):
            token = token.strip()
            if not token:
                continue
            start = ts
            # 结束时间 = 下一个 token 的开始时间，或用小偏移估算
            end = timestamps[i + 1] if i + 1 < len(timestamps) else ts + 0.1
            # 均匀分配 token 时长给每个字
            ch_duration = (end - start) / len(token)
            for j, ch in enumerate(token):
                ch_start = start + j * ch_duration
                char_ts.append(CharTimestamp(
                    char=ch,
                    start=ch_start,
                    end=ch_start + ch_duration,
                ))

        return char_ts if char_ts else None

    @staticmethod
    def _reset_session_segment(session: StreamingSession) -> None:
        """重置片段累积状态"""
        session.current_text = ""
        session.vad_speech_start = -1.0
        session.locked_text = ""
        session.pending_text = ""
        session.pending_stable_count = 0

    @staticmethod
    def _apply_prefix_locking(session: StreamingSession, full_text: str) -> str:
        """前缀锁定（与 FunASREngine 相同逻辑）"""
        locked = session.locked_text
        if full_text.startswith(locked):
            extension = full_text[len(locked):]
            if extension == session.pending_text and extension:
                session.pending_stable_count += 1
                if session.pending_stable_count >= session.STABILITY_THRESHOLD:
                    session.locked_text = full_text
                    session.pending_text = ""
                    session.pending_stable_count = 0
            else:
                session.pending_text = extension
                session.pending_stable_count = 1 if extension else 0
            return full_text
        else:
            common_len = 0
            for i in range(min(len(locked), len(full_text))):
                if locked[i] == full_text[i]:
                    common_len = i + 1
                else:
                    break
            session.locked_text = full_text[:common_len]
            session.pending_text = full_text[common_len:]
            session.pending_stable_count = 0
            return full_text

    def unload(self) -> None:
        if self._recognizer is not None:
            logger.info("卸载 sherpa-onnx 流式模型...")
            del self._recognizer
            self._recognizer = None


# ---------------------------------------------------------------------------
# 引擎信息查询
# ---------------------------------------------------------------------------

def _resolve_relative_model_path(config_path: Path) -> Path:
    """将配置中的相对路径解析为绝对路径（与引擎类使用相同逻辑）"""
    if config_path.is_absolute():
        return config_path
    models_dir = get_settings().paths.models_dir
    rel = str(config_path).replace("\\", "/")
    if rel.startswith("models/"):
        rel = rel[len("models/"):]
    return (models_dir / rel).resolve()


def list_available_engines() -> list[dict]:
    """返回可用的流式 ASR 引擎列表及其模型状态"""
    settings = get_settings().streaming
    engines = []

    # FunASR
    funasr_dir = _resolve_relative_model_path(settings.funasr_model_dir)
    funasr_parent = funasr_dir.parent  # streaming/funasr/
    engines.append({
        "id": "funasr",
        "name": "FunASR Paraformer (CER 3.5% ≈1G)",
        "description": "阿里达摩院 Paraformer 流式中文 ASR（需要 PyTorch）",
        "installed": funasr_dir.exists(),
        "model_dir": str(funasr_parent),
    })

    # sherpa-onnx
    sherpa_dir = _resolve_relative_model_path(settings.sherpa_model_dir)
    sherpa_encoder = sherpa_dir / "encoder.int8.onnx"
    if not sherpa_encoder.exists():
        sherpa_encoder = sherpa_dir / "encoder.onnx"
    engines.append({
        "id": "sherpa-onnx",
        "name": "sherpa-onnx Paraformer (CER 4% ≈0.5G)",
        "description": "ONNX Runtime 流式三语 ASR（中/粤/英，无需 PyTorch）",
        "installed": sherpa_encoder.exists(),
        "model_dir": str(sherpa_dir),
    })

    return engines


# ---------------------------------------------------------------------------
# 工厂函数 + 单例
# ---------------------------------------------------------------------------

_engine: StreamingASREngine | None = None
_engine_type: str | None = None


def get_streaming_asr_engine(engine_type: str | None = None) -> StreamingASREngine:
    """
    获取流式 ASR 引擎单例

    Args:
        engine_type: 指定引擎类型 ("funasr" / "sherpa-onnx")，
                     None 表示使用配置中的默认值
    """
    global _engine, _engine_type

    if engine_type is None:
        engine_type = get_settings().streaming.asr_engine

    # 如果切换了引擎类型，重置
    if _engine is not None and _engine_type != engine_type:
        _engine.unload()
        _engine = None

    if _engine is None:
        if engine_type == "funasr":
            _engine = FunASREngine()
        elif engine_type == "sherpa-onnx":
            _engine = SherpaOnnxEngine()
        else:
            raise ValueError(f"未知的 ASR 引擎: {engine_type}")

        _engine_type = engine_type
        logger.info(f"流式 ASR 引擎: {engine_type}")

    return _engine


def reset_streaming_asr_engine() -> None:
    """重置引擎单例（用于引擎切换或测试）"""
    global _engine, _engine_type
    if _engine is not None:
        _engine.unload()
    _engine = None
    _engine_type = None
