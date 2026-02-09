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
from ..models import StreamingSegment

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

        # ── 1. 累积 ASR 文本 ──
        new_text = ""
        if result:
            for r in result:
                text = r.get("text", "").strip()
                if text:
                    new_text += text

        if new_text:
            if not session.current_text:
                session.current_segment_start = chunk_end_time - (chunk_samples / sample_rate)
            session.current_text += new_text
            session.last_text_time = chunk_end_time

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
                segment = StreamingSegment(
                    id=session.segment_counter,
                    start=session.current_segment_start,
                    end=chunk_end_time,
                    text=final_text,
                    temp_speaker="SPEAKER_00",
                )
                results.append((segment, True))
                session.segment_counter += 1
                session.current_text = ""
                session.last_text_time = 0.0
        elif vad_speech_end and session.current_text:
            # VAD 检测到语音结束 → 确认当前片段
            segment_end = min(vad_end_time, chunk_end_time) if vad_end_time > 0 else session.last_text_time
            final_text = self._add_punctuation(session.current_text)
            segment = StreamingSegment(
                id=session.segment_counter,
                start=session.current_segment_start,
                end=segment_end,
                text=final_text,
                temp_speaker="SPEAKER_00",
            )
            results.append((segment, True))
            session.segment_counter += 1
            session.current_text = ""
            session.last_text_time = 0.0
            logger.debug(
                f"VAD 分段: segment {segment.id}, "
                f"vad_end={vad_end_time:.1f}s, {segment.text[:30]}..."
            )
        elif silence_duration >= FALLBACK_SILENCE_SECONDS:
            # Fallback: VAD 未触发但文字已静默很久
            segment_end = session.last_text_time
            final_text = self._add_punctuation(session.current_text)
            segment = StreamingSegment(
                id=session.segment_counter,
                start=session.current_segment_start,
                end=segment_end,
                text=final_text,
                temp_speaker="SPEAKER_00",
            )
            results.append((segment, True))
            session.segment_counter += 1
            session.current_text = ""
            session.last_text_time = 0.0
            logger.debug(
                f"Fallback 分段: segment {segment.id}, "
                f"silence={silence_duration:.1f}s, {segment.text[:30]}..."
            )
        elif session.current_text:
            # 发送部分更新（同一 segment_id，前端更新显示）
            segment = StreamingSegment(
                id=session.segment_counter,
                start=session.current_segment_start,
                end=chunk_end_time,
                text=session.current_text,
                temp_speaker="SPEAKER_00",
            )
            results.append((segment, False))

        return results

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
            rule2_min_trailing_silence=1.2,
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

        if is_final:
            # 录音结束：返回当前文本作为最终片段
            if text:
                segment = StreamingSegment(
                    id=session.segment_counter,
                    start=session.current_segment_start,
                    end=chunk_end_time,
                    text=text,
                    temp_speaker="SPEAKER_00",
                )
                results.append((segment, True))
                session.segment_counter += 1
        elif self._recognizer.is_endpoint(stream):
            # 端点检测：当前句子完成
            if text:
                segment = StreamingSegment(
                    id=session.segment_counter,
                    start=session.current_segment_start,
                    end=chunk_end_time,
                    text=text,
                    temp_speaker="SPEAKER_00",
                )
                results.append((segment, True))
                session.segment_counter += 1
            # 重置流，开始新句子
            self._recognizer.reset(stream)
            session.current_segment_start = chunk_end_time
        elif text:
            # 部分识别：更新当前片段（同一 segment_id）
            if not session.current_text:
                session.current_segment_start = chunk_end_time - (chunk_samples / sample_rate)
            session.current_text = text  # 记录用于跟踪
            segment = StreamingSegment(
                id=session.segment_counter,
                start=session.current_segment_start,
                end=chunk_end_time,
                text=text,
                temp_speaker="SPEAKER_00",
            )
            results.append((segment, False))

        return results

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
