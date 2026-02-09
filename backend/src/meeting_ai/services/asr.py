"""
语音转写服务（ASR）— 多引擎架构

支持引擎:
  - faster-whisper  (CTranslate2，99 语言)
  - funasr          (SenseVoice / Paraformer-Large，中文最优)
  - fireredasr      (FireRedASR-AED，中文 SOTA)

使用方式:
    engine = get_asr_engine("sensevoice-small")
    result = engine.transcribe(Path("audio.wav"))
"""

from __future__ import annotations

import gc
import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import get_settings
from ..logger import get_logger
from ..models import Segment, TranscriptResult

logger = get_logger("services.asr")


# ---------------------------------------------------------------------------
# VAD pre-segmentation (shared across engines)
# ---------------------------------------------------------------------------

_vad_model = None


def _get_vad_model():
    """懒加载 fsmn-vad 模型（单例）"""
    global _vad_model
    if _vad_model is not None:
        return _vad_model

    models_dir = get_settings().paths.models_dir
    vad_dir = models_dir / "streaming" / "funasr" / "fsmn-vad"
    if not vad_dir.exists():
        logger.warning(f"fsmn-vad 模型未找到: {vad_dir}")
        return None

    from funasr import AutoModel
    logger.info(f"加载 fsmn-vad 模型: {vad_dir}")
    _vad_model = AutoModel(
        model=str(vad_dir),
        disable_update=True,
    )
    return _vad_model


def _run_vad(
    audio_path: str | Path,
    *,
    max_single_segment_time: int = 15000,
    min_segment_ms: int = 200,
) -> list[tuple[float, float]] | None:
    """
    用 fsmn-vad 对音频做语音活动检测，返回语音段 [(start_sec, end_sec), ...].

    返回 None 表示 VAD 不可用或失败。
    """
    vad_model = _get_vad_model()
    if vad_model is None:
        return None

    try:
        res = vad_model.generate(
            input=str(audio_path),
            cache={},
            max_single_segment_time=max_single_segment_time,
        )
        # res 格式: [{"text": [[start_ms, end_ms], [start_ms, end_ms], ...]}]
        if not res or not isinstance(res, list):
            return None

        raw_segments = res[0].get("text", []) if isinstance(res[0], dict) else []
        if not raw_segments:
            return None

        vad_segs: list[tuple[float, float]] = []
        for pair in raw_segments:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                start_ms, end_ms = pair
                if end_ms - start_ms >= min_segment_ms:
                    vad_segs.append((start_ms / 1000.0, end_ms / 1000.0))

        logger.info(f"VAD 检测到 {len(vad_segs)} 个语音段")
        return vad_segs if vad_segs else None

    except Exception as e:
        logger.warning(f"VAD 检测失败: {e}")
        return None


def unload_vad_model():
    """释放 VAD 模型"""
    global _vad_model
    if _vad_model is not None:
        logger.info("释放 fsmn-vad 模型")
        _vad_model = None
        gc.collect()


# ---------------------------------------------------------------------------
# ct-punc punctuation restoration (for engines that don't produce punctuation)
# ---------------------------------------------------------------------------

_punc_model = None


def _get_punc_model():
    """懒加载 ct-punc 标点恢复模型（单例）"""
    global _punc_model
    if _punc_model is not None:
        return _punc_model

    models_dir = get_settings().paths.models_dir
    punc_dir = models_dir / "streaming" / "funasr" / "ct-punc"
    if not punc_dir.exists():
        logger.warning(f"ct-punc 模型未找到: {punc_dir}")
        return None

    from funasr import AutoModel
    logger.info(f"加载 ct-punc 标点模型: {punc_dir}")
    _punc_model = AutoModel(
        model=str(punc_dir),
        disable_update=True,
    )
    return _punc_model


def _add_punctuation(text: str) -> str:
    """用 ct-punc 给无标点文本添加标点"""
    if not text or not text.strip():
        return text

    punc_model = _get_punc_model()
    if punc_model is None:
        return text

    try:
        res = punc_model.generate(input=text)
        if res and isinstance(res, list) and isinstance(res[0], dict):
            return res[0].get("text", text)
    except Exception as e:
        logger.warning(f"标点恢复失败: {e}")

    return text


def unload_punc_model():
    """释放标点模型"""
    global _punc_model
    if _punc_model is not None:
        logger.info("释放 ct-punc 模型")
        _punc_model = None
        gc.collect()


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class ASREngine(ABC):
    """语音转写引擎抽象基类"""

    @abstractmethod
    def load(self, model_dir: Path) -> None:
        """加载模型"""
        ...

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path | str,
        language: str | None = None,
    ) -> TranscriptResult:
        """转写音频文件，返回 TranscriptResult"""
        ...

    @abstractmethod
    def unload(self) -> None:
        """释放模型，回收显存"""
        ...


# ---------------------------------------------------------------------------
# Engine: faster-whisper
# ---------------------------------------------------------------------------

class FasterWhisperEngine(ASREngine):
    """基于 CTranslate2 的 Whisper 引擎"""

    def __init__(self) -> None:
        self._model = None

    def load(self, model_dir: Path) -> None:
        from faster_whisper import WhisperModel

        settings = get_settings().asr
        device = settings.device
        compute_type = settings.compute_type

        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and compute_type == "float16":
                compute_type = "int8"

        logger.info(
            f"加载 Whisper 模型: {model_dir.name} "
            f"(device={device}, compute={compute_type})"
        )
        self._model = WhisperModel(
            str(model_dir), device=device, compute_type=compute_type,
        )

    def transcribe(
        self,
        audio_path: Path | str,
        language: str | None = None,
    ) -> TranscriptResult:
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        settings = get_settings().asr
        lang = language or settings.language or "zh"

        logger.info(f"[faster-whisper] 开始转写: {audio_path.name}")

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=lang,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
            word_timestamps=settings.word_timestamps,
        )

        segments = []
        for seg in segments_iter:
            segments.append(Segment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                speaker=None,
            ))

        logger.info(
            f"[faster-whisper] 转写完成: 语言={info.language}, "
            f"概率={info.language_probability:.2f}, 片段数={len(segments)}"
        )

        return TranscriptResult(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segments,
        )

    def unload(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Engine: FunASR (SenseVoice / Paraformer-Large)
# ---------------------------------------------------------------------------

class FunASRFileEngine(ASREngine):
    """FunASR 引擎，支持 SenseVoice 和 Paraformer-Large"""

    def __init__(self) -> None:
        self._model = None
        self._is_sensevoice = False
        self._model_dir: Path | None = None

    def load(self, model_dir: Path) -> None:
        from funasr import AutoModel

        self._model_dir = model_dir
        self._is_sensevoice = "sensevoice" in model_dir.name.lower()

        device = get_settings().asr.device
        if device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            device = "cuda:0"

        # 共享 VAD 模型（用于长音频分段）
        models_dir = get_settings().paths.models_dir
        vad_dir = models_dir / "streaming" / "funasr" / "fsmn-vad"

        model_kwargs: dict = {
            "model": str(model_dir),
            "device": device,
            "disable_update": True,
        }

        if vad_dir.exists():
            model_kwargs["vad_model"] = str(vad_dir)
            model_kwargs["vad_kwargs"] = {"max_single_segment_time": 60000}

        if self._is_sensevoice:
            model_kwargs["trust_remote_code"] = True
        else:
            # Paraformer: 附加标点恢复模型
            punc_dir = models_dir / "streaming" / "funasr" / "ct-punc"
            if punc_dir.exists():
                model_kwargs["punc_model"] = str(punc_dir)

        engine_name = "SenseVoice" if self._is_sensevoice else "Paraformer"
        logger.info(f"加载 {engine_name} 模型: {model_dir.name} (device={device})")

        self._model = AutoModel(**model_kwargs)

    def transcribe(
        self,
        audio_path: Path | str,
        language: str | None = None,
    ) -> TranscriptResult:
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        import soundfile as sf
        audio_info = sf.info(str(audio_path))
        duration = audio_info.frames / audio_info.samplerate

        engine_name = "SenseVoice" if self._is_sensevoice else "Paraformer"
        logger.info(f"[{engine_name}] 开始转写: {audio_path.name} ({duration:.1f}s)")

        # 优先尝试 VAD 预分段：先切再逐段转写，获得精确时间戳
        segments = self._transcribe_with_vad(audio_path, duration, language)

        if segments is None:
            # VAD 不可用，回退到整文件转写
            logger.info(f"[{engine_name}] VAD 不可用，使用整文件转写")
            segments = self._transcribe_whole_file(audio_path, duration, language)

        logger.info(f"[{engine_name}] 转写完成: 片段数={len(segments)}")

        return TranscriptResult(
            language=language or "zh",
            language_probability=1.0,
            duration=duration,
            segments=segments,
        )

    def _transcribe_whole_file(
        self,
        audio_path: Path,
        duration: float,
        language: str | None = None,
    ) -> list[Segment]:
        """整文件转写（原始方式，作为 VAD 回退）"""
        generate_kwargs: dict = {
            "input": str(audio_path),
            "cache": {},
            "batch_size_s": 300,
        }

        if self._is_sensevoice:
            generate_kwargs["language"] = language or "auto"
            generate_kwargs["use_itn"] = True
            generate_kwargs["merge_vad"] = True
            generate_kwargs["merge_length_s"] = 15

        res = self._model.generate(**generate_kwargs)
        return self._parse_funasr_output(res, duration)

    def _transcribe_with_vad(
        self,
        audio_path: Path,
        duration: float,
        language: str | None = None,
    ) -> list[Segment] | None:
        """VAD 预分段转写：先切成短片段，逐段转写，精确保留时间戳"""
        vad_segs = _run_vad(audio_path, max_single_segment_time=15000)
        if not vad_segs:
            return None

        import soundfile as sf
        audio, sr = sf.read(str(audio_path))

        engine_name = "SenseVoice" if self._is_sensevoice else "Paraformer"
        logger.info(
            f"[{engine_name}] VAD 预分段: {len(vad_segs)} 段, "
            f"逐段转写中..."
        )

        segments: list[Segment] = []
        seg_id = 0

        for i, (vad_start, vad_end) in enumerate(vad_segs):
            # 提取 VAD 片段对应的音频
            start_sample = int(vad_start * sr)
            end_sample = int(vad_end * sr)
            chunk = audio[start_sample:end_sample]

            if len(chunk) < sr * 0.1:  # 跳过 <0.1s 的极短片段
                continue

            # 写临时 WAV 文件
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            chunk_path = f.name
            f.close()

            try:
                sf.write(chunk_path, chunk, sr)

                # 对每个短片段调用 FunASR — 短片段不需要内部 VAD
                generate_kwargs: dict = {
                    "input": chunk_path,
                    "cache": {},
                }

                if self._is_sensevoice:
                    generate_kwargs["language"] = language or "auto"
                    generate_kwargs["use_itn"] = True

                res = self._model.generate(**generate_kwargs)

                # 解析文本
                text = self._extract_text_from_result(res)
                if not text:
                    continue

                # 创建 Segment，时间戳用 VAD 的准确起止
                segments.append(Segment(
                    id=seg_id,
                    start=round(vad_start, 3),
                    end=round(vad_end, 3),
                    text=text,
                    speaker=None,
                ))
                seg_id += 1

            except Exception as e:
                logger.warning(f"[{engine_name}] VAD 段 {i} 转写失败: {e}")
            finally:
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass

        if not segments:
            return None

        logger.info(
            f"[{engine_name}] VAD 预分段转写完成: "
            f"{len(vad_segs)} VAD段 → {len(segments)} 文本段"
        )
        return segments

    def _extract_text_from_result(self, res: list) -> str:
        """从 FunASR generate() 输出中提取纯文本"""
        if not res:
            return ""

        for item in res:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "")
            if not text or not text.strip():
                continue

            # SenseVoice 后处理
            if self._is_sensevoice:
                try:
                    from funasr.utils.postprocess_utils import rich_transcription_postprocess
                    text = rich_transcription_postprocess(text)
                except ImportError:
                    pass
                # 去除 emoji
                text = re.sub(
                    r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F'
                    r'\U0000200D\U00002600-\U000026FF]+',
                    '', text,
                )

            return text.strip()

        return ""

    def _parse_funasr_output(
        self, res: list, duration: float,
    ) -> list[Segment]:
        """解析 FunASR model.generate() 的输出"""
        segments: list[Segment] = []
        seg_id = 0
        # 用于无时间戳时的位置追踪
        last_end = 0.0

        for item in res:
            if not isinstance(item, dict):
                continue

            text = item.get("text", "")
            if not text or not text.strip():
                continue

            # SenseVoice 输出需要后处理（去除特殊标记）
            if self._is_sensevoice:
                try:
                    from funasr.utils.postprocess_utils import rich_transcription_postprocess
                    text = rich_transcription_postprocess(text)
                except ImportError:
                    pass
                # rich_transcription_postprocess 会把 <|HAPPY|> 等标签转为 emoji
                # 在会议纪要场景中不需要，全部去掉
                text = re.sub(
                    r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F'
                    r'\U0000200D\U00002600-\U000026FF]+',
                    '', text,
                )

            text = text.strip()
            if not text:
                continue

            # 提取时间戳（FunASR 使用毫秒）
            timestamp = item.get("timestamp", [])
            if timestamp:
                start = timestamp[0][0] / 1000.0
                end = timestamp[-1][1] / 1000.0
            else:
                # 无时间戳：按文本长度估算持续时间（约每字 0.3s）
                estimated_dur = len(text) * 0.3
                start = last_end
                end = min(start + estimated_dur, duration)

            last_end = end

            segments.append(Segment(
                id=seg_id,
                start=start,
                end=end,
                text=text,
                speaker=None,
            ))
            seg_id += 1

        return segments

    def unload(self) -> None:
        self._model = None
        self._model_dir = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Engine: FireRedASR-AED
# ---------------------------------------------------------------------------

class FireRedASREngine(ASREngine):
    """小红书 FireRedASR-AED 引擎（中文 SOTA，60s 输入限制）"""

    def __init__(self) -> None:
        self._model = None
        self._model_dir: Path | None = None

    def load(self, model_dir: Path) -> None:
        import os
        import torch
        from fireredasr.models.fireredasr import (
            FireRedAsr,
            ASRFeatExtractor,
            ChineseCharEnglishSpmTokenizer,
        )
        from fireredasr.models.fireredasr_aed import FireRedAsrAed

        self._model_dir = model_dir
        logger.info(f"加载 FireRedASR 模型: {model_dir.name}")

        # from_pretrained 只支持在线下载，本地加载需要手动组装
        # load_fireredasr_aed_model 缺少 weights_only=False，PyTorch 2.6+ 会报错
        md = str(model_dir)
        feat_extractor = ASRFeatExtractor(os.path.join(md, "cmvn.ark"))
        package = torch.load(
            os.path.join(md, "model.pth.tar"),
            map_location="cpu",
            weights_only=False,
        )
        model = FireRedAsrAed.from_args(package["args"])
        model.load_state_dict(package["model_state_dict"], strict=True)
        model.eval()
        tokenizer = ChineseCharEnglishSpmTokenizer(
            os.path.join(md, "dict.txt"),
            os.path.join(md, "train_bpe1000.model"),
        )
        self._model = FireRedAsr("aed", feat_extractor, model, tokenizer)

    def transcribe(
        self,
        audio_path: Path | str,
        language: str | None = None,
    ) -> TranscriptResult:
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        import soundfile as sf
        audio_info = sf.info(str(audio_path))
        duration = audio_info.frames / audio_info.samplerate

        logger.info(f"[FireRedASR] 开始转写: {audio_path.name} ({duration:.1f}s)")

        decode_params = {
            "use_gpu": 1,
            "beam_size": 3,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.25,
            "aed_length_penalty": 0.6,
            "eos_penalty": 1.0,
        }

        # 优先用 VAD 分段，回退到固定 55s 分块
        segments = self._transcribe_with_vad(audio_path, duration, decode_params)

        if segments is None:
            # VAD 不可用
            if duration > 60:
                segments = self._transcribe_chunked(audio_path, duration, decode_params)
            else:
                segments = self._transcribe_single(
                    audio_path, 0.0, duration, decode_params, seg_id_start=0,
                )

        logger.info(f"[FireRedASR] 转写完成: 片段数={len(segments)}")

        return TranscriptResult(
            language="zh",
            language_probability=1.0,
            duration=duration,
            segments=segments,
        )

    def _transcribe_with_vad(
        self,
        audio_path: Path,
        duration: float,
        decode_params: dict,
    ) -> list[Segment] | None:
        """VAD 预分段转写：用 fsmn-vad 切段后逐段转写"""
        # FireRedASR 60s 限制 — VAD 段不超过 55s
        vad_segs = _run_vad(audio_path, max_single_segment_time=55000)
        if not vad_segs:
            return None

        import soundfile as sf
        audio, sr = sf.read(str(audio_path))

        logger.info(
            f"[FireRedASR] VAD 预分段: {len(vad_segs)} 段, 逐段转写中..."
        )

        segments: list[Segment] = []
        seg_id = 0

        for i, (vad_start, vad_end) in enumerate(vad_segs):
            start_sample = int(vad_start * sr)
            end_sample = int(vad_end * sr)
            chunk = audio[start_sample:end_sample]

            if len(chunk) < sr * 0.1:
                continue

            # 写临时 WAV
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            chunk_path = f.name
            f.close()

            try:
                sf.write(chunk_path, chunk, sr)
                chunk_segs = self._transcribe_single(
                    Path(chunk_path),
                    offset=vad_start,
                    duration=vad_end - vad_start,
                    decode_params=decode_params,
                    seg_id_start=seg_id,
                )
                segments.extend(chunk_segs)
                seg_id += len(chunk_segs)
            except Exception as e:
                logger.warning(f"[FireRedASR] VAD 段 {i} 转写失败: {e}")
            finally:
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass

        if not segments:
            return None

        logger.info(
            f"[FireRedASR] VAD 预分段转写完成: "
            f"{len(vad_segs)} VAD段 → {len(segments)} 文本段"
        )
        return segments

    def _transcribe_single(
        self,
        audio_path: Path,
        offset: float,
        duration: float,
        decode_params: dict,
        seg_id_start: int = 0,
    ) -> list[Segment]:
        """转写单段音频（≤60s）"""
        results = self._model.transcribe(
            [f"utt_{seg_id_start}"],
            [str(audio_path)],
            decode_params,
        )
        text = ""
        if results and isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], dict):
                text = results[0].get("text", "")
            elif isinstance(results[0], str):
                text = results[0]

        text = text.strip()
        if not text:
            return []

        # FireRedASR 不返回标点，用 ct-punc 恢复
        text = _add_punctuation(text)

        return [Segment(
            id=seg_id_start,
            start=round(offset, 3),
            end=round(offset + duration, 3),
            text=text,
            speaker=None,
        )]

    def _transcribe_chunked(
        self,
        audio_path: Path,
        total_duration: float,
        decode_params: dict,
    ) -> list[Segment]:
        """分块转写长音频（>60s，VAD 不可用时的回退方案）"""
        import soundfile as sf

        audio, sr = sf.read(str(audio_path))
        chunk_seconds = 55  # 安全边界
        segments: list[Segment] = []
        seg_id = 0

        for start_sec in range(0, int(total_duration) + 1, chunk_seconds):
            end_sec = min(start_sec + chunk_seconds, total_duration)
            if end_sec - start_sec < 0.5:
                break

            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            chunk = audio[start_sample:end_sample]

            # Windows 安全的临时文件
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            chunk_path = f.name
            f.close()

            try:
                sf.write(chunk_path, chunk, sr)
                chunk_segs = self._transcribe_single(
                    Path(chunk_path),
                    offset=float(start_sec),
                    duration=float(end_sec - start_sec),
                    decode_params=decode_params,
                    seg_id_start=seg_id,
                )
                segments.extend(chunk_segs)
                seg_id += len(chunk_segs)
            finally:
                os.unlink(chunk_path)

        return segments

    def unload(self) -> None:
        self._model = None
        self._model_dir = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Engine type detection
# ---------------------------------------------------------------------------

def detect_engine_type(model_dir: Path) -> str | None:
    """根据目录内容自动检测 ASR 引擎类型"""
    if not model_dir.is_dir():
        return None

    files = {f.name for f in model_dir.iterdir() if f.is_file()}

    # faster-whisper: CTranslate2 格式 (model.bin + vocabulary.json)
    if ("model.bin" in files or "model.safetensors" in files) and "vocabulary.json" in files:
        return "faster-whisper"

    # FunASR: configuration.json 或 model.py + config.yaml
    if "configuration.json" in files:
        return "funasr"
    if "model.py" in files and ("config.yaml" in files or "config.json" in files):
        return "funasr"

    # FireRedASR: spm.model (SentencePiece)
    if "spm.model" in files:
        return "fireredasr"

    # 目录名回退
    name = model_dir.name.lower()
    if "whisper" in name:
        return "faster-whisper"
    if "sensevoice" in name or "paraformer" in name:
        return "funasr"
    if "firered" in name:
        return "fireredasr"

    return None


# ---------------------------------------------------------------------------
# Model dir resolution
# ---------------------------------------------------------------------------

def _resolve_model_dir(model_name: str) -> Path:
    """解析模型名称为目录路径，按优先级查找"""
    models_dir = get_settings().paths.models_dir

    # 1. models/asr/{name}/
    d = models_dir / "asr" / model_name
    if d.exists():
        return d

    # 2. models/whisper/faster-whisper-{name}/  （兼容旧的 name="medium"）
    d = models_dir / "whisper" / f"faster-whisper-{model_name}"
    if d.exists():
        return d

    # 3. models/whisper/{name}/  （完整目录名）
    d = models_dir / "whisper" / model_name
    if d.exists():
        return d

    raise FileNotFoundError(
        f"ASR 模型未找到: {model_name}\n"
        f"搜索路径:\n"
        f"  - {models_dir / 'asr' / model_name}\n"
        f"  - {models_dir / 'whisper' / f'faster-whisper-{model_name}'}\n"
        f"  - {models_dir / 'whisper' / model_name}"
    )


# ---------------------------------------------------------------------------
# Factory (singleton with model switching)
# ---------------------------------------------------------------------------

_engine: ASREngine | None = None
_engine_model: str | None = None

_ENGINE_MAP = {
    "faster-whisper": FasterWhisperEngine,
    "funasr": FunASRFileEngine,
    "fireredasr": FireRedASREngine,
}


def get_asr_engine(model_name: str | None = None) -> ASREngine:
    """获取 ASR 引擎（单例 + 模型切换）"""
    global _engine, _engine_model

    if model_name is None:
        model_name = get_settings().asr.model_name

    # 模型切换：释放旧引擎
    if _engine is not None and _engine_model != model_name:
        logger.info(f"ASR 引擎切换: {_engine_model} → {model_name}")
        _engine.unload()
        _engine = None

    if _engine is None:
        model_dir = _resolve_model_dir(model_name)
        engine_type = detect_engine_type(model_dir)

        if engine_type is None:
            raise ValueError(f"无法识别 ASR 引擎类型: {model_dir}")

        engine_cls = _ENGINE_MAP.get(engine_type)
        if engine_cls is None:
            raise ValueError(f"不支持的 ASR 引擎: {engine_type}")

        _engine = engine_cls()
        _engine.load(model_dir)
        _engine_model = model_name
        logger.info(f"ASR 引擎就绪: {model_name} (type={engine_type})")

    return _engine


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

# 保留旧的 ASRService 别名，供 __init__.py 和 process.py 向后兼容
ASRService = ASREngine


def get_asr_service() -> ASREngine:
    """向后兼容别名，等价于 get_asr_engine()"""
    return get_asr_engine()
