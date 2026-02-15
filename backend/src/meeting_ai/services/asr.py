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
from ..models import CharTimestamp, Segment, TranscriptResult

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
        logger.debug(f"VAD generate() 返回: type={type(res)}, len={len(res) if isinstance(res, list) else 'N/A'}")

        # res 格式: [{"text": [[start_ms, end_ms], [start_ms, end_ms], ...]}]
        if not res or not isinstance(res, list):
            logger.warning(f"VAD 返回格式错误: res={type(res)}")
            return None

        if not res[0]:
            logger.warning("VAD res[0] 为空")
            return None

        logger.debug(f"VAD res[0] type={type(res[0])}, keys={res[0].keys() if isinstance(res[0], dict) else 'not dict'}")

        raw_segments = res[0].get("text", []) if isinstance(res[0], dict) else []
        if not raw_segments:
            logger.warning(f"VAD 未检测到语音段: raw_segments={raw_segments}")
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
# Forced Alignment (用于没有原生时间戳的 ASR 引擎)
# ---------------------------------------------------------------------------

_fa_model = None  # Forced Alignment 参考模型（Paraformer-Large）
_fa_model_name = None


def _get_fa_model():
    """
    懒加载强制对齐参考模型（单例）

    优先使用 Paraformer-Large，回退到 SenseVoice-Small
    """
    global _fa_model, _fa_model_name
    if _fa_model is not None:
        return _fa_model, _fa_model_name

    from funasr import AutoModel
    models_dir = get_settings().paths.models_dir

    # 优先: Paraformer-Large (字级时间戳最准)
    paraformer_dir = models_dir / "asr" / "paraformer-large"
    if paraformer_dir.exists():
        logger.info(f"[强制对齐] 加载 Paraformer-Large: {paraformer_dir}")
        _fa_model = AutoModel(
            model=str(paraformer_dir),
            disable_update=True,
            disable_pbar=True,
        )
        _fa_model_name = "paraformer-large"
        return _fa_model, _fa_model_name

    # 回退: SenseVoice-Small
    sensevoice_dir = models_dir / "asr" / "sensevoice-small"
    if sensevoice_dir.exists():
        logger.info(f"[强制对齐] 加载 SenseVoice-Small: {sensevoice_dir}")
        _fa_model = AutoModel(
            model=str(sensevoice_dir),
            disable_update=True,
            disable_pbar=True,
        )
        _fa_model_name = "sensevoice-small"
        return _fa_model, _fa_model_name

    logger.warning("[强制对齐] 无可用参考模型 (需 paraformer-large 或 sensevoice-small)")
    return None, None


def _lcs_align_timestamps(
    target_text: str,
    ref_text: str,
    ref_timestamps: list[tuple[int, int]],  # [(start_ms, end_ms), ...]
) -> list[CharTimestamp]:
    """
    用 LCS（最长公共子序列）动态规划算法对齐两个文本的时间戳

    Args:
        target_text: 目标文本（需要时间戳的文本，如 FireRedASR 输出）
        ref_text: 参考文本（有时间戳的文本，如 Paraformer 输出）
        ref_timestamps: 参考文本的字级时间戳列表

    Returns:
        target_text 的字级时间戳（匹配字复用参考时间戳，不匹配字线性插值）
    """
    if not target_text or not ref_text or not ref_timestamps:
        return []

    target_chars = list(target_text)
    ref_chars = list(ref_text)
    n, m = len(target_chars), len(ref_chars)

    # DP 表: lcs[i][j] = LCS 长度
    lcs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target_chars[i - 1] == ref_chars[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])

    # 回溯找对齐关系
    alignment: dict[int, int] = {}  # target_idx -> ref_idx
    i, j = n, m
    while i > 0 and j > 0:
        if target_chars[i - 1] == ref_chars[j - 1]:
            alignment[i - 1] = j - 1
            i -= 1
            j -= 1
        elif lcs[i - 1][j] > lcs[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # 生成时间戳
    char_ts: list[CharTimestamp] = []
    for idx, ch in enumerate(target_chars):
        if idx in alignment:
            # 匹配字：复用参考时间戳
            ref_idx = alignment[idx]
            if ref_idx < len(ref_timestamps):
                start_ms, end_ms = ref_timestamps[ref_idx]
                char_ts.append(CharTimestamp(
                    char=ch,
                    start=start_ms / 1000.0,
                    end=end_ms / 1000.0,
                ))
            else:
                # 超界，用线性插值
                char_ts.append(CharTimestamp(char=ch, start=0.0, end=0.0))
        else:
            # 不匹配字：线性插值
            # 找前后最近的匹配字
            prev_idx = next((k for k in range(idx - 1, -1, -1) if k in alignment), None)
            next_idx = next((k for k in range(idx + 1, n) if k in alignment), None)

            if prev_idx is not None and next_idx is not None:
                # 前后都有匹配，插值
                prev_ref = alignment[prev_idx]
                next_ref = alignment[next_idx]
                _, prev_end_ms = ref_timestamps[prev_ref]
                next_start_ms, _ = ref_timestamps[next_ref]
                gap_chars = next_idx - prev_idx
                char_dur_ms = (next_start_ms - prev_end_ms) / gap_chars
                offset_chars = idx - prev_idx
                start_ms = prev_end_ms + offset_chars * char_dur_ms
                end_ms = start_ms + char_dur_ms
            elif prev_idx is not None:
                # 只有前面有匹配，延长
                prev_ref = alignment[prev_idx]
                _, prev_end_ms = ref_timestamps[prev_ref]
                start_ms = prev_end_ms + 50 * (idx - prev_idx)
                end_ms = start_ms + 50
            elif next_idx is not None:
                # 只有后面有匹配，前移
                next_ref = alignment[next_idx]
                next_start_ms, _ = ref_timestamps[next_ref]
                end_ms = next_start_ms - 50 * (next_idx - idx)
                start_ms = end_ms - 50
            else:
                # 完全没有匹配（极端情况），假设均匀分布
                start_ms = idx * 200
                end_ms = start_ms + 200

            char_ts.append(CharTimestamp(
                char=ch,
                start=round(start_ms / 1000.0, 3),
                end=round(end_ms / 1000.0, 3),
            ))

    return char_ts


def _forced_align(
    audio_path: Path,
    target_text: str,
    segment_start: float,
    segment_end: float,
) -> list[CharTimestamp]:
    """
    强制对齐：用参考 ASR 模型（Paraformer）转写同一段音频，
    然后用 LCS 算法将目标文本对齐到参考时间戳

    Args:
        audio_path: 音频文件路径
        target_text: 需要时间戳的文本
        segment_start: 音频段起始时间（秒）
        segment_end: 音频段结束时间（秒）

    Returns:
        字级时间戳列表
    """
    fa_model, model_name = _get_fa_model()
    if fa_model is None:
        logger.warning("[强制对齐] 无参考模型，跳过")
        return []

    try:
        # 用参考模型转写同一段音频
        # 注意：不同参数可能影响是否返回 timestamp
        res = fa_model.generate(
            input=str(audio_path),
            cache={},
            batch_size_s=300,
        )

        logger.debug(f"[强制对齐] 参考模型返回: type={type(res)}, len={len(res) if isinstance(res, list) else 'N/A'}")

        if not res or not isinstance(res, list):
            logger.warning(f"[强制对齐] 参考模型未返回列表: {type(res)}")
            return []

        if not res or not isinstance(res[0], dict):
            logger.warning(f"[强制对齐] 参考模型未返回字典: {type(res[0]) if res else 'empty'}")
            return []

        logger.debug(f"[强制对齐] res[0] keys: {res[0].keys()}")

        ref_text = res[0].get("text", "")
        ref_timestamps = res[0].get("timestamp", [])

        if not ref_text:
            logger.warning("[强制对齐] 参考模型未返回文本")
            return []

        if not ref_timestamps:
            logger.warning(f"[强制对齐] 参考模型未返回时间戳 (text_len={len(ref_text)})")
            return []

        # LCS 对齐
        logger.debug(f"[强制对齐] target={target_text[:20]}..., ref={ref_text[:20]}...")
        char_ts = _lcs_align_timestamps(target_text, ref_text, ref_timestamps)

        # 调整时间偏移（参考模型转写的是整个文件，需要加上 segment_start）
        for ts in char_ts:
            ts.start += segment_start
            ts.end += segment_start

        logger.info(f"[强制对齐] 成功: {len(char_ts)} 个字符")
        return char_ts

    except Exception as e:
        logger.warning(f"[强制对齐] 失败: {e}")
        return []


def unload_fa_model():
    """释放强制对齐参考模型"""
    global _fa_model, _fa_model_name
    if _fa_model is not None:
        logger.info(f"释放强制对齐模型: {_fa_model_name}")
        _fa_model = None
        _fa_model_name = None
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

        # 强制开启 word_timestamps 以支持字级对齐
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=lang,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
            word_timestamps=True,
        )

        segments = []
        all_char_ts: list[list[CharTimestamp]] = []

        for seg in segments_iter:
            segments.append(Segment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                speaker=None,
            ))

            # 提取词级时间戳
            seg_char_ts: list[CharTimestamp] = []
            if seg.words:
                for w in seg.words:
                    seg_char_ts.append(CharTimestamp(
                        char=w.word.strip(),
                        start=w.start,
                        end=w.end,
                    ))
            all_char_ts.append(seg_char_ts)

        has_ts = sum(1 for ts in all_char_ts if ts)
        logger.info(
            f"[faster-whisper] 转写完成: 语言={info.language}, "
            f"概率={info.language_probability:.2f}, 片段数={len(segments)}, "
            f"词级时间戳={has_ts}/{len(segments)}"
        )

        return TranscriptResult(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segments,
            char_timestamps=all_char_ts if has_ts > 0 else None,
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
        result = self._transcribe_with_vad(audio_path, duration, language)

        if result is None:
            # VAD 不可用，回退到整文件转写
            logger.info(f"[{engine_name}] VAD 不可用，使用整文件转写")
            result = self._transcribe_whole_file(audio_path, duration, language)

        logger.info(f"[{engine_name}] 转写完成: 片段数={len(result.segments)}")
        return result

    def _transcribe_whole_file(
        self,
        audio_path: Path,
        duration: float,
        language: str | None = None,
    ) -> TranscriptResult:
        """
        整文件转写（作为 VAD 回退）

        返回: TranscriptResult（包含字级时间戳）
        """
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
        logger.debug(f"[FunASR] generate() 返回类型: {type(res)}, 长度: {len(res) if isinstance(res, list) else 'N/A'}")
        if res and isinstance(res, list) and len(res) > 0:
            logger.debug(f"[FunASR] 第一项类型: {type(res[0])}, 内容keys: {res[0].keys() if isinstance(res[0], dict) else 'N/A'}")
            if isinstance(res[0], dict) and 'timestamp' in res[0]:
                ts = res[0]['timestamp']
                logger.debug(f"[FunASR] timestamp类型: {type(ts)}, 长度: {len(ts) if isinstance(ts, list) else 'N/A'}")

        segments, char_timestamps = self._parse_funasr_output_with_timestamps(res, duration)
        logger.debug(f"[FunASR] _parse返回: {len(segments)}个段, char_timestamps长度: {len(char_timestamps)}")
        if char_timestamps:
            logger.debug(f"[FunASR] 第一段char_ts数量: {len(char_timestamps[0]) if char_timestamps[0] else 0}")

        has_ts = sum(1 for ts in char_timestamps if ts) if char_timestamps else 0

        # 如果没有任何原生时间戳，尝试强制对齐
        if has_ts == 0 and segments:
            logger.info(f"[FunASR] 整文件转写无原生时间戳, 尝试强制对齐")
            for i, seg in enumerate(segments):
                if not seg.text:
                    continue
                # 对每个 segment 做强制对齐
                aligned_ts = _forced_align(audio_path, seg.text, seg.start, seg.end)
                if aligned_ts:
                    char_timestamps[i] = aligned_ts
                    logger.debug(f"[FunASR] 段{i}强制对齐成功: {len(aligned_ts)}字")
                else:
                    logger.debug(f"[FunASR] 段{i}强制对齐失败, 保留线性插值")
            has_ts = sum(1 for ts in char_timestamps if ts) if char_timestamps else 0

        logger.info(f"[FunASR] 整文件转写: {len(segments)}段, {has_ts}/{len(segments)}段有字级时间戳")
        return TranscriptResult(
            language=language or "zh",
            language_probability=1.0,
            duration=duration,
            segments=segments,
            char_timestamps=char_timestamps if has_ts > 0 else None,
        )

    def _transcribe_with_vad(
        self,
        audio_path: Path,
        duration: float,
        language: str | None = None,
    ) -> TranscriptResult | None:
        """VAD 预分段转写：先切成短片段，逐段转写，提取字级时间戳"""
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
        all_char_ts: list[list[CharTimestamp]] = []
        seg_id = 0

        for i, (vad_start, vad_end) in enumerate(vad_segs):
            start_sample = int(vad_start * sr)
            end_sample = int(vad_end * sr)
            chunk = audio[start_sample:end_sample]

            if len(chunk) < sr * 0.1:
                continue

            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            chunk_path = f.name
            f.close()

            try:
                sf.write(chunk_path, chunk, sr)

                generate_kwargs: dict = {
                    "input": chunk_path,
                    "cache": {},
                }
                if self._is_sensevoice:
                    generate_kwargs["language"] = language or "auto"
                    generate_kwargs["use_itn"] = True

                res = self._model.generate(**generate_kwargs)

                # 提取文本和字级时间戳
                text, char_ts = self._extract_text_and_timestamps(res, vad_start)
                if not text:
                    continue

                # 如果没有原生字级时间戳，尝试强制对齐 → 回退到线性插值
                if not char_ts and text:
                    logger.debug(f"[{engine_name}] VAD段{i}无原生timestamp, 尝试强制对齐")
                    # 优先：强制对齐（使用 Paraformer 参考模型）
                    char_ts = _forced_align(Path(chunk_path), text, vad_start, vad_end)

                    # 回退：线性插值
                    if not char_ts:
                        logger.debug(f"[{engine_name}] VAD段{i}强制对齐失败, 使用线性插值")
                        chars = list(text)
                        seg_duration = vad_end - vad_start
                        char_duration = seg_duration / len(chars) if chars else 0

                        for j, ch in enumerate(chars):
                            ch_start = vad_start + j * char_duration
                            ch_end = vad_start + (j + 1) * char_duration
                            char_ts.append(CharTimestamp(
                                char=ch,
                                start=round(ch_start, 3),
                                end=round(ch_end, 3),
                            ))

                segments.append(Segment(
                    id=seg_id,
                    start=round(vad_start, 3),
                    end=round(vad_end, 3),
                    text=text,
                    speaker=None,
                ))
                all_char_ts.append(char_ts)
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

        has_ts = sum(1 for ts in all_char_ts if ts)
        logger.info(
            f"[{engine_name}] VAD 预分段转写完成: "
            f"{len(vad_segs)} VAD段 → {len(segments)} 文本段, "
            f"{has_ts}/{len(segments)} 段有字级时间戳"
        )

        return TranscriptResult(
            language=language or "zh",
            language_probability=1.0,
            duration=duration,
            segments=segments,
            char_timestamps=all_char_ts if has_ts > 0 else None,
        )

    def _extract_text_from_result(self, res: list) -> str:
        """从 FunASR generate() 输出中提取纯文本"""
        text, _ = self._extract_text_and_timestamps(res, time_offset=0.0)
        return text

    def _extract_text_and_timestamps(
        self,
        res: list,
        time_offset: float = 0.0,
    ) -> tuple[str, list[CharTimestamp]]:
        """
        从 FunASR generate() 输出中提取文本和字级时间戳。

        time_offset: 加到每个时间戳上的偏移（VAD 段在整个音频中的起始时间）。
        返回: (text, char_timestamps)
        """
        if not res:
            logger.debug("[FunASR] _extract_text_and_timestamps: res为空")
            return "", []

        for item in res:
            if not isinstance(item, dict):
                logger.debug(f"[FunASR] _extract: item不是dict, 类型={type(item)}")
                continue
            raw_text = item.get("text", "")
            if not raw_text or not raw_text.strip():
                logger.debug("[FunASR] _extract: text为空")
                continue

            # 提取原始时间戳（毫秒级，对应原始 raw_text 的每个字符）
            raw_timestamps = item.get("timestamp", [])
            logger.debug(f"[FunASR] _extract: text长度={len(raw_text)}, timestamp存在={bool(raw_timestamps)}, timestamp长度={len(raw_timestamps) if raw_timestamps else 0}")

            # SenseVoice 后处理
            if self._is_sensevoice:
                try:
                    from funasr.utils.postprocess_utils import rich_transcription_postprocess
                    raw_text = rich_transcription_postprocess(raw_text)
                except ImportError:
                    pass
                raw_text = re.sub(
                    r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F'
                    r'\U0000200D\U00002600-\U000026FF]+',
                    '', raw_text,
                )

            text = raw_text.strip()
            if not text:
                return "", []

            # 构建字级时间戳
            char_ts: list[CharTimestamp] = []
            if raw_timestamps and len(raw_timestamps) > 0:
                # FunASR timestamp 格式: [[start_ms, end_ms], ...]
                # 对应原始（后处理前）文本的每个字符
                text_chars = list(text)
                if len(raw_timestamps) == len(text_chars):
                    # 长度匹配，直接对应
                    logger.debug(f"[FunASR] _extract: timestamp长度匹配, 直接对应 ({len(text_chars)}字)")
                    for ch, (start_ms, end_ms) in zip(text_chars, raw_timestamps):
                        char_ts.append(CharTimestamp(
                            char=ch,
                            start=start_ms / 1000.0 + time_offset,
                            end=end_ms / 1000.0 + time_offset,
                        ))
                elif len(raw_timestamps) >= 2:
                    # 长度不匹配（后处理改变了文本）
                    # 用首尾时间戳做线性插值
                    logger.debug(f"[FunASR] _extract: timestamp长度不匹配({len(raw_timestamps)}!={len(text_chars)}), 线性插值")
                    total_start = raw_timestamps[0][0] / 1000.0 + time_offset
                    total_end = raw_timestamps[-1][1] / 1000.0 + time_offset
                    total_dur = total_end - total_start
                    char_dur = total_dur / len(text_chars) if text_chars else 0

                    for i, ch in enumerate(text_chars):
                        ch_start = total_start + i * char_dur
                        ch_end = total_start + (i + 1) * char_dur
                        char_ts.append(CharTimestamp(
                            char=ch, start=round(ch_start, 3), end=round(ch_end, 3),
                        ))
            else:
                logger.debug("[FunASR] _extract: 无timestamp数据")

            logger.debug(f"[FunASR] _extract: 返回text长度={len(text)}, char_ts数量={len(char_ts)}")
            return text, char_ts

        logger.debug("[FunASR] _extract: 无有效item, 返回空")
        return "", []

    def _parse_funasr_output(
        self, res: list, duration: float,
    ) -> list[Segment]:
        """解析 FunASR model.generate() 的输出（不含字级时间戳）"""
        segments, _ = self._parse_funasr_output_with_timestamps(res, duration)
        return segments

    def _parse_funasr_output_with_timestamps(
        self, res: list, duration: float,
    ) -> tuple[list[Segment], list[list[CharTimestamp]]]:
        """
        解析 FunASR model.generate() 的输出，同时提取字级时间戳。

        返回: (segments, char_timestamps)
        """
        logger.debug(f"[FunASR] _parse_funasr_output_with_timestamps: 收到{len(res)}项")
        segments: list[Segment] = []
        all_char_ts: list[list[CharTimestamp]] = []
        seg_id = 0
        last_end = 0.0

        for item in res:
            if not isinstance(item, dict):
                continue

            # 使用 _extract_text_and_timestamps 提取文本和字级时间戳
            text, char_ts = self._extract_text_and_timestamps([item], time_offset=0.0)
            if not text:
                continue

            # 提取片段时间戳
            timestamp = item.get("timestamp", [])
            if timestamp:
                start = timestamp[0][0] / 1000.0
                end = timestamp[-1][1] / 1000.0
            else:
                # 无时间戳：按文本长度估算持续时间
                estimated_dur = len(text) * 0.3
                start = last_end
                end = min(start + estimated_dur, duration)

            last_end = end

            # 如果没有原生字级时间戳，生成线性插值时间戳
            if not char_ts and text:
                logger.debug(f"[FunASR] _parse: 段{seg_id}无原生timestamp, 生成线性插值")
                chars = list(text)
                seg_duration = end - start
                char_duration = seg_duration / len(chars) if chars else 0

                for i, ch in enumerate(chars):
                    ch_start = start + i * char_duration
                    ch_end = start + (i + 1) * char_duration
                    char_ts.append(CharTimestamp(
                        char=ch,
                        start=round(ch_start, 3),
                        end=round(ch_end, 3),
                    ))

            segments.append(Segment(
                id=seg_id,
                start=start,
                end=end,
                text=text,
                speaker=None,
            ))
            all_char_ts.append(char_ts)
            logger.debug(f"[FunASR] _parse: 段{seg_id}: text长度={len(text)}, char_ts数量={len(char_ts)}")
            seg_id += 1

        logger.debug(f"[FunASR] _parse: 返回{len(segments)}段, {len(all_char_ts)}个char_ts列表")
        return segments, all_char_ts

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
        result = self._transcribe_with_vad(audio_path, duration, decode_params)

        if result is None:
            # VAD 不可用，使用整文件/分块转写 + 生成线性插值时间戳
            logger.info("[FireRedASR] VAD 不可用，使用整文件转写")
            if duration > 60:
                segments = self._transcribe_chunked(audio_path, duration, decode_params)
            else:
                segments = self._transcribe_single(
                    audio_path, 0.0, duration, decode_params, seg_id_start=0,
                )

            # 为每个 segment 生成字级时间戳（强制对齐 → 线性插值回退）
            all_char_ts: list[list[CharTimestamp]] = []
            for i, seg in enumerate(segments):
                # 优先：强制对齐
                char_ts = _forced_align(audio_path, seg.text, seg.start, seg.end)
                if not char_ts:
                    # 回退：线性插值
                    logger.debug(f"[FireRedASR] 段{i}强制对齐失败, 使用线性插值")
                    char_ts = self._generate_char_timestamps_linear(
                        seg.text, seg.start, seg.end
                    )
                all_char_ts.append(char_ts)

            has_ts = sum(1 for ts in all_char_ts if ts)
            fa_count = sum(1 for seg, ts in zip(segments, all_char_ts) if ts and any(ch.char for ch in ts))
            logger.info(f"[FireRedASR] 整文件转写: {len(segments)}段, {has_ts}/{len(segments)}段有字级时间戳 (强制对齐+线性插值)")
            result = TranscriptResult(
                language="zh",
                language_probability=1.0,
                duration=duration,
                segments=segments,
                char_timestamps=all_char_ts if has_ts > 0 else None,
            )

        logger.info(f"[FireRedASR] 转写完成: 片段数={len(result.segments)}")
        return result

    def _transcribe_with_vad(
        self,
        audio_path: Path,
        duration: float,
        decode_params: dict,
    ) -> TranscriptResult | None:
        """VAD 预分段转写 + 生成字级时间戳（线性插值）"""
        vad_segs = _run_vad(audio_path, max_single_segment_time=55000)
        if not vad_segs:
            return None

        import soundfile as sf
        audio, sr = sf.read(str(audio_path))

        logger.info(
            f"[FireRedASR] VAD 预分段: {len(vad_segs)} 段, 逐段转写中..."
        )

        segments: list[Segment] = []
        all_char_ts: list[list[CharTimestamp]] = []
        seg_id = 0

        for i, (vad_start, vad_end) in enumerate(vad_segs):
            start_sample = int(vad_start * sr)
            end_sample = int(vad_end * sr)
            chunk = audio[start_sample:end_sample]

            if len(chunk) < sr * 0.1:
                continue

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

                # 为每个 segment 生成字级时间戳（强制对齐 → 线性插值回退）
                for seg in chunk_segs:
                    # 优先：强制对齐
                    char_ts = _forced_align(Path(chunk_path), seg.text, seg.start, seg.end)
                    if not char_ts:
                        # 回退：线性插值
                        char_ts = self._generate_char_timestamps_linear(
                            seg.text, seg.start, seg.end
                        )
                    all_char_ts.append(char_ts)

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

        has_ts = sum(1 for ts in all_char_ts if ts)
        logger.info(
            f"[FireRedASR] VAD 预分段转写完成: "
            f"{len(vad_segs)} VAD段 → {len(segments)} 文本段, "
            f"{has_ts}/{len(segments)} 段有字级时间戳"
        )

        return TranscriptResult(
            language="zh",
            language_probability=1.0,
            duration=duration,
            segments=segments,
            char_timestamps=all_char_ts if has_ts > 0 else None,
        )

    def _generate_char_timestamps_linear(
        self, text: str, start: float, end: float
    ) -> list[CharTimestamp]:
        """
        为文本生成字级时间戳（线性插值）

        FireRedASR 没有原生字级时间戳，使用均匀分布作为近似。
        """
        if not text:
            return []

        chars = list(text)
        duration = end - start
        char_duration = duration / len(chars) if chars else 0

        char_ts = []
        for i, ch in enumerate(chars):
            char_start = start + i * char_duration
            char_end = start + (i + 1) * char_duration
            char_ts.append(CharTimestamp(
                char=ch,
                start=round(char_start, 3),
                end=round(char_end, 3),
            ))

        return char_ts

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
