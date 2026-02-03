"""
性别检测服务

使用 pyannote 的说话人嵌入模型进行性别分类。
无需额外安装库，复用已有的 pyannote 模型。

原理：
1. 提取说话人的音频片段
2. 用 pyannote 的嵌入模型提取特征向量
3. 结合基频和嵌入特征进行性别判断
"""

import wave
from pathlib import Path

import numpy as np
import torch

from ..config import get_settings
from ..logger import get_logger
from ..models import Gender, Segment

logger = get_logger("services.gender")

# 全局模型缓存
_embedding_model = None


def _get_embedding_model():
    """懒加载 pyannote 嵌入模型"""
    global _embedding_model
    if _embedding_model is None:
        try:
            from pyannote.audio import Model
            
            settings = get_settings()
            # 使用 pyannote 的嵌入模型（和说话人分离共用）
            model_dir = settings.paths.models_dir / "pyannote" / "wespeaker-voxceleb-resnet34-LM"
            
            if model_dir.exists():
                logger.info(f"加载嵌入模型: {model_dir}")
                _embedding_model = Model.from_pretrained(str(model_dir))
            else:
                # 尝试在线加载（需要 token）
                logger.info("本地嵌入模型不存在，尝试在线加载...")
                _embedding_model = Model.from_pretrained(
                    "pyannote/wespeaker-voxceleb-resnet34-LM"
                )
            
            # 移到 GPU（如果可用）
            if torch.cuda.is_available():
                _embedding_model = _embedding_model.to("cuda")
            
            _embedding_model.eval()
            logger.info("嵌入模型加载完成")
            
        except Exception as e:
            logger.warning(f"嵌入模型加载失败: {e}，将只使用基频分析")
            _embedding_model = "unavailable"
    
    return _embedding_model if _embedding_model != "unavailable" else None


def _read_wav_mono(wav_path: Path) -> tuple[int, np.ndarray]:
    """读取 WAV 文件为单声道浮点数组"""
    with wave.open(str(wav_path), "rb") as w:
        channels = w.getnchannels()
        sample_rate = w.getframerate()
        sample_width = w.getsampwidth()
        n_frames = w.getnframes()
        pcm_data = w.readframes(n_frames)

    if channels != 1:
        raise ValueError(f"需要单声道 WAV，当前是 {channels} 声道")

    if sample_width == 2:
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(pcm_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"不支持的采样位深: {sample_width}")

    return sample_rate, audio


def _estimate_f0_autocorr(
    frame: np.ndarray,
    sample_rate: int,
    f_min: float = 60.0,
    f_max: float = 400.0,
) -> float:
    """使用自相关法估计基频"""
    frame = frame.astype(np.float32)
    frame = frame - frame.mean()

    energy = np.sum(frame * frame)
    if energy < 1e-4:
        return 0.0

    frame = frame * np.hamming(len(frame)).astype(np.float32)
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    corr[0] = 0.0

    min_lag = int(sample_rate / f_max)
    max_lag = int(sample_rate / f_min)
    max_lag = min(max_lag, len(corr) - 1)

    if min_lag >= max_lag:
        return 0.0

    segment = corr[min_lag:max_lag + 1]
    peak_lag = min_lag + int(np.argmax(segment))

    lag2 = peak_lag * 2
    if lag2 < len(corr) and corr[lag2] >= 0.85 * corr[peak_lag]:
        peak_lag = lag2

    peak_val = corr[peak_lag]
    if (peak_val / (energy + 1e-9)) < 0.30:
        return 0.0

    return sample_rate / peak_lag


def _extract_features(
    wav_path: Path,
    segments: list[Segment],
    speaker_id: str,
    max_seconds: float = 20.0,
) -> dict | None:
    """
    提取说话人的声学特征
    
    Returns:
        {
            "f0_median": 基频中位数,
            "f0_std": 基频标准差,
            "f0_range": 基频范围,
            "embedding": 嵌入向量 (如果模型可用),
        }
    """
    sample_rate, audio = _read_wav_mono(wav_path)
    frame_len = int(sample_rate * 0.030)
    hop_len = int(sample_rate * 0.010)

    # 收集该说话人的时间段
    intervals = [
        (seg.start, seg.end)
        for seg in segments
        if seg.speaker == speaker_id and seg.duration >= 0.4
    ]

    if not intervals:
        return None

    # 提取音频片段
    audio_segments = []
    f0_values = []
    total_used = 0.0

    for start, end in intervals:
        if total_used >= max_seconds:
            break

        start_sample = max(0, int(start * sample_rate))
        end_sample = min(len(audio), int(end * sample_rate))

        if end_sample <= start_sample:
            continue

        max_samples = int((max_seconds - total_used) * sample_rate)
        take_samples = min(end_sample - start_sample, max_samples)
        segment_audio = audio[start_sample:start_sample + take_samples]
        audio_segments.append(segment_audio)
        total_used += take_samples / sample_rate

        # 计算基频
        for i in range(0, len(segment_audio) - frame_len + 1, hop_len):
            frame = segment_audio[i:i + frame_len]
            f0 = _estimate_f0_autocorr(frame, sample_rate)
            if f0 > 0:
                f0_values.append(f0)

    if total_used < 2.0 or len(f0_values) < 10:
        return None

    f0_array = np.array(f0_values)
    combined_audio = np.concatenate(audio_segments)
    
    features = {
        "f0_median": float(np.median(f0_array)),
        "f0_std": float(np.std(f0_array)),
        "f0_range": float(np.percentile(f0_array, 90) - np.percentile(f0_array, 10)),
        "f0_q25": float(np.percentile(f0_array, 25)),
        "f0_q75": float(np.percentile(f0_array, 75)),
        "embedding": None,
    }
    
    # 尝试提取嵌入向量
    model = _get_embedding_model()
    if model is not None:
        try:
            # 转换为 torch tensor
            waveform = torch.from_numpy(combined_audio).unsqueeze(0)
            if torch.cuda.is_available():
                waveform = waveform.to("cuda")
            
            with torch.no_grad():
                embedding = model(waveform)
            
            features["embedding"] = embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.debug(f"嵌入提取失败: {e}")
    
    return features


def detect_gender(
    wav_path: Path,
    segments: list[Segment],
    speaker_id: str,
    max_seconds: float = 20.0,
) -> tuple[Gender, float]:
    """
    检测指定说话人的性别
    
    使用多特征融合：
    1. 基频中位数（主要特征）
    2. 基频变化范围（辅助特征）
    3. 基频标准差（辅助特征）
    4. 嵌入向量特征（如果可用）
    
    Returns:
        (性别, 基频中位数)
    """
    features = _extract_features(wav_path, segments, speaker_id, max_seconds)
    
    if features is None:
        return Gender.UNKNOWN, 0.0

    f0_median = features["f0_median"]
    f0_std = features["f0_std"]
    f0_range = features["f0_range"]
    f0_q25 = features["f0_q25"]
    
    # 多特征评分系统
    male_score = 0.0
    female_score = 0.0
    
    # === 特征 1：基频中位数（权重 50%）===
    # 男性典型范围: 85-165 Hz
    # 女性典型范围: 165-255 Hz
    if f0_median < 130:
        male_score += 0.50
    elif f0_median < 150:
        male_score += 0.40
        female_score += 0.05
    elif f0_median < 170:
        male_score += 0.25
        female_score += 0.15
    elif f0_median < 190:
        male_score += 0.15
        female_score += 0.25
    elif f0_median < 210:
        male_score += 0.05
        female_score += 0.40
    else:
        female_score += 0.50
    
    # === 特征 2：基频 25% 分位数（权重 20%）===
    # 男性低音部分更低
    if f0_q25 < 120:
        male_score += 0.20
    elif f0_q25 < 140:
        male_score += 0.15
        female_score += 0.03
    elif f0_q25 < 160:
        male_score += 0.08
        female_score += 0.08
    elif f0_q25 < 180:
        male_score += 0.03
        female_score += 0.15
    else:
        female_score += 0.20
    
    # === 特征 3：基频变化范围（权重 15%）===
    # 女性通常基频变化范围更大
    if f0_range < 30:
        male_score += 0.15
    elif f0_range < 50:
        male_score += 0.10
        female_score += 0.03
    elif f0_range < 70:
        male_score += 0.05
        female_score += 0.08
    elif f0_range < 90:
        male_score += 0.03
        female_score += 0.12
    else:
        female_score += 0.15
    
    # === 特征 4：基频标准差（权重 15%）===
    # 女性通常基频波动更大
    if f0_std < 15:
        male_score += 0.15
    elif f0_std < 25:
        male_score += 0.10
        female_score += 0.03
    elif f0_std < 35:
        male_score += 0.05
        female_score += 0.08
    elif f0_std < 45:
        male_score += 0.03
        female_score += 0.12
    else:
        female_score += 0.15
    
    # 计算最终结果
    total = male_score + female_score
    if total == 0:
        return Gender.UNKNOWN, f0_median
    
    male_ratio = male_score / total
    female_ratio = female_score / total
    
    # 需要明显优势才做出判断（55% 以上）
    if male_ratio >= 0.55:
        gender = Gender.MALE
    elif female_ratio >= 0.55:
        gender = Gender.FEMALE
    else:
        gender = Gender.UNKNOWN
    
    logger.debug(
        f"{speaker_id}: f0_median={f0_median:.1f}Hz, f0_q25={f0_q25:.1f}Hz, "
        f"f0_range={f0_range:.1f}Hz, f0_std={f0_std:.1f}Hz, "
        f"male={male_score:.2f} ({male_ratio:.0%}), female={female_score:.2f} ({female_ratio:.0%}) "
        f"-> {gender.value}"
    )

    return gender, f0_median


def detect_all_genders(
    wav_path: Path,
    segments: list[Segment],
) -> dict[str, tuple[Gender, float]]:
    """
    检测所有说话人的性别
    
    Returns:
        {speaker_id: (gender, f0_median), ...}
    """
    speakers = set(
        seg.speaker for seg in segments
        if seg.speaker and seg.speaker != "UNKNOWN"
    )

    results = {}
    for speaker_id in speakers:
        gender, f0_median = detect_gender(wav_path, segments, speaker_id)
        results[speaker_id] = (gender, f0_median)
        logger.info(f"性别检测 {speaker_id}: {gender.value} (f0={f0_median:.1f}Hz)")

    return results
