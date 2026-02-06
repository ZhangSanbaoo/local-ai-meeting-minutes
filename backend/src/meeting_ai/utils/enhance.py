"""
音频增强预处理模块

在语音识别和说话人分离之前对音频进行预处理，提高识别准确率。

主要功能：
1. 降噪 (Noise Reduction) - 去除背景噪音
2. 去混响 (Dereverberation) - 减少房间回声
3. 音量归一化 (Normalization) - 统一音量水平
4. 语音增强 (Speech Enhancement) - 突出人声
"""

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..logger import get_logger

logger = get_logger("utils.enhance")

# 全局模型缓存
_denoiser = None
_separator = None


def _check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _read_wav(wav_path: Path) -> tuple[int, np.ndarray]:
    """读取 WAV 文件"""
    with wave.open(str(wav_path), "rb") as w:
        sample_rate = w.getframerate()
        n_frames = w.getnframes()
        pcm_data = w.readframes(n_frames)
        channels = w.getnchannels()
        sample_width = w.getsampwidth()

    if sample_width == 2:
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(pcm_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"不支持的采样位深: {sample_width}")

    # 如果是立体声，转为单声道
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    return sample_rate, audio


def _write_wav(wav_path: Path, sample_rate: int, audio: np.ndarray) -> None:
    """写入 WAV 文件"""
    # 确保是单声道
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    
    # 归一化到 [-1, 1]
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95
    
    # 转换为 int16
    pcm_data = (audio * 32767).astype(np.int16).tobytes()
    
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_data)


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    音量归一化
    
    Args:
        audio: 音频数据
        target_db: 目标音量（dB）
        
    Returns:
        归一化后的音频
    """
    # 计算当前 RMS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio
    
    # 计算当前 dB
    current_db = 20 * np.log10(rms + 1e-8)
    
    # 计算增益
    gain = 10 ** ((target_db - current_db) / 20)
    
    # 应用增益，但限制最大值
    normalized = audio * gain
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def reduce_noise_simple(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    简单降噪（基于频谱减法）
    
    适用于平稳噪声（如风扇、空调）
    """
    try:
        import noisereduce as nr
        
        # 使用 noisereduce 库
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            prop_decrease=0.8,  # 降噪强度
            stationary=True,  # 假设噪声是平稳的
        )
        return reduced.astype(np.float32)
    except ImportError:
        logger.warning("noisereduce 未安装，跳过降噪。请运行: pip install noisereduce")
        return audio


def reduce_noise_deep(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    深度学习降噪（使用 SpeechBrain 或 noisereduce 增强模式）
    
    效果更好，但需要额外安装
    """
    try:
        # 尝试使用 noisereduce 的非平稳降噪模式（效果更好）
        import noisereduce as nr
        
        logger.info("使用 noisereduce 深度降噪模式...")
        
        # 非平稳降噪，对复杂噪声更有效
        # 注意: n_jobs=1 避免 Windows 多进程序列化问题
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            prop_decrease=0.9,  # 更强的降噪
            stationary=False,   # 非平稳噪声模式
            n_fft=2048,
            hop_length=512,
            n_jobs=1,  # Windows 兼容：单线程模式避免序列化错误
        )
        return reduced.astype(np.float32)
        
    except ImportError:
        logger.warning("noisereduce 未安装，跳过深度降噪。请运行: pip install noisereduce")
        return audio
    except Exception as e:
        logger.warning(f"深度降噪失败: {e}，使用简单降噪")
        return reduce_noise_simple(audio, sample_rate)


def separate_vocals(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    人声分离（使用 Demucs 或 Spleeter）
    
    从音频中提取人声，去除背景音乐和音效
    """
    global _separator
    
    try:
        # 尝试使用 demucs
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        
        if _separator is None:
            logger.info("加载 Demucs 人声分离模型...")
            _separator = get_model('htdemucs')
            _separator.eval()
            if torch.cuda.is_available():
                _separator.cuda()
            logger.info("Demucs 加载完成")
        
        # Demucs 需要特定格式
        if sample_rate != 44100:
            import librosa
            audio_44k = librosa.resample(audio, orig_sr=sample_rate, target_sr=44100)
        else:
            audio_44k = audio
        
        # 转换为 tensor
        wav = torch.tensor(audio_44k).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        if torch.cuda.is_available():
            wav = wav.cuda()
        
        # 分离
        with torch.no_grad():
            sources = apply_model(_separator, wav)
        
        # 提取人声 (vocals)
        # demucs 输出顺序: drums, bass, other, vocals
        vocals = sources[0, 3].cpu().numpy()
        
        # 重采样回原始采样率
        if sample_rate != 44100:
            vocals = librosa.resample(vocals, orig_sr=44100, target_sr=sample_rate)
        
        return vocals.astype(np.float32)
        
    except ImportError:
        logger.warning("Demucs 未安装，跳过人声分离。如需分离人声请运行: pip install demucs")
        return audio
    except Exception as e:
        logger.warning(f"人声分离失败: {e}")
        return audio


def enhance_audio(
    input_path: Path,
    output_path: Path,
    denoise: bool = True,
    normalize: bool = True,
    separate_voice: bool = False,
    deep_denoise: bool = False,
) -> Path:
    """
    音频增强主函数
    
    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        denoise: 是否降噪
        normalize: 是否归一化音量
        separate_voice: 是否分离人声（去除背景音乐/音效）
        deep_denoise: 是否使用深度学习降噪（效果更好但更慢）
        
    Returns:
        处理后的音频路径
    """
    logger.info(f"开始音频增强: {input_path}")
    
    # 读取音频
    sample_rate, audio = _read_wav(input_path)
    logger.info(f"原始音频: {len(audio)/sample_rate:.1f}秒, {sample_rate}Hz")
    
    # 1. 人声分离（如果需要）
    if separate_voice:
        logger.info("正在分离人声...")
        audio = separate_vocals(audio, sample_rate)
    
    # 2. 降噪
    if denoise:
        logger.info("正在降噪...")
        if deep_denoise:
            audio = reduce_noise_deep(audio, sample_rate)
        else:
            audio = reduce_noise_simple(audio, sample_rate)
    
    # 3. 音量归一化
    if normalize:
        logger.info("正在归一化音量...")
        audio = normalize_audio(audio)
    
    # 写入输出文件
    _write_wav(output_path, sample_rate, audio)
    logger.info(f"音频增强完成: {output_path}")
    
    return output_path


def auto_enhance(
    input_path: Path,
    output_path: Path | None = None,
    quality_threshold: float = 0.5,
) -> Path:
    """
    自动检测音频质量并决定增强策略
    
    Args:
        input_path: 输入音频路径
        output_path: 输出路径（默认在同目录下创建）
        quality_threshold: 质量阈值，低于此值才增强
        
    Returns:
        处理后的音频路径（如果不需要增强则返回原路径）
    """
    sample_rate, audio = _read_wav(input_path)
    
    # 简单的音频质量评估
    # 1. 信噪比估计（基于静音段和语音段的能量比）
    frame_length = int(sample_rate * 0.025)
    hop_length = int(sample_rate * 0.010)
    
    energies = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        energy = np.sqrt(np.mean(frame ** 2))
        energies.append(energy)
    
    energies = np.array(energies)
    
    # 估计噪声水平（最低 10% 的能量）
    noise_level = np.percentile(energies, 10)
    signal_level = np.percentile(energies, 90)
    
    if noise_level > 0:
        snr_estimate = 20 * np.log10(signal_level / noise_level)
    else:
        snr_estimate = 60  # 非常干净
    
    logger.info(f"估计信噪比: {snr_estimate:.1f} dB")
    
    # 判断是否需要增强
    needs_denoise = snr_estimate < 20  # 信噪比低于 20dB 需要降噪
    needs_normalize = signal_level < 0.1 or signal_level > 0.9  # 音量过低或过高
    
    if not needs_denoise and not needs_normalize:
        logger.info("音频质量良好，无需增强")
        return input_path
    
    # 确定输出路径
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"
    
    return enhance_audio(
        input_path,
        output_path,
        denoise=needs_denoise,
        normalize=needs_normalize,
        deep_denoise=snr_estimate < 10,  # 信噪比非常低才用深度学习
    )
