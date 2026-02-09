"""
音频处理工具

提供音频格式转换、重采样等基础功能。
"""

import subprocess
from pathlib import Path

from ..logger import get_logger

logger = get_logger("utils.audio")


def ensure_wav_16k_mono(
    input_path: Path,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    确保音频是 16kHz 单声道 WAV 格式

    大多数语音处理模型（Whisper、pyannote）都需要这种格式。

    Args:
        input_path: 输入音频文件
        output_path: 输出路径（默认在同目录下创建 _16k_mono.wav）
        overwrite: 是否覆盖已存在的文件

    Returns:
        转换后的文件路径
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix(".16k_mono.wav")
    else:
        output_path = Path(output_path)

    # 如果输出已存在且不覆盖，直接返回
    if output_path.exists() and not overwrite:
        logger.debug(f"使用已存在的文件: {output_path}")
        return output_path

    logger.info(f"转换音频格式: {input_path} -> {output_path}")

    # 使用 ffmpeg 转换
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出
        "-i", str(input_path),
        "-ar", "16000",  # 采样率 16kHz
        "-ac", "1",      # 单声道
        "-c:a", "pcm_s16le",  # 16-bit PCM
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        logger.debug(f"ffmpeg 输出: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg 转换失败: {e.stderr}")
        raise RuntimeError(f"音频转换失败: {e.stderr}") from e
    except FileNotFoundError:
        logger.error("找不到 ffmpeg，请确保已安装")
        raise RuntimeError("找不到 ffmpeg，请安装: sudo apt install ffmpeg")

    return output_path


def get_audio_duration(audio_path: Path) -> float:
    """
    获取音频时长（秒）

    Args:
        audio_path: 音频文件路径

    Returns:
        时长（秒）
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"无法获取音频时长: {e}")
        return 0.0


def get_audio_info(audio_path: Path) -> dict:
    """
    获取音频文件信息

    Args:
        audio_path: 音频文件路径

    Returns:
        包含 duration, sample_rate, channels 等信息的字典
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,duration:format=duration",
        "-of", "json",
        str(audio_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=True,
        )
        import json
        data = json.loads(result.stdout)

        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        return {
            "duration": float(fmt.get("duration", stream.get("duration", 0))),
            "sample_rate": int(stream.get("sample_rate", 0)),
            "channels": int(stream.get("channels", 0)),
        }
    except Exception as e:
        logger.warning(f"无法获取音频信息: {e}")
        return {"duration": 0.0, "sample_rate": 0, "channels": 0}
