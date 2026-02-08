"""
共享 pytest fixtures
"""

import os
import wave
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_audio():
    """生成 1 秒 16kHz 正弦波 + 噪声的 numpy 数组"""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    clean = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.05 * np.random.randn(sr).astype(np.float32)
    return clean + noise, sr


@pytest.fixture
def sample_wav_file(sample_audio):
    """写入临时 WAV 文件，yield 路径，自动清理"""
    audio, sr = sample_audio
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = Path(f.name)
    f.close()

    pcm = (audio * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)

    yield path

    if path.exists():
        os.unlink(path)


@pytest.fixture
def tmp_output_path():
    """临时输出文件路径，测试后自动清理"""
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = Path(f.name)
    f.close()
    # 删除文件，只需要路径
    if path.exists():
        os.unlink(path)

    yield path

    if path.exists():
        os.unlink(path)
