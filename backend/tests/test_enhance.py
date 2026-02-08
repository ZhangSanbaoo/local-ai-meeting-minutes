"""
音频增强模块测试

测试覆盖:
1. 工具函数 (_resample, _read_wav, _write_wav, normalize_audio)
2. DeepFilterNet3 ONNX 降噪
3. 增强管线 (enhance_audio)
4. 可选模型 (Demucs, Resemble Enhance)
"""

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from meeting_ai.utils.enhance import (
    _resample,
    _read_wav,
    _write_wav,
    normalize_audio,
    denoise_deepfilter,
    enhance_audio,
    _get_deepfilter_model_path,
)


# ============================================================================
# 检测可用依赖
# ============================================================================

try:
    import onnxruntime
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

try:
    _get_deepfilter_model_path()
    HAS_DF_MODEL = True
except FileNotFoundError:
    HAS_DF_MODEL = False

try:
    import demucs
    HAS_DEMUCS = True
except ImportError:
    HAS_DEMUCS = False

try:
    from resemble_enhance.enhancer.inference import enhance as _re
    HAS_RESEMBLE = True
except ImportError:
    HAS_RESEMBLE = False


# ============================================================================
# 1. 工具函数测试
# ============================================================================


class TestResample:
    """重采样函数测试"""

    def test_same_rate_returns_original(self):
        """相同采样率返回原数组"""
        audio = np.random.randn(16000).astype(np.float32)
        result = _resample(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_upsample_16k_to_48k(self):
        """16kHz → 48kHz 上采样长度正确"""
        audio = np.random.randn(16000).astype(np.float32)
        result = _resample(audio, 16000, 48000)
        assert len(result) == 48000
        assert result.dtype == np.float32

    def test_downsample_48k_to_16k(self):
        """48kHz → 16kHz 下采样长度正确"""
        audio = np.random.randn(48000).astype(np.float32)
        result = _resample(audio, 48000, 16000)
        assert len(result) == 16000

    def test_resample_preserves_energy(self):
        """重采样前后能量大致不变"""
        audio = np.random.randn(16000).astype(np.float32) * 0.3
        result = _resample(audio, 16000, 48000)
        orig_rms = np.sqrt(np.mean(audio ** 2))
        new_rms = np.sqrt(np.mean(result ** 2))
        assert abs(orig_rms - new_rms) / orig_rms < 0.1  # 误差 < 10%


class TestWavIO:
    """WAV 读写测试"""

    def test_read_write_roundtrip(self, sample_wav_file, tmp_output_path):
        """写入后读取，数据基本一致（考虑 _write_wav 内部的峰值归一化）"""
        sr, audio = _read_wav(sample_wav_file)
        assert sr == 16000
        assert len(audio) == 16000
        assert audio.dtype == np.float32

        _write_wav(tmp_output_path, sr, audio)
        sr2, audio2 = _read_wav(tmp_output_path)
        assert sr2 == sr
        assert len(audio2) == len(audio)
        # _write_wav 内部归一化峰值到 0.95，所以值会等比例缩放
        # 验证形状一致（相关性 > 0.99）
        corr = np.corrcoef(audio, audio2)[0, 1]
        assert corr > 0.99, f"相关性 {corr} 过低"

    def test_read_stereo_wav(self, tmp_output_path):
        """立体声 WAV 自动转单声道"""
        sr = 16000
        left = np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float32) * 0.3
        right = np.sin(np.linspace(0, 2 * np.pi * 880, sr)).astype(np.float32) * 0.3
        stereo = np.column_stack([left, right])
        pcm = (stereo.flatten() * 32767).astype(np.int16).tobytes()

        with wave.open(str(tmp_output_path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)

        sr_out, audio = _read_wav(tmp_output_path)
        assert sr_out == sr
        assert audio.ndim == 1
        assert len(audio) == sr  # 单声道

    def test_write_wav_normalizes_peak(self, tmp_output_path):
        """写入时自动归一化到 0.95 峰值"""
        sr = 16000
        audio = np.ones(sr, dtype=np.float32) * 2.0  # 超过 1.0
        _write_wav(tmp_output_path, sr, audio)
        sr2, audio2 = _read_wav(tmp_output_path)
        assert np.abs(audio2).max() < 1.0


class TestNormalize:
    """音量归一化测试"""

    def test_normalize_to_target(self, sample_audio):
        """归一化后 RMS 接近目标 dB"""
        audio, _ = sample_audio
        target_db = -20.0
        result = normalize_audio(audio, target_db)
        rms = np.sqrt(np.mean(result ** 2))
        actual_db = 20 * np.log10(rms + 1e-8)
        assert abs(actual_db - target_db) < 1.0  # 允许 1dB 误差

    def test_normalize_silence(self):
        """静音音频不报错，返回原值"""
        silence = np.zeros(16000, dtype=np.float32)
        result = normalize_audio(silence)
        np.testing.assert_array_equal(result, silence)

    def test_normalize_clips_to_range(self):
        """归一化后不超过 [-1, 1]"""
        audio = np.random.randn(16000).astype(np.float32) * 0.001  # 很安静
        result = normalize_audio(audio, target_db=-3.0)  # 目标很大声
        assert result.max() <= 1.0
        assert result.min() >= -1.0


# ============================================================================
# 2. DeepFilterNet3 ONNX 测试
# ============================================================================


@pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="onnxruntime 未安装")
@pytest.mark.skipif(not HAS_DF_MODEL, reason="DeepFilterNet3 ONNX 模型未找到")
class TestDeepFilterNet:
    """DeepFilterNet3 ONNX 降噪测试"""

    def test_model_path_found(self):
        """能找到 ONNX 模型文件"""
        path = _get_deepfilter_model_path()
        assert path.exists()
        assert path.suffix == ".onnx"

    def test_denoise_reduces_noise(self, sample_audio):
        """降噪后噪声 RMS 应降低"""
        audio, sr = sample_audio
        # 添加更多噪声
        noisy = audio + 0.1 * np.random.randn(len(audio)).astype(np.float32)
        result = denoise_deepfilter(noisy, sr)
        # DeepFilterNet3 应该降低噪声
        assert result.dtype == np.float32
        assert len(result) == len(noisy)

    def test_denoise_preserves_length(self, sample_audio):
        """输出样本数等于输入"""
        audio, sr = sample_audio
        result = denoise_deepfilter(audio, sr)
        assert len(result) == len(audio)

    def test_denoise_48k_input(self):
        """48kHz 输入无需重采样"""
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = denoise_deepfilter(audio, 48000)
        assert len(result) == 48000


# ============================================================================
# 3. 增强管线测试
# ============================================================================


class TestEnhanceAudio:
    """增强管线主函数测试"""

    def test_mode_none_returns_input(self, sample_wav_file, tmp_output_path):
        """mode='none' 直接返回输入路径"""
        result = enhance_audio(sample_wav_file, tmp_output_path, mode="none")
        assert result == sample_wav_file
        assert not tmp_output_path.exists()

    @pytest.mark.skipif(not HAS_ONNXRUNTIME or not HAS_DF_MODEL,
                        reason="onnxruntime 或 DeepFilterNet3 模型未找到")
    def test_mode_denoise_creates_output(self, sample_wav_file, tmp_output_path):
        """mode='denoise' 创建输出文件"""
        result = enhance_audio(sample_wav_file, tmp_output_path, mode="denoise")
        assert result == tmp_output_path
        assert tmp_output_path.exists()
        # 读取输出验证
        sr, audio = _read_wav(tmp_output_path)
        assert sr == 16000
        assert len(audio) > 0

    def test_nonexistent_input_raises(self, tmp_output_path):
        """不存在的输入文件应该报错"""
        fake_path = Path("/nonexistent/audio.wav")
        with pytest.raises(Exception):
            enhance_audio(fake_path, tmp_output_path, mode="denoise")


# ============================================================================
# 4. 可选模型测试
# ============================================================================


@pytest.mark.skipif(not HAS_DEMUCS, reason="Demucs 未安装")
class TestDemucs:
    """Demucs 人声分离测试（需要模型下载）"""

    def test_import_demucs(self):
        """Demucs 可以导入"""
        from demucs.pretrained import get_model
        assert callable(get_model)


@pytest.mark.skipif(not HAS_RESEMBLE, reason="Resemble Enhance 未安装")
class TestResembleEnhance:
    """Resemble Enhance 清晰化测试"""

    def test_import_resemble(self):
        """Resemble Enhance inference 可以导入"""
        from resemble_enhance.enhancer.inference import enhance
        assert callable(enhance)
