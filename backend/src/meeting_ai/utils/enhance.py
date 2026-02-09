"""
专业级音频增强预处理模块

在语音识别和说话人分离之前对音频进行预处理，提高识别准确率。

增强管线（按处理顺序）：
1. Demucs v4 — 人声分离（去除背景音乐/特效声音）
2. DeepFilterNet3 — 专业降噪 + 去混响（CPU 实时）
3. Resemble Enhance — 语音清晰化 + 超分辨率（GPU 加速）
4. 音量归一化
"""

import wave
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ..logger import get_logger

logger = get_logger("utils.enhance")


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """使用 scipy.signal.resample_poly 进行高质量重采样（librosa 在某些环境下会挂起）"""
    if orig_sr == target_sr:
        return audio
    g = gcd(target_sr, orig_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)

# ---------------------------------------------------------------------------
# 全局模型缓存（懒加载）
# ---------------------------------------------------------------------------
_df_session = None  # ONNX Runtime session for DeepFilterNet3
_demucs_model = None

# ---------------------------------------------------------------------------
# 音频 I/O 工具
# ---------------------------------------------------------------------------


def _read_wav(wav_path: Path) -> tuple[int, np.ndarray]:
    """读取 WAV 文件，返回 (sample_rate, float32 单声道音频)"""
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

    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    return sample_rate, audio


def _write_wav(wav_path: Path, sample_rate: int, audio: np.ndarray) -> None:
    """写入 WAV 文件（单声道 int16）"""
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95

    pcm_data = (audio * 32767).astype(np.int16).tobytes()

    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_data)


# ---------------------------------------------------------------------------
# 音量归一化
# ---------------------------------------------------------------------------


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """RMS 音量归一化到目标 dB"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio

    current_db = 20 * np.log10(rms + 1e-8)
    gain = 10 ** ((target_db - current_db) / 20)
    normalized = audio * gain
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized


# ---------------------------------------------------------------------------
# 1. Demucs v4 — 人声分离
# ---------------------------------------------------------------------------


def separate_vocals(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Demucs v4 人声分离 — 从音频中提取人声，去除背景音乐和音效

    模型: htdemucs (Hybrid Transformer Demucs)
    输出: drums, bass, other, vocals → 只取 vocals
    """
    global _demucs_model

    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        if _demucs_model is None:
            logger.info("加载 Demucs v4 人声分离模型 (htdemucs)...")
            _demucs_model = get_model("htdemucs")
            _demucs_model.eval()
            if torch.cuda.is_available():
                _demucs_model.cuda()
            logger.info("Demucs 加载完成")

        # Demucs 训练于 44.1kHz，需要 resample
        audio_44k = _resample(audio, sample_rate, 44100)

        # 转换为 tensor: (batch, channels, samples)
        # Demucs htdemucs 需要立体声 (2 channels)，单声道需要复制为双通道
        wav = torch.tensor(audio_44k, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        wav = wav.expand(-1, 2, -1).clone()  # (1, 1, N) → (1, 2, N)
        if torch.cuda.is_available():
            wav = wav.cuda()

        with torch.no_grad():
            sources = apply_model(_demucs_model, wav)

        # 提取人声 (index 3 = vocals)
        vocals = sources[0, 3].cpu().numpy()
        if vocals.ndim > 1:
            vocals = vocals.mean(axis=0)

        # resample 回原始采样率
        vocals = _resample(vocals, 44100, sample_rate)

        logger.info("Demucs 人声分离完成")
        return vocals.astype(np.float32)

    except ImportError:
        logger.warning("Demucs 未安装，跳过人声分离。请运行: pip install demucs")
        return audio
    except Exception as e:
        logger.warning(f"Demucs 人声分离失败: {e}")
        return audio


# ---------------------------------------------------------------------------
# 2. DeepFilterNet3 — 专业降噪 + 去混响
# ---------------------------------------------------------------------------


def _get_deepfilter_model_path() -> Path:
    """获取 DeepFilterNet3 ONNX 模型路径"""
    # 按优先级搜索
    candidates = [
        Path(__file__).parent.parent.parent.parent.parent / "models" / "deepfilter" / "denoiser_model.onnx",
        Path("models/deepfilter/denoiser_model.onnx"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "找不到 DeepFilterNet3 ONNX 模型。"
        "请将 denoiser_model.onnx 放到 models/deepfilter/ 目录"
    )


def denoise_deepfilter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    DeepFilterNet3 专业降噪 + 去混响 (ONNX Runtime 推理)

    特点: ~16MB ONNX 模型, CPU/GPU, PESQ 3.17, 48kHz 全带宽
    无需 Rust 编译，使用 onnxruntime 直接推理
    """
    global _df_session

    try:
        import onnxruntime as ort

        if _df_session is None:
            model_path = _get_deepfilter_model_path()
            logger.info(f"加载 DeepFilterNet3 ONNX 模型: {model_path}")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            # 尝试 CUDA, 如果不支持则回退 CPU
            # (模型的 FusedConv+Sigmoid 在某些 CUDA EP 版本不兼容)
            if ort.get_device() == "GPU":
                try:
                    _df_session = ort.InferenceSession(
                        str(model_path), sess_options,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                except Exception:
                    logger.info("DeepFilterNet3 CUDA 不兼容, 回退 CPU")
                    _df_session = None

            if _df_session is None:
                _df_session = ort.InferenceSession(
                    str(model_path), sess_options,
                    providers=["CPUExecutionProvider"],
                )

            logger.info(f"DeepFilterNet3 加载完成 (provider: {_df_session.get_providers()[0]})")

        # DeepFilterNet3 训练于 48kHz
        DF_SR = 48000
        HOP_SIZE = 480
        FFT_SIZE = 960
        STATE_SIZE = 45304

        # resample 到 48kHz
        audio_48k = _resample(audio, sample_rate, DF_SR) if sample_rate != DF_SR else audio.copy()

        audio_48k = audio_48k.astype(np.float32)

        # 填充 padding (对齐 hop_size)
        orig_len = len(audio_48k)
        hop_pad = (HOP_SIZE - orig_len % HOP_SIZE) % HOP_SIZE
        orig_len += hop_pad
        audio_padded = np.pad(audio_48k, (0, FFT_SIZE + hop_pad), mode="constant")

        # 分帧
        n_frames = len(audio_padded) // HOP_SIZE
        frames = [audio_padded[i * HOP_SIZE: (i + 1) * HOP_SIZE] for i in range(n_frames)]

        # 逐帧推理
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        atten_lim_db = np.zeros(1, dtype=np.float32)
        enhanced_frames = []

        for frame in frames:
            out = _df_session.run(None, {
                "input_frame": frame,
                "states": state,
                "atten_lim_db": atten_lim_db,
            })
            enhanced_frames.append(out[0])
            state = out[1]

        # 拼接并裁剪
        enhanced = np.concatenate(enhanced_frames)
        d = FFT_SIZE - HOP_SIZE  # = 480, 算法延迟
        enhanced = enhanced[d: orig_len + d]

        # resample 回原始采样率
        if sample_rate != DF_SR:
            enhanced = _resample(enhanced, DF_SR, sample_rate)

        logger.info("DeepFilterNet3 降噪完成")
        return enhanced.astype(np.float32)

    except ImportError:
        logger.warning(
            "onnxruntime 未安装，跳过专业降噪。请运行: pip install onnxruntime-gpu"
        )
        return audio
    except FileNotFoundError as e:
        logger.warning(str(e))
        return audio
    except Exception as e:
        logger.warning(f"DeepFilterNet3 降噪失败: {e}")
        return audio


# ---------------------------------------------------------------------------
# 3. Resemble Enhance — 语音清晰化 + 超分辨率
# ---------------------------------------------------------------------------


def enhance_clarity(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Resemble Enhance 语音清晰化 + 超分辨率

    两阶段管线:
    - Stage 1 (Denoiser): UNet 降噪器，快速
    - Stage 2 (Enhancer): CFM (Conditional Flow Matching) 生成模型，
      将语音提升至 44.1kHz 录音室品质

    需要 GPU 加速（CFM 推理较慢）
    模型权重自动下载到 HuggingFace cache
    """
    try:
        import sys
        import pathlib
        import torch

        # resemble-enhance 模型权重用 pickle 序列化，内含 PosixPath 对象
        # Windows 上 PosixPath 无法实例化，需要在整个 import + 推理期间 patch
        _posix_patched = False
        if sys.platform == "win32":
            pathlib.PosixPath = pathlib.WindowsPath
            _posix_patched = True

        try:
            from resemble_enhance.enhancer.inference import enhance as resemble_enhance

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 转换为 1D torch tensor
            audio_tensor = torch.from_numpy(audio).float()

            logger.info(f"Resemble Enhance 语音清晰化开始 (device={device})...")

            # enhance() 内部会自动下载模型、处理 resample、分块推理
            # 返回 (enhanced_wav, sr) — enhanced_wav 是 CPU tensor
            enhanced, new_sr = resemble_enhance(
                dwav=audio_tensor,
                sr=sample_rate,
                device=device,
                nfe=32,             # CFM 采样步数 (默认 32, 速度/质量平衡)
                solver="midpoint",  # ODE solver
                lambd=0.9,          # 降噪程度 (0=纯增强, 1=最大降噪)
                tau=0.5,            # CFM 温度 (0=保守, 1=最大增强)
            )
        finally:
            # 恢复 PosixPath（无论成功还是失败）
            if _posix_patched:
                pathlib.PosixPath = type('PosixPath', (pathlib.PurePosixPath,), {})

        result = enhanced.cpu().numpy()

        # Resemble Enhance 输出可能是 44.1kHz, resample 回原始采样率
        if new_sr != sample_rate:
            result = _resample(result, new_sr, sample_rate)

        # 释放 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Resemble Enhance 语音清晰化完成")
        return result.astype(np.float32)

    except ImportError:
        logger.warning(
            "Resemble Enhance 未安装，跳过语音清晰化。"
            "请运行: pip install resemble-enhance"
        )
        return audio
    except Exception as e:
        logger.warning(f"Resemble Enhance 语音清晰化失败: {e}")
        return audio


# ---------------------------------------------------------------------------
# 主入口 — 专业音频增强管线
# ---------------------------------------------------------------------------


def enhance_audio(
    input_path: Path,
    output_path: Path,
    mode: str = "none",
) -> Path:
    """
    专业音频增强管线

    Args:
        input_path: 输入音频路径 (16kHz WAV)
        output_path: 输出音频路径
        mode: 增强模式
            - "none": 不增强
            - "denoise": DeepFilterNet3 降噪+去混响
            - "enhance": DeepFilterNet3 + Resemble Enhance 降噪+清晰化
            - "vocal": Demucs v4 人声分离
            - "full": Demucs + DeepFilterNet3 + Resemble Enhance 完整增强

    Returns:
        处理后的音频路径
    """
    if mode == "none":
        return input_path

    logger.info(f"开始音频增强: mode={mode}, 输入={input_path}")

    sample_rate, audio = _read_wav(input_path)
    logger.info(f"原始音频: {len(audio)/sample_rate:.1f}秒, {sample_rate}Hz")

    # 1. 人声分离 (vocal / full)
    if mode in ("vocal", "full"):
        logger.info("阶段 1/4: Demucs 人声分离...")
        audio = separate_vocals(audio, sample_rate)

    # 2. DeepFilterNet3 降噪 + 去混响 (denoise / enhance / full)
    if mode in ("denoise", "enhance", "full"):
        logger.info("阶段 2/4: DeepFilterNet3 降噪+去混响...")
        audio = denoise_deepfilter(audio, sample_rate)

    # 3. Resemble Enhance 语音清晰化 (enhance / full)
    if mode in ("enhance", "full"):
        logger.info("阶段 3/4: Resemble Enhance 语音清晰化...")
        audio = enhance_clarity(audio, sample_rate)

    # 4. 音量归一化 (始终)
    logger.info("阶段 4/4: 音量归一化...")
    audio = normalize_audio(audio)

    _write_wav(output_path, sample_rate, audio)
    logger.info(f"音频增强完成: {output_path}")

    return output_path
