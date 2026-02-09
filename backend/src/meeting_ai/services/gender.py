"""
性别检测服务

多引擎支持：
- f0: 基频分析（内置，零依赖）
- ecapa-gender: ECAPA-TDNN 模型（需下载到 models/gender/ecapa-gender/）
- wav2vec2-gender: Wav2Vec2 模型（需下载到 models/gender/wav2vec2-gender/）

使用工厂模式，根据引擎名称动态选择检测器。
"""

import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from ..config import get_settings
from ..logger import get_logger
from ..models import Gender, Segment

logger = get_logger("services.gender")


# ============================================================================
# 抽象基类
# ============================================================================

class GenderDetector(ABC):
    """性别检测器抽象基类"""

    @abstractmethod
    def detect(
        self,
        wav_path: Path,
        segments: list[Segment],
        speaker_id: str,
        max_seconds: float = 20.0,
    ) -> tuple[Gender, float]:
        """
        检测指定说话人的性别

        Returns:
            (性别, 置信度/f0_median)
        """
        ...


# ============================================================================
# 公共工具函数
# ============================================================================

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


def _extract_speaker_audio(
    wav_path: Path,
    segments: list[Segment],
    speaker_id: str,
    max_seconds: float = 20.0,
) -> tuple[int, np.ndarray | None]:
    """提取指定说话人的音频片段，拼接后返回"""
    sample_rate, audio = _read_wav_mono(wav_path)

    intervals = [
        (seg.start, seg.end)
        for seg in segments
        if seg.speaker == speaker_id and seg.duration >= 0.4
    ]

    if not intervals:
        return sample_rate, None

    audio_segments = []
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

    if total_used < 2.0 or not audio_segments:
        return sample_rate, None

    return sample_rate, np.concatenate(audio_segments)


# ============================================================================
# F0 基频分析引擎（内置）
# ============================================================================

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


class F0GenderDetector(GenderDetector):
    """基频分析性别检测器（内置，零依赖）"""

    # 全局嵌入模型缓存
    _embedding_model = None

    @classmethod
    def _get_embedding_model(cls):
        """懒加载 pyannote 嵌入模型"""
        if cls._embedding_model is None:
            try:
                from pyannote.audio import Model

                settings = get_settings()
                model_dir = settings.paths.models_dir / "pyannote" / "wespeaker-voxceleb-resnet34-LM"

                if model_dir.exists():
                    logger.info(f"加载嵌入模型: {model_dir}")
                    cls._embedding_model = Model.from_pretrained(str(model_dir))
                else:
                    logger.info("本地嵌入模型不存在，跳过")
                    cls._embedding_model = "unavailable"
                    return None

                if torch.cuda.is_available():
                    cls._embedding_model = cls._embedding_model.to("cuda")

                cls._embedding_model.eval()
                logger.info("嵌入模型加载完成")

            except Exception as e:
                logger.warning(f"嵌入模型加载失败: {e}，将只使用基频分析")
                cls._embedding_model = "unavailable"

        return cls._embedding_model if cls._embedding_model != "unavailable" else None

    def _extract_features(
        self,
        wav_path: Path,
        segments: list[Segment],
        speaker_id: str,
        max_seconds: float = 20.0,
    ) -> dict | None:
        """提取说话人的声学特征"""
        sample_rate, audio = _read_wav_mono(wav_path)
        frame_len = int(sample_rate * 0.030)
        hop_len = int(sample_rate * 0.010)

        intervals = [
            (seg.start, seg.end)
            for seg in segments
            if seg.speaker == speaker_id and seg.duration >= 0.4
        ]

        if not intervals:
            return None

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

        model = self._get_embedding_model()
        if model is not None:
            try:
                waveform = torch.from_numpy(combined_audio).unsqueeze(0)
                if torch.cuda.is_available():
                    waveform = waveform.to("cuda")

                with torch.no_grad():
                    embedding = model(waveform)

                features["embedding"] = embedding.cpu().numpy().flatten()
            except Exception as e:
                logger.debug(f"嵌入提取失败: {e}")

        return features

    def detect(
        self,
        wav_path: Path,
        segments: list[Segment],
        speaker_id: str,
        max_seconds: float = 20.0,
    ) -> tuple[Gender, float]:
        features = self._extract_features(wav_path, segments, speaker_id, max_seconds)

        if features is None:
            return Gender.UNKNOWN, 0.0

        f0_median = features["f0_median"]
        f0_std = features["f0_std"]
        f0_range = features["f0_range"]
        f0_q25 = features["f0_q25"]

        male_score = 0.0
        female_score = 0.0

        # 特征 1：基频中位数（权重 50%）
        if f0_median < 130:
            male_score += 0.50
        elif f0_median < 155:
            male_score += 0.40
            female_score += 0.05
        elif f0_median < 180:
            male_score += 0.30
            female_score += 0.10
        elif f0_median < 200:
            male_score += 0.10
            female_score += 0.30
        elif f0_median < 220:
            male_score += 0.05
            female_score += 0.40
        else:
            female_score += 0.50

        # 特征 2：基频 25% 分位数（权重 20%）
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

        # 特征 3：基频变化范围（权重 15%）
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

        # 特征 4：基频标准差（权重 15%）
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

        total = male_score + female_score
        if total == 0:
            return Gender.UNKNOWN, f0_median

        male_ratio = male_score / total
        female_ratio = female_score / total

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


# ============================================================================
# ECAPA-TDNN 引擎（JaesungHuh/voice-gender-classifier）
# ============================================================================

class ECAPAGenderDetector(GenderDetector):
    """ECAPA-TDNN 性别检测器 (JaesungHuh/voice-gender-classifier)

    该模型使用 PyTorchModelHubMixin，不是标准 transformers 模型。
    需要内联模型架构来加载 model.safetensors 权重。
    """

    def __init__(self, model_dir: Path):
        self._model_dir = model_dir
        self._model = None
        self._device = "cpu"

    def _load_model(self):
        if self._model is not None:
            return

        import math

        import torch.nn.functional as F
        import torchaudio
        from huggingface_hub import PyTorchModelHubMixin

        logger.info(f"加载 ECAPA-Gender 模型: {self._model_dir}")

        # --- 模型架构 (来自 github.com/JaesungHuh/voice-gender-classifier) ---
        class _SEModule(torch.nn.Module):
            def __init__(self, channels, bottleneck=128):
                super().__init__()
                self.se = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool1d(1),
                    torch.nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
                    torch.nn.Sigmoid(),
                )

            def forward(self, x):
                return x * self.se(x)

        class _Bottle2neck(torch.nn.Module):
            def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
                super().__init__()
                width = int(math.floor(planes / scale))
                self.conv1 = torch.nn.Conv1d(inplanes, width * scale, kernel_size=1)
                self.bn1 = torch.nn.BatchNorm1d(width * scale)
                self.nums = scale - 1
                convs, bns = [], []
                num_pad = math.floor(kernel_size / 2) * dilation
                for _ in range(self.nums):
                    convs.append(torch.nn.Conv1d(width, width, kernel_size=kernel_size,
                                                 dilation=dilation, padding=num_pad))
                    bns.append(torch.nn.BatchNorm1d(width))
                self.convs = torch.nn.ModuleList(convs)
                self.bns = torch.nn.ModuleList(bns)
                self.conv3 = torch.nn.Conv1d(width * scale, planes, kernel_size=1)
                self.bn3 = torch.nn.BatchNorm1d(planes)
                self.relu = torch.nn.ReLU()
                self.width = width
                self.se = _SEModule(planes)

            def forward(self, x):
                residual = x
                out = self.bn1(self.relu(self.conv1(x)))
                spx = torch.split(out, self.width, 1)
                for i in range(self.nums):
                    sp = spx[i] if i == 0 else sp + spx[i]
                    sp = self.bns[i](self.relu(self.convs[i](sp)))
                    out = sp if i == 0 else torch.cat((out, sp), 1)
                out = torch.cat((out, spx[self.nums]), 1)
                out = self.se(self.bn3(self.relu(self.conv3(out))))
                return out + residual

        class _ECAPA_gender(torch.nn.Module, PyTorchModelHubMixin):
            def __init__(self, C=1024):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
                self.relu = torch.nn.ReLU()
                self.bn1 = torch.nn.BatchNorm1d(C)
                self.layer1 = _Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
                self.layer2 = _Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
                self.layer3 = _Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
                self.layer4 = torch.nn.Conv1d(3 * C, 1536, kernel_size=1)
                self.attention = torch.nn.Sequential(
                    torch.nn.Conv1d(4608, 256, kernel_size=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(256),
                    torch.nn.Tanh(),
                    torch.nn.Conv1d(256, 1536, kernel_size=1),
                    torch.nn.Softmax(dim=2),
                )
                self.bn5 = torch.nn.BatchNorm1d(3072)
                self.fc6 = torch.nn.Linear(3072, 192)
                self.bn6 = torch.nn.BatchNorm1d(192)
                self.fc7 = torch.nn.Linear(192, 2)

            def logtorchfbank(self, x):
                flipped = torch.FloatTensor([-0.97, 1.0]).unsqueeze(0).unsqueeze(0).to(x.device)
                x = x.unsqueeze(1)
                x = F.pad(x, (1, 0), "reflect")
                x = F.conv1d(x, flipped).squeeze(1)
                mel = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                    f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80,
                ).to(x.device)(x) + 1e-6
                mel = mel.log()
                return mel - torch.mean(mel, dim=-1, keepdim=True)

            def forward(self, x):
                x = self.logtorchfbank(x)
                x = self.bn1(self.relu(self.conv1(x)))
                x1 = self.layer1(x)
                x2 = self.layer2(x + x1)
                x3 = self.layer3(x + x1 + x2)
                x = self.relu(self.layer4(torch.cat((x1, x2, x3), dim=1)))
                t = x.size()[-1]
                global_x = torch.cat(
                    (x,
                     torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                     torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)),
                    dim=1,
                )
                w = self.attention(global_x)
                mu = torch.sum(x * w, dim=2)
                sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
                x = self.bn5(torch.cat((mu, sg), 1))
                x = self.relu(self.bn6(self.fc6(x)))
                return self.fc7(x)

        # --- 模型架构结束 ---

        self._model = _ECAPA_gender.from_pretrained(str(self._model_dir))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("ECAPA-Gender 模型加载完成")

    def detect(
        self,
        wav_path: Path,
        segments: list[Segment],
        speaker_id: str,
        max_seconds: float = 20.0,
    ) -> tuple[Gender, float]:
        self._load_model()

        sample_rate, combined_audio = _extract_speaker_audio(
            wav_path, segments, speaker_id, max_seconds
        )
        if combined_audio is None:
            return Gender.UNKNOWN, 0.0

        # 模型期望 16kHz 单声道
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            combined_audio = resample_poly(combined_audio, 16000, sample_rate).astype(np.float32)

        waveform = torch.from_numpy(combined_audio).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(waveform)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        male_prob = float(probs[0])    # index 0 = male
        female_prob = float(probs[1])  # index 1 = female

        if male_prob > female_prob and male_prob > 0.55:
            gender = Gender.MALE
        elif female_prob > male_prob and female_prob > 0.55:
            gender = Gender.FEMALE
        else:
            gender = Gender.UNKNOWN

        confidence = max(male_prob, female_prob)
        logger.debug(
            f"{speaker_id}: ECAPA male={male_prob:.3f}, female={female_prob:.3f} -> {gender.value}"
        )

        return gender, confidence


# ============================================================================
# Wav2Vec2 引擎（alefiury/wav2vec2-large-xlsr-53）
# ============================================================================

class Wav2Vec2GenderDetector(GenderDetector):
    """Wav2Vec2 性别检测器"""

    def __init__(self, model_dir: Path):
        self._model_dir = model_dir
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return

        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        logger.info(f"加载 Wav2Vec2-Gender 模型: {self._model_dir}")
        self._processor = AutoFeatureExtractor.from_pretrained(str(self._model_dir))
        self._model = AutoModelForAudioClassification.from_pretrained(str(self._model_dir))

        if torch.cuda.is_available():
            self._model = self._model.to("cuda")

        self._model.eval()
        logger.info("Wav2Vec2-Gender 模型加载完成")

    def detect(
        self,
        wav_path: Path,
        segments: list[Segment],
        speaker_id: str,
        max_seconds: float = 20.0,
    ) -> tuple[Gender, float]:
        self._load_model()

        sample_rate, combined_audio = _extract_speaker_audio(
            wav_path, segments, speaker_id, max_seconds
        )
        if combined_audio is None:
            return Gender.UNKNOWN, 0.0

        inputs = self._processor(
            combined_audio, sampling_rate=sample_rate, return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        labels = self._model.config.id2label

        male_prob = 0.0
        female_prob = 0.0
        for idx, label in labels.items():
            if "male" in label.lower() and "female" not in label.lower():
                male_prob = float(probs[idx])
            elif "female" in label.lower():
                female_prob = float(probs[idx])

        if male_prob > female_prob and male_prob > 0.55:
            gender = Gender.MALE
        elif female_prob > male_prob and female_prob > 0.55:
            gender = Gender.FEMALE
        else:
            gender = Gender.UNKNOWN

        confidence = max(male_prob, female_prob)
        logger.debug(
            f"{speaker_id}: Wav2Vec2 male={male_prob:.3f}, female={female_prob:.3f} -> {gender.value}"
        )

        return gender, confidence


# ============================================================================
# 工厂函数
# ============================================================================

_detector: GenderDetector | None = None
_detector_engine: str | None = None


def get_gender_detector(engine_name: str | None = None) -> GenderDetector:
    """获取性别检测器（工厂 + 缓存）"""
    global _detector, _detector_engine

    if engine_name is None:
        engine_name = get_settings().gender.engine

    if _detector is not None and _detector_engine == engine_name:
        return _detector

    # 切换引擎：释放旧模型的 GPU 显存
    if _detector is not None:
        logger.info(f"性别检测引擎切换: {_detector_engine} → {engine_name}")
        _old = _detector
        _detector = None
        del _old
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    if engine_name == "f0":
        _detector = F0GenderDetector()
    else:
        settings = get_settings()
        model_dir = settings.paths.models_dir / "gender" / engine_name

        if not model_dir.exists():
            logger.warning(f"性别检测模型不存在: {model_dir}，回退到 f0")
            _detector = F0GenderDetector()
            engine_name = "f0"
        elif "ecapa" in engine_name.lower():
            _detector = ECAPAGenderDetector(model_dir)
        elif "wav2vec2" in engine_name.lower():
            _detector = Wav2Vec2GenderDetector(model_dir)
        else:
            # 通用 transformers 模型（尝试 AutoModel）
            logger.info(f"尝试以通用 transformers 模型加载: {engine_name}")
            _detector = ECAPAGenderDetector(model_dir)  # 假设兼容

    _detector_engine = engine_name
    return _detector


def detect_gender(
    wav_path: Path,
    segments: list[Segment],
    speaker_id: str,
    max_seconds: float = 20.0,
    engine_name: str | None = None,
) -> tuple[Gender, float]:
    """检测指定说话人的性别"""
    detector = get_gender_detector(engine_name)
    return detector.detect(wav_path, segments, speaker_id, max_seconds)


def detect_all_genders(
    wav_path: Path,
    segments: list[Segment],
    engine_name: str | None = None,
) -> dict[str, tuple[Gender, float]]:
    """
    检测所有说话人的性别

    Returns:
        {speaker_id: (gender, f0_median/confidence), ...}
    """
    speakers = set(
        seg.speaker for seg in segments
        if seg.speaker and seg.speaker != "UNKNOWN"
    )

    results = {}
    for speaker_id in speakers:
        gender, metric = detect_gender(wav_path, segments, speaker_id, engine_name=engine_name)
        results[speaker_id] = (gender, metric)
        logger.info(f"性别检测 {speaker_id}: {gender.value} (metric={metric:.3f})")

    return results
