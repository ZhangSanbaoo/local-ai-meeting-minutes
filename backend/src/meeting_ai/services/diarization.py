"""
说话人分离服务

使用 pyannote-audio 进行说话人分离（Speaker Diarization）。
回答"谁在什么时候说话"这个问题。

输入：音频文件
输出：每个说话人的时间段列表
"""

import os
import warnings

# 抑制不影响功能的警告
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*triton.*")
import shutil
import tempfile
from pathlib import Path

# ============================================================
# 在导入 pyannote 之前，先清理环境和 patch
# ============================================================

# 1. 清理损坏的代理环境变量（包含换行符会导致 httpx 报错）
for _proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    if _proxy_var in os.environ:
        _proxy_value = os.environ[_proxy_var]
        if "\n" in _proxy_value or "\r" in _proxy_value:
            del os.environ[_proxy_var]

# 2. Monkey-patch pyannote 的 get_plda 函数，暂时禁用 PLDA
# pyannote 3.1.1+ 默认会尝试从 HuggingFace 下载 PLDA
try:
    from pyannote.audio.pipelines.utils import getter as _pyannote_getter

    def _patched_get_plda(plda=None, token=None, cache_dir=None):
        # 暂时禁用 PLDA，返回 None
        return None

    _pyannote_getter.get_plda = _patched_get_plda

    # 同时 patch speaker_diarization 模块中的引用
    from pyannote.audio.pipelines import speaker_diarization as _sd_module
    _sd_module.get_plda = _patched_get_plda
except ImportError:
    pass  # pyannote 未安装

# 3. Monkey-patch pyannote 的音频加载，使用 soundfile 替代 torchcodec
# torchcodec 与 PyTorch nightly 不兼容，所以我们用 soundfile 作为 fallback
try:
    import soundfile as sf
    from pyannote.audio.core import io as _pyannote_io
    from dataclasses import dataclass

    @dataclass
    class _FakeAudioStreamMetadata:
        """模拟 torchcodec 的 AudioStreamMetadata"""
        sample_rate: int
        num_channels: int
        num_frames: int

        @property
        def duration_seconds(self) -> float:
            return self.num_frames / self.sample_rate if self.sample_rate > 0 else 0.0

        @property
        def duration_seconds_from_header(self) -> float:
            return self.duration_seconds

    def _get_audio_info_sf(audio_path):
        """使用 soundfile 获取音频信息"""
        info = sf.info(str(audio_path))
        return info.samplerate, info.channels, info.frames

    def _patched_get_audio_metadata(file):
        """使用 soundfile 获取音频元数据"""
        audio_path = file["audio"] if isinstance(file, dict) else file
        sample_rate, num_channels, num_frames = _get_audio_info_sf(audio_path)
        return _FakeAudioStreamMetadata(
            sample_rate=sample_rate,
            num_channels=num_channels,
            num_frames=num_frames,
        )

    # 替换原函数
    _pyannote_io.get_audio_metadata = _patched_get_audio_metadata

    # 模拟 torchcodec 的 AudioSamples
    @dataclass
    class _FakeAudioSamples:
        """模拟 torchcodec 的 AudioSamples"""
        data: "torch.Tensor"
        sample_rate: int
        pts_seconds: float = 0.0
        duration_seconds: float = 0.0

    # 同时需要提供 AudioDecoder 的替代
    class _FakeAudioDecoder:
        """模拟 torchcodec 的 AudioDecoder - 完整实现"""
        def __init__(self, path):
            import torch
            self._path = str(path)
            # 读取音频数据
            data, sr = sf.read(self._path, dtype='float32')
            self._sample_rate = sr
            # soundfile 返回 (frames, channels)，需要转为 (channels, frames)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)  # mono -> (1, frames)
            else:
                data = data.T  # (frames, channels) -> (channels, frames)
            self._data = torch.from_numpy(data)
            self._num_channels = self._data.shape[0]
            self._num_frames = self._data.shape[1]
            self._duration = self._num_frames / self._sample_rate if self._sample_rate > 0 else 0.0

        @property
        def metadata(self):
            return _FakeAudioStreamMetadata(
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                num_frames=self._num_frames,
            )

        def get_all_samples(self):
            """返回所有音频样本"""
            return _FakeAudioSamples(
                data=self._data,
                sample_rate=self._sample_rate,
                pts_seconds=0.0,
                duration_seconds=self._duration,
            )

        def get_samples_played_in_range(self, start_seconds: float, end_seconds: float):
            """返回指定时间范围内的音频样本"""
            start_frame = int(start_seconds * self._sample_rate)
            end_frame = int(end_seconds * self._sample_rate)
            # 边界检查
            start_frame = max(0, start_frame)
            end_frame = min(self._num_frames, end_frame)
            # 提取数据
            data = self._data[:, start_frame:end_frame]
            duration = (end_frame - start_frame) / self._sample_rate
            return _FakeAudioSamples(
                data=data,
                sample_rate=self._sample_rate,
                pts_seconds=start_seconds,
                duration_seconds=duration,
            )

        def get_samples_played_at(self, seconds: float):
            """返回指定时间点的音频样本（单帧）"""
            frame = int(seconds * self._sample_rate)
            frame = max(0, min(self._num_frames - 1, frame))
            data = self._data[:, frame:frame+1]
            return _FakeAudioSamples(
                data=data,
                sample_rate=self._sample_rate,
                pts_seconds=seconds,
                duration_seconds=1.0 / self._sample_rate,
            )

        def __iter__(self):
            """支持迭代"""
            yield self.get_all_samples()

        def seek(self, pts_seconds: float):
            """跳转到指定位置（空实现，因为我们已经加载了全部数据）"""
            pass

    # 注入到 pyannote.audio.core.io 模块
    _pyannote_io.AudioDecoder = _FakeAudioDecoder
    _pyannote_io.AudioStreamMetadata = _FakeAudioStreamMetadata
    _pyannote_io.AudioSamples = _FakeAudioSamples

except ImportError as e:
    print(f"[WARNING] 无法 patch pyannote 音频加载: {e}")

# ============================================================

import torch
import yaml
from pyannote.audio import Pipeline

from ..config import get_settings
from ..logger import get_logger
from ..models import DiarizationResult, Segment, SpeakerInfo

logger = get_logger("services.diarization")


class DiarizationService:
    """
    说话人分离服务

    支持引擎：
    - pyannote-audio 系列（config.yaml 标志）

    使用方式：
        service = DiarizationService("pyannote-3.1")
        result = service.diarize("meeting.wav")

        for segment in result.segments:
            print(f"{segment.speaker}: {segment.start:.1f}s - {segment.end:.1f}s")
    """

    def __init__(self, model_name: str | None = None) -> None:
        """初始化服务（不立即加载模型）"""
        self._pipeline = None
        self._settings = get_settings().diarization
        self._model_name = model_name or self._settings.engine

    def _resolve_model_dir(self) -> Path:
        """解析模型目录"""
        settings = get_settings()
        return settings.paths.models_dir / "diarization" / self._model_name

    @property
    def pipeline(self):
        """懒加载 pipeline"""
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self):
        """自动检测模型类型并加载"""
        model_dir = self._resolve_model_dir()

        logger.info(f"加载说话人分离模型: {model_dir} (engine={self._model_name})")

        if not model_dir.exists():
            raise FileNotFoundError(
                f"模型目录不存在: {model_dir}\n"
                f"请先下载模型到该目录。"
            )

        config_file = model_dir / "config.yaml"

        if config_file.exists():
            # 验证 config.yaml 是否是 pyannote 格式（含 pipeline key）
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if isinstance(config, dict) and "pipeline" in config:
                return self._load_pyannote(model_dir, config_file)

        raise FileNotFoundError(
            f"未知模型格式: {model_dir}\n"
            f"需要 config.yaml (pyannote-audio 格式)"
        )

    def _load_pyannote(self, model_dir: Path, config_file: Path):
        """加载 pyannote 系列模型"""
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        needs_path_fix = False
        if "pipeline" in config and "params" in config["pipeline"]:
            params = config["pipeline"]["params"]
            for key in ["segmentation", "embedding"]:
                if key in params and str(params[key]).startswith(".."):
                    needs_path_fix = True
                    break

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        if needs_path_fix:
            pipeline = self._load_with_fixed_paths(model_dir, config, device)
        else:
            try:
                pipeline = Pipeline.from_pretrained(model_dir)
            except TypeError:
                pipeline = Pipeline.from_pretrained(str(model_dir))
            pipeline = pipeline.to(device)

        return pipeline

    def _load_with_fixed_paths(self, model_dir: Path, config: dict, device) -> Pipeline:
        """
        修复配置中的相对路径后加载模型
        """
        params = config["pipeline"]["params"]

        # 修复 segmentation 路径
        if "segmentation" in params and str(params["segmentation"]).startswith(".."):
            old_path = params["segmentation"]
            abs_path = (model_dir / old_path).resolve()
            if abs_path.exists():
                params["segmentation"] = str(abs_path)
                logger.info(f"修复 segmentation 路径: {old_path} -> {abs_path}")

        # 修复 embedding 路径
        if "embedding" in params and str(params["embedding"]).startswith(".."):
            old_path = params["embedding"]
            abs_path = (model_dir / old_path).resolve()
            if abs_path.exists():
                params["embedding"] = str(abs_path)
                logger.info(f"修复 embedding 路径: {old_path} -> {abs_path}")

        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp(prefix="pyannote_"))
        logger.info(f"创建临时配置目录: {temp_dir}")

        try:
            # 复制模型目录的所有文件到临时目录（除了 config.yaml）
            for item in model_dir.iterdir():
                dest = temp_dir / item.name
                if item.name == "config.yaml":
                    continue
                elif item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

            # 写入修复后的配置
            with open(temp_dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            # 从临时目录加载
            try:
                pipeline = Pipeline.from_pretrained(temp_dir)
            except TypeError:
                pipeline = Pipeline.from_pretrained(str(temp_dir))

            pipeline = pipeline.to(device)
            return pipeline

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def unload(self) -> None:
        """释放模型，回收 GPU 显存"""
        if self._pipeline is not None:
            logger.info("卸载说话人分离模型...")
            del self._pipeline
            self._pipeline = None
            import gc
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def diarize(
        self,
        audio_path: Path | str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """
        对音频进行说话人分离
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        logger.info(f"开始说话人分离: {audio_path.name}")

        min_spk = min_speakers or self._settings.min_speakers
        max_spk = max_speakers or self._settings.max_speakers

        pipeline = self.pipeline

        # pyannote Pipeline — 返回 Annotation 对象
        diarization_output = pipeline(
            audio_path,
            min_speakers=min_spk,
            max_speakers=max_spk,
        )
        return self._parse_pyannote_output(diarization_output)

    def _parse_pyannote_output(self, diarization_output) -> DiarizationResult:
        """解析 pyannote 输出"""
        segments = []
        speaker_stats: dict[str, dict] = {}

        if hasattr(diarization_output, 'speaker_diarization'):
            annotation = diarization_output.speaker_diarization
        else:
            annotation = diarization_output

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segment = Segment(
                id=len(segments),
                start=turn.start,
                end=turn.end,
                text="",
                speaker=speaker,
            )
            segments.append(segment)

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_duration": 0.0,
                    "segment_count": 0,
                }
            speaker_stats[speaker]["total_duration"] += segment.duration
            speaker_stats[speaker]["segment_count"] += 1

        speakers = {}
        for speaker_id, stats in speaker_stats.items():
            speakers[speaker_id] = SpeakerInfo(
                id=speaker_id,
                display_name=speaker_id,
                total_duration=stats["total_duration"],
                segment_count=stats["segment_count"],
            )

        logger.info(
            f"说话人分离完成: {len(speakers)} 个说话人, {len(segments)} 个片段"
        )

        return DiarizationResult(
            speakers=speakers,
            segments=segments,
        )


# ============================================================================
# 工厂函数（支持引擎切换）
# ============================================================================

_service: DiarizationService | None = None
_service_model: str | None = None


def get_diarization_service(model_name: str | None = None) -> DiarizationService:
    """获取说话人分离服务（硬编码使用 pyannote-3.1）"""
    global _service, _service_model

    # 硬编码使用 pyannote-3.1
    model_name = "pyannote-3.1"

    if _service is None:
        _service = DiarizationService(model_name)
        _service_model = model_name

    return _service
