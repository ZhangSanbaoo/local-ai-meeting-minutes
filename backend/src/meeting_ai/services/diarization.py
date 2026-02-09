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


class _3DSpeakerDiarizer:
    """
    使用 CAM++ 说话人嵌入 + fsmn-vad + 层次聚类的说话人分离

    流程: 音频 → VAD 切段 → 嵌入提取 → 聚类 → SPEAKER_XX 标签
    """

    # 嵌入提取的窗口参数
    WINDOW_SEC = 1.5       # 每个嵌入窗口的长度（秒）
    STEP_SEC = 0.75        # 窗口步进（秒），50% 重叠
    MIN_SEG_SEC = 0.3      # 最短有效片段（秒）

    def __init__(self, sv_model, vad_model=None):
        self.sv_model = sv_model
        self.vad_model = vad_model

    def diarize(
        self,
        audio_path: Path | str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """执行说话人分离"""
        import numpy as np
        import soundfile as sf_lib
        from scipy.cluster.hierarchy import fcluster, linkage

        audio, sr = sf_lib.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]  # 立体声取左声道
        duration_s = len(audio) / sr

        # ---- Step 1: VAD ----
        vad_segs = self._run_vad(audio_path, duration_s)
        logger.info(f"VAD 检测到 {len(vad_segs)} 个语音片段")

        if not vad_segs:
            return DiarizationResult(speakers={}, segments=[])

        # ---- Step 2: 滑窗切分 + 嵌入提取 ----
        windows = self._build_windows(vad_segs)
        if not windows:
            return DiarizationResult(speakers={}, segments=[])

        embeddings = self._extract_embeddings(audio, sr, windows)
        if not embeddings:
            return DiarizationResult(speakers={}, segments=[])

        emb_array = np.stack([e for _, e in embeddings])
        valid_windows = [w for w, _ in embeddings]
        n = len(emb_array)

        logger.info(f"提取了 {n} 个嵌入向量 (dim={emb_array.shape[1]})")

        # ---- Step 3: 聚类 ----
        if n == 1:
            labels = [0]
        else:
            # L2 归一化 → 余弦距离
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            emb_normed = emb_array / norms

            Z = linkage(emb_normed, method="ward")

            # 先用距离阈值切分，再用 min/max_speakers 约束
            threshold = 1.2
            labels_arr = fcluster(Z, t=threshold, criterion="distance") - 1
            n_clusters = len(set(labels_arr))

            if max_speakers and n_clusters > max_speakers:
                labels_arr = fcluster(Z, t=max_speakers, criterion="maxclust") - 1
            elif min_speakers and n_clusters < min_speakers:
                labels_arr = fcluster(Z, t=min_speakers, criterion="maxclust") - 1

            labels = labels_arr.tolist()

        # ---- Step 4: 窗口标签 → VAD 片段标签 ----
        segments, speakers = self._build_result(vad_segs, valid_windows, labels)

        logger.info(
            f"说话人分离完成 (3D-Speaker): {len(speakers)} 个说话人, "
            f"{len(segments)} 个片段"
        )

        return DiarizationResult(speakers=speakers, segments=segments)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _run_vad(self, audio_path, duration_s: float) -> list[list[float]]:
        """运行 VAD，返回 [[start_s, end_s], ...]"""
        if self.vad_model is not None:
            try:
                res = self.vad_model.generate(input=str(audio_path))
                if res and res[0].get("value"):
                    # fsmn-vad 返回毫秒
                    return [[s / 1000.0, e / 1000.0] for s, e in res[0]["value"]]
            except Exception as e:
                logger.warning(f"fsmn-vad 失败，回退固定窗口: {e}")

        # 回退: 固定窗口切分
        window = 1.5
        segs = []
        pos = 0.0
        while pos < duration_s:
            end = min(pos + window, duration_s)
            segs.append([pos, end])
            pos = end
        return segs

    def _build_windows(self, vad_segs: list[list[float]]) -> list[list[float]]:
        """将 VAD 片段切成固定长度的嵌入窗口"""
        windows = []
        for start, end in vad_segs:
            dur = end - start
            if dur < self.MIN_SEG_SEC:
                continue
            if dur <= self.WINDOW_SEC:
                windows.append([start, end])
            else:
                # 滑窗
                pos = start
                while pos + self.MIN_SEG_SEC <= end:
                    w_end = min(pos + self.WINDOW_SEC, end)
                    windows.append([pos, w_end])
                    pos += self.STEP_SEC
        return windows

    def _extract_embeddings(
        self, audio, sr: int, windows: list[list[float]]
    ):
        """批量提取嵌入向量，返回 [(window, embedding), ...]"""
        import numpy as np

        results = []
        # 分批，避免一次性 GPU 内存过大
        batch_size = 32
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i : i + batch_size]
            batch_audio = []
            for start, end in batch_windows:
                s_idx = int(start * sr)
                e_idx = int(end * sr)
                seg = audio[s_idx:e_idx]
                if len(seg) < int(self.MIN_SEG_SEC * sr):
                    continue
                batch_audio.append(seg)

            if not batch_audio:
                continue

            try:
                res = self.sv_model.generate(input=batch_audio)
                if res and "spk_embedding" in res[0]:
                    emb_tensor = res[0]["spk_embedding"]
                    # torch.Tensor → numpy
                    if hasattr(emb_tensor, "cpu"):
                        emb_np = emb_tensor.cpu().numpy()
                    else:
                        emb_np = np.array(emb_tensor)

                    if emb_np.ndim == 1:
                        emb_np = emb_np.reshape(1, -1)

                    for j, w in enumerate(batch_windows[: emb_np.shape[0]]):
                        results.append((w, emb_np[j]))
            except Exception as e:
                logger.warning(f"嵌入提取失败 (batch {i}): {e}")
                continue

        return results

    def _build_result(
        self,
        vad_segs: list[list[float]],
        windows: list[list[float]],
        labels: list[int],
    ) -> tuple[list[Segment], dict[str, SpeakerInfo]]:
        """将窗口级标签映射回 VAD 片段，构建 DiarizationResult"""
        import numpy as np
        from collections import Counter

        # 给每个 VAD 片段分配说话人标签（多数投票）
        seg_labels = []
        for vs, ve in vad_segs:
            votes = []
            for (ws, we), lbl in zip(windows, labels):
                # 窗口与 VAD 片段的交集
                overlap = min(ve, we) - max(vs, ws)
                if overlap > 0.05:  # 至少 50ms 重叠
                    votes.append(lbl)
            if votes:
                seg_labels.append(Counter(votes).most_common(1)[0][0])
            else:
                seg_labels.append(0)

        # 合并相邻同一说话人的片段
        merged_segments: list[Segment] = []
        speaker_stats: dict[str, dict] = {}

        for i, ((start, end), lbl) in enumerate(zip(vad_segs, seg_labels)):
            spk = f"SPEAKER_{lbl:02d}"

            # 与前一片段合并（同说话人 + 间隔 < 0.5s）
            if (
                merged_segments
                and merged_segments[-1].speaker == spk
                and start - merged_segments[-1].end < 0.5
            ):
                merged_segments[-1].end = end
            else:
                merged_segments.append(
                    Segment(
                        id=len(merged_segments),
                        start=start,
                        end=end,
                        text="",
                        speaker=spk,
                    )
                )

            if spk not in speaker_stats:
                speaker_stats[spk] = {"total_duration": 0.0, "segment_count": 0}
            speaker_stats[spk]["total_duration"] += end - start
            speaker_stats[spk]["segment_count"] += 1

        # 重新编号 segment id
        for idx, seg in enumerate(merged_segments):
            seg.id = idx

        speakers = {}
        for spk_id, stats in speaker_stats.items():
            speakers[spk_id] = SpeakerInfo(
                id=spk_id,
                display_name=spk_id,
                total_duration=stats["total_duration"],
                segment_count=stats["segment_count"],
            )

        return merged_segments, speakers


class DiarizationService:
    """
    说话人分离服务

    支持多引擎：
    - pyannote 系列（config.yaml 标志）
    - 3D-Speaker / ModelScope 系列（configuration.json 标志）

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
        configuration_file = model_dir / "configuration.json"

        if config_file.exists():
            # 验证 config.yaml 是否是 pyannote 格式（含 pipeline key）
            # ModelScope 下载的模型也可能带 config.yaml，但内容不同
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if isinstance(config, dict) and "pipeline" in config:
                return self._load_pyannote(model_dir, config_file)
            # config.yaml 不是 pyannote 格式，继续检测

        if configuration_file.exists():
            # 3D-Speaker / ModelScope 系列
            return self._load_3d_speaker(model_dir)

        raise FileNotFoundError(
            f"未知模型格式: {model_dir}\n"
            f"需要 config.yaml (pyannote) 或 configuration.json (ModelScope)"
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

    def _load_3d_speaker(self, model_dir: Path):
        """加载 3D-Speaker CAM++ 模型（嵌入提取 + VAD + 聚类 = 说话人分离）"""
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError(
                "3D-Speaker 模型需要 funasr 库。请运行: pip install funasr"
            ) from e

        settings = get_settings()
        vad_dir = settings.paths.models_dir / "streaming" / "funasr" / "fsmn-vad"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"加载 CAM++ 说话人嵌入模型: {model_dir} (device={device})")

        sv_model = AutoModel(model=str(model_dir), device=device, disable_update=True)

        vad_model = None
        if vad_dir.exists():
            logger.info(f"加载 fsmn-vad: {vad_dir}")
            vad_model = AutoModel(model=str(vad_dir), device=device, disable_update=True)
        else:
            logger.warning(f"fsmn-vad 未找到 ({vad_dir})，将使用固定窗口切分")

        return _3DSpeakerDiarizer(sv_model, vad_model)

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

        # 根据 pipeline 类型选择调用方式
        if isinstance(pipeline, _3DSpeakerDiarizer):
            # CAM++ 嵌入 + VAD + 聚类
            return pipeline.diarize(audio_path, min_spk, max_spk)
        elif hasattr(pipeline, 'itertracks') or isinstance(pipeline, Pipeline):
            # pyannote Pipeline — 返回 Annotation 对象
            diarization_output = pipeline(
                audio_path,
                min_speakers=min_spk,
                max_speakers=max_spk,
            )
            return self._parse_pyannote_output(diarization_output)
        else:
            # ModelScope pipeline — 返回格式不同
            diarization_output = pipeline(str(audio_path))
            return self._parse_modelscope_output(diarization_output)

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

    def _parse_modelscope_output(self, output) -> DiarizationResult:
        """解析 ModelScope / 3D-Speaker 输出"""
        segments = []
        speaker_stats: dict[str, dict] = {}

        # ModelScope 返回格式: {"text": "...", "sentences": [{"start": ms, "end": ms, "spk": 0}, ...]}
        if isinstance(output, dict):
            sentences = output.get("sentences", output.get("text", []))
            if isinstance(sentences, list):
                for item in sentences:
                    if isinstance(item, dict):
                        start = item.get("start", 0) / 1000.0  # ms -> s
                        end = item.get("end", 0) / 1000.0
                        spk = f"SPEAKER_{item.get('spk', item.get('speaker', 0)):02d}"
                    else:
                        continue

                    segment = Segment(
                        id=len(segments),
                        start=start,
                        end=end,
                        text="",
                        speaker=spk,
                    )
                    segments.append(segment)

                    if spk not in speaker_stats:
                        speaker_stats[spk] = {"total_duration": 0.0, "segment_count": 0}
                    speaker_stats[spk]["total_duration"] += segment.duration
                    speaker_stats[spk]["segment_count"] += 1

        speakers = {}
        for speaker_id, stats in speaker_stats.items():
            speakers[speaker_id] = SpeakerInfo(
                id=speaker_id,
                display_name=speaker_id,
                total_duration=stats["total_duration"],
                segment_count=stats["segment_count"],
            )

        logger.info(
            f"说话人分离完成 (ModelScope): {len(speakers)} 个说话人, {len(segments)} 个片段"
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
    """获取说话人分离服务（支持按模型名切换）"""
    global _service, _service_model

    if model_name is None:
        model_name = get_settings().diarization.engine

    if _service is not None and _service_model != model_name:
        logger.info(f"说话人分离引擎切换: {_service_model} → {model_name}")
        _service.unload()
        _service = None

    if _service is None:
        _service = DiarizationService(model_name)
        _service_model = model_name

    return _service
