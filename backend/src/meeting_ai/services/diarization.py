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


class _FunASRFullCampPlusDiarizer:
    """
    完整的 3D-Speaker CAM++ Diarization 实现（官方流程）

    使用 FunASR 手动组装，不依赖 ModelScope Pipeline

    流程:
    1. VAD 分段 (fsmn-vad)
    2. Speaker embedding 提取 (CAM++)
    3. Change locator 优化边界 (transformer)
    4. HDBSCAN 聚类
    """

    def __init__(self, speaker_model, change_locator=None, vad_model=None):
        """
        Args:
            speaker_model: CAM++ speaker embedding 模型
            change_locator: speaker change point 定位模型（可选）
            vad_model: fsmn-vad 模型（可选）
        """
        self.speaker_model = speaker_model
        self.change_locator = change_locator
        self.vad_model = vad_model

    def diarize(
        self,
        audio_path: Path | str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """执行完整的说话人分离"""
        import numpy as np
        import soundfile as sf_lib
        from hdbscan import HDBSCAN

        logger.info(f"开始 3D-Speaker 说话人分离（FunASR 实现）")

        # 加载音频
        audio, sr = sf_lib.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        duration_s = len(audio) / sr

        # Step 1: VAD 分段
        logger.info("Step 1/4: VAD 语音活动检测...")
        vad_segments = self._run_vad(audio_path, duration_s)
        logger.info(f"VAD 检测到 {len(vad_segments)} 个语音片段")

        if not vad_segments:
            return DiarizationResult(speakers={}, segments=[])

        # Step 2: 提取 speaker embeddings
        logger.info("Step 2/4: 提取说话人特征...")
        embeddings_data = self._extract_embeddings(audio, sr, vad_segments)
        if not embeddings_data:
            return DiarizationResult(speakers={}, segments=[])

        segment_embeddings = [e for _, e in embeddings_data]
        emb_array = np.stack(segment_embeddings)
        logger.info(f"提取了 {len(emb_array)} 个嵌入向量 (dim={emb_array.shape[1]})")

        # Step 3: Change point detection (可选)
        if self.change_locator is not None:
            logger.info("Step 3/4: 说话人转换点检测...")
            vad_segments = self._refine_boundaries(audio, sr, vad_segments, emb_array)
            logger.info(f"边界优化后: {len(vad_segments)} 个片段")
        else:
            logger.info("Step 3/4: 跳过（无 change locator 模型）")

        # Step 4: HDBSCAN 聚类
        logger.info("Step 4/4: HDBSCAN 聚类...")
        labels = self._cluster_embeddings(emb_array, min_speakers, max_speakers)
        n_speakers = len(set(labels))
        logger.info(f"聚类完成: 识别出 {n_speakers} 个说话人")

        # 构建结果
        segments, speakers = self._build_result(vad_segments, labels)

        logger.info(
            f"说话人分离完成 (3D-Speaker 官方): {len(speakers)} 个说话人, "
            f"{len(segments)} 个片段"
        )

        return DiarizationResult(speakers=speakers, segments=segments)

    def _run_vad(self, audio_path, duration_s: float) -> list[tuple[float, float]]:
        """VAD 分段（高敏感度配置）"""
        if self.vad_model is not None:
            try:
                res = self.vad_model.generate(
                    input=str(audio_path),
                    max_end_silence_time=300,  # 300ms 静音结束（更敏感，检测更多切换点）
                    speech_noise_thres=0.4,    # 降低阈值，更容易检测到语音
                )
                if res and res[0].get("value"):
                    # fsmn-vad 返回毫秒
                    segments = [(s / 1000.0, e / 1000.0) for s, e in res[0]["value"]]
                    logger.debug(f"VAD (高敏感): 检测到 {len(segments)} 个片段")
                    return segments
            except Exception as e:
                logger.warning(f"VAD 失败，使用固定窗口: {e}")

        # 回退: 固定 1.5 秒窗口（比之前的3秒更细）
        segments = []
        pos = 0.0
        while pos < duration_s:
            end = min(pos + 1.5, duration_s)
            segments.append((pos, end))
            pos += 0.75  # 50% 重叠
        return segments

    def _extract_embeddings(
        self, audio, sr: int, segments: list[tuple[float, float]]
    ) -> list[tuple[tuple[float, float], "np.ndarray"]]:
        """
        为每个 VAD 片段提取 speaker embedding

        对于长片段（>2s），使用滑窗提取多个 embeddings 以提高时间分辨率
        """
        import numpy as np

        results = []
        WINDOW_SIZE = 1.5  # 滑窗大小 1.5秒
        STEP_SIZE = 0.75   # 步进 0.75秒（50% 重叠）
        MIN_LEN = 0.3      # 最小片段长度

        for start, end in segments:
            duration = end - start

            # 短片段：直接提取
            if duration <= WINDOW_SIZE + 0.5:
                s_idx = int(start * sr)
                e_idx = int(end * sr)
                seg_audio = audio[s_idx:e_idx]

                if len(seg_audio) < int(MIN_LEN * sr):
                    continue

                emb = self._extract_single_embedding(seg_audio, start, end)
                if emb is not None:
                    results.append(((start, end), emb))
            else:
                # 长片段：滑窗提取多个 embeddings
                pos = start
                while pos + MIN_LEN <= end:
                    win_end = min(pos + WINDOW_SIZE, end)
                    s_idx = int(pos * sr)
                    e_idx = int(win_end * sr)
                    seg_audio = audio[s_idx:e_idx]

                    if len(seg_audio) >= int(MIN_LEN * sr):
                        emb = self._extract_single_embedding(seg_audio, pos, win_end)
                        if emb is not None:
                            results.append(((pos, win_end), emb))

                    pos += STEP_SIZE
                    if pos >= end:
                        break

        logger.debug(f"从 {len(segments)} 个 VAD 片段提取了 {len(results)} 个 embeddings")
        return results

    def _extract_single_embedding(self, seg_audio, start, end):
        """提取单个 embedding"""
        import numpy as np

        try:
            res = self.speaker_model.generate(input=[seg_audio])
            if res and "spk_embedding" in res[0]:
                emb_tensor = res[0]["spk_embedding"]
                if hasattr(emb_tensor, "cpu"):
                    emb_np = emb_tensor.cpu().numpy()
                else:
                    emb_np = np.array(emb_tensor)

                if emb_np.ndim == 2:
                    emb_np = emb_np[0]

                return emb_np
        except Exception as e:
            logger.debug(f"提取 embedding 失败 ({start:.1f}-{end:.1f}s): {e}")
            return None

    def _refine_boundaries(
        self, audio, sr: int, segments: list[tuple[float, float]], embeddings
    ) -> list[tuple[float, float]]:
        """使用 change locator 优化边界（TODO: 需要实现）"""
        # 这里可以实现 transformer change locator 的逻辑
        # 目前先返回原始 segments
        logger.debug("边界优化暂未实现，使用原始 VAD 边界")
        return segments

    def _cluster_embeddings(
        self, embeddings, min_speakers: int | None, max_speakers: int | None
    ) -> list[int]:
        """
        使用 HDBSCAN 聚类

        优化策略：更小的 min_cluster_size，更激进的分割
        """
        import numpy as np
        from hdbscan import HDBSCAN

        n = len(embeddings)
        logger.debug(f"HDBSCAN 聚类: {n} 个 embeddings")

        # L2 归一化 → 余弦距离
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        emb_normed = embeddings / norms

        # HDBSCAN 聚类参数优化
        # min_cluster_size: 降低以允许更小的聚类（更激进的分割）
        # min_samples: 降低以增加敏感度
        min_cluster_size = max(2, min(5, n // 8))  # 最多 5，最少 2
        min_samples = max(1, min_cluster_size // 2)

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            allow_single_cluster=False,  # 不允许单聚类（强制分割）
        )

        labels = clusterer.fit_predict(emb_normed)
        n_clusters_raw = len(set(labels)) - (1 if -1 in labels else 0)
        logger.debug(f"HDBSCAN 初始聚类: {n_clusters_raw} 个聚类, {list(labels).count(-1)} 个噪声点")

        # 处理噪声点（label = -1）
        if -1 in labels:
            for i, lbl in enumerate(labels):
                if lbl == -1:
                    # 找最近的非噪声点
                    dists = np.linalg.norm(emb_normed - emb_normed[i], axis=1)
                    dists[i] = np.inf
                    nearest_valid = np.where(labels != -1)[0]
                    if len(nearest_valid) > 0:
                        nearest_idx = nearest_valid[np.argmin(dists[nearest_valid])]
                        labels[i] = labels[nearest_idx]
                    else:
                        labels[i] = 0

        # 如果聚类数超过 max_speakers，用层次聚类合并
        n_clusters = len(set(labels))
        if max_speakers and n_clusters > max_speakers:
            logger.debug(f"聚类数 {n_clusters} > max_speakers {max_speakers}，进行合并")
            from scipy.cluster.hierarchy import fcluster, linkage
            Z = linkage(emb_normed, method='ward')
            labels = fcluster(Z, t=max_speakers, criterion='maxclust') - 1
            labels = labels.tolist()

        n_final = len(set(labels))
        logger.debug(f"最终聚类: {n_final} 个说话人")

        return labels if isinstance(labels, list) else labels.tolist()

    def _build_result(
        self, segments: list[tuple[float, float]], labels: list[int]
    ) -> tuple[list[Segment], dict[str, SpeakerInfo]]:
        """
        构建最终结果

        合并策略：只合并间隔很小（<200ms）的同说话人片段，保留更多切换点
        """
        merged_segments: list[Segment] = []
        speaker_stats: dict[str, dict] = {}

        for (start, end), lbl in zip(segments, labels):
            spk = f"SPEAKER_{lbl:02d}"

            # 合并相邻同说话人片段（降低阈值以保留更多切换点）
            if (
                merged_segments
                and merged_segments[-1].speaker == spk
                and start - merged_segments[-1].end < 0.2  # 200ms 内合并（之前是500ms）
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

        # 重新编号
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


class _3DSpeakerDiarizer:
    """
    使用 CAM++ 说话人嵌入 + fsmn-vad + 层次聚类的说话人分离（旧实现）

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
            # 聚类阈值折中配置（原值 1.2 太高，0.8 太低，用 1.0）
            threshold = 1.0
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
                # 配置 VAD 参数以提高对短片段的敏感度
                # max_end_silence_time: 静音超过此值（毫秒）才结束当前段落
                # speech_noise_thres: 语音/噪声阈值，越小越敏感
                res = self.vad_model.generate(
                    input=str(audio_path),
                    max_end_silence_time=500,  # 500ms（折中：不过度合并也不过度切分）
                    speech_noise_thres=0.55,   # 折中敏感度
                )
                if res and res[0].get("value"):
                    # fsmn-vad 返回毫秒
                    vad_segs = [[s / 1000.0, e / 1000.0] for s, e in res[0]["value"]]
                    logger.debug(f"VAD (敏感模式) 检测到 {len(vad_segs)} 个片段")
                    return vad_segs
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

            # 与前一片段合并（同说话人 + 间隔 < 0.3s）
            # 降低合并间隔以保留更多短片段（原值 0.5s）
            if (
                merged_segments
                and merged_segments[-1].speaker == spk
                and start - merged_segments[-1].end < 0.3
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
        """
        加载 3D-Speaker / ModelScope 说话人分离模型

        根据 configuration.json 中的 task 字段选择加载方式：
        - "speaker-diarization": 使用 ModelScope Pipeline API（官方推荐）
        - "speaker-verification": 使用手动 VAD + 嵌入 + 聚类管线
        """
        import json

        config_file = model_dir / "configuration.json"
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        task = config.get("model", {}).get("task") or config.get("task", "")

        if task == "speaker-diarization":
            # 使用 ModelScope 官方 Pipeline API
            logger.info(f"加载说话人分离模型: {model_dir} (engine=3d-speaker-campplus)")
            return self._load_modelscope_pipeline(model_dir)
        elif task == "speaker-verification":
            # 使用手动 VAD + 嵌入 + 聚类管线（兼容旧模型）
            logger.info(f"加载说话人嵌入模型: {model_dir} (engine=verification-based)")
            return self._load_verification_based_diarizer(model_dir)
        else:
            raise ValueError(
                f"未知的 task 类型: {task}\n"
                f"支持的类型: speaker-diarization, speaker-verification"
            )

    def _load_modelscope_pipeline(self, model_dir: Path):
        """
        使用 FunASR 手动实现完整的 3D-Speaker 流程

        不使用 ModelScope Pipeline（会卡死），改用 FunASR 分别加载三个子模型
        """
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError(
                "FunASR 实现需要 funasr 库。请运行: pip install funasr"
            ) from e

        logger.info(f"使用 FunASR 手动实现 3D-Speaker 流程")

        settings = get_settings()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 从 ModelScope cache 加载三个子模型
        cache_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "damo"

        # 1. Speaker embedding 模型 (CAM++)
        speaker_model_dir = cache_dir / "speech_campplus_sv_zh-cn_16k-common"
        if not speaker_model_dir.exists():
            raise FileNotFoundError(
                f"Speaker模型未找到: {speaker_model_dir}\n"
                f"请运行: python backend/scripts/download_3dspeaker_submodels.py"
            )
        logger.info(f"加载 CAM++ Speaker Embedding: {speaker_model_dir}")
        speaker_model = AutoModel(model=str(speaker_model_dir), device=device, disable_update=True)

        # 2. Change locator 模型 (可选)
        change_locator = None
        change_model_dir = cache_dir / "speech_campplus-transformer_scl_zh-cn_16k-common"
        if change_model_dir.exists():
            logger.info(f"加载 Change Locator: {change_model_dir}")
            try:
                change_locator = AutoModel(model=str(change_model_dir), device=device, disable_update=True)
            except Exception as e:
                logger.warning(f"Change Locator 加载失败（跳过）: {e}")

        # 3. VAD 模型
        vad_model = None
        vad_dir = cache_dir / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
        if not vad_dir.exists():
            # 尝试从 models/streaming 加载
            vad_dir = settings.paths.models_dir / "streaming" / "funasr" / "fsmn-vad"
        if vad_dir.exists():
            logger.info(f"加载 FSMN-VAD: {vad_dir}")
            vad_model = AutoModel(model=str(vad_dir), device=device, disable_update=True)
        else:
            logger.warning("VAD 模型未找到，将使用固定窗口分段")

        logger.info("FunASR 3D-Speaker 组件加载完成")

        # 返回完整实现
        return _FunASRFullCampPlusDiarizer(
            speaker_model=speaker_model,
            change_locator=change_locator,
            vad_model=vad_model,
        )

    def _load_verification_based_diarizer(self, model_dir: Path):
        """加载基于 verification 模型的手动 diarization 管线（兼容旧模型）"""
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError(
                "verification-based diarization 需要 funasr 库。请运行: pip install funasr"
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
        if isinstance(pipeline, (_3DSpeakerDiarizer, _FunASRFullCampPlusDiarizer)):
            # FunASR 实现（新旧两种）
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
            # ModelScope pipeline — 返回格式不同（已废弃，会卡死）
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

    # 处理 None 或空字符串，使用配置默认值
    if not model_name or not model_name.strip():
        model_name = get_settings().diarization.engine

    if _service is not None and _service_model != model_name:
        logger.info(f"说话人分离引擎切换: {_service_model} → {model_name}")
        _service.unload()
        _service = None

    if _service is None:
        _service = DiarizationService(model_name)
        _service_model = model_name

    return _service
