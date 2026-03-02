"""
在线说话人分离 (Online Speaker Diarization)

基于 diart 核心算法的实时说话人追踪：
  segmentation-3.0（帧级检测）+ WeSpeaker（embedding）+ 在线余弦聚类

使用方式:
    diarizer = get_online_diarizer()
    diarizer.load()
    state = diarizer.create_state()

    for pcm_chunk in audio_stream:
        speaker = diarizer.feed_chunk(state, pcm_chunk)
        # speaker = "SPEAKER_00" / "SPEAKER_01" / None

    diarizer.unload()

核心算法提取自 diart (MIT License, Copyright (c) 2021 Universite Paris-Saclay / CNRS)
参考: https://github.com/juanmc2005/diart
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.distance import cdist
from scipy.optimize import linear_sum_assignment

from ..config import get_settings
from ..logger import get_logger

logger = get_logger("services.online_diarization")

# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 16000
WINDOW_DURATION = 5.0       # 滑动窗口 (秒)
STEP_DURATION = 0.5         # 步进 (秒)
LATENCY = 0.5               # 输出延迟 (秒)
WINDOW_SAMPLES = int(WINDOW_DURATION * SAMPLE_RATE)  # 80000
STEP_SAMPLES = int(STEP_DURATION * SAMPLE_RATE)      # 8000

# 聚类参数 (基于 diart 默认值，适度调优)
TAU_ACTIVE = 0.6            # 说话人活动阈值 (diart 默认值)
RHO_UPDATE = 0.3            # 质心更新最低语音比例
DELTA_NEW = 1.1             # 新说话人距离阈值 (diart=1.0, 略微提高)
GAMMA = 3.0                 # 重叠语音惩罚指数
BETA = 10.0                 # 重叠语音 softmax 温度
MAX_SPEAKERS = 20           # 最大全局说话人数
MIN_TURN_DURATION = 0.5     # 最短说话人回合 (秒)


# ============================================================================
# Speaker Mapping (simplified from diart/mapping.py)
# ============================================================================

class SpeakerMap:
    """局部说话人 → 全局说话人映射矩阵"""

    INVALID = 1e10  # 最小化目标的无效值

    def __init__(self, cost_matrix: np.ndarray):
        self.cost_matrix = cost_matrix
        self.num_local = cost_matrix.shape[0]
        self.num_global = cost_matrix.shape[1]
        # 计算有效 (非 INVALID) 的行/列
        self._mapped_local = set(
            np.where(np.min(cost_matrix, axis=1) < self.INVALID)[0]
        )
        self._mapped_global = set(
            np.where(np.min(cost_matrix, axis=0) < self.INVALID)[0]
        )
        self._assignments: list[int] | None = None

    @staticmethod
    def from_distances(
        embeddings: np.ndarray,
        centers: np.ndarray,
        metric: str = "cosine",
    ) -> SpeakerMap:
        """从 embedding 和质心计算距离矩阵"""
        dist_matrix = cdist(embeddings, centers, metric=metric)
        return SpeakerMap(dist_matrix)

    @staticmethod
    def hard(
        num_local: int,
        num_global: int,
        assignments: list[tuple[int, int]],
    ) -> SpeakerMap:
        """直接指定映射关系"""
        cost = np.full((num_local, num_global), SpeakerMap.INVALID)
        for src, tgt in assignments:
            cost[src, tgt] = 0.0
        return SpeakerMap(cost)

    @property
    def optimal_assignments(self) -> list[int]:
        """匈牙利算法最优匹配 (最小化)"""
        if self._assignments is None:
            _, col_ind = linear_sum_assignment(self.cost_matrix)
            self._assignments = list(col_ind)
        return self._assignments

    def valid_assignments(self) -> tuple[list[int], list[int]]:
        """返回有效的 (local, global) 配对列表"""
        sources, targets = [], []
        for src, tgt in enumerate(self.optimal_assignments):
            if src in self._mapped_local:
                sources.append(src)
                targets.append(tgt)
        return sources, targets

    def is_local_mapped(self, speaker: int) -> bool:
        return speaker in self._mapped_local

    def is_global_mapped(self, speaker: int) -> bool:
        return speaker in self._mapped_global

    def unmap_speakers(
        self,
        local_speakers: np.ndarray | list[int] | None = None,
        global_speakers: np.ndarray | list[int] | None = None,
    ) -> SpeakerMap:
        """将指定说话人设为无效"""
        cost = self.cost_matrix.copy()
        if local_speakers is not None:
            for s in local_speakers:
                cost[s, :] = self.INVALID
        if global_speakers is not None:
            for s in global_speakers:
                cost[:, s] = self.INVALID
        return SpeakerMap(cost)

    def unmap_threshold(self, threshold: float) -> SpeakerMap:
        """距离超过阈值的映射设为无效"""
        cost = self.cost_matrix.copy()
        sources, targets = self.valid_assignments()
        for src, tgt in zip(sources, targets):
            if cost[src, tgt] >= threshold:
                cost[src, :] = self.INVALID
        return SpeakerMap(cost)

    def set_local(self, local_spk: int, global_spk: int) -> SpeakerMap:
        """设置指定映射"""
        cost = self.cost_matrix.copy()
        cost[local_spk, global_spk] = 0.0
        return SpeakerMap(cost)

    def apply(self, local_scores: np.ndarray) -> np.ndarray:
        """
        将局部说话人分数映射到全局空间

        local_scores: (frames, num_local)
        返回: (frames, num_global)
        """
        num_frames = local_scores.shape[0]
        global_scores = np.zeros((num_frames, self.num_global))
        for src, tgt in zip(*self.valid_assignments()):
            global_scores[:, tgt] = local_scores[:, src]
        return global_scores


# ============================================================================
# Online Speaker Clustering (from diart/blocks/clustering.py)
# ============================================================================

class OnlineSpeakerClustering:
    """
    在线说话人聚类

    维护全局说话人质心，将每个窗口的局部说话人映射到全局说话人。
    """

    def __init__(
        self,
        tau_active: float = TAU_ACTIVE,
        rho_update: float = RHO_UPDATE,
        delta_new: float = DELTA_NEW,
        max_speakers: int = MAX_SPEAKERS,
    ):
        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.max_speakers = max_speakers
        self.centers: np.ndarray | None = None
        self.active_centers: set[int] = set()

    @property
    def num_known(self) -> int:
        return len(self.active_centers)

    @property
    def num_free(self) -> int:
        return self.max_speakers - self.num_known

    @property
    def inactive_centers(self) -> list[int]:
        return [c for c in range(self.max_speakers) if c not in self.active_centers]

    def _next_free_center(self) -> int | None:
        for c in range(self.max_speakers):
            if c not in self.active_centers:
                return c
        return None

    def _init_centers(self, dim: int):
        self.centers = np.zeros((self.max_speakers, dim))
        self.active_centers = set()

    def _add_center(self, embedding: np.ndarray) -> int:
        pos = self._next_free_center()
        self.centers[pos] = embedding
        self.active_centers.add(pos)
        return pos

    def _update_centers(
        self, assignments: list[tuple[int, int]], embeddings: np.ndarray
    ):
        """累加更新质心 (余弦距离下方向比幅度重要)"""
        for local_spk, global_spk in assignments:
            self.centers[global_spk] += embeddings[local_spk]

    def identify(
        self,
        segmentation: SlidingWindowFeature,
        embeddings: np.ndarray,
    ) -> SpeakerMap:
        """
        将局部说话人映射到全局说话人

        segmentation: (frames, local_speakers) 活动概率矩阵
        embeddings: (local_speakers, emb_dim) 说话人向量
        """
        # 找活跃说话人: max activation >= tau_active
        active_speakers = np.where(
            np.max(segmentation.data, axis=0) >= self.tau_active
        )[0]
        # 长时说话人: mean activation >= rho_update
        long_speakers = np.where(
            np.mean(segmentation.data, axis=0) >= self.rho_update
        )[0]
        # 过滤 NaN embedding
        valid_emb = np.where(~np.isnan(embeddings).any(axis=1))[0]
        active_speakers = np.intersect1d(active_speakers, valid_emb)

        num_local = segmentation.data.shape[1]

        # 第一次调用: 初始化质心
        if self.centers is None:
            self._init_centers(embeddings.shape[1])
            assignments = [
                (spk, self._add_center(embeddings[spk]))
                for spk in active_speakers
            ]
            return SpeakerMap.hard(num_local, self.max_speakers, assignments)

        # 计算距离矩阵 → 初始映射
        dist_map = SpeakerMap.from_distances(embeddings, self.centers, "cosine")

        # 屏蔽非活跃说话人
        inactive_local = [
            s for s in range(num_local) if s not in active_speakers
        ]
        dist_map = dist_map.unmap_speakers(inactive_local, self.inactive_centers)

        # 距离阈值过滤
        valid_map = dist_map.unmap_threshold(self.delta_new)

        # 未匹配的活跃说话人
        missed = [
            s for s in active_speakers if not valid_map.is_local_mapped(s)
        ]

        # 尝试分配新质心或匹配已有
        new_center_speakers = []
        for spk in missed:
            if len(new_center_speakers) < self.num_free and spk in long_speakers:
                new_center_speakers.append(spk)
            else:
                # 尝试分配到最近的已占用质心
                preferences = np.argsort(dist_map.cost_matrix[spk, :])
                preferences = [g for g in preferences if g in self.active_centers]
                _, assigned_globals = valid_map.valid_assignments()
                free = [g for g in preferences if g not in assigned_globals]
                if free:
                    valid_map = valid_map.set_local(spk, free[0])

        # 更新已知质心
        to_update = [
            (ls, gs)
            for ls, gs in zip(*valid_map.valid_assignments())
            if ls not in missed and ls in long_speakers
        ]
        self._update_centers(to_update, embeddings)

        # 添加新质心
        for spk in new_center_speakers:
            valid_map = valid_map.set_local(spk, self._add_center(embeddings[spk]))

        return valid_map

    def __call__(
        self,
        segmentation: SlidingWindowFeature,
        embeddings: np.ndarray,
    ) -> SlidingWindowFeature:
        """
        聚类并返回全局空间的分割结果

        返回: SlidingWindowFeature (frames, max_speakers) 全局说话人分数
        """
        speaker_map = self.identify(segmentation, embeddings)
        global_scores = speaker_map.apply(segmentation.data)
        return SlidingWindowFeature(global_scores, segmentation.sliding_window)


# ============================================================================
# Delayed Aggregation (from diart/blocks/aggregation.py)
# ============================================================================

class DelayedAggregation:
    """
    延迟聚合: Hamming 窗口加权平均 + 时间对齐

    多个重叠窗口的预测通过 Hamming 窗口加权平均，
    平滑输出并减少边界效应。
    """

    def __init__(
        self,
        step: float = STEP_DURATION,
        latency: float = LATENCY,
    ):
        self.step = step
        self.latency = latency
        self.num_overlapping = int(round(latency / step))

    def __call__(
        self, buffers: list[SlidingWindowFeature]
    ) -> SlidingWindowFeature:
        """
        从缓冲区中聚合输出

        buffers: 最近的预测列表 (每个 frames x speakers)
        返回: 当前步进区域的聚合结果
        """
        last = buffers[-1]
        start = last.extent.end - self.latency
        focus = Segment(start, start + self.step)

        num_frames = buffers[0].data.shape[0]
        hamming_list = []
        data_list = []

        for buf in buffers:
            cropped = buf.crop(focus, mode="loose", fixed=focus.duration)
            h = np.expand_dims(np.hamming(num_frames), axis=-1)
            h_swf = SlidingWindowFeature(h, buf.sliding_window)
            h_cropped = h_swf.crop(focus, mode="loose", fixed=focus.duration)
            hamming_list.append(h_cropped.data)
            data_list.append(cropped.data)

        hamming_arr = np.stack(hamming_list)
        data_arr = np.stack(data_list)
        aggregated = np.sum(hamming_arr * data_arr, axis=0) / np.sum(hamming_arr, axis=0)

        # 特殊处理: 第一个窗口从 t=0 开始
        if len(buffers) == 1 and last.extent.start == 0:
            first_region = Segment(0, focus.end)
            first_output = np.array(
                buffers[0].crop(
                    first_region, mode="loose", fixed=first_region.duration
                )
            )
            n = aggregated.shape[0]
            first_output[-n:] = aggregated
            resolution = focus.end / first_output.shape[0]
            return SlidingWindowFeature(
                first_output,
                SlidingWindow(start=0, duration=resolution, step=resolution),
            )

        resolution = focus.duration / aggregated.shape[0]
        return SlidingWindowFeature(
            aggregated,
            SlidingWindow(start=focus.start, duration=resolution, step=resolution),
        )


# ============================================================================
# Binarize (from diart/blocks/utils.py)
# ============================================================================

class Binarize:
    """将说话人概率矩阵二值化为 pyannote Annotation"""

    def __init__(self, threshold: float = TAU_ACTIVE):
        self.threshold = threshold

    def __call__(self, segmentation: SlidingWindowFeature) -> Annotation:
        num_frames, num_speakers = segmentation.data.shape
        timestamps = segmentation.sliding_window
        is_active = segmentation.data > self.threshold
        # 末尾添加非活跃帧，关闭所有 speaker turn
        is_active = np.append(is_active, [[False] * num_speakers], axis=0)
        start_times = np.zeros(num_speakers) + timestamps[0].middle

        annotation = Annotation(modality="speech")
        for t in range(num_frames):
            # 检测 onset (不活跃 → 活跃)
            onsets = np.logical_and(~is_active[t], is_active[t + 1])
            start_times[onsets] = timestamps[t + 1].middle
            # 检测 offset (活跃 → 不活跃)
            offsets = np.logical_and(is_active[t], ~is_active[t + 1])
            for spk in np.where(offsets)[0]:
                region = Segment(start_times[spk], timestamps[t + 1].middle)
                annotation[region, spk] = f"SPEAKER_{spk:02d}"

        return annotation


# ============================================================================
# Overlap-Aware Speaker Embedding (from diart/blocks/embedding.py)
# ============================================================================

def _overlapped_speech_penalty(
    segmentation: torch.Tensor, gamma: float = GAMMA, beta: float = BETA
) -> torch.Tensor:
    """
    diart 论文 Equation 2: 重叠语音惩罚

    segmentation: (batch, frames, speakers) 或 (frames, speakers)
    返回: 同形状的权重矩阵
    """
    probs = torch.softmax(beta * segmentation, dim=-1)
    weights = torch.pow(segmentation, gamma) * torch.pow(probs, gamma)
    weights[weights < 1e-8] = 1e-8
    return weights


# ============================================================================
# Online Diarizer (main class)
# ============================================================================

@dataclass
class DiarizationState:
    """
    一次录音会话的在线分离状态

    每次 start_recording 时创建新实例。
    """
    # 滑动窗口缓冲区 (float32, 单声道)
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    # 时间追踪
    total_samples: int = 0
    last_processed_step: int = -1
    # 聚类状态
    clustering: OnlineSpeakerClustering = field(
        default_factory=OnlineSpeakerClustering
    )
    # 聚合缓冲区
    pred_buffer: list[SlidingWindowFeature] = field(default_factory=list)
    chunk_buffer: list[SlidingWindowFeature] = field(default_factory=list)
    # 聚合 + 二值化
    aggregation: DelayedAggregation = field(default_factory=DelayedAggregation)
    binarize: Binarize = field(default_factory=Binarize)
    # 当前说话人 (内部追踪，不对外暴露身份)
    current_speaker: str = "SPEAKER_00"
    # 说话人切换滞后
    speaker_turn_start: float = 0.0     # 当前说话人回合开始时间 (秒)
    # 变化检测标志
    speaker_changed: bool = False       # True 表示检测到说话人切换


class OnlineDiarizer:
    """
    在线说话人分离器

    使用 segmentation-3.0 + WeSpeaker 实现 diart 风格的
    帧级说话人追踪 + 在线余弦聚类。
    """

    def __init__(self):
        self._seg_model = None
        self._emb_model = None
        self._powerset = None
        self._device = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, device: str | None = None) -> None:
        """加载 segmentation + embedding 模型到 GPU"""
        if self._loaded:
            return

        from pyannote.audio import Model
        from pyannote.audio.utils.powerset import Powerset

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        settings = get_settings()
        models_dir = settings.paths.models_dir / "pyannote"

        # 1. 加载 segmentation-3.0
        seg_path = models_dir / "segmentation-3.0"
        logger.info(f"加载 segmentation 模型: {seg_path}")
        seg_model = Model.from_pretrained(str(seg_path))
        seg_model.eval().to(self._device)

        # Powerset 解码器 (7类 → 3说话人多标签)
        specs = seg_model.specifications
        self._powerset = Powerset(
            len(specs.classes), specs.powerset_max_classes
        )
        self._seg_model = seg_model

        # 2. 加载 WeSpeaker embedding
        emb_path = models_dir / "wespeaker-voxceleb-resnet34-LM"
        logger.info(f"加载 embedding 模型: {emb_path}")
        emb_model = Model.from_pretrained(str(emb_path))
        emb_model.eval().to(self._device)
        self._emb_model = emb_model

        self._loaded = True
        logger.info(
            f"在线分离模型加载完成 (device={self._device})"
        )

    def unload(self) -> None:
        """释放 GPU 显存"""
        if not self._loaded:
            return

        del self._seg_model
        del self._emb_model
        del self._powerset
        self._seg_model = None
        self._emb_model = None
        self._powerset = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("在线分离模型已释放")

    def create_state(self) -> DiarizationState:
        """为新录音会话创建状态"""
        return DiarizationState()

    def _segmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        运行 segmentation 模型

        waveform: (batch, channels, samples)
        返回: (batch, frames, local_speakers) 多标签概率
        """
        with torch.no_grad():
            raw_output = self._seg_model(waveform.to(self._device))
            # powerset 解码: (batch, frames, 7) → (batch, frames, 3)
            # Powerset.mapping 需要和输出在同一设备
            if self._powerset.mapping.device != raw_output.device:
                self._powerset.mapping = self._powerset.mapping.to(raw_output.device)
            multilabel = self._powerset.to_multilabel(raw_output)
        return multilabel.cpu()

    def _embedding(
        self,
        waveform: torch.Tensor,
        segmentation: torch.Tensor,
    ) -> np.ndarray:
        """
        提取 overlap-aware speaker embedding

        waveform: (batch, channels, samples)
        segmentation: (batch, frames, speakers)
        返回: (speakers, emb_dim) numpy array
        """
        with torch.no_grad():
            # Overlap-aware 权重 (diart Eq. 2)
            weights = _overlapped_speech_penalty(segmentation)  # (batch, frames, spk)

            batch_size = waveform.shape[0]
            num_speakers = weights.shape[2]

            # 重复波形给每个说话人
            # (batch, ch, samples) → (batch, spk, samples) → (batch*spk, 1, samples)
            wave_rep = waveform.repeat(1, num_speakers, 1)
            wave_flat = wave_rep.reshape(batch_size * num_speakers, 1, -1)

            # 权重展平: (batch, frames, spk) → (batch*spk, frames)
            weights_flat = weights.permute(0, 2, 1).reshape(batch_size * num_speakers, -1)

            # WeSpeaker embedding
            emb = self._emb_model(
                wave_flat.to(self._device),
                weights_flat.to(self._device),
            )

            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)

            # (batch*spk, dim) → (batch, spk, dim)
            emb = emb.reshape(batch_size, num_speakers, -1)

            # L2 归一化
            emb = emb / (torch.norm(emb, p=2, dim=-1, keepdim=True) + 1e-8)

        return emb.squeeze(0).cpu().numpy()  # (speakers, dim)

    def feed_chunk(
        self, state: DiarizationState, pcm_int16: bytes
    ) -> str | None:
        """
        喂入 PCM 音频，返回当前说话人 ID

        pcm_int16: int16 PCM 音频 bytes (16kHz 单声道)
        返回: "SPEAKER_XX" 或 None (尚未累积足够音频)
        """
        if not self._loaded:
            return None

        # PCM int16 → float32
        audio = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        state.total_samples += len(audio)

        # 追加到缓冲区
        state.audio_buffer = np.concatenate([state.audio_buffer, audio])

        # 检查是否该处理一个新步进
        current_step = (state.total_samples - WINDOW_SAMPLES) // STEP_SAMPLES
        if current_step <= state.last_processed_step:
            return None  # 还没积累够一个步进
        if state.total_samples < WINDOW_SAMPLES:
            return None  # 还没积累够第一个窗口

        state.last_processed_step = current_step

        # 取最新的 WINDOW_SAMPLES 作为当前窗口
        window_audio = state.audio_buffer[-WINDOW_SAMPLES:]
        # 裁剪缓冲区: 保留 window + 一点余量
        keep = WINDOW_SAMPLES + STEP_SAMPLES
        if len(state.audio_buffer) > keep:
            state.audio_buffer = state.audio_buffer[-keep:]

        # 计算窗口的绝对开始时间
        window_start = (state.total_samples - WINDOW_SAMPLES) / SAMPLE_RATE

        # 构造 SlidingWindowFeature
        waveform_swf = SlidingWindowFeature(
            window_audio.reshape(-1, 1),  # (samples, 1)
            SlidingWindow(
                start=window_start,
                duration=1.0 / SAMPLE_RATE,
                step=1.0 / SAMPLE_RATE,
            ),
        )

        # (1, 1, samples) — batch=1, channels=1
        waveform_tensor = torch.from_numpy(window_audio).unsqueeze(0).unsqueeze(0)

        # 1. Segmentation → (1, frames, local_speakers)
        seg_output = self._segmentation(waveform_tensor)
        seg_data = seg_output.squeeze(0).numpy()  # (frames, local_speakers)

        seg_resolution = WINDOW_DURATION / seg_data.shape[0]
        seg_sw = SlidingWindow(
            start=window_start, duration=seg_resolution, step=seg_resolution
        )
        seg_swf = SlidingWindowFeature(seg_data, seg_sw)

        # 2. Embedding → (local_speakers, emb_dim)
        emb = self._embedding(waveform_tensor, seg_output)

        # 3. Online clustering → global-space segmentation
        permuted_seg = state.clustering(seg_swf, emb)

        # 4. 缓冲区管理
        state.chunk_buffer.append(waveform_swf)
        state.pred_buffer.append(permuted_seg)

        # 5. 聚合 + 二值化
        agg_prediction = state.aggregation(state.pred_buffer)
        annotation = state.binarize(agg_prediction)

        # 裁剪缓冲区
        if len(state.chunk_buffer) > state.aggregation.num_overlapping:
            state.chunk_buffer = state.chunk_buffer[1:]
            state.pred_buffer = state.pred_buffer[1:]

        # 6. 从 Annotation 中提取最新时刻的说话人 (简单滞后)
        speaker = self._extract_current_speaker(annotation)
        state.speaker_changed = False  # 每次重置

        if speaker is not None and speaker != state.current_speaker:
            current_time = state.total_samples / SAMPLE_RATE
            turn_duration = current_time - state.speaker_turn_start
            if turn_duration >= MIN_TURN_DURATION:
                logger.debug(
                    f"说话人切换: {state.current_speaker} → {speaker} "
                    f"(持续 {turn_duration:.1f}s)"
                )
                state.current_speaker = speaker
                state.speaker_turn_start = current_time
                state.speaker_changed = True

        return state.speaker_changed

    def _extract_current_speaker(self, annotation: Annotation) -> str | None:
        """从 Annotation 中提取最新活跃的说话人"""
        if len(annotation) == 0:
            return None

        # 找最晚结束的 segment
        latest_end = 0.0
        latest_speaker = None
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            if segment.end >= latest_end:
                latest_end = segment.end
                latest_speaker = speaker
        return latest_speaker


# ============================================================================
# Singleton
# ============================================================================

_instance: OnlineDiarizer | None = None


def get_online_diarizer() -> OnlineDiarizer:
    """获取单例 OnlineDiarizer"""
    global _instance
    if _instance is None:
        _instance = OnlineDiarizer()
    return _instance
