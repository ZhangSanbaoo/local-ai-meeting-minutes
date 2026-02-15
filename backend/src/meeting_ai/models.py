"""
核心数据模型

定义整个项目中使用的数据结构，确保类型安全和数据验证。
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, Field, computed_field


class CharTimestamp(NamedTuple):
    """单个字/词的时间戳"""
    char: str
    start: float  # 秒
    end: float    # 秒


class Gender(str, Enum):
    """性别枚举"""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class NameKind(str, Enum):
    """命名类型"""

    NAME = "name"       # 真实姓名（如"张教授"）
    ROLE = "role"       # 角色（如"主持人"）
    GENDER = "gender"   # 性别标识（如"男性01"）
    UNKNOWN = "unknown" # 未知


class StreamingState(str, Enum):
    """实时录音状态"""

    IDLE = "idle"             # 空闲
    RECORDING = "recording"   # 录音中
    PROCESSING = "processing" # 后处理中
    DONE = "done"             # 完成


class StreamingStatus(BaseModel):
    """
    实时录音状态信息

    用于 GUI 实时显示录音状态。
    """

    state: StreamingState = Field(default=StreamingState.IDLE, description="当前状态")
    duration_seconds: float = Field(default=0.0, ge=0, description="录音时长（秒）")
    volume_db: float = Field(default=-60.0, description="当前音量（dB）")
    is_speech: bool = Field(default=False, description="当前是否检测到语音")
    current_speaker: str = Field(default="SPEAKER_00", description="当前说话人")
    segment_count: int = Field(default=0, ge=0, description="已转写片段数")


class StreamingSegment(BaseModel):
    """
    实时转写片段

    录音过程中实时生成的转写片段，带临时说话人标识。
    """

    id: int = Field(description="片段序号")
    start: float = Field(ge=0, description="开始时间（秒）")
    end: float = Field(ge=0, description="结束时间（秒）")
    text: str = Field(default="", description="转写文本")
    temp_speaker: str = Field(description="临时说话人ID（如 SPEAKER_00）")
    # embedding 不序列化到 JSON，仅用于运行时匹配
    embedding: Any | None = Field(default=None, exclude=True, description="说话人 embedding")

    @computed_field
    @property
    def duration(self) -> float:
        """片段时长（秒）"""
        return self.end - self.start

    def format_time(self) -> str:
        """格式化开始时间，如 '01:23'"""
        m, s = divmod(int(self.start), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"


class Segment(BaseModel):
    """
    转写片段

    表示一段连续的语音及其转写文本。
    """

    id: int = Field(description="片段序号")
    start: float = Field(ge=0, description="开始时间（秒）")
    end: float = Field(ge=0, description="结束时间（秒）")
    text: str = Field(default="", description="转写文本")
    speaker: str | None = Field(default=None, description="说话人ID（如 SPEAKER_00）")

    @computed_field
    @property
    def duration(self) -> float:
        """片段时长（秒）"""
        return self.end - self.start

    def format_time(self) -> str:
        """格式化时间范围，如 '01:23 - 01:45'"""
        def fmt(seconds: float) -> str:
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h:02d}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        return f"{fmt(self.start)} - {fmt(self.end)}"


class SpeakerInfo(BaseModel):
    """
    说话人信息

    包含说话人的显示名称、性别、命名来源等信息。
    """

    id: str = Field(description="说话人ID（如 SPEAKER_00）")
    display_name: str = Field(description="显示名称")
    gender: Gender = Field(default=Gender.UNKNOWN, description="性别")
    kind: NameKind = Field(default=NameKind.UNKNOWN, description="命名类型")
    confidence: float = Field(default=0.0, ge=0, le=1, description="命名置信度")
    evidence: list[str] = Field(default_factory=list, description="命名依据")

    # 音频特征（可选）
    f0_median: float | None = Field(default=None, description="基频中位数(Hz)")
    total_duration: float = Field(default=0.0, description="总发言时长(秒)")
    segment_count: int = Field(default=0, description="发言片段数")


class TranscriptResult(BaseModel):
    """
    转写结果

    ASR 阶段的输出，包含语言检测结果和所有片段。
    """

    language: str = Field(description="检测到的语言代码")
    language_probability: float = Field(ge=0, le=1, description="语言检测置信度")
    duration: float = Field(ge=0, description="音频总时长（秒）")
    segments: list[Segment] = Field(default_factory=list, description="转写片段列表")

    # 字级时间戳（内部使用，不序列化到 JSON）
    # char_timestamps[i] 对应 segments[i] 的逐字时间戳
    char_timestamps: list[list[CharTimestamp]] | None = Field(
        default=None, exclude=True, description="逐字时间戳（内部对齐用）",
    )

    @computed_field
    @property
    def full_text(self) -> str:
        """完整转写文本"""
        return "\n".join(s.text for s in self.segments if s.text)

    @computed_field
    @property
    def segment_count(self) -> int:
        """片段数量"""
        return len(self.segments)


class DiarizationResult(BaseModel):
    """
    说话人分离结果

    包含说话人信息和带说话人标注的片段。
    """

    speakers: dict[str, SpeakerInfo] = Field(
        default_factory=dict, description="说话人信息字典"
    )
    segments: list[Segment] = Field(default_factory=list, description="带说话人的片段")

    @computed_field
    @property
    def speaker_count(self) -> int:
        """说话人数量"""
        return len(self.speakers)

    def get_speaker_segments(self, speaker_id: str) -> list[Segment]:
        """获取指定说话人的所有片段"""
        return [s for s in self.segments if s.speaker == speaker_id]


class MeetingSummary(BaseModel):
    """
    会议总结

    LLM 生成的会议摘要和要点。
    """

    title: str = Field(default="", description="会议主题")
    summary: str = Field(default="", description="会议摘要")
    key_points: list[str] = Field(default_factory=list, description="要点列表")
    action_items: list[str] = Field(default_factory=list, description="待办事项")
    decisions: list[str] = Field(default_factory=list, description="决议事项")


class JobStatus(str, Enum):
    """任务状态"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobMeta(BaseModel):
    """
    任务元信息

    跟踪一个转写任务的完整状态。
    """

    job_id: str = Field(description="任务ID")
    status: JobStatus = Field(default=JobStatus.PENDING, description="任务状态")

    # 输入
    input_file: str = Field(description="输入文件名")
    input_path: Path | None = Field(default=None, description="输入文件路径")

    # 配置快照
    asr_model: str = Field(default="small", description="ASR 模型")
    device: str = Field(default="cpu", description="计算设备")

    # 结果
    language: str | None = Field(default=None, description="检测到的语言")
    duration: float | None = Field(default=None, description="音频时长")
    segment_count: int | None = Field(default=None, description="片段数量")
    speaker_count: int | None = Field(default=None, description="说话人数量")

    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    completed_at: datetime | None = Field(default=None, description="完成时间")

    # 错误信息
    error: str | None = Field(default=None, description="错误信息")

    def mark_completed(
        self,
        language: str,
        duration: float,
        segment_count: int,
        speaker_count: int,
    ) -> None:
        """标记任务完成"""
        self.status = JobStatus.COMPLETED
        self.language = language
        self.duration = duration
        self.segment_count = segment_count
        self.speaker_count = speaker_count
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """标记任务失败"""
        self.status = JobStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()


class Job(BaseModel):
    """
    完整的任务对象

    包含元信息和所有处理结果。
    """

    meta: JobMeta
    transcript: TranscriptResult | None = None
    diarization: DiarizationResult | None = None
    summary: MeetingSummary | None = None

    # 输出路径
    output_dir: Path | None = None

    def save(self) -> None:
        """保存任务到输出目录"""
        if self.output_dir is None:
            raise ValueError("output_dir is not set")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存元信息
        (self.output_dir / "meta.json").write_text(
            self.meta.model_dump_json(indent=2), encoding="utf-8"
        )

        # 保存转写结果
        if self.transcript is not None:
            (self.output_dir / "transcript.json").write_text(
                self.transcript.model_dump_json(indent=2), encoding="utf-8"
            )
            (self.output_dir / "transcript.txt").write_text(
                self.transcript.full_text, encoding="utf-8"
            )

        # 保存说话人分离结果
        if self.diarization is not None:
            (self.output_dir / "diarization.json").write_text(
                self.diarization.model_dump_json(indent=2), encoding="utf-8"
            )

        # 保存会议总结
        if self.summary is not None:
            (self.output_dir / "summary.json").write_text(
                self.summary.model_dump_json(indent=2), encoding="utf-8"
            )

    @classmethod
    def load(cls, output_dir: Path) -> "Job":
        """从输出目录加载任务"""
        import json

        meta_path = output_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Job not found: {output_dir}")

        meta = JobMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))

        job = cls(meta=meta, output_dir=output_dir)

        # 加载转写结果
        transcript_path = output_dir / "transcript.json"
        if transcript_path.exists():
            job.transcript = TranscriptResult.model_validate_json(
                transcript_path.read_text(encoding="utf-8")
            )

        # 加载说话人分离结果
        diarization_path = output_dir / "diarization.json"
        if diarization_path.exists():
            job.diarization = DiarizationResult.model_validate_json(
                diarization_path.read_text(encoding="utf-8")
            )

        # 加载会议总结
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            job.summary = MeetingSummary.model_validate_json(
                summary_path.read_text(encoding="utf-8")
            )

        return job
