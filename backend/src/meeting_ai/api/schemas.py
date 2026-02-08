"""
API 请求/响应模型
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# 枚举类型
# ============================================================================

class JobStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EnhanceMode(str, Enum):
    """音频增强模式"""
    NONE = "none"
    DENOISE = "denoise"      # DeepFilterNet3 降噪+去混响
    ENHANCE = "enhance"      # DeepFilterNet3 + Resemble Enhance 降噪+清晰化
    VOCAL = "vocal"          # Demucs v4 人声分离
    FULL = "full"            # Demucs + DeepFilterNet3 + Resemble Enhance 完整增强


# ============================================================================
# 请求模型
# ============================================================================

class ProcessRequest(BaseModel):
    """音频处理请求"""
    whisper_model: str = Field(default="medium", description="Whisper 模型名称")
    llm_model: Optional[str] = Field(default=None, description="LLM 模型路径，None 表示不使用")
    enable_naming: bool = Field(default=True, description="启用智能命名")
    enable_correction: bool = Field(default=True, description="启用错别字校正")
    enable_summary: bool = Field(default=True, description="启用会议总结")
    enhance_mode: EnhanceMode = Field(default=EnhanceMode.NONE, description="音频增强模式")


class StreamingStartRequest(BaseModel):
    """实时录音开始请求"""
    whisper_model: str = Field(default="medium", description="Whisper 模型名称")
    llm_model: Optional[str] = Field(default=None, description="LLM 模型路径")
    enable_naming: bool = Field(default=True, description="启用智能命名")
    enable_correction: bool = Field(default=True, description="启用错别字校正")
    enable_summary: bool = Field(default=True, description="启用会议总结")


class SegmentUpdateRequest(BaseModel):
    """更新片段请求"""
    text: Optional[str] = Field(default=None, description="新的文本内容")
    speaker_name: Optional[str] = Field(default=None, description="新的说话人名称")


class SpeakerRenameRequest(BaseModel):
    """重命名说话人请求"""
    speaker_id: str = Field(..., description="说话人 ID")
    new_name: str = Field(..., description="新名称")


class SummaryUpdateRequest(BaseModel):
    """更新总结请求"""
    summary: str = Field(..., description="新的总结内容")


class SegmentSplitRequest(BaseModel):
    """分割片段请求"""
    split_position: int = Field(..., description="分割位置（文本中的字符索引）")
    new_speaker: Optional[str] = Field(default=None, description="后半部分的新说话人 ID")


class SegmentMergeRequest(BaseModel):
    """合并片段请求"""
    segment_ids: list[int] = Field(..., description="要合并的片段 ID 列表（必须连续）")


# ============================================================================
# 响应模型
# ============================================================================

class SegmentResponse(BaseModel):
    """对话片段响应"""
    id: int
    start: float
    end: float
    text: str
    speaker: str
    speaker_name: Optional[str] = None

    class Config:
        from_attributes = True


class SpeakerResponse(BaseModel):
    """说话人信息响应"""
    id: str
    display_name: str
    gender: Optional[str] = None
    total_duration: float = 0
    segment_count: int = 0

    class Config:
        from_attributes = True


class JobResponse(BaseModel):
    """任务响应"""
    job_id: str
    status: JobStatus
    progress: float = Field(default=0, ge=0, le=1)
    message: str = ""
    created_at: datetime
    completed_at: Optional[datetime] = None


class ProcessResultResponse(BaseModel):
    """处理结果响应"""
    job_id: str
    status: JobStatus
    segments: list[SegmentResponse] = []
    speakers: dict[str, SpeakerResponse] = {}
    summary: str = ""
    audio_url: Optional[str] = None
    audio_original_url: Optional[str] = None
    duration: float = 0
    output_dir: str = ""


class HistoryItemResponse(BaseModel):
    """历史记录项"""
    id: str
    name: str
    created_at: datetime
    duration: float = 0
    segment_count: int = 0
    has_summary: bool = False


class HistoryListResponse(BaseModel):
    """历史记录列表响应"""
    items: list[HistoryItemResponse]
    total: int


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    name: str
    display_name: str
    path: str
    size_mb: Optional[float] = None


class ModelsListResponse(BaseModel):
    """模型列表响应"""
    whisper_models: list[ModelInfoResponse]
    llm_models: list[ModelInfoResponse]
    diarization_models: list[ModelInfoResponse]
    gender_models: list[ModelInfoResponse]


class StreamingEngineResponse(BaseModel):
    """流式 ASR 引擎信息"""
    id: str
    name: str
    description: str
    installed: bool
    model_dir: str


class StreamingEnginesListResponse(BaseModel):
    """流式引擎列表响应"""
    engines: list[StreamingEngineResponse]
    current: str


class SystemInfoResponse(BaseModel):
    """系统信息响应"""
    version: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    models_dir: str
    output_dir: str


# ============================================================================
# WebSocket 消息模型
# ============================================================================

class WSMessageType(str, Enum):
    """WebSocket 消息类型"""
    # 服务端 -> 客户端
    PROGRESS = "progress"
    SEGMENT = "segment"
    STATUS = "status"
    ERROR = "error"
    RESULT = "result"
    # 客户端 -> 服务端
    START = "start"
    STOP = "stop"
    AUDIO_CHUNK = "audio_chunk"


class WSProgressMessage(BaseModel):
    """进度消息"""
    type: str = WSMessageType.PROGRESS
    progress: float
    message: str


class WSSegmentMessage(BaseModel):
    """新片段消息"""
    type: str = WSMessageType.SEGMENT
    segment: SegmentResponse


class WSStatusMessage(BaseModel):
    """状态消息"""
    type: str = WSMessageType.STATUS
    is_recording: bool = False
    duration_seconds: float = 0
    volume_db: float = -60
    is_speech: bool = False
    current_speaker: str = ""
    segment_count: int = 0


class WSErrorMessage(BaseModel):
    """错误消息"""
    type: str = WSMessageType.ERROR
    error: str
    detail: Optional[str] = None


class WSResultMessage(BaseModel):
    """结果消息"""
    type: str = WSMessageType.RESULT
    result: ProcessResultResponse
