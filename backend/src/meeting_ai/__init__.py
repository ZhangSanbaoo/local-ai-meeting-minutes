"""
Meeting AI - 离线会议纪要工具

功能：
- 语音转写（ASR）：使用 faster-whisper
- 说话人分离（Diarization）：使用 pyannote-audio
- 智能命名：使用本地 LLM
- 会议总结：使用本地 LLM
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import Settings, get_settings
from .models import (
    DiarizationResult,
    Gender,
    Job,
    JobMeta,
    JobStatus,
    MeetingSummary,
    NameKind,
    Segment,
    SpeakerInfo,
    StreamingSegment,
    StreamingState,
    StreamingStatus,
    TranscriptResult,
)
from .services import (
    ASRService,
    DiarizationService,
    NamingService,
    align_transcript_with_speakers,
    detect_all_genders,
    detect_gender,
    fix_unknown_speakers,
    get_asr_service,
    get_diarization_service,
    get_naming_service,
    merge_adjacent_segments,
)

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    # 配置
    "Settings",
    "get_settings",
    # 数据模型
    "Gender",
    "NameKind",
    "JobStatus",
    "StreamingState",
    "StreamingStatus",
    "StreamingSegment",
    "Segment",
    "SpeakerInfo",
    "TranscriptResult",
    "DiarizationResult",
    "MeetingSummary",
    "JobMeta",
    "Job",
    # 服务
    "DiarizationService",
    "get_diarization_service",
    "ASRService",
    "get_asr_service",
    "NamingService",
    "get_naming_service",
    "align_transcript_with_speakers",
    "fix_unknown_speakers",
    "merge_adjacent_segments",
    "detect_gender",
    "detect_all_genders",
]
