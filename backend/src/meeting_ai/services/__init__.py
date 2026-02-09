# Services module - 服务层（ASR、Diarization、LLM 等）

from .alignment import (
    align_transcript_with_speakers,
    fix_unknown_speakers,
    merge_adjacent_segments,
)
from .asr import (
    ASREngine,
    ASRService,
    get_asr_engine,
    get_asr_service,
    unload_punc_model,
    unload_vad_model,
)
from .correction import correct_segments, correct_text, correct_transcript_batch
from .diarization import DiarizationService, get_diarization_service
from .gender import detect_all_genders, detect_gender
from .llm import get_llm, reset_llm
from .naming import NamingService, get_naming_service
from .streaming_asr import (
    StreamingASREngine,
    StreamingSession,
    get_streaming_asr_engine,
    reset_streaming_asr_engine,
)
from .summary import format_summary_markdown, summarize_meeting

__all__ = [
    # Diarization
    "DiarizationService",
    "get_diarization_service",
    # ASR
    "ASREngine",
    "ASRService",
    "get_asr_engine",
    "get_asr_service",
    "unload_vad_model",
    "unload_punc_model",
    # Alignment
    "align_transcript_with_speakers",
    "fix_unknown_speakers",
    "merge_adjacent_segments",
    # Gender
    "detect_gender",
    "detect_all_genders",
    # LLM
    "get_llm",
    "reset_llm",
    # Naming
    "NamingService",
    "get_naming_service",
    # Correction
    "correct_text",
    "correct_segments",
    "correct_transcript_batch",
    # Streaming ASR
    "StreamingASREngine",
    "StreamingSession",
    "get_streaming_asr_engine",
    "reset_streaming_asr_engine",
    # Summary
    "summarize_meeting",
    "format_summary_markdown",
]
