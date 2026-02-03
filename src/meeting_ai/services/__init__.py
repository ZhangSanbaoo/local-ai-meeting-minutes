# Services module - 服务层（ASR、Diarization、LLM 等）

from .alignment import (
    align_transcript_with_speakers,
    fix_unknown_speakers,
    merge_adjacent_segments,
)
from .asr import ASRService, get_asr_service
from .correction import correct_segments, correct_text, correct_transcript_batch
from .diarization import DiarizationService, get_diarization_service
from .gender import detect_all_genders, detect_gender
from .naming import NamingService, get_naming_service
from .summary import format_summary_markdown, summarize_meeting

__all__ = [
    # Diarization
    "DiarizationService",
    "get_diarization_service",
    # ASR
    "ASRService",
    "get_asr_service",
    # Alignment
    "align_transcript_with_speakers",
    "fix_unknown_speakers",
    "merge_adjacent_segments",
    # Gender
    "detect_gender",
    "detect_all_genders",
    # Naming
    "NamingService",
    "get_naming_service",
    # Correction
    "correct_text",
    "correct_segments",
    "correct_transcript_batch",
    # Summary
    "summarize_meeting",
    "format_summary_markdown",
]
