# Utils module - 工具函数

from .audio import ensure_wav_16k_mono, get_audio_info

__all__ = ["ensure_wav_16k_mono", "get_audio_info"]

# 音频增强模块（可选依赖）
try:
    from .enhance import auto_enhance, enhance_audio
    __all__.extend(["enhance_audio", "auto_enhance"])
except ImportError:
    pass

# VAD 模块（可选依赖）
try:
    from .vad import VADWrapper, SpeechSegmenter, calculate_volume_db
    __all__.extend(["VADWrapper", "SpeechSegmenter", "calculate_volume_db"])
except ImportError:
    pass
