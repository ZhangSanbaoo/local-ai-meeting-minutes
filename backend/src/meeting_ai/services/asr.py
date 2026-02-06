"""
语音转写服务（ASR）

使用 faster-whisper 进行语音识别。
把音频转换成带时间戳的文字片段。
"""

from pathlib import Path

from faster_whisper import WhisperModel

from ..config import get_settings
from ..logger import get_logger
from ..models import Segment, TranscriptResult

logger = get_logger("services.asr")


class ASRService:
    """
    语音转写服务
    
    使用方式：
        service = ASRService()
        result = service.transcribe("audio.wav")
        
        print(f"语言: {result.language}")
        for seg in result.segments:
            print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}")
    """

    def __init__(self) -> None:
        """初始化服务（不立即加载模型）"""
        self._model: WhisperModel | None = None
        self._settings = get_settings().asr

    @property
    def model(self) -> WhisperModel:
        """懒加载 Whisper 模型"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> WhisperModel:
        """加载本地 Whisper 模型"""
        settings = get_settings()
        
        model_name = self._settings.model_name
        device = self._settings.device
        compute_type = self._settings.compute_type

        # 处理 device = "auto"
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # CPU 只支持 int8 或 float32
            if device == "cpu" and compute_type == "float16":
                compute_type = "int8"

        logger.info(f"加载 Whisper 模型: {model_name} (device={device}, compute={compute_type})")

        # 本地模型路径
        local_model_dir = settings.paths.models_dir / "whisper" / f"faster-whisper-{model_name}"
        
        if not local_model_dir.exists():
            raise FileNotFoundError(
                f"Whisper 模型不存在: {local_model_dir}\n"
                f"请先下载模型到该目录。"
            )

        logger.info(f"使用本地模型: {local_model_dir}")

        model = WhisperModel(
            str(local_model_dir),
            device=device,
            compute_type=compute_type,
        )

        return model

    def transcribe(
        self,
        audio_path: Path | str,
        language: str | None = None,
    ) -> TranscriptResult:
        """
        转写音频文件
        
        Args:
            audio_path: 音频文件路径
            language: 指定语言（如 "zh", "en"），None 表示自动检测
            
        Returns:
            TranscriptResult: 包含语言信息和所有转写片段
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        logger.info(f"开始转写: {audio_path.name}")

        # 使用配置或参数，默认中文以确保正确的标点符号输出
        lang = language or self._settings.language or "zh"

        # 运行转写
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=lang,
            beam_size=self._settings.beam_size,
            vad_filter=self._settings.vad_filter,
            word_timestamps=self._settings.word_timestamps,
        )

        # 收集结果
        segments = []
        for seg in segments_iter:
            segment = Segment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                speaker=None,  # ASR 不知道说话人，需要后续对齐
            )
            segments.append(segment)

        logger.info(
            f"转写完成: 语言={info.language}, "
            f"概率={info.language_probability:.2f}, "
            f"片段数={len(segments)}"
        )

        return TranscriptResult(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segments,
        )


# 模块级别的单例
_service: ASRService | None = None


def get_asr_service() -> ASRService:
    """获取 ASR 服务的单例"""
    global _service
    if _service is None:
        _service = ASRService()
    return _service
