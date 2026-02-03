"""
说话人分离服务

使用 pyannote-audio 进行说话人分离（Speaker Diarization）。
回答"谁在什么时候说话"这个问题。

输入：音频文件
输出：每个说话人的时间段列表
"""

from pathlib import Path

import torch
from pyannote.audio import Pipeline

from ..config import get_settings
from ..logger import get_logger
from ..models import DiarizationResult, Segment, SpeakerInfo

logger = get_logger("services.diarization")


class DiarizationService:
    """
    说话人分离服务
    
    使用方式：
        service = DiarizationService()
        result = service.diarize("meeting.wav")
        
        for segment in result.segments:
            print(f"{segment.speaker}: {segment.start:.1f}s - {segment.end:.1f}s")
    """

    def __init__(self) -> None:
        """初始化服务（不立即加载模型）"""
        self._pipeline: Pipeline | None = None
        self._settings = get_settings().diarization

    @property
    def pipeline(self) -> Pipeline:
        """
        懒加载 pyannote pipeline
        
        第一次访问时加载模型，之后复用。
        这样可以避免在导入模块时就加载大模型。
        """
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self) -> Pipeline:
        """从本地目录加载 pyannote 模型"""
        settings = get_settings()
        model_dir = self._settings.model_dir
        
        # 如果是相对路径，相对于项目根目录
        if not model_dir.is_absolute():
            model_dir = settings.paths.root_dir / model_dir

        logger.info(f"加载说话人分离模型: {model_dir}")

        if not model_dir.exists():
            raise FileNotFoundError(
                f"模型目录不存在: {model_dir}\n"
                f"请先下载模型到该目录，参考 README.md 中的说明。"
            )

        # 检查必要的配置文件
        config_file = model_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(
                f"模型配置文件不存在: {config_file}\n"
                f"请确保模型下载完整。"
            )

        # 选择计算设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        # 加载 pipeline
        # pyannote 3.x 从本地加载，直接传路径
        try:
            # 新版本 API
            pipeline = Pipeline.from_pretrained(model_dir)
        except TypeError:
            # 兼容旧版本
            pipeline = Pipeline.from_pretrained(str(model_dir))
        
        pipeline = pipeline.to(device)

        return pipeline

    def diarize(
        self,
        audio_path: Path | str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """
        对音频进行说话人分离
        
        Args:
            audio_path: 音频文件路径（支持 wav, mp3, m4a 等）
            min_speakers: 最少说话人数（可选，用于约束聚类）
            max_speakers: 最多说话人数（可选，用于约束聚类）
            
        Returns:
            DiarizationResult: 包含说话人信息和时间段
            
        Example:
            >>> service = DiarizationService()
            >>> result = service.diarize("meeting.wav", max_speakers=3)
            >>> print(f"检测到 {result.speaker_count} 个说话人")
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        logger.info(f"开始说话人分离: {audio_path.name}")

        # 使用配置的默认值
        min_spk = min_speakers or self._settings.min_speakers
        max_spk = max_speakers or self._settings.max_speakers

        # 运行 pipeline
        # pyannote 的输入可以是文件路径
        diarization_output = self.pipeline(
            audio_path,
            min_speakers=min_spk,
            max_speakers=max_spk,
        )

        # 解析结果
        segments = []
        speaker_stats: dict[str, dict] = {}  # 统计每个说话人的信息

        # pyannote 3.x 返回 DiarizeOutput 对象，需要访问 speaker_diarization 属性
        # 旧版本直接返回 Annotation 对象
        if hasattr(diarization_output, 'speaker_diarization'):
            # 新版本：DiarizeOutput
            annotation = diarization_output.speaker_diarization
        else:
            # 旧版本：直接是 Annotation
            annotation = diarization_output

        # itertracks(yield_label=True) 返回 (Segment, track_name, speaker_label)
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segment = Segment(
                id=len(segments),
                start=turn.start,
                end=turn.end,
                text="",  # 这个阶段还没有文字，只有时间
                speaker=speaker,
            )
            segments.append(segment)

            # 统计说话人信息
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_duration": 0.0,
                    "segment_count": 0,
                }
            speaker_stats[speaker]["total_duration"] += segment.duration
            speaker_stats[speaker]["segment_count"] += 1

        # 构建说话人信息
        speakers = {}
        for speaker_id, stats in speaker_stats.items():
            speakers[speaker_id] = SpeakerInfo(
                id=speaker_id,
                display_name=speaker_id,  # 暂时用 ID 作为显示名
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


# 模块级别的单例，方便直接使用
_service: DiarizationService | None = None


def get_diarization_service() -> DiarizationService:
    """获取说话人分离服务的单例"""
    global _service
    if _service is None:
        _service = DiarizationService()
    return _service
