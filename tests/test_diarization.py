"""
说话人分离服务测试

注意：完整测试需要下载模型，这里只测试基本逻辑。
"""

import pytest
from pathlib import Path


def test_diarization_service_init():
    """测试服务初始化（不加载模型）"""
    from meeting_ai.services.diarization import DiarizationService
    
    # 初始化不应该加载模型
    service = DiarizationService()
    
    # pipeline 应该是 None（懒加载）
    assert service._pipeline is None


def test_diarization_result_model():
    """测试 DiarizationResult 数据模型"""
    from meeting_ai.models import DiarizationResult, Segment, SpeakerInfo
    
    # 创建测试数据
    segments = [
        Segment(id=0, start=0.0, end=5.0, text="", speaker="SPEAKER_00"),
        Segment(id=1, start=5.0, end=10.0, text="", speaker="SPEAKER_01"),
        Segment(id=2, start=10.0, end=15.0, text="", speaker="SPEAKER_00"),
    ]
    
    speakers = {
        "SPEAKER_00": SpeakerInfo(
            id="SPEAKER_00",
            display_name="SPEAKER_00",
            total_duration=10.0,
            segment_count=2,
        ),
        "SPEAKER_01": SpeakerInfo(
            id="SPEAKER_01",
            display_name="SPEAKER_01",
            total_duration=5.0,
            segment_count=1,
        ),
    }
    
    result = DiarizationResult(speakers=speakers, segments=segments)
    
    # 测试属性
    assert result.speaker_count == 2
    assert len(result.segments) == 3
    
    # 测试获取特定说话人的片段
    speaker_00_segments = result.get_speaker_segments("SPEAKER_00")
    assert len(speaker_00_segments) == 2
    
    # 测试序列化
    json_str = result.model_dump_json()
    assert "SPEAKER_00" in json_str
    assert "SPEAKER_01" in json_str


def test_segment_format_time():
    """测试时间格式化"""
    from meeting_ai.models import Segment
    
    # 不到一小时
    seg1 = Segment(id=0, start=65.0, end=125.0, speaker="A")
    assert seg1.format_time() == "01:05 - 02:05"
    
    # 超过一小时
    seg2 = Segment(id=0, start=3665.0, end=3725.0, speaker="A")
    assert seg2.format_time() == "01:01:05 - 01:02:05"


@pytest.mark.skipif(
    not Path("models/pyannote/speaker-diarization-3.1").exists(),
    reason="需要下载 pyannote 模型才能运行此测试"
)
def test_diarization_with_real_audio():
    """
    使用真实音频测试（需要模型）
    
    运行此测试前需要：
    1. 下载模型
    2. 准备测试音频文件
    """
    # TODO: 添加真实音频测试
    pass
