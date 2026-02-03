"""
基础测试 - 验证项目结构和导入
"""

import pytest


def test_import_package():
    """测试包可以正常导入"""
    import meeting_ai

    assert meeting_ai.__version__ == "0.1.0"


def test_import_config():
    """测试配置模块"""
    from meeting_ai.config import Settings, get_settings

    settings = get_settings()
    assert settings.app_name == "Meeting AI"
    assert settings.version == "0.1.0"


def test_import_models():
    """测试数据模型"""
    from meeting_ai.models import (
        Gender,
        JobStatus,
        NameKind,
        Segment,
        SpeakerInfo,
    )

    # 测试 Segment
    seg = Segment(id=1, start=0.0, end=1.5, text="Hello")
    assert seg.duration == 1.5

    # 测试 SpeakerInfo
    speaker = SpeakerInfo(
        id="SPEAKER_00",
        display_name="张教授",
        gender=Gender.MALE,
        kind=NameKind.NAME,
        confidence=0.85,
    )
    assert speaker.display_name == "张教授"


def test_settings_paths():
    """测试路径配置"""
    from meeting_ai.config import get_settings

    settings = get_settings()
    
    # 路径应该是 Path 对象
    assert hasattr(settings.paths, "data_dir")
    assert hasattr(settings.paths, "models_dir")
    assert hasattr(settings.paths, "output_dir")


def test_segment_format_time():
    """测试时间格式化"""
    from meeting_ai.models import Segment

    # 短时间
    seg1 = Segment(id=1, start=65.0, end=125.5, text="test")
    assert seg1.format_time() == "01:05 - 02:05"

    # 长时间（超过1小时）
    seg2 = Segment(id=2, start=3665.0, end=3725.0, text="test")
    assert seg2.format_time() == "01:01:05 - 01:02:05"
