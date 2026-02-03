"""
配置管理模块

使用 pydantic-settings 实现：
- 类型安全的配置
- 环境变量自动加载
- .env 文件支持
- 配置验证
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathSettings(BaseSettings):
    """路径相关配置"""

    # 项目根目录（自动检测）
    root_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # 数据目录
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))
    output_dir: Path = Field(default=Path("outputs"))
    cache_dir: Path = Field(default=Path(".cache"))

    model_config = SettingsConfigDict(env_prefix="MEETING_AI_")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 将相对路径转换为绝对路径
        for field_name in ["data_dir", "models_dir", "output_dir", "cache_dir"]:
            path = getattr(self, field_name)
            if not path.is_absolute():
                setattr(self, field_name, self.root_dir / path)


class ASRSettings(BaseSettings):
    """语音识别配置"""

    # Whisper 模型
    model_name: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "small"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    compute_type: Literal["int8", "float16", "float32"] = "int8"

    # 识别参数
    language: str | None = None  # None = 自动检测
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False

    model_config = SettingsConfigDict(env_prefix="MEETING_AI_ASR_")


class DiarizationSettings(BaseSettings):
    """说话人分离配置"""

    # 本地模型目录（相对于项目根目录或绝对路径）
    model_dir: Path = Field(default=Path("models/pyannote/speaker-diarization-3.1"))

    # 聚类参数
    min_speakers: int | None = None  # 最少说话人数（可选）
    max_speakers: int | None = None  # 最多说话人数（可选）
    
    # 合并阈值：embedding 余弦相似度超过此值认为是同一人
    embedding_threshold: float = 0.75

    model_config = SettingsConfigDict(env_prefix="MEETING_AI_DIAR_")


class LLMSettings(BaseSettings):
    """本地 LLM 配置"""

    # 是否启用 LLM 命名
    enabled: bool = True

    # 模型路径（GGUF 格式，相对于项目根目录或绝对路径）
    model_path: Path | None = Field(default=Path("models/llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf"))

    # 推理参数
    n_ctx: int = 4096              # 上下文长度
    n_threads: int | None = None   # CPU 线程数，None = 自动
    n_gpu_layers: int = -1         # GPU 层数，-1 = 全部放 GPU
    temperature: float = 0.1       # 温度，越低越确定
    max_tokens: int = 1024         # 最大生成 token 数

    # 命名置信度阈值（低于此值使用性别兜底）
    confidence_threshold: float = 0.6

    model_config = SettingsConfigDict(env_prefix="MEETING_AI_LLM_")

    @field_validator("model_path", mode="before")
    @classmethod
    def validate_model_path(cls, v):
        if v is None or v == "":
            return None
        return Path(v)


class Settings(BaseSettings):
    """主配置类 - 聚合所有子配置"""

    # 应用信息
    app_name: str = "Meeting AI"
    version: str = "0.1.0"
    debug: bool = False

    # 日志级别
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # 子配置
    paths: PathSettings = Field(default_factory=PathSettings)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    diarization: DiarizationSettings = Field(default_factory=DiarizationSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    model_config = SettingsConfigDict(
        env_prefix="MEETING_AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # 支持 MEETING_AI_ASR__MODEL_NAME 这种格式
        extra="ignore",
    )

    def ensure_dirs(self) -> None:
        """确保必要的目录存在"""
        for path in [
            self.paths.data_dir,
            self.paths.models_dir,
            self.paths.output_dir,
            self.paths.cache_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# 全局单例
_settings: Settings | None = None


def get_settings() -> Settings:
    """获取配置单例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """重新加载配置（用于测试）"""
    global _settings
    _settings = Settings()
    return _settings
