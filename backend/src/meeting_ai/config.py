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

    model_config = SettingsConfigDict(
        env_prefix="MEETING_AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 将相对路径转换为绝对路径（并解析 .. 等）
        for field_name in ["data_dir", "models_dir", "output_dir", "cache_dir"]:
            path = getattr(self, field_name)
            if not path.is_absolute():
                # 使用 resolve() 来正确处理 ../xxx 这种路径
                setattr(self, field_name, (self.root_dir / path).resolve())


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

    # 引擎名称（对应 models/diarization/{engine}/ 目录名）
    engine: str = Field(default="pyannote-3.1")

    # 聚类参数
    min_speakers: int | None = None  # 最少说话人数（可选）
    max_speakers: int | None = None  # 最多说话人数（可选）

    # 合并阈值：embedding 余弦相似度超过此值认为是同一人
    embedding_threshold: float = 0.75

    model_config = SettingsConfigDict(env_prefix="MEETING_AI_DIAR_")


class GenderSettings(BaseSettings):
    """性别检测配置"""

    # 引擎名称: f0 / ecapa-gender / wav2vec2-gender
    # f0 = 内置基频分析（零依赖），其他引擎对应 models/gender/{engine}/ 目录
    engine: str = Field(default="f0")

    model_config = SettingsConfigDict(env_prefix="MEETING_AI_GENDER_")


class StreamingSettings(BaseSettings):
    """实时流式转写配置"""

    # 音频参数
    sample_rate: int = 16000
    channels: int = 1

    # ASR 引擎选择
    asr_engine: Literal["funasr", "sherpa-onnx"] = "funasr"

    # ── 通用设备 ─────────────────────────────────────────────────
    device: str = Field(
        default="auto",
        description="流式 ASR 计算设备 (auto/cpu/cuda:0)",
    )

    # ── FunASR 配置 ──────────────────────────────────────────────
    funasr_model_dir: Path = Field(
        default=Path("models/streaming/funasr/paraformer-zh-streaming"),
        description="FunASR 流式 ASR 模型目录",
    )
    funasr_punc_dir: Path = Field(
        default=Path("models/streaming/funasr/ct-punc"),
        description="FunASR 标点恢复模型目录",
    )
    funasr_vad_dir: Path = Field(
        default=Path("models/streaming/funasr/fsmn-vad"),
        description="FunASR VAD 语音活动检测模型目录",
    )
    funasr_vad_silence_ms: int = Field(
        default=800,
        description="VAD 判定语音结束的静默时长 (ms)，对应 fsmn-vad 的 max_end_silence_time",
    )
    funasr_chunk_size: list[int] = Field(
        default=[1, 10, 5],
        description="Paraformer 流式 chunk 配置 [动态标志, chunk, 右看] "
        "(单位: 60ms帧; 首位=1 启用动态右看，静默时自动扩展上下文提升精度)",
    )
    funasr_encoder_chunk_look_back: int = Field(
        default=4,
        description="编码器左看 chunk 数",
    )
    funasr_decoder_chunk_look_back: int = Field(
        default=1,
        description="解码器左看 chunk 数",
    )

    # ── sherpa-onnx 配置 ─────────────────────────────────────────
    sherpa_model_dir: Path = Field(
        default=Path("models/streaming/sherpa-onnx"),
        description="sherpa-onnx 模型目录",
    )

    # ── 录音限制 ─────────────────────────────────────────────────
    max_recording_seconds: int = Field(default=3600, description="最大录音时长 (秒)")

    # ── 前端音频发送间隔 ──────────────────────────────────────────
    chunk_duration_ms: int = Field(
        default=300,
        description="前端每次发送的音频 chunk 时长 (ms)",
    )

    model_config = SettingsConfigDict(env_prefix="MEETING_AI_STREAMING_")


class LLMSettings(BaseSettings):
    """本地 LLM 配置"""

    # 是否启用 LLM 命名
    enabled: bool = True

    # 模型路径（GGUF 格式，相对于 models_dir）
    model_path: Path | None = Field(default=Path("llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf"))

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
    gender: GenderSettings = Field(default_factory=GenderSettings)
    streaming: StreamingSettings = Field(default_factory=StreamingSettings)
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
