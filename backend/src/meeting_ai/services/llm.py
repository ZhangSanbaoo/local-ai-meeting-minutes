"""
共享 LLM 服务

提供统一的 LLM 实例，避免重复加载。
"""

import re
from pathlib import Path

from ..config import get_settings
from ..logger import get_logger

logger = get_logger("services.llm")

# 全局 LLM 单例
_llm = None
_llm_state = "unloaded"  # unloaded, disabled, loaded


def get_llm():
    """
    获取共享的 LLM 实例

    Returns:
        Llama 实例，如果未启用或加载失败则返回 None
    """
    global _llm, _llm_state

    if _llm_state == "disabled":
        return None

    if _llm_state == "loaded":
        return _llm

    # 首次加载
    from llama_cpp import Llama

    settings = get_settings()
    llm_settings = settings.llm

    if not llm_settings.enabled:
        logger.info("LLM 未启用")
        _llm_state = "disabled"
        return None

    model_path = llm_settings.model_path
    if model_path is None:
        logger.warning("LLM 模型路径未配置")
        _llm_state = "disabled"
        return None

    # 处理相对路径
    if not model_path.is_absolute():
        if str(model_path).startswith("models"):
            relative_path = Path(*model_path.parts[1:])
            model_path = settings.paths.models_dir / relative_path
        else:
            model_path = settings.paths.models_dir / model_path

    if not model_path.exists():
        logger.warning(f"LLM 模型不存在: {model_path}")
        _llm_state = "disabled"
        return None

    logger.info(f"加载 LLM 模型: {model_path}")

    _llm = Llama(
        model_path=str(model_path),
        n_ctx=llm_settings.n_ctx,
        n_threads=llm_settings.n_threads,
        n_gpu_layers=llm_settings.n_gpu_layers,
        verbose=False,
    )
    _llm_state = "loaded"
    logger.info("LLM 模型加载完成")

    return _llm


def get_llm_if_loaded():
    """
    获取已加载的 LLM 实例（不触发自动加载）

    仅在 LLM 已经加载时返回实例，否则返回 None。
    用于聊天等需要用户手动加载 LLM 的场景。
    """
    if _llm_state == "loaded":
        return _llm
    return None


def get_llm_status() -> dict:
    """
    获取 LLM 当前状态

    Returns:
        {"state": "unloaded"|"disabled"|"loaded", "model_name": str|None}
    """
    model_name = None
    if _llm_state == "loaded" and _llm is not None:
        model_path = getattr(_llm, "model_path", None)
        if model_path:
            model_name = Path(model_path).stem
    return {"state": _llm_state, "model_name": model_name}


def reset_llm():
    """重置 LLM 状态（用于配置更改后重新加载）"""
    global _llm, _llm_state
    _llm = None
    _llm_state = "unloaded"


# Qwen3 等模型会输出 <think>...</think> 思考标签
_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """移除 LLM 输出中的 <think>...</think> 深度思考标签（Qwen3 等模型）"""
    if not text or "<think>" not in text.lower():
        return text
    return _THINK_TAG_RE.sub("", text).strip()
