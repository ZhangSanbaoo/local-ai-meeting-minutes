"""
文本校正服务

使用本地 LLM 修复 Whisper 转写中的错别字。
"""

from ..config import get_settings
from ..logger import get_logger
from ..models import Segment

logger = get_logger("services.correction")

# 全局 LLM 缓存
_llm = None


def _get_llm():
    """懒加载 LLM 模型"""
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        
        settings = get_settings()
        llm_settings = settings.llm
        
        if not llm_settings.enabled:
            logger.info("LLM 未启用，跳过文本校正")
            _llm = "disabled"
            return None
        
        model_path = llm_settings.model_path
        if model_path is None:
            logger.warning("LLM 模型路径未配置")
            _llm = "disabled"
            return None
        
        if not model_path.is_absolute():
            model_path = settings.paths.root_dir / model_path
        
        if not model_path.exists():
            logger.warning(f"LLM 模型不存在: {model_path}")
            _llm = "disabled"
            return None
        
        logger.info(f"加载 LLM 模型用于文本校正: {model_path}")
        
        _llm = Llama(
            model_path=str(model_path),
            n_ctx=llm_settings.n_ctx,
            n_threads=llm_settings.n_threads,
            n_gpu_layers=llm_settings.n_gpu_layers,
            verbose=False,
        )
        logger.info("LLM 模型加载完成")
    
    return _llm if _llm != "disabled" else None


def correct_text(text: str) -> str:
    """
    使用 LLM 修复文本中的错别字
    
    Args:
        text: 原始文本
        
    Returns:
        修复后的文本
    """
    llm = _get_llm()
    if llm is None:
        return text
    
    if not text or len(text.strip()) < 5:
        return text
    
    settings = get_settings()
    
    prompt = f"""请修复以下中文文本中的错别字和语法错误。只修复明显的错误，不要改变原意或添加内容。
如果文本没有错误，直接返回原文。只输出修复后的文本，不要任何解释。

原文：{text}

修复后："""

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=len(text) * 2,  # 足够容纳修复后的文本
        )
        
        corrected = response["choices"][0]["message"]["content"].strip()
        
        # 基本验证：修复后的文本长度不应该差太多
        if corrected and 0.5 < len(corrected) / len(text) < 2.0:
            # 去除可能的引号
            if corrected.startswith('"') and corrected.endswith('"'):
                corrected = corrected[1:-1]
            if corrected.startswith('"') and corrected.endswith('"'):
                corrected = corrected[1:-1]
            return corrected
        else:
            logger.debug(f"校正结果异常，使用原文: {len(text)} -> {len(corrected)}")
            return text
            
    except Exception as e:
        logger.warning(f"文本校正失败: {e}")
        return text


def correct_segments(segments: list[Segment], batch_size: int = 5) -> list[Segment]:
    """
    批量校正片段中的文本
    
    Args:
        segments: 片段列表
        batch_size: 每批处理的片段数（用于合并短文本，提高效率）
        
    Returns:
        校正后的片段列表
    """
    llm = _get_llm()
    if llm is None:
        return segments
    
    corrected_segments = []
    
    for seg in segments:
        if seg.text and len(seg.text.strip()) >= 5:
            corrected_text = correct_text(seg.text)
            if corrected_text != seg.text:
                logger.debug(f"校正: '{seg.text[:30]}...' -> '{corrected_text[:30]}...'")
            new_seg = seg.model_copy()
            new_seg.text = corrected_text
            corrected_segments.append(new_seg)
        else:
            corrected_segments.append(seg)
    
    return corrected_segments


def correct_transcript_batch(texts: list[str]) -> list[str]:
    """
    批量校正多段文本（更高效）
    
    把多段文本合并成一个请求，减少 LLM 调用次数。
    
    Args:
        texts: 文本列表
        
    Returns:
        校正后的文本列表
    """
    llm = _get_llm()
    if llm is None:
        return texts
    
    if not texts:
        return texts
    
    # 过滤空文本
    valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) >= 5]
    if not valid_indices:
        return texts
    
    # 构建批量请求
    numbered_texts = []
    for idx, i in enumerate(valid_indices):
        numbered_texts.append(f"{idx + 1}. {texts[i]}")
    
    combined = "\n".join(numbered_texts)
    
    prompt = f"""请修复以下中文文本中的错别字和语法错误。每行是一段独立的文本，请保持编号格式。
只修复明显的错误，不要改变原意或添加内容。如果某行没有错误，保持原样。

{combined}

请输出修复后的文本（保持相同的编号格式）："""

    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=len(combined) * 2,
        )
        
        result = response["choices"][0]["message"]["content"].strip()
        
        # 解析结果
        corrected_texts = texts.copy()
        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 尝试解析 "1. xxx" 格式
            if ". " in line:
                try:
                    num_str, text = line.split(". ", 1)
                    num = int(num_str) - 1
                    if 0 <= num < len(valid_indices):
                        original_idx = valid_indices[num]
                        corrected_texts[original_idx] = text
                except (ValueError, IndexError):
                    continue
        
        return corrected_texts
        
    except Exception as e:
        logger.warning(f"批量文本校正失败: {e}")
        return texts
