"""
会议 AI 对话服务

基于会议转写内容与用户进行对话。
"""

import re
from collections.abc import Generator

from ..logger import get_logger
from .llm import get_llm_if_loaded

logger = get_logger("services.chat")

# Qwen3 <think> 标签正则
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def build_meeting_context(
    segments: list[dict],
    speakers: dict[str, dict],
    max_chars: int = 4000,
) -> str:
    """
    从 result.json 原始 dict 构建会议上下文文本

    Args:
        segments: 片段列表 (raw dict from result.json)
        speakers: 说话人信息 (raw dict from result.json)
        max_chars: 最大字符数

    Returns:
        格式化的对话文本
    """
    lines = []
    total_chars = 0

    for seg in segments:
        text = seg.get("text", "")
        if not text:
            continue

        speaker_id = seg.get("speaker", "UNKNOWN")
        speaker_info = speakers.get(speaker_id, {})
        display_name = speaker_info.get("display_name", speaker_id)

        line = f"{display_name}: {text}"

        if total_chars + len(line) > max_chars:
            lines.append("... (内容过长，已截断)")
            break

        lines.append(line)
        total_chars += len(line)

    return "\n".join(lines)


def build_chat_messages(
    user_message: str,
    history: list[dict],
    meeting_context: str,
    max_history: int = 10,
) -> list[dict]:
    """
    组装聊天消息列表 (system + history + user)

    Args:
        user_message: 用户当前消息
        history: 对话历史 [{role, content}, ...]
        meeting_context: 会议上下文文本
        max_history: 最多保留的历史轮数

    Returns:
        消息列表，可直接传给 LLM
    """
    system_prompt = (
        "你是一个会议助手 AI。基于以下会议转写内容回答用户的问题。\n"
        "请简洁、准确地回答，引用会议中的具体内容。\n"
        "如果问题与会议内容无关，礼貌地说明。\n\n"
        f"=== 会议内容 ===\n{meeting_context}\n=== 会议内容结束 ==="
    )

    messages = [{"role": "system", "content": system_prompt}]

    # 只保留最近的历史
    recent_history = history[-max_history * 2:] if history else []
    messages.extend(recent_history)

    messages.append({"role": "user", "content": user_message})

    return messages


def chat_stream(messages: list[dict]) -> Generator[str, None, None]:
    """
    流式聊天，逐 token yield

    处理 Qwen3 的 <think>...</think> 标签：
    - 遇到 <think> 开始跳过
    - 遇到 </think> 恢复输出

    Args:
        messages: 完整消息列表

    Yields:
        文本 token
    """
    llm = get_llm_if_loaded()
    if llm is None:
        yield "[错误] LLM 未加载，请先在 AI 对话面板中选择并加载模型"
        return

    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            stream=True,
        )

        in_think = False
        buffer = ""

        for chunk in response:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if not token:
                continue

            buffer += token

            # 检查是否进入/退出 think 标签
            while buffer:
                if in_think:
                    # 在 think 块内，查找 </think>
                    match = _THINK_CLOSE.search(buffer)
                    if match:
                        # 跳过 think 内容和闭合标签
                        buffer = buffer[match.end():]
                        in_think = False
                    else:
                        # 还没找到闭合标签，继续等待
                        # 但只保留最后 20 字符以防内存泄漏
                        if len(buffer) > 20:
                            buffer = buffer[-20:]
                        break
                else:
                    # 不在 think 块内，查找 <think>
                    match = _THINK_OPEN.search(buffer)
                    if match:
                        # 输出 <think> 之前的内容
                        before = buffer[:match.start()]
                        if before:
                            yield before
                        buffer = buffer[match.end():]
                        in_think = True
                    else:
                        # 没有 think 标签，但可能是部分匹配 "<thi..."
                        # 安全输出到最后一个 '<' 之前
                        last_lt = buffer.rfind("<")
                        if last_lt >= 0 and last_lt > len(buffer) - 10:
                            # 可能是不完整的标签，保留
                            to_yield = buffer[:last_lt]
                            buffer = buffer[last_lt:]
                            if to_yield:
                                yield to_yield
                            break
                        else:
                            yield buffer
                            buffer = ""
                            break

        # 输出剩余 buffer
        if buffer and not in_think:
            yield buffer

    except Exception as e:
        logger.error(f"聊天流式生成失败: {e}")
        yield f"\n[错误] 生成失败: {e}"
