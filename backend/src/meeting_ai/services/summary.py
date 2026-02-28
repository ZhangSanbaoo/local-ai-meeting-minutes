"""
会议总结服务

使用 LLM 对会议内容进行智能总结。
"""

from ..config import get_settings
from ..logger import get_logger
from ..models import MeetingSummary, Segment, SpeakerInfo
from .llm import get_llm, strip_think_tags

logger = get_logger("services.summary")


def _build_dialog_text(
    segments: list[Segment],
    speakers: dict[str, SpeakerInfo],
    max_chars: int = 6000,
) -> str:
    """
    构建对话文本
    
    Args:
        segments: 片段列表
        speakers: 说话人信息
        max_chars: 最大字符数（防止超出 LLM 上下文）
        
    Returns:
        格式化的对话文本
    """
    lines = []
    total_chars = 0
    
    for seg in segments:
        if not seg.text:
            continue
        
        speaker_id = seg.speaker or "UNKNOWN"
        speaker_info = speakers.get(speaker_id)
        display_name = speaker_info.display_name if speaker_info else speaker_id
        
        line = f"{display_name}: {seg.text}"
        
        if total_chars + len(line) > max_chars:
            lines.append("... (内容过长，已截断)")
            break
        
        lines.append(line)
        total_chars += len(line)
    
    return "\n".join(lines)


def _parse_summary_response(response: str) -> dict:
    """
    解析 LLM 的总结响应
    
    尝试从响应中提取结构化信息。
    """
    result = {
        "title": "",
        "summary": "",
        "key_points": [],
        "action_items": [],
        "decisions": [],
        "follow_ups": [],
        "key_data": [],
    }
    
    lines = response.strip().split("\n")
    current_section = None
    current_content = []
    
    def save_current_section():
        """保存当前章节的内容"""
        if not current_section or not current_content:
            return
        
        content = "\n".join(current_content).strip()
        if not content or content == "无":
            return
        
        if current_section == "title":
            result["title"] = content.lstrip("- •·").strip()
        elif current_section == "summary":
            result["summary"] = content
        elif current_section in ["key_points", "details"]:
            # 解析列表项
            for line in current_content:
                line = line.strip()
                if line and line != "无":
                    point = line.lstrip("- •·0123456789.").strip()
                    if point and len(point) > 2:
                        result["key_points"].append(point)
        elif current_section == "action_items":
            for line in current_content:
                line = line.strip()
                if line and line != "无" and "无明确" not in line:
                    item = line.lstrip("- •·0123456789.").strip()
                    if item and len(item) > 2:
                        result["action_items"].append(item)
        elif current_section == "decisions":
            for line in current_content:
                line = line.strip()
                if line and line != "无":
                    decision = line.lstrip("- •·0123456789.").strip()
                    if decision and len(decision) > 2:
                        result["decisions"].append(decision)
        elif current_section == "follow_ups":
            for line in current_content:
                line = line.strip()
                if line and line != "无":
                    item = line.lstrip("- •·0123456789.").strip()
                    if item and len(item) > 2:
                        result["follow_ups"].append(item)
        elif current_section == "key_data":
            for line in current_content:
                line = line.strip()
                if line and line != "无":
                    data = line.lstrip("- •·0123456789.").strip()
                    if data and len(data) > 2:
                        result["key_data"].append(data)
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # 检测章节标题 — 只有以 # 或 ** 开头的行才可能是标题
        # 避免内容行中包含关键词被误判为新章节
        is_header_line = (
            line_stripped.startswith("#")
            or line_stripped.startswith("**")
            or line_stripped.startswith("##")
        )
        lower_line = line_stripped.lower().replace(" ", "").replace("：", ":").replace("*", "").replace("#", "")
        
        new_section = None
        if "主题" in lower_line or "title" in lower_line:
            new_section = "title"
        elif "摘要" in lower_line or "概述" in lower_line:
            new_section = "summary"
        elif "详细内容" in lower_line or "详细" in lower_line:
            new_section = "details"
        elif "任务分配" in lower_line or "任务" in lower_line or "action" in lower_line:
            new_section = "action_items"
        elif "决议" in lower_line or "决定" in lower_line or "结论" in lower_line or "decision" in lower_line:
            new_section = "decisions"
        elif "待跟进" in lower_line or "跟进" in lower_line or "follow" in lower_line:
            new_section = "follow_ups"
        elif "关键数据" in lower_line or "数据" in lower_line or "数字" in lower_line:
            new_section = "key_data"
        elif "要点" in lower_line or "keypoint" in lower_line or "主要内容" in lower_line:
            new_section = "key_points"
        
        if new_section and is_header_line:
            save_current_section()
            current_section = new_section
            current_content = []
            # 尝试从同一行提取内容（如 "主题: xxx"）
            if ":" in line_stripped:
                after_colon = line_stripped.split(":", 1)[1].strip()
                if after_colon and after_colon != "无":
                    current_content.append(after_colon)
        else:
            current_content.append(line_stripped)
    
    # 保存最后一个章节
    save_current_section()
    
    # 合并 follow_ups 到 action_items
    if result["follow_ups"]:
        result["action_items"].extend(result["follow_ups"])
    
    # 合并 key_data 到 key_points（如果有的话）
    if result["key_data"]:
        result["key_points"].extend([f"📊 {d}" for d in result["key_data"]])

    # 去重（保持顺序）
    def _deduplicate(items: list[str]) -> list[str]:
        seen = set()
        unique = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    for key in ["key_points", "action_items", "decisions", "follow_ups", "key_data"]:
        result[key] = _deduplicate(result[key])

    return result


def summarize_meeting(
    segments: list[Segment],
    speakers: dict[str, SpeakerInfo],
    duration: float = 0.0,
) -> MeetingSummary | None:
    """
    生成会议总结
    
    Args:
        segments: 对话片段列表
        speakers: 说话人信息字典
        duration: 会议时长（秒）
        
    Returns:
        MeetingSummary 对象，失败返回 None
    """
    llm = get_llm()
    if llm is None:
        logger.warning("LLM 不可用，无法生成会议总结")
        return None
    
    # 构建对话文本
    dialog_text = _build_dialog_text(segments, speakers)
    
    if not dialog_text or len(dialog_text) < 50:
        logger.warning("对话内容太短，无法生成总结")
        return None
    
    # 构建参与者列表
    participants = [info.display_name for info in speakers.values()]
    participants_str = "、".join(participants) if participants else "未知"
    
    # 格式化时长
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    duration_str = f"{minutes}分{seconds}秒" if minutes > 0 else f"{seconds}秒"
    
    # 优化后的 Prompt - 更详细的提取
    prompt = f"""请仔细分析以下会议/对话内容，生成详细的会议纪要。

基本信息：
- 时长：{duration_str}
- 参与者：{participants_str}

对话内容：
{dialog_text}

请按以下格式输出详细的会议纪要：

## 主题
（一句话概括会议主题）

## 摘要
（3-5句话概括会议的主要内容和结论）

## 详细内容
（按讨论顺序，列出每个议题的详细内容，包括：
- 讨论了什么问题
- 得出了什么结论
- 提到的具体数据、金额、比例等）

## 任务分配
（如果会议中有分配任务，请按以下格式列出每条任务：
- 【任务】具体任务内容 | 【负责人】姓名 | 【截止】日期或时限
例如：
- 【任务】准备下周的报告 | 【负责人】张三 | 【截止】下周一
- 【任务】联系供应商确认价格 | 【负责人】李四 | 【截止】本周五
如果没有明确的任务分配，写"无明确任务分配"）

## 决议事项
（列出会议中确定的决定、结论、共识）

## 待跟进事项
（需要后续跟进或确认的事项）

## 关键数据
（会议中提到的重要数字、日期、金额、百分比等）

请确保：
1. 不遗漏任何重要信息
2. 保留具体的人名、日期、数字
3. 如果某个部分没有相关内容，写"无"
4. 使用清晰的列表格式

请直接输出会议纪要："""

    logger.info("正在生成会议总结...")
    
    try:
        settings = get_settings()
        response = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # 降低温度，更准确
            max_tokens=2048,  # 增加 token 数，允许更详细的输出
        )
        
        content = strip_think_tags(response["choices"][0]["message"]["content"]).strip()
        logger.debug(f"LLM 响应:\n{content}")
        
        # 解析响应
        parsed = _parse_summary_response(content)
        
        # 如果解析失败，使用原始响应作为摘要
        if not parsed["title"] and not parsed["key_points"]:
            logger.warning("无法解析结构化总结，使用原始响应")
            parsed["summary"] = content
        
        summary = MeetingSummary(
            title=parsed.get("title", "会议记录"),
            summary=parsed.get("summary", ""),
            key_points=parsed.get("key_points", []),
            action_items=parsed.get("action_items", []),
            decisions=parsed.get("decisions", []),
        )
        
        logger.info(f"会议总结生成完成: {summary.title}")
        return summary
        
    except Exception as e:
        logger.error(f"生成会议总结失败: {e}")
        return None


def format_summary_markdown(
    summary: MeetingSummary,
    speakers: dict[str, SpeakerInfo],
    duration: float = 0.0,
) -> str:
    """
    将 MeetingSummary 格式化为 Markdown
    
    Args:
        summary: 会议总结对象
        speakers: 说话人信息
        duration: 会议时长（秒）
        
    Returns:
        Markdown 格式的字符串
    """
    lines = []
    
    # 标题
    title = summary.title or "会议纪要"
    lines.append(f"# {title}")
    lines.append("")
    
    # 基本信息
    lines.append("## 基本信息")
    lines.append("")
    
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    if minutes > 0:
        duration_str = f"{minutes}分{seconds}秒"
    else:
        duration_str = f"{seconds}秒"
    lines.append(f"- **时长**：{duration_str}")
    
    participants = [info.display_name for info in speakers.values()]
    if participants:
        lines.append(f"- **参与者**：{', '.join(participants)}")
    
    lines.append(f"- **说话人数**：{len(speakers)}人")
    lines.append("")
    
    # 摘要
    if summary.summary:
        lines.append("## 摘要")
        lines.append("")
        lines.append(summary.summary)
        lines.append("")
    
    # 主要内容/要点
    if summary.key_points:
        lines.append("## 主要内容")
        lines.append("")
        for i, point in enumerate(summary.key_points, 1):
            # 检查是否是数据项（带📊标记的）
            if point.startswith("📊"):
                lines.append(f"- {point}")
            else:
                lines.append(f"{i}. {point}")
        lines.append("")
    
    # 决议事项
    if summary.decisions:
        lines.append("## 决议事项")
        lines.append("")
        for decision in summary.decisions:
            lines.append(f"- ✅ {decision}")
        lines.append("")
    
    # 任务分配/待办事项
    if summary.action_items:
        lines.append("## 任务与待办")
        lines.append("")
        for item in summary.action_items:
            # 检查是否是结构化格式（包含【任务】【负责人】【截止】）
            if "【任务】" in item or "【负责人】" in item:
                lines.append(f"- [ ] {item}")
            else:
                lines.append(f"- [ ] {item}")
        lines.append("")
    
    # 如果没有任何内容，显示提示
    if not summary.summary and not summary.key_points and not summary.decisions:
        lines.append("*会议内容较短，无法生成详细总结。*")
        lines.append("")
    
    # 分隔线和生成时间
    lines.append("---")
    from datetime import datetime
    lines.append(f"*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    return "\n".join(lines)
