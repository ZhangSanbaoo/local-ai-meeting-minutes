"""
对齐服务

把 ASR 转写结果和说话人分离结果对齐。
为每个文字片段分配说话人。
"""

from ..logger import get_logger
from ..models import DiarizationResult, Segment, TranscriptResult

logger = get_logger("services.alignment")


def align_transcript_with_speakers(
    transcript: TranscriptResult,
    diarization: DiarizationResult,
) -> list[Segment]:
    """
    把转写片段和说话人对齐
    
    原理：
    1. 对于每个 ASR 片段，计算其中点时间
    2. 找到该中点落在哪个说话人的时间范围内
    3. 如果没有匹配，标记为 "UNKNOWN"
    
    Args:
        transcript: ASR 转写结果
        diarization: 说话人分离结果
        
    Returns:
        带有说话人标记的片段列表
    """
    logger.info(
        f"对齐: {len(transcript.segments)} 个转写片段, "
        f"{len(diarization.segments)} 个说话人片段"
    )

    # 构建说话人时间线：[(start, end, speaker), ...]
    speaker_timeline = [
        (seg.start, seg.end, seg.speaker)
        for seg in diarization.segments
        if seg.speaker is not None
    ]
    # 按开始时间排序
    speaker_timeline.sort(key=lambda x: x[0])

    def find_speaker_at(time: float) -> str:
        """找到某个时间点对应的说话人"""
        for start, end, speaker in speaker_timeline:
            if start <= time <= end:
                return speaker
        return "UNKNOWN"

    # 对齐每个 ASR 片段
    aligned_segments = []
    for seg in transcript.segments:
        # 计算片段中点
        mid_time = (seg.start + seg.end) / 2
        
        # 找到对应的说话人
        speaker = find_speaker_at(mid_time)
        
        # 创建带说话人的新片段
        aligned_segment = Segment(
            id=seg.id,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker=speaker,
        )
        aligned_segments.append(aligned_segment)

    # 统计结果
    unknown_count = sum(1 for s in aligned_segments if s.speaker == "UNKNOWN")
    if unknown_count > 0:
        logger.warning(f"有 {unknown_count} 个片段无法匹配说话人")

    logger.info(f"对齐完成: {len(aligned_segments)} 个片段")

    return aligned_segments


def fix_unknown_speakers(
    segments: list[Segment],
    max_gap: float = 2.0,
) -> list[Segment]:
    """
    修复 UNKNOWN 说话人
    
    策略：
    1. 如果前后是同一个说话人 → 用那个说话人
    2. 如果前后不同 → 用时间更近的那个
    3. 如果是开头/结尾的 UNKNOWN → 用最近的已知说话人
    
    Args:
        segments: 片段列表
        max_gap: 最大时间间隔（秒），超过此间隔不修复
        
    Returns:
        修复后的片段列表
    """
    if not segments:
        return []

    # 复制一份，避免修改原始数据
    fixed = [seg.model_copy() for seg in segments]
    
    # 按时间排序
    fixed.sort(key=lambda s: s.start)
    
    for i, seg in enumerate(fixed):
        if seg.speaker != "UNKNOWN":
            continue
        
        # 找前一个非 UNKNOWN 片段
        prev_speaker = None
        prev_gap = float('inf')
        for j in range(i - 1, -1, -1):
            if fixed[j].speaker != "UNKNOWN":
                prev_speaker = fixed[j].speaker
                prev_gap = seg.start - fixed[j].end
                break
        
        # 找后一个非 UNKNOWN 片段
        next_speaker = None
        next_gap = float('inf')
        for j in range(i + 1, len(fixed)):
            if fixed[j].speaker != "UNKNOWN":
                next_speaker = fixed[j].speaker
                next_gap = fixed[j].start - seg.end
                break
        
        # 决定用哪个说话人
        new_speaker = None
        
        if prev_speaker == next_speaker and prev_speaker is not None:
            # 前后相同 → 直接用
            new_speaker = prev_speaker
        elif prev_speaker is not None and next_speaker is not None:
            # 前后不同 → 用时间更近的
            if prev_gap <= next_gap and prev_gap <= max_gap:
                new_speaker = prev_speaker
            elif next_gap <= max_gap:
                new_speaker = next_speaker
        elif prev_speaker is not None and prev_gap <= max_gap:
            # 只有前面有
            new_speaker = prev_speaker
        elif next_speaker is not None and next_gap <= max_gap:
            # 只有后面有
            new_speaker = next_speaker
        
        if new_speaker is not None:
            logger.debug(f"修复 UNKNOWN: [{seg.start:.1f}-{seg.end:.1f}] -> {new_speaker}")
            fixed[i].speaker = new_speaker
    
    # 统计修复结果
    remaining_unknown = sum(1 for s in fixed if s.speaker == "UNKNOWN")
    original_unknown = sum(1 for s in segments if s.speaker == "UNKNOWN")
    if original_unknown > 0:
        fixed_count = original_unknown - remaining_unknown
        logger.info(f"修复 UNKNOWN: {fixed_count}/{original_unknown} 个片段")
    
    return fixed


def merge_adjacent_segments(
    segments: list[Segment],
    max_gap: float = 0.3,
    max_duration: float = 60.0,
) -> list[Segment]:
    """
    合并相邻的同一说话人片段

    如果两个片段：
    1. 是同一个说话人
    2. 间隔小于 max_gap 秒
    3. 合并后时长不超过 max_duration 秒

    则合并为一个片段。

    Args:
        segments: 片段列表
        max_gap: 最大间隔（秒）
        max_duration: 最大片段时长（秒），超过此时长会强制分段

    Returns:
        合并后的片段列表
    """
    if not segments:
        return []

    # 按时间排序
    sorted_segs = sorted(segments, key=lambda s: s.start)

    merged = [sorted_segs[0].model_copy()]

    for seg in sorted_segs[1:]:
        last = merged[-1]

        # 检查是否可以合并
        same_speaker = last.speaker == seg.speaker
        small_gap = (seg.start - last.end) <= max_gap
        # 检查合并后时长是否超过限制
        merged_duration = seg.end - last.start
        within_duration = merged_duration <= max_duration

        if same_speaker and small_gap and within_duration:
            # 合并：扩展时间范围，拼接文本
            last.end = seg.end
            if seg.text:
                last.text = f"{last.text} {seg.text}".strip()
        else:
            # 不合并：添加新片段
            merged.append(seg.model_copy())

    logger.info(f"合并片段: {len(segments)} -> {len(merged)}")

    return merged


def split_long_segments(
    segments: list[Segment],
    max_duration: float = 30.0,
    split_on_pause: float = 1.0,
) -> list[Segment]:
    """
    分割过长的片段

    如果片段时长超过 max_duration，尝试在合适的位置分割：
    1. 优先在句号、问号、感叹号处分割
    2. 其次在逗号处分割
    3. 最后在空格处分割

    Args:
        segments: 片段列表
        max_duration: 最大片段时长（秒）
        split_on_pause: 如果有时间间隔信息，超过此值（秒）分割

    Returns:
        分割后的片段列表
    """
    if not segments:
        return []

    result = []
    next_id = 0

    for seg in segments:
        duration = seg.end - seg.start

        # 如果时长在限制内，直接添加
        if duration <= max_duration:
            new_seg = seg.model_copy()
            new_seg.id = next_id
            result.append(new_seg)
            next_id += 1
            continue

        # 需要分割
        text = seg.text or ""
        if not text:
            new_seg = seg.model_copy()
            new_seg.id = next_id
            result.append(new_seg)
            next_id += 1
            continue

        # 计算需要分成多少段
        num_parts = max(2, int(duration / max_duration) + 1)
        target_len = len(text) // num_parts

        # 尝试在标点处分割
        split_points = []

        # 优先级：句号/问号/感叹号 > 逗号/分号 > 空格
        high_priority = ["。", "？", "！", ".", "?", "!"]
        mid_priority = ["，", "；", "、", ",", ";"]
        low_priority = [" "]

        # 查找所有可能的分割点
        for i, char in enumerate(text):
            if char in high_priority:
                split_points.append((i + 1, 3))  # 位置，优先级
            elif char in mid_priority:
                split_points.append((i + 1, 2))
            elif char in low_priority:
                split_points.append((i + 1, 1))

        # 选择最佳分割点
        if split_points:
            # 按目标位置和优先级选择
            selected_points = [0]  # 开始位置
            last_point = 0

            for _ in range(num_parts - 1):
                target_pos = last_point + target_len

                # 找到目标位置附近优先级最高的分割点
                best_point = None
                best_score = -1

                for pos, priority in split_points:
                    if pos <= last_point + 5:  # 太近了跳过
                        continue
                    if pos >= len(text) - 5:  # 太远了跳过
                        continue

                    # 计算得分：优先级高 + 距离目标位置近
                    distance = abs(pos - target_pos)
                    if distance > target_len * 0.5:  # 距离太远，降低优先级
                        score = priority - distance / target_len
                    else:
                        score = priority + (1 - distance / target_len)

                    if score > best_score:
                        best_score = score
                        best_point = pos

                if best_point:
                    selected_points.append(best_point)
                    last_point = best_point

            selected_points.append(len(text))

            # 根据分割点创建新片段
            time_per_char = duration / len(text) if text else 0

            for i in range(len(selected_points) - 1):
                start_char = selected_points[i]
                end_char = selected_points[i + 1]
                part_text = text[start_char:end_char].strip()

                if not part_text:
                    continue

                part_start = seg.start + start_char * time_per_char
                part_end = seg.start + end_char * time_per_char

                new_seg = Segment(
                    id=next_id,
                    start=part_start,
                    end=part_end,
                    text=part_text,
                    speaker=seg.speaker,
                )
                result.append(new_seg)
                next_id += 1
        else:
            # 没有找到分割点，强制按长度分割
            time_per_char = duration / len(text) if text else 0
            for i in range(num_parts):
                start_char = i * target_len
                end_char = (i + 1) * target_len if i < num_parts - 1 else len(text)
                part_text = text[start_char:end_char].strip()

                if not part_text:
                    continue

                part_start = seg.start + start_char * time_per_char
                part_end = seg.start + end_char * time_per_char

                new_seg = Segment(
                    id=next_id,
                    start=part_start,
                    end=part_end,
                    text=part_text,
                    speaker=seg.speaker,
                )
                result.append(new_seg)
                next_id += 1

    if len(result) != len(segments):
        logger.info(f"分割长片段: {len(segments)} -> {len(result)}")

    return result
