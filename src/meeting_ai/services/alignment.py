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
    max_gap: float = 0.5,
) -> list[Segment]:
    """
    合并相邻的同一说话人片段
    
    如果两个片段：
    1. 是同一个说话人
    2. 间隔小于 max_gap 秒
    
    则合并为一个片段。
    
    Args:
        segments: 片段列表
        max_gap: 最大间隔（秒）
        
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
        
        if same_speaker and small_gap:
            # 合并：扩展时间范围，拼接文本
            last.end = seg.end
            if seg.text:
                last.text = f"{last.text} {seg.text}".strip()
        else:
            # 不合并：添加新片段
            merged.append(seg.model_copy())

    logger.info(f"合并片段: {len(segments)} -> {len(merged)}")

    return merged
