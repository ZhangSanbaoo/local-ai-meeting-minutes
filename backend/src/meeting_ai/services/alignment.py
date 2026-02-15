"""
对齐服务

把 ASR 转写结果和说话人分离结果对齐。
为每个文字片段分配说话人。

支持三种对齐策略（按优先级）：
1. 字级对齐：有 char_timestamps 时，逐字查 diarization → 精确切分
2. 中点匹配：短片段（<5s），用中点匹配（快速准确）
3. 句级分割：长片段（≥5s），按标点拆句再按时间分配说话人
"""

import re

from ..logger import get_logger
from ..models import CharTimestamp, DiarizationResult, Segment, TranscriptResult

logger = get_logger("services.alignment")

# 超过此时长的 ASR 片段，使用重叠分割而非中点匹配
_LONG_SEGMENT_THRESHOLD = 5.0


def align_transcript_with_speakers(
    transcript: TranscriptResult,
    diarization: DiarizationResult,
) -> list[Segment]:
    """
    把转写片段和说话人对齐

    策略优先级：
    1. 字级对齐（有 char_timestamps）→ 最精确
    2. 中点匹配（短片段）→ 快速准确
    3. 句级分割（长片段）→ 保证句子完整
    """
    has_char_ts = (
        transcript.char_timestamps is not None
        and len(transcript.char_timestamps) == len(transcript.segments)
    )

    logger.info(
        f"对齐: {len(transcript.segments)} 个转写片段, "
        f"{len(diarization.segments)} 个说话人片段, "
        f"字级时间戳={'有' if has_char_ts else '无'}"
    )

    # 构建说话人时间线
    speaker_timeline = [
        (seg.start, seg.end, seg.speaker)
        for seg in diarization.segments
        if seg.speaker is not None
    ]
    speaker_timeline.sort(key=lambda x: x[0])

    if not speaker_timeline:
        logger.warning("说话人时间线为空")
        return [
            Segment(id=i, start=s.start, end=s.end, text=s.text, speaker="UNKNOWN")
            for i, s in enumerate(transcript.segments)
        ]

    aligned_segments = []
    next_id = 0

    for seg_idx, seg in enumerate(transcript.segments):
        # 优先级 1：字级对齐
        char_ts = None
        if has_char_ts:
            char_ts = transcript.char_timestamps[seg_idx]

        if char_ts and len(char_ts) > 0:
            sub_segs = _align_by_char_timestamps(
                seg, char_ts, speaker_timeline, next_id,
            )
            aligned_segments.extend(sub_segs)
            next_id += len(sub_segs)
            continue

        # 优先级 2 & 3：中点匹配 / 句级分割
        duration = seg.end - seg.start
        if duration < _LONG_SEGMENT_THRESHOLD:
            mid_time = (seg.start + seg.end) / 2
            speaker = _find_speaker_at(speaker_timeline, mid_time)
            aligned_segments.append(Segment(
                id=next_id, start=seg.start, end=seg.end,
                text=seg.text, speaker=speaker,
            ))
            next_id += 1
        else:
            sub_segs = _split_by_speakers(seg, speaker_timeline, next_id)
            aligned_segments.extend(sub_segs)
            next_id += len(sub_segs)

    unknown_count = sum(1 for s in aligned_segments if s.speaker == "UNKNOWN")
    if unknown_count > 0:
        logger.warning(f"有 {unknown_count} 个片段无法匹配说话人")

    logger.info(f"对齐完成: {len(aligned_segments)} 个片段")
    return aligned_segments


# ---------------------------------------------------------------------------
# 字级对齐（最精确）
# ---------------------------------------------------------------------------

def _align_by_char_timestamps(
    seg: Segment,
    char_ts: list[CharTimestamp],
    timeline: list[tuple],
    id_start: int,
) -> list[Segment]:
    """
    用字级时间戳逐字匹配说话人，然后合并连续同说话人的字为一段。

    这是最精确的对齐方式：每个字独立查 diarization 时间线。
    """
    if not char_ts:
        speaker = _find_speaker_at(timeline, (seg.start + seg.end) / 2)
        return [Segment(
            id=id_start, start=seg.start, end=seg.end,
            text=seg.text, speaker=speaker,
        )]

    # 为每个字/词分配说话人
    char_speakers: list[tuple[str, float, float, str]] = []  # (char, start, end, speaker)
    for ct in char_ts:
        mid = (ct.start + ct.end) / 2
        speaker = _find_speaker_at(timeline, mid)
        char_speakers.append((ct.char, ct.start, ct.end, speaker))

    # 合并连续同说话人的字
    groups: list[tuple[str, float, float, str]] = []  # (text, start, end, speaker)
    for char_text, c_start, c_end, speaker in char_speakers:
        if groups and groups[-1][3] == speaker:
            prev_text, prev_start, _, prev_speaker = groups[-1]
            groups[-1] = (prev_text + char_text, prev_start, c_end, prev_speaker)
        else:
            groups.append((char_text, c_start, c_end, speaker))

    # 生成 Segment
    results = []
    for text, t_start, t_end, speaker in groups:
        text = text.strip()
        if text:
            results.append(Segment(
                id=id_start + len(results),
                start=round(t_start, 3),
                end=round(t_end, 3),
                text=text,
                speaker=speaker,
            ))

    if results:
        if len(results) > 1:
            logger.debug(
                f"字级对齐: '{seg.text[:20]}...' → {len(results)} 段 "
                f"({len(char_ts)} 字)"
            )
        return results

    # 回退
    speaker = _find_speaker_at(timeline, (seg.start + seg.end) / 2)
    return [Segment(
        id=id_start, start=seg.start, end=seg.end,
        text=seg.text, speaker=speaker,
    )]


def _find_speaker_at(timeline: list[tuple], time: float) -> str:
    """找到某个时间点对应的说话人"""
    for start, end, speaker in timeline:
        if start <= time <= end:
            return speaker
    # 找最近的说话人
    min_dist = float("inf")
    closest = "UNKNOWN"
    for start, end, speaker in timeline:
        dist = min(abs(time - start), abs(time - end))
        if dist < min_dist:
            min_dist = dist
            closest = speaker
    # 距离太远就放弃
    return closest if min_dist < 2.0 else "UNKNOWN"


def _split_text_to_sentences(text: str) -> list[str]:
    """
    按标点把文本拆成句子，保持标点跟着前一句

    优先按句末标点（。？！）拆分，
    若拆分结果太少（<3），则用逗号等进一步拆分。

    返回 [] 表示无标点可拆。
    """
    # 先按句末标点拆
    parts = re.split(r'(?<=[。？！.?!])', text)
    parts = [p for p in parts if p.strip()]

    if len(parts) >= 3:
        return parts

    # 句末标点太少，改用所有标点拆
    parts = re.split(r'(?<=[。？！，；、.?!,;])', text)
    parts = [p for p in parts if p.strip()]

    if len(parts) >= 2:
        return parts

    # 无可用标点
    return []


def _find_speaker_in_merged(
    merged: list[tuple],
    time: float,
) -> str:
    """在合并后的说话人区间列表中找某时刻的说话人"""
    for start, end, speaker in merged:
        if start <= time <= end:
            return speaker
    # 找最近的
    min_dist = float("inf")
    closest = merged[0][2]
    for start, end, speaker in merged:
        dist = min(abs(time - start), abs(time - end))
        if dist < min_dist:
            min_dist = dist
            closest = speaker
    return closest


def _split_by_char_ratio(
    text: str,
    seg: Segment,
    merged: list[tuple],
    id_start: int,
) -> list[Segment]:
    """按字符比例分割文本（无标点时的回退方案）"""
    total_dur = sum(e - s for s, e, _ in merged)
    if total_dur <= 0:
        return [Segment(
            id=id_start, start=seg.start, end=seg.end,
            text=text, speaker=merged[0][2],
        )]

    results = []
    char_pos = 0
    for i, (ov_start, ov_end, speaker) in enumerate(merged):
        ratio = (ov_end - ov_start) / total_dur
        if i < len(merged) - 1:
            char_count = round(len(text) * ratio)
            part_text = text[char_pos:char_pos + char_count].strip()
            char_pos += char_count
        else:
            part_text = text[char_pos:].strip()

        if part_text:
            results.append(Segment(
                id=id_start + len(results),
                start=ov_start,
                end=ov_end,
                text=part_text,
                speaker=speaker,
            ))

    return results if results else [Segment(
        id=id_start, start=seg.start, end=seg.end,
        text=text, speaker=merged[0][2],
    )]


def _split_by_speakers(
    seg: Segment,
    timeline: list[tuple],
    id_start: int,
) -> list[Segment]:
    """
    将一个长 ASR 片段按说话人边界分割

    策略：
    1. 先把文本按标点拆成句子
    2. 给每个句子按字符位置估算时间
    3. 用句子中点匹配说话人
    4. 合并连续同说话人的句子

    这样保证文本只在句子/从句边界处断开，不会在任意位置撕裂。
    若文本无标点，回退到字符比例分割。
    """
    # 找出所有重叠的说话人区间
    overlaps = []
    for d_start, d_end, speaker in timeline:
        ov_start = max(seg.start, d_start)
        ov_end = min(seg.end, d_end)
        if ov_end > ov_start:
            overlaps.append((ov_start, ov_end, speaker))

    if not overlaps:
        return [Segment(
            id=id_start, start=seg.start, end=seg.end,
            text=seg.text, speaker="UNKNOWN",
        )]

    # 合并连续同说话人的重叠区间
    merged = [overlaps[0]]
    for ov_start, ov_end, speaker in overlaps[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        if speaker == prev_speaker and ov_start - prev_end < 0.3:
            merged[-1] = (prev_start, ov_end, speaker)
        else:
            merged.append((ov_start, ov_end, speaker))

    # 只有一个说话人，不用分割
    if len(merged) == 1:
        return [Segment(
            id=id_start, start=seg.start, end=seg.end,
            text=seg.text, speaker=merged[0][2],
        )]

    text = seg.text or ""
    if not text:
        return [Segment(
            id=id_start, start=seg.start, end=seg.end,
            text=text, speaker=merged[0][2],
        )]

    # 尝试按标点拆句
    sentences = _split_text_to_sentences(text)

    if not sentences:
        # 无标点（如 FireRedASR），回退到字符比例分割
        logger.debug(f"无标点，回退字符比例分割 ({len(text)} 字, {len(merged)} 说话人区间)")
        return _split_by_char_ratio(text, seg, merged, id_start)

    # 按字符位置估算每个句子的时间，用中点匹配说话人
    total_chars = sum(len(s) for s in sentences)
    duration = seg.end - seg.start

    if total_chars <= 0 or duration <= 0:
        return [Segment(
            id=id_start, start=seg.start, end=seg.end,
            text=text, speaker=merged[0][2],
        )]

    assigned = []  # [(text, start_time, end_time, speaker)]
    char_offset = 0
    for sent in sentences:
        t_start = seg.start + (char_offset / total_chars) * duration
        t_end = seg.start + ((char_offset + len(sent)) / total_chars) * duration
        t_mid = (t_start + t_end) / 2
        speaker = _find_speaker_in_merged(merged, t_mid)
        assigned.append((sent, t_start, t_end, speaker))
        char_offset += len(sent)

    # 合并连续同说话人的句子
    groups = []
    for sent, t_start, t_end, speaker in assigned:
        if groups and groups[-1][3] == speaker:
            prev = groups[-1]
            groups[-1] = (prev[0] + sent, prev[1], t_end, speaker)
        else:
            groups.append((sent, t_start, t_end, speaker))

    # 生成 Segment
    results = []
    for text_part, t_start, t_end, speaker in groups:
        text_part = text_part.strip()
        if text_part:
            results.append(Segment(
                id=id_start + len(results),
                start=t_start,
                end=t_end,
                text=text_part,
                speaker=speaker,
            ))

    if results:
        logger.debug(
            f"句级分割: {len(sentences)} 句 → {len(results)} 段"
            f"（{len(merged)} 个说话人区间）"
        )
        return results

    return [Segment(
        id=id_start, start=seg.start, end=seg.end,
        text=text, speaker=merged[0][2],
    )]


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
    """
    if not segments:
        return []

    fixed = [seg.model_copy() for seg in segments]
    fixed.sort(key=lambda s: s.start)

    for i, seg in enumerate(fixed):
        if seg.speaker != "UNKNOWN":
            continue

        prev_speaker = None
        prev_gap = float('inf')
        for j in range(i - 1, -1, -1):
            if fixed[j].speaker != "UNKNOWN":
                prev_speaker = fixed[j].speaker
                prev_gap = seg.start - fixed[j].end
                break

        next_speaker = None
        next_gap = float('inf')
        for j in range(i + 1, len(fixed)):
            if fixed[j].speaker != "UNKNOWN":
                next_speaker = fixed[j].speaker
                next_gap = fixed[j].start - seg.end
                break

        new_speaker = None

        if prev_speaker == next_speaker and prev_speaker is not None:
            new_speaker = prev_speaker
        elif prev_speaker is not None and next_speaker is not None:
            if prev_gap <= next_gap and prev_gap <= max_gap:
                new_speaker = prev_speaker
            elif next_gap <= max_gap:
                new_speaker = next_speaker
        elif prev_speaker is not None and prev_gap <= max_gap:
            new_speaker = prev_speaker
        elif next_speaker is not None and next_gap <= max_gap:
            new_speaker = next_speaker

        if new_speaker is not None:
            logger.debug(f"修复 UNKNOWN: [{seg.start:.1f}-{seg.end:.1f}] -> {new_speaker}")
            fixed[i].speaker = new_speaker

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
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.start)
    merged = [sorted_segs[0].model_copy()]

    for seg in sorted_segs[1:]:
        last = merged[-1]
        same_speaker = last.speaker == seg.speaker
        small_gap = (seg.start - last.end) <= max_gap
        merged_duration = seg.end - last.start
        within_duration = merged_duration <= max_duration

        if same_speaker and small_gap and within_duration:
            last.end = seg.end
            if seg.text:
                last.text = f"{last.text} {seg.text}".strip()
        else:
            merged.append(seg.model_copy())

    # 重新编号
    for i, seg in enumerate(merged):
        seg.id = i

    logger.info(f"合并片段: {len(segments)} -> {len(merged)}")
    return merged


def split_long_segments(
    segments: list[Segment],
    max_duration: float = 30.0,
    split_on_pause: float = 1.0,
) -> list[Segment]:
    """
    分割过长的片段
    """
    if not segments:
        return []

    result = []
    next_id = 0

    for seg in segments:
        duration = seg.end - seg.start

        if duration <= max_duration:
            new_seg = seg.model_copy()
            new_seg.id = next_id
            result.append(new_seg)
            next_id += 1
            continue

        text = seg.text or ""
        if not text:
            new_seg = seg.model_copy()
            new_seg.id = next_id
            result.append(new_seg)
            next_id += 1
            continue

        num_parts = max(2, int(duration / max_duration) + 1)
        target_len = len(text) // num_parts

        split_points = []
        high_priority = ["。", "？", "！", ".", "?", "!"]
        mid_priority = ["，", "；", "、", ",", ";"]
        low_priority = [" "]

        for i, char in enumerate(text):
            if char in high_priority:
                split_points.append((i + 1, 3))
            elif char in mid_priority:
                split_points.append((i + 1, 2))
            elif char in low_priority:
                split_points.append((i + 1, 1))

        if split_points:
            selected_points = [0]
            last_point = 0

            for _ in range(num_parts - 1):
                target_pos = last_point + target_len
                best_point = None
                best_score = -1

                for pos, priority in split_points:
                    if pos <= last_point + 5:
                        continue
                    if pos >= len(text) - 5:
                        continue

                    distance = abs(pos - target_pos)
                    if distance > target_len * 0.5:
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
