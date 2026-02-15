"""
全面分析所有 outputs，指出问题
"""
import sys
sys.path.insert(0, "src")

import json
from pathlib import Path
from collections import defaultdict

output_file = open("../outputs_analysis.txt", "w", encoding="utf-8")

def log(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        # Skip console output on encoding error
        pass
    output_file.write(msg + "\n")
    output_file.flush()

# 收集所有测试结果
outputs = Path("../outputs")
all_tests = sorted(outputs.iterdir(), key=lambda d: d.stat().st_mtime)

log("=" * 100)
log("全面分析所有 outputs 中的问题")
log("=" * 100)
log(f"\n共找到 {len(all_tests)} 个测试目录\n")

# 统计信息
stats = {
    "total": 0,
    "with_result": 0,
    "segment_counts": [],
    "speaker_misalignment": 0,
    "long_segments": 0,
    "mixed_speaker_segments": 0,
}

detailed_issues = []

for test_dir in all_tests:
    if not test_dir.is_dir():
        continue

    result_file = test_dir / "result.json"
    if not result_file.exists():
        continue

    stats["total"] += 1
    stats["with_result"] += 1

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    seg_count = len(data["segments"])
    stats["segment_counts"].append((test_dir.name, seg_count))

    # 分析问题
    issues = []

    # 1. 检查段落数异常
    if seg_count < 5:
        issues.append(f"⚠️ 段落数过少 ({seg_count}个) - 可能说话人合并过度")
    elif seg_count > 50:
        issues.append(f"⚠️ 段落数过多 ({seg_count}个) - 可能过度分割")

    # 2. 检查长段落（超过 30s）
    long_segs = []
    for seg in data["segments"]:
        duration = seg["end"] - seg["start"]
        if duration > 30:
            long_segs.append((seg["id"], duration, len(seg["text"])))

    if long_segs:
        stats["long_segments"] += len(long_segs)
        issues.append(f"⚠️ 有 {len(long_segs)} 个超长片段 (>30s)")

    # 3. 检查可能的说话人混合（启发式检测）
    mixed_segs = []
    for seg in data["segments"]:
        text = seg["text"]
        # 检测问答模式："...吗？" 后面紧跟回答
        if "吗？" in text or "吗?" in text:
            # 找到问号位置
            for q in ["吗？", "吗?"]:
                if q in text:
                    idx = text.index(q)
                    # 问号后还有超过 20 个字
                    if len(text) - idx - len(q) > 20:
                        mixed_segs.append({
                            "id": seg["id"],
                            "speaker": seg["speaker"],
                            "text_len": len(text),
                            "question": text[:idx+len(q)],
                            "answer": text[idx+len(q):idx+len(q)+30]
                        })
                        break

    if mixed_segs:
        stats["mixed_speaker_segments"] += len(mixed_segs)
        issues.append(f"🔴 检测到 {len(mixed_segs)} 个疑似问答混合片段")

    # 4. 检查说话人分配是否合理
    speaker_stats = defaultdict(lambda: {"count": 0, "duration": 0, "avg_duration": 0})
    for seg in data["segments"]:
        spk = seg["speaker"]
        duration = seg["end"] - seg["start"]
        speaker_stats[spk]["count"] += 1
        speaker_stats[spk]["duration"] += duration

    for spk, stats_dict in speaker_stats.items():
        stats_dict["avg_duration"] = stats_dict["duration"] / stats_dict["count"]

    # 检测异常：某个说话人只有 1-2 段但时长很长
    for spk, spk_stats in speaker_stats.items():
        if spk_stats["count"] <= 2 and spk_stats["duration"] > 30:
            issues.append(f"⚠️ {spk} 只有 {spk_stats['count']} 段但时长 {spk_stats['duration']:.1f}s")

    # 5. 检查说话人信息
    for spk_id, spk_info in data["speakers"].items():
        display_name = spk_info["display_name"]
        gender = spk_info["gender"]
        kind = spk_info["kind"]

        # 检查性别识别
        if display_name == "张教授" and gender != "male":
            issues.append(f"🔴 张教授性别识别错误: {gender} (应为 male)")
        if display_name == "主持人" and gender != "female":
            issues.append(f"⚠️ 主持人性别识别可能错误: {gender}")

    # 记录详细问题
    if issues:
        detailed_issues.append({
            "test": test_dir.name,
            "seg_count": seg_count,
            "issues": issues,
            "long_segs": long_segs,
            "mixed_segs": mixed_segs,
            "speaker_stats": dict(speaker_stats),
        })

# 输出汇总统计
log("\n" + "=" * 100)
log("汇总统计")
log("=" * 100)
log(f"\n总测试数: {stats['total']}")
log(f"有结果的: {stats['with_result']}")
log(f"检测到问题的: {len(detailed_issues)}")
log(f"说话人混合片段总数: {stats['mixed_speaker_segments']}")
log(f"超长片段总数: {stats['long_segments']}")

# 段落数分布
log("\n段落数分布:")
seg_dist = defaultdict(int)
for name, count in stats["segment_counts"]:
    seg_dist[count] += 1

for count in sorted(seg_dist.keys()):
    log(f"  {count:3d} 段: {seg_dist[count]} 个测试")

# 详细问题报告
log("\n" + "=" * 100)
log("详细问题报告（按严重性排序）")
log("=" * 100)

# 按问题严重性排序
def severity_score(issue_dict):
    score = 0
    for issue in issue_dict["issues"]:
        if "🔴" in issue:
            score += 10
        elif "⚠️" in issue:
            score += 1
    score += len(issue_dict["mixed_segs"]) * 5
    score += len(issue_dict["long_segs"]) * 2
    if issue_dict["seg_count"] < 5:
        score += 20  # 段落数过少是严重问题
    return score

detailed_issues.sort(key=severity_score, reverse=True)

for idx, issue_dict in enumerate(detailed_issues, 1):
    log(f"\n[{idx}] {issue_dict['test']} ({issue_dict['seg_count']} 段)")
    log("─" * 100)

    # 基本问题
    for issue in issue_dict["issues"]:
        log(f"  {issue}")

    # 说话人统计
    log(f"\n  说话人统计:")
    for spk, spk_stats in issue_dict["speaker_stats"].items():
        log(f"    {spk}: {spk_stats['count']} 段, {spk_stats['duration']:.1f}s 总时长, "
            f"{spk_stats['avg_duration']:.1f}s 平均")

    # 超长片段详情
    if issue_dict["long_segs"]:
        log(f"\n  超长片段详情:")
        for seg_id, duration, text_len in issue_dict["long_segs"]:
            log(f"    Segment {seg_id}: {duration:.1f}s, {text_len} 字")

    # 混合片段详情
    if issue_dict["mixed_segs"]:
        log(f"\n  疑似问答混合片段:")
        for mixed in issue_dict["mixed_segs"]:
            log(f"    Segment {mixed['id']} ({mixed['speaker']}, {mixed['text_len']}字):")
            log(f"      问: {mixed['question']}")
            log(f"      答: {mixed['answer']}...")

# 特别关注：汉语水平考试听力测试
log("\n" + "=" * 100)
log("特别分析：汉语水平考试听力测试（同一音频的多次测试）")
log("=" * 100)

hsk_tests = [item for item in stats["segment_counts"]
             if "汉语水平考试听力" in item[0]]

if hsk_tests:
    log(f"\n找到 {len(hsk_tests)} 个相同音频的测试结果:")
    log("\n测试名称 | 段落数 | 差异")
    log("─" * 100)

    counts = [count for _, count in hsk_tests]
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)

    for name, count in hsk_tests:
        diff = abs(count - avg_count)
        status = "✅ 正常" if diff < 10 else "⚠️ 偏离" if diff < 20 else "🔴 异常"
        log(f"{name} | {count:3d} | {status}")

    log(f"\n统计: 最少 {min_count} 段, 最多 {max_count} 段, 平均 {avg_count:.1f} 段")
    log(f"变异系数: {(max_count - min_count) / avg_count * 100:.1f}%")

    if max_count / min_count > 2:
        log("\n🔴 严重问题: 最大/最小比值 > 2，说明结果极不稳定！")

# 读取一个具体的坏例子
log("\n" + "=" * 100)
log("具体案例分析：最差结果")
log("=" * 100)

if detailed_issues:
    worst = detailed_issues[0]
    test_dir = outputs / worst["test"]
    result_file = test_dir / "result.json"

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    log(f"\n测试: {worst['test']}")
    log(f"段落数: {worst['seg_count']}")
    log(f"\n完整分段列表:")
    log("─" * 100)

    for seg in data["segments"]:
        duration = seg["end"] - seg["start"]
        text_preview = seg["text"][:80] + ("..." if len(seg["text"]) > 80 else "")
        log(f"[{seg['id']:2d}] {seg['start']:6.1f}-{seg['end']:6.1f} ({duration:5.1f}s) {seg['speaker']}")
        log(f"     {text_preview}")

output_file.close()
log("\n分析完成！结果已保存到 ../outputs_analysis.txt")
