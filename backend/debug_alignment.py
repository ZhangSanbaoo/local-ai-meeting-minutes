"""
Debug char-level alignment issues
"""
import sys
sys.path.insert(0, "src")

import json
from pathlib import Path

# Output to file to avoid encoding issues
output_file = open("../alignment_debug.txt", "w", encoding="utf-8")

def log(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        pass
    output_file.write(msg + "\n")
    output_file.flush()

# 检查最近的几个测试结果
outputs = Path("../outputs")
test_dirs = sorted(outputs.glob("汉语水平考试听力_trimmed_2026021*"))[-5:]

log("=" * 80)
log("分析最近 5 次测试的对齐结果")
log("=" * 80)

for test_dir in test_dirs:
    result_file = test_dir / "result.json"
    if not result_file.exists():
        continue

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    seg_count = len(data["segments"])
    spk0_count = sum(1 for s in data["segments"] if s["speaker"] == "SPEAKER_00")
    spk1_count = sum(1 for s in data["segments"] if s["speaker"] == "SPEAKER_01")

    log(f"\n{test_dir.name}:")
    log(f"  总片段: {seg_count}")
    log(f"  SPEAKER_00: {spk0_count} 段")
    log(f"  SPEAKER_01: {spk1_count} 段")

    # 检查是否有多说话人混合的情况
    for i, seg in enumerate(data["segments"]):
        text = seg["text"]
        # 检查是否包含明显的对话标记
        if "请问" in text and "青少年" in text and len(text) > 100:
            log(f"  [WARN] Segment {i} ({seg['speaker']}): 包含完整对话（{len(text)}字）")
            log(f"      前50字: {text[:50]}")
        if "吗？" in text[:30] and any(x in text for x in ["这是", "可能是"]):
            log(f"  [WARN] Segment {i} ({seg['speaker']}): 疑似问答混合")
            log(f"      前80字: {text[:80]}")

# 找到最差和最好的结果，详细分析
log("\n" + "=" * 80)
log("详细分析：最差结果 vs 最好结果")
log("=" * 80)

seg_counts = []
for test_dir in test_dirs:
    result_file = test_dir / "result.json"
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        seg_counts.append((len(data["segments"]), test_dir, data))

seg_counts.sort(key=lambda x: x[0])
worst = seg_counts[0]
best = seg_counts[-1]

log(f"\n最差结果: {worst[1].name} ({worst[0]} 个片段)")
for i, seg in enumerate(worst[2]["segments"]):
    duration = seg["end"] - seg["start"]
    log(f"  [{i}] {seg['start']:.1f}-{seg['end']:.1f} ({duration:.1f}s) {seg['speaker']}")
    log(f"      {seg['text'][:80]}...")

log(f"\n最好结果: {best[1].name} ({best[0]} 个片段)")
for i, seg in enumerate(best[2]["segments"]):
    duration = seg["end"] - seg["start"]
    log(f"  [{i}] {seg['start']:.1f}-{seg['end']:.1f} ({duration:.1f}s) {seg['speaker']}")
    log(f"      {seg['text'][:80]}...")

output_file.close()
log("\nResults saved to ../alignment_debug.txt")
