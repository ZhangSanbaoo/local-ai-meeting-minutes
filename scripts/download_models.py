from huggingface_hub import snapshot_download
import os
TOKEN = os.environ.get("HF_TOKEN")  # 从环境变量读取，不硬编码
if not TOKEN:
    print("请先设置 HF_TOKEN 环境变量")
    exit(1)

models = [
    ("pyannote/speaker-diarization-3.1", "./models/pyannote/speaker-diarization-3.1"),
    ("pyannote/segmentation-3.0", "./models/pyannote/segmentation-3.0"),
    ("pyannote/wespeaker-voxceleb-resnet34-LM", "./models/pyannote/wespeaker-voxceleb-resnet34-LM"),
]

for repo_id, local_dir in models:
    print(f"下载 {repo_id}...")
    snapshot_download(repo_id, local_dir=local_dir, token=TOKEN)
    print(f"完成: {local_dir}")