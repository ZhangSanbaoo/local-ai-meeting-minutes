"""
下载正确的 3D-Speaker Diarization 模型

之前下载的是 speaker-verification 模型，现在下载真正的 diarization 模型
"""

import os
from pathlib import Path
from modelscope import snapshot_download

# 设置镜像（如果需要）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# 项目根目录
root = Path(__file__).parent.parent.parent
models_dir = root / "models" / "diarization"
models_dir.mkdir(parents=True, exist_ok=True)

print("正在下载 3D-Speaker Diarization 模型...")
print("模型: damo/speech_campplus_speaker-diarization_common")

try:
    model_dir = snapshot_download(
        'damo/speech_campplus_speaker-diarization_common',
        cache_dir=str(models_dir),
        revision='v1.0.0'
    )
    print(f"\n✅ 下载成功！")
    print(f"模型路径: {model_dir}")

    # 检查关键文件
    model_path = Path(model_dir)
    print("\n文件列表:")
    for f in model_path.iterdir():
        print(f"  - {f.name}")

except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n备选方案:")
    print("1. 访问 https://modelscope.cn/models/damo/speech_campplus_speaker-diarization_common")
    print("2. 手动下载模型文件")
    print("3. 解压到 models/diarization/speech_campplus_speaker-diarization_common/")
