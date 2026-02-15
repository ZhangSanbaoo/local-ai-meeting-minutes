"""
预下载 3D-Speaker Diarization 的三个子模型

ModelScope Pipeline 在首次运行时会自动下载这三个模型，
但没有进度显示且速度慢。这个脚本可以提前下载并显示进度。
"""

import os
from pathlib import Path
from modelscope import snapshot_download

# 设置镜像加速（可选，如果国内网络慢）
# os.environ['MODELSCOPE_CACHE'] = str(Path.home() / '.cache' / 'modelscope')

print("=" * 60)
print("开始下载 3D-Speaker Diarization 的三个子模型")
print("=" * 60)

# 三个子模型
submodels = [
    {
        "name": "speaker_model (CAM++ 说话人嵌入)",
        "model_id": "damo/speech_campplus_sv_zh-cn_16k-common",
        "revision": "v2.0.2",
    },
    {
        "name": "change_locator (说话人转换点定位)",
        "model_id": "damo/speech_campplus-transformer_scl_zh-cn_16k-common",
        "revision": "v1.0.0",
    },
    {
        "name": "vad_model (FSMN VAD)",
        "model_id": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "revision": "v2.0.4",
    },
]

for i, model_info in enumerate(submodels, 1):
    print(f"\n[{i}/3] 正在下载: {model_info['name']}")
    print(f"模型ID: {model_info['model_id']}")
    print(f"版本: {model_info['revision']}")
    print("-" * 60)

    try:
        model_dir = snapshot_download(
            model_info['model_id'],
            revision=model_info['revision'],
        )
        print(f"[OK] 下载完成: {model_dir}")
    except Exception as e:
        print(f"[FAIL] 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 如果在国内，可以设置镜像:")
        print("   export MODELSCOPE_CACHE=~/.cache/modelscope")
        print("3. 访问 https://modelscope.cn 手动下载")
        continue

print("\n" + "=" * 60)
print("所有子模型下载完成！")
print("现在可以重新运行说话人分离任务，应该会立即开始处理。")
print("=" * 60)
