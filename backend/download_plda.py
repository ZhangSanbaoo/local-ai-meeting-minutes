"""
下载 PLDA 模型到本地

运行前请先访问以下链接接受协议：
https://huggingface.co/pyannote/speaker-diarization-community-1
"""

import os
from pathlib import Path

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# 清理可能损坏的代理环境变量
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    if proxy_var in os.environ:
        val = os.environ[proxy_var]
        if "\n" in val or "\r" in val:
            del os.environ[proxy_var]

from huggingface_hub import hf_hub_download, list_repo_files

# 从环境变量或直接设置 token（清理换行符）
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_TOKEN = HF_TOKEN.strip().replace("\n", "").replace("\r", "")

# 目标目录
REPO_ID = "pyannote/speaker-diarization-community-1"
LOCAL_DIR = Path(__file__).parent.parent / "models" / "pyannote" / "plda-community-1"

def main():
    print(f"下载 PLDA 模型从: {REPO_ID}")
    print(f"保存到: {LOCAL_DIR}")
    print()

    # 创建目录
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # 列出仓库中的文件
    print("获取文件列表...")
    try:
        files = list_repo_files(REPO_ID, token=HF_TOKEN)
        print(f"仓库中共有 {len(files)} 个文件")

        # 过滤出 plda 相关文件
        plda_files = [f for f in files if "plda" in f.lower() or f.endswith(".npz")]

        if not plda_files:
            # 下载所有文件
            print("未找到明确的 plda 文件，列出所有文件：")
            for f in files:
                print(f"  - {f}")
            plda_files = files

        print(f"\n将下载以下文件:")
        for f in plda_files:
            print(f"  - {f}")
        print()

        # 下载文件
        for filename in plda_files:
            print(f"下载: {filename}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    token=HF_TOKEN,
                    local_dir=LOCAL_DIR,
                )
                print(f"  ✓ 已保存到: {downloaded_path}")
            except Exception as e:
                print(f"  ✗ 下载失败: {e}")

        print("\n下载完成!")
        print(f"文件保存在: {LOCAL_DIR}")

    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保:")
        print("1. 已访问 https://huggingface.co/pyannote/speaker-diarization-community-1 接受协议")
        print("2. HF_TOKEN 正确且有效")

if __name__ == "__main__":
    main()
