#!/usr/bin/env python3
"""
下载所有可选 AI 模型

使用方法:
  python backend/scripts/download_all_models.py              # 交互选择
  python backend/scripts/download_all_models.py --all        # 下载全部
  python backend/scripts/download_all_models.py --asr        # 只下载 ASR 模型
  python backend/scripts/download_all_models.py --diar       # 只下载说话人分离模型
  python backend/scripts/download_all_models.py --gender     # 只下载性别检测模型
  python backend/scripts/download_all_models.py --list       # 列出所有模型信息

注意:
  - 从项目根目录运行（不是 backend/ 目录）
  - 总下载量约 10-15 GB，请确保磁盘空间充足
  - 部分模型从 HuggingFace 下载，部分从 ModelScope 下载
  - pyannote-community-1 需要手动配置（见脚本末尾说明）
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ============================================================================
# 模型定义
# ============================================================================

# 项目根目录 (从 backend/scripts/ 往上两级)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class ModelDef:
    """模型定义"""

    def __init__(
        self,
        name: str,
        category: str,
        target_dir: Path,
        source: str,  # "huggingface" or "modelscope"
        repo_id: str,
        description: str,
        size_hint: str,
        note: str = "",
    ):
        self.name = name
        self.category = category
        self.target_dir = target_dir
        self.source = source
        self.repo_id = repo_id
        self.description = description
        self.size_hint = size_hint
        self.note = note

    @property
    def exists(self) -> bool:
        if not self.target_dir.exists():
            return False
        # 目录存在且非空
        return any(self.target_dir.iterdir())


# ── ASR 模型 ──

ASR_MODELS = [
    ModelDef(
        name="sensevoice-small",
        category="asr",
        target_dir=MODELS_DIR / "asr" / "sensevoice-small",
        source="huggingface",
        repo_id="FunAudioLLM/SenseVoiceSmall",
        description="SenseVoice-Small · CER 3.0% · 234M 参数",
        size_hint="~1 GB",
        note="5 语言，推理速度是 Whisper 的 15 倍",
    ),
    ModelDef(
        name="paraformer-large",
        category="asr",
        target_dir=MODELS_DIR / "asr" / "paraformer-large",
        source="modelscope",
        repo_id="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        description="Paraformer-Large · CER 1.7% · 220M 参数",
        size_hint="~1 GB",
        note="中文专精，模型小精度高",
    ),
    ModelDef(
        name="fireredasr-aed",
        category="asr",
        target_dir=MODELS_DIR / "asr" / "fireredasr-aed",
        source="huggingface",
        repo_id="FireRedTeam/FireRedASR-AED-L",
        description="FireRedASR-AED · CER 0.6% · 1.1B 参数",
        size_hint="~4 GB",
        note="中文 SOTA，60s 音频长度限制（已做分段处理）",
    ),
]

# ── 说话人分离模型 ──

DIAR_MODELS = [
    ModelDef(
        name="pyannote-3.1",
        category="diar",
        target_dir=MODELS_DIR / "diarization" / "pyannote-3.1",
        source="huggingface",
        repo_id="pyannote/speaker-diarization-3.1",
        description="pyannote 3.1 · MIT License · DER ~11%",
        size_hint="~500 MB",
        note="pyannote 3.1 生产级模型。需要 HF_TOKEN（接受许可后即可）",
    ),
    ]

# ── 性别检测模型 ──

GENDER_MODELS = [
    ModelDef(
        name="ecapa-gender",
        category="gender",
        target_dir=MODELS_DIR / "gender" / "ecapa-gender",
        source="huggingface",
        repo_id="JaesungHuh/voice-gender-classifier",
        description="ECAPA-TDNN 性别分类 · 准确率 ~97%",
        size_hint="~300 MB",
        note="ECAPA-TDNN 声纹模型，小巧精准",
    ),
    ModelDef(
        name="wav2vec2-gender",
        category="gender",
        target_dir=MODELS_DIR / "gender" / "wav2vec2-gender",
        source="huggingface",
        repo_id="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
        description="Wav2Vec2 性别分类 · 准确率 ~95%",
        size_hint="~1.2 GB",
        note="Wav2Vec2 大模型微调，噪声环境鲁棒",
    ),
]

ALL_MODELS = ASR_MODELS + DIAR_MODELS + GENDER_MODELS


# ============================================================================
# 下载函数
# ============================================================================


HF_MIRROR = "https://hf-mirror.com"


def download_from_huggingface(repo_id: str, target_dir: Path) -> bool:
    """从 HuggingFace 下载模型（自动回退到镜像 + git clone）"""
    token = os.environ.get("HF_TOKEN")

    print(f"  来源: HuggingFace ({repo_id})")
    print(f"  目标: {target_dir}")
    if token:
        print("  认证: 使用 HF_TOKEN")

    target_dir.mkdir(parents=True, exist_ok=True)

    # 方法 1: huggingface_hub snapshot_download
    try:
        from huggingface_hub import snapshot_download
        print("  尝试 huggingface_hub...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            token=token,
        )
        return True
    except ImportError:
        print("  [!] huggingface_hub 未安装，尝试镜像...")
    except Exception as e:
        print(f"  [!] huggingface_hub 失败: {e}")
        print("  尝试镜像 hf-mirror.com...")

    # 方法 2: huggingface_hub + 镜像
    try:
        from huggingface_hub import snapshot_download
        old_endpoint = os.environ.get("HF_ENDPOINT")
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                token=token,
            )
            return True
        except Exception as e:
            print(f"  [!] 镜像下载也失败: {e}")
            print("  尝试 git clone...")
        finally:
            if old_endpoint:
                os.environ["HF_ENDPOINT"] = old_endpoint
            else:
                os.environ.pop("HF_ENDPOINT", None)
    except ImportError:
        pass

    # 方法 3: git clone + git lfs pull
    return _git_clone_huggingface(repo_id, target_dir, token)


def _git_clone_huggingface(repo_id: str, target_dir: Path, token: str | None = None) -> bool:
    """用 git clone 从 HF 镜像下载"""
    # 清空目标目录
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    clone_url = f"{HF_MIRROR}/{repo_id}"
    if token:
        # 带认证的 URL
        clone_url = f"https://user:{token}@hf-mirror.com/{repo_id}"

    print(f"  git clone {HF_MIRROR}/{repo_id}")

    # 先 clone（不拉 LFS），再单独拉 LFS
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    result = subprocess.run(
        ["git", "clone", "--depth=1", clone_url, str(target_dir)],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        print(f"  [!] git clone 失败: {result.stderr.strip()}")
        return False

    print("  git lfs pull...")
    result = subprocess.run(
        ["git", "lfs", "pull"],
        capture_output=True, text=True, cwd=str(target_dir),
    )
    if result.returncode != 0:
        print(f"  [!] git lfs pull 失败，尝试 curl 回退...")
        if not _curl_lfs_fallback(repo_id, target_dir):
            return False

    # 清理 .git 目录节省空间
    git_dir = target_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)

    return True


def _curl_lfs_fallback(repo_id: str, target_dir: Path) -> bool:
    """用 curl 逐个下载 LFS 指针文件指向的实际文件"""
    # 找出所有 LFS 指针文件
    lfs_files = []
    for fpath in target_dir.rglob("*"):
        if fpath.is_file() and fpath.stat().st_size < 200:
            try:
                content = fpath.read_text(encoding="utf-8")
                if content.startswith("version https://git-lfs.github.com/spec/"):
                    rel = fpath.relative_to(target_dir).as_posix()
                    lfs_files.append(rel)
            except (UnicodeDecodeError, OSError):
                pass

    if not lfs_files:
        return True

    print(f"  curl 回退: {len(lfs_files)} 个 LFS 文件")
    all_ok = True
    for rel_path in lfs_files:
        url = f"{HF_MIRROR}/{repo_id}/resolve/main/{rel_path}"
        dest = target_dir / rel_path
        print(f"    下载 {rel_path}...")
        result = subprocess.run(
            ["curl", "-L", "--retry", "3", "--retry-delay", "5",
             "-o", str(dest), url],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"    [!] 失败: {rel_path}")
            all_ok = False

    return all_ok


def download_from_modelscope(repo_id: str, target_dir: Path) -> bool:
    """从 ModelScope 下载模型"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        # 尝试新版 API
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("  [!] 缺少 modelscope，请安装: pip install modelscope")
            return False

    print(f"  来源: ModelScope ({repo_id})")
    print(f"  目标: {target_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # ModelScope snapshot_download 下载到 cache，然后我们复制过来
    cache_dir = snapshot_download(model_id=repo_id)
    cache_path = Path(cache_dir)

    if cache_path != target_dir and cache_path.exists():
        # 复制文件到目标目录
        for item in cache_path.iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    return True


def download_model(model: ModelDef) -> bool:
    """下载单个模型"""
    print(f"\n{'='*60}")
    print(f"  {model.description}")
    if model.note:
        print(f"  {model.note}")
    print(f"  预估大小: {model.size_hint}")
    print(f"{'='*60}")

    if model.exists:
        print(f"  [跳过] 已存在: {model.target_dir}")
        return True

    start_time = time.time()
    try:
        if model.source == "huggingface":
            success = download_from_huggingface(model.repo_id, model.target_dir)
        elif model.source == "modelscope":
            success = download_from_modelscope(model.repo_id, model.target_dir)
        else:
            print(f"  [!] 未知来源: {model.source}")
            return False

        elapsed = time.time() - start_time
        if success:
            print(f"  [OK] 下载完成 ({elapsed:.1f}s)")
        return success

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [失败] {e} ({elapsed:.1f}s)")
        # 清理不完整的下载
        if model.target_dir.exists():
            shutil.rmtree(model.target_dir, ignore_errors=True)
        return False


# ============================================================================
# 命令行入口
# ============================================================================


def list_models():
    """列出所有模型信息"""
    categories = {
        "asr": ("ASR 语音识别", ASR_MODELS),
        "diar": ("说话人分离", DIAR_MODELS),
        "gender": ("性别检测", GENDER_MODELS),
    }

    for cat_id, (cat_name, models) in categories.items():
        print(f"\n{'─'*60}")
        print(f"  {cat_name}")
        print(f"{'─'*60}")
        for m in models:
            status = "[OK] 已下载" if m.exists else "[--] 未下载"
            print(f"  {status}  {m.name}")
            print(f"          {m.description}")
            print(f"          来源: {m.source} ({m.repo_id})")
            print(f"          大小: {m.size_hint}")
            if m.note:
                print(f"          备注: {m.note}")
            print()

    # 未公开的模型
    print(f"{'─'*60}")
    print("  尚未公开的模型")
    print(f"{'─'*60}")
    print("  [--]  sensevoice-large")
    print("          SenseVoice-Large (CER 2.1%, 1.6B) 尚未公开发布")
    print("          请关注: https://github.com/FunAudioLLM/SenseVoice")
    print()


def interactive_select() -> list[ModelDef]:
    """交互式选择要下载的模型"""
    print("\n请选择要下载的模型类别:")
    print("  1. 全部下载 (~7 GB)")
    print("  2. ASR 语音识别 (~6 GB)")
    print("  3. 说话人分离 (~100 MB)")
    print("  4. 性别检测 (~1.5 GB)")
    print("  5. 退出")
    print()

    choice = input("输入选项 (可多选，用逗号分隔，如 2,4): ").strip()
    if not choice or "5" in choice:
        return []

    models = []
    if "1" in choice:
        models = list(ALL_MODELS)
    else:
        if "2" in choice:
            models.extend(ASR_MODELS)
        if "3" in choice:
            models.extend(DIAR_MODELS)
        if "4" in choice:
            models.extend(GENDER_MODELS)

    return models


def main():
    parser = argparse.ArgumentParser(description="下载会议 AI 所需的可选模型")
    parser.add_argument("--all", action="store_true", help="下载全部模型")
    parser.add_argument("--asr", action="store_true", help="下载 ASR 语音识别模型")
    parser.add_argument("--diar", action="store_true", help="下载说话人分离模型")
    parser.add_argument("--gender", action="store_true", help="下载性别检测模型")
    parser.add_argument("--list", action="store_true", help="列出所有模型信息")
    parser.add_argument("-y", "--yes", action="store_true", help="跳过确认，直接下载")
    args = parser.parse_args()

    print("=" * 60)
    print("  会议纪要 AI — 模型下载工具")
    print(f"  模型目录: {MODELS_DIR}")
    print("=" * 60)

    if args.list:
        list_models()
        return

    # 确定要下载的模型
    if args.all:
        models = list(ALL_MODELS)
    elif args.asr or args.diar or args.gender:
        models = []
        if args.asr:
            models.extend(ASR_MODELS)
        if args.diar:
            models.extend(DIAR_MODELS)
        if args.gender:
            models.extend(GENDER_MODELS)
    else:
        models = interactive_select()

    if not models:
        print("\n未选择任何模型，退出。")
        return

    # 统计
    to_download = [m for m in models if not m.exists]
    already = [m for m in models if m.exists]

    if already:
        print(f"\n已存在 (跳过): {len(already)} 个")
        for m in already:
            print(f"  [OK] {m.name}")

    if not to_download:
        print("\n所有模型已就绪，无需下载。")
        return

    print(f"\n待下载: {len(to_download)} 个")
    for m in to_download:
        print(f"  -> {m.name} ({m.size_hint})")

    print()
    if not args.yes:
        confirm = input("确认开始下载？(y/N): ").strip().lower()
        if confirm != "y":
            print("已取消。")
            return

    # 下载
    results = {}
    for m in to_download:
        success = download_model(m)
        results[m.name] = success

    # 汇总
    print(f"\n{'='*60}")
    print("  下载结果汇总")
    print(f"{'='*60}")

    success_count = sum(1 for v in results.values() if v)
    fail_count = sum(1 for v in results.values() if not v)

    for name, success in results.items():
        status = "[OK] 成功" if success else "[--] 失败"
        print(f"  {status}  {name}")

    print(f"\n  成功: {success_count}  失败: {fail_count}  跳过: {len(already)}")

    if fail_count > 0:
        print("\n  失败的模型可以重新运行此脚本下载。")

    # 提示
    print(f"\n{'─'*60}")
    print("  注意事项:")
    print(f"{'─'*60}")
    print("""
  * pyannote-community-1 需要 HF_TOKEN 环境变量（已接受许可协议）
    设置方法: set HF_TOKEN=hf_xxxx  (在 https://huggingface.co/settings/tokens 创建)
    许可页面: https://huggingface.co/pyannote/speaker-diarization-community-1

  * SenseVoice-Large (1.6B, CER 2.1%) 尚未公开发布，暂不可下载。
    请关注: https://github.com/FunAudioLLM/SenseVoice
""")


if __name__ == "__main__":
    main()
