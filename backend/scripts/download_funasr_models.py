"""Download FunASR streaming models to models/funasr/

Models:
- paraformer-zh-streaming: Streaming ASR (~300MB)
- fsmn-vad: Voice Activity Detection (~10MB)
- ct-punc: Punctuation restoration (~1GB)

Usage:
    python backend/scripts/download_funasr_models.py
"""

import os
import sys
from pathlib import Path

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

MODELS_DIR = PROJECT_ROOT / "models" / "funasr"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model IDs on ModelScope
MODELS = [
    {
        "name": "paraformer-zh-streaming",
        "model_id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "desc": "Streaming ASR (Paraformer)",
    },
    {
        "name": "fsmn-vad",
        "model_id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "desc": "Voice Activity Detection",
    },
    {
        "name": "ct-punc",
        "model_id": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        "desc": "Punctuation Restoration",
    },
]


def download_models():
    from funasr import AutoModel

    for m in MODELS:
        target = MODELS_DIR / m["name"]
        if target.exists() and any(target.iterdir()):
            print(f"[SKIP] {m['name']} already exists at {target}")
            continue

        print(f"\n[DOWNLOAD] {m['desc']}: {m['model_id']}")
        print(f"  → Target: {target}")

        # AutoModel.from_pretrained will download to ModelScope cache,
        # but we want them in our local models/ directory.
        # Use modelscope snapshot_download directly for explicit control.
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            local_path = snapshot_download(
                m["model_id"],
                cache_dir=str(MODELS_DIR),
            )
            # snapshot_download creates a nested structure, symlink or rename
            print(f"  ✓ Downloaded to: {local_path}")

            # Create a convenient alias if needed
            downloaded = Path(local_path)
            if downloaded != target and not target.exists():
                # Create symlink or just note the path
                print(f"  Note: Model at {downloaded}, expected at {target}")
                print(f"  Creating symlink...")
                try:
                    target.symlink_to(downloaded)
                    print(f"  ✓ Symlink created: {target} → {downloaded}")
                except OSError:
                    print(f"  ! Symlink failed (need admin). Will use direct path.")

        except ImportError:
            # Fallback: let AutoModel handle download
            print("  modelscope.hub not available, using AutoModel...")
            try:
                model = AutoModel(model=m["model_id"], model_path=str(target))
                print(f"  ✓ Downloaded via AutoModel")
            except Exception as e:
                print(f"  ✗ AutoModel failed: {e}")
                print(f"  Trying direct download...")
                _download_via_automodel_init(m["model_id"], target)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            raise

    print("\n" + "=" * 50)
    print("All models downloaded!")
    print(f"Location: {MODELS_DIR}")
    print("=" * 50)


def _download_via_automodel_init(model_id: str, target: Path):
    """Fallback: use AutoModel which auto-downloads to cache."""
    from funasr import AutoModel

    # This downloads to ModelScope cache and returns the model
    # We just need the download to happen
    _ = AutoModel(model=model_id)
    print(f"  ✓ Downloaded to ModelScope cache")
    print(f"  Note: Will reference via model_id at runtime")


if __name__ == "__main__":
    print(f"FunASR Model Downloader")
    print(f"Target directory: {MODELS_DIR}")
    print(f"Models to download: {len(MODELS)}")
    download_models()
