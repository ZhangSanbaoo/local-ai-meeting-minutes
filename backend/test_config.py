#!/usr/bin/env python
"""测试配置路径"""
import sys
from pathlib import Path

# 添加 src 到路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from meeting_ai.config import get_settings

settings = get_settings()

print("=" * 50)
print("路径配置测试")
print("=" * 50)
print(f"root_dir:    {settings.paths.root_dir}")
print(f"models_dir:  {settings.paths.models_dir}")
print(f"output_dir:  {settings.paths.output_dir}")
print(f"data_dir:    {settings.paths.data_dir}")
print()
print(f"models_dir 存在: {settings.paths.models_dir.exists()}")
print(f"output_dir 存在: {settings.paths.output_dir.exists()}")
print()

# 检查 whisper 模型
whisper_dir = settings.paths.models_dir / "whisper"
print(f"whisper_dir: {whisper_dir}")
print(f"whisper_dir 存在: {whisper_dir.exists()}")

if whisper_dir.exists():
    print("Whisper 模型:")
    for d in whisper_dir.iterdir():
        if d.is_dir():
            print(f"  - {d.name}")

# 检查 LLM 模型
llm_dir = settings.paths.models_dir / "llm"
print(f"\nllm_dir: {llm_dir}")
print(f"llm_dir 存在: {llm_dir.exists()}")

if llm_dir.exists():
    print("LLM 模型:")
    for f in llm_dir.glob("*.gguf"):
        print(f"  - {f.name}")
