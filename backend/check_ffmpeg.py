import subprocess
import shutil
import os

print("检查 FFmpeg...")

# 检查 PATH 中的 ffmpeg
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    print(f"ffmpeg 路径: {ffmpeg_path}")
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    first_line = result.stdout.split('\n')[0] if result.stdout else result.stderr.split('\n')[0]
    print(f"版本: {first_line}")
else:
    print("ffmpeg 未在 PATH 中找到")

# 检查 conda 环境中的 ffmpeg
conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    conda_ffmpeg = os.path.join(conda_prefix, "Library", "bin", "ffmpeg.exe")
    if os.path.exists(conda_ffmpeg):
        print(f"\nConda ffmpeg: {conda_ffmpeg}")
        result = subprocess.run([conda_ffmpeg, "-version"], capture_output=True, text=True)
        first_line = result.stdout.split('\n')[0] if result.stdout else ""
        print(f"版本: {first_line}")

# 检查 DLL
print("\n检查 FFmpeg DLL...")
dll_dirs = [
    os.path.join(conda_prefix, "Library", "bin") if conda_prefix else "",
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
]
for dll_dir in dll_dirs:
    if not dll_dir or not os.path.exists(dll_dir):
        continue
    print(f"\n目录: {dll_dir}")
    for f in os.listdir(dll_dir):
        if "avcodec" in f.lower() or "avformat" in f.lower() or "avutil" in f.lower():
            print(f"  {f}")
