print("检查 torchcodec...")
try:
    import torchcodec
    print(f"torchcodec 版本: {torchcodec.__version__}")
    from torchcodec.decoders import AudioDecoder, AudioStreamMetadata
    print("AudioDecoder 导入成功")
except Exception as e:
    print(f"torchcodec 导入失败: {e}")

print("\n检查 torchaudio...")
try:
    import torchaudio
    print(f"torchaudio 版本: {torchaudio.__version__}")
except Exception as e:
    print(f"torchaudio 导入失败: {e}")

print("\n检查 pyannote...")
try:
    import pyannote.audio
    print(f"pyannote.audio 版本: {pyannote.audio.__version__}")
except Exception as e:
    print(f"pyannote.audio 导入失败: {e}")
