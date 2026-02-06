"""
增量 WAV 写入器

在录音过程中持续将 PCM 数据追加到 WAV 文件。
录音结束后调用 finalize() 写入正确的文件头。

使用方式:
    writer = IncrementalWavWriter("output.wav", sample_rate=16000)
    writer.write_chunk(pcm_bytes_1)
    writer.write_chunk(pcm_bytes_2)
    writer.finalize()
"""

from __future__ import annotations

import struct
from pathlib import Path

from ..logger import get_logger

logger = get_logger("utils.wav_writer")


class IncrementalWavWriter:
    """增量 WAV 文件写入器"""

    def __init__(
        self,
        path: Path | str,
        sample_rate: int = 16000,
        channels: int = 1,
        bits_per_sample: int = 16,
    ) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits_per_sample = bits_per_sample
        self.bytes_per_sample = bits_per_sample // 8
        self._data_size = 0
        self._closed = False

        # 确保目录存在
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # 打开文件并写入占位 WAV 头（44 bytes）
        self._file = open(self.path, "wb")
        self._write_header_placeholder()

    def _write_header_placeholder(self) -> None:
        """写入占位 WAV 头（44 bytes 全零，finalize 时覆盖）"""
        self._file.write(b"\x00" * 44)
        self._file.flush()

    def write_chunk(self, pcm_bytes: bytes) -> None:
        """追加 PCM 数据"""
        if self._closed:
            raise RuntimeError("WAV writer 已关闭")
        if pcm_bytes:
            self._file.write(pcm_bytes)
            self._data_size += len(pcm_bytes)

    def finalize(self) -> Path:
        """写入正确的 WAV 头并关闭文件"""
        if self._closed:
            return self.path

        # 回到文件开头写入正确的 WAV 头
        self._file.seek(0)
        self._write_wav_header()
        self._file.close()
        self._closed = True

        duration = self._data_size / (self.sample_rate * self.channels * self.bytes_per_sample)
        logger.info(f"WAV 文件已写入: {self.path} ({duration:.1f}s, {self._data_size} bytes)")
        return self.path

    def _write_wav_header(self) -> None:
        """写入标准 44-byte WAV 头"""
        byte_rate = self.sample_rate * self.channels * self.bytes_per_sample
        block_align = self.channels * self.bytes_per_sample
        file_size = 36 + self._data_size  # RIFF chunk size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            file_size,
            b"WAVE",
            b"fmt ",
            16,  # fmt chunk size
            1,   # PCM format
            self.channels,
            self.sample_rate,
            byte_rate,
            block_align,
            self.bits_per_sample,
            b"data",
            self._data_size,
        )
        self._file.write(header)

    @property
    def duration_seconds(self) -> float:
        """当前已写入的音频时长（秒）"""
        if self._data_size == 0:
            return 0.0
        return self._data_size / (self.sample_rate * self.channels * self.bytes_per_sample)

    @property
    def data_size(self) -> int:
        """已写入的 PCM 数据大小（bytes）"""
        return self._data_size

    def close(self) -> None:
        """关闭文件（不写入头，用于异常时清理）"""
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finalize()
        else:
            self.close()
        return False
