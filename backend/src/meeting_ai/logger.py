"""
日志模块

特点：
- 使用 rich 美化终端输出
- 支持文件日志
- 结构化日志格式
- 按模块分级别控制
"""

import logging
import sys
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# 自定义主题
CUSTOM_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "success": "green",
        "debug": "dim",
    }
)

# 全局 Console 实例
console = Console(theme=CUSTOM_THEME)


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_file: Path | None = None,
    show_path: bool = False,
) -> logging.Logger:
    """
    配置日志系统

    Args:
        level: 日志级别
        log_file: 可选的日志文件路径
        show_path: 是否显示文件路径（调试时有用）

    Returns:
        配置好的 logger
    """
    # 获取根 logger
    logger = logging.getLogger("meeting_ai")
    logger.setLevel(getattr(logging, level))

    # 清除已有的 handlers
    logger.handlers.clear()

    # Rich handler（终端输出）
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=show_path,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
    )
    rich_handler.setLevel(getattr(logging, level))
    logger.addHandler(rich_handler)

    # 文件 handler（可选）
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取子模块的 logger

    Args:
        name: 模块名（会自动加上 meeting_ai. 前缀）

    Returns:
        子模块的 logger

    Example:
        >>> logger = get_logger("asr")
        >>> logger.info("开始转写...")
    """
    if name.startswith("meeting_ai."):
        return logging.getLogger(name)
    return logging.getLogger(f"meeting_ai.{name}")


# 便捷函数：直接打印带样式的消息
def print_info(message: str) -> None:
    """打印信息消息"""
    console.print(f"[info]ℹ {message}[/info]")


def print_success(message: str) -> None:
    """打印成功消息"""
    console.print(f"[success]✓ {message}[/success]")


def print_warning(message: str) -> None:
    """打印警告消息"""
    console.print(f"[warning]⚠ {message}[/warning]")


def print_error(message: str) -> None:
    """打印错误消息"""
    console.print(f"[error]✗ {message}[/error]")


def print_step(step: int, total: int, message: str) -> None:
    """打印步骤进度"""
    console.print(f"[info][{step}/{total}][/info] {message}")
