"""日志系统模块

基于 Loguru 的日志配置，支持控制台彩色输出和文件轮转日志。
提供 setup_logging() 初始化日志系统，get_logger() 获取模块级 logger 实例。

Author: LogicYe
Date: 2026-05-19
"""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from utils.settings import settings


def setup_logging(
    log_level:  Optional[str] = None,
    log_file:   Optional[str] = None,
    rotation:   Optional[str] = None,
    retention:  Optional[str] = None,
) -> None:
    """配置日志系统

    初始化 Loguru 日志器，添加控制台彩色输出和文件轮转日志两个 handler。
    所有参数均可选，未传入时从全局 settings 配置中读取默认值。

    Args:
        log_level: 日志级别，默认从配置读取
        log_file:  日志文件路径，默认从配置读取
        rotation:  日志轮转规则，默认从配置读取
        retention: 日志保留时间，默认从配置读取
    """
    # 使用配置中的默认值
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    rotation = rotation or settings.log_rotation
    retention = retention or settings.log_retention

    # 移除默认的 handler
    logger.remove()

    # ==================== 控制台日志 ====================
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ==================== 文件日志 ====================
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )


def get_logger(name: str):
    """获取指定名称的模块级 logger

    通过 bind(name=name) 创建带模块标识的 logger 实例，
    便于在日志输出中区分不同模块的来源。

    Args:
        name: logger 标识名称，通常传入调用模块的 __name__

    Returns:
        绑定 name 后的 loguru.Logger 实例
    """
    return logger.bind(name=name)
