"""训练日志记录模块——同时输出到控制台和 .log 文件，每条带时间戳

用途: 满足课程要求(4)——提供完整训练过程证据，日志文件可直接截图放入报告
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_path: str) -> logging.Logger:
    """创建一个同时输出到控制台和文件的 logger

    参数:
        name:     logger 名称，用于区分不同训练任务
        log_path: 日志文件路径，自动创建父目录
    返回:
        logging.Logger 实例，同时写入控制台和指定文件
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 时间戳格式: [2026-05-14 22:01:03] 便于追溯每次训练的起止时间
    fmt = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件 handler——写入 .log 文件，可用作训练证据截图
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
