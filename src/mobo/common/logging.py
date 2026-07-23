"""全局日志。

保留原 ``Common/tools.py`` 中 ``GlobalLogger`` 的单例与日志格式，但做了两点
工程化调整（不改变日志本身的行为）：

1. **不在 import / 构造时劫持 ``sys.stdout``**：原实现会在构造时执行
   ``sys.stdout = self``，导致 import 即产生全局副作用，破坏 pytest 的输出
   捕获。现改为显式调用 :meth:`GlobalLogger.install_stdout_redirect`（通常仅
   在 CLI 入口调用），并提供 :meth:`GlobalLogger.restore_stdout` 还原。
2. **``logs/`` 目录惰性创建**：仅在真正需要写文件时创建，避免 import 副作用。
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

from .paths import LOGS_DIR


class _LazyDirFileHandler(logging.FileHandler):
    """在首次打开日志文件时才创建其所在目录，避免 import 期副作用。"""

    def _open(self):
        os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)
        return super()._open()



class GlobalLogger:
    """全局单例日志类，控制台+文件输出，可选捕获 print。"""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: str = "logs", log_name: str = ""):
        """初始化：只执行一次。

        与旧实现的区别：不再在此处劫持 ``sys.stdout``/``sys.stderr``，也不再
        于 import 时创建日志目录（改为写文件时惰性创建）。

        :param log_dir: 日志目录
        :param log_name: 日志文件名（为空时按日期生成）
        """
        if GlobalLogger._initialized:
            return
        GlobalLogger._initialized = True

        # 日志目录
        self.log_dir = log_dir

        # 日志文件名（按日期）
        if log_name == "":
            log_name = f"run_{datetime.now().strftime('%Y%m%d')}.log"
        self.log_path = os.path.join(log_dir, log_name)

        # 记录原始 stdout/stderr，供 restore 使用
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._redirected = False

        # 配置 logger
        self.logger = logging.getLogger("GlobalLogger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # 清空旧handler

        # 格式
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. 文件输出（delay=True + 惰性建目录：首次写日志时才创建文件与目录）
        file_handler = _LazyDirFileHandler(self.log_path, encoding="utf-8", delay=True)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 2. 控制台输出
        console_handler = logging.StreamHandler(self._orig_stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def install_stdout_redirect(self):
        """将 ``print`` 输出重定向到日志（显式开启，通常在 CLI 入口调用）。"""
        if not self._redirected:
            self._orig_stdout = sys.stdout
            self._orig_stderr = sys.stderr
            sys.stdout = self
            sys.stderr = self
            self._redirected = True

    def restore_stdout(self):
        """还原 ``sys.stdout``/``sys.stderr``。"""
        if self._redirected:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            self._redirected = False

    def write(self, message):
        """重定向print输出"""
        message = message.strip()
        if message:
            self.logger.info(message)

    def flush(self):
        pass

    # 常用日志方法
    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)


logger = GlobalLogger(log_dir=str(LOGS_DIR))


__all__ = ["GlobalLogger", "logger"]
