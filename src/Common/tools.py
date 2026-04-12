import logging
import sys
import os
from datetime import datetime

# 路径相关定义
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CUR_DIR)
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

os.makedirs(LOGS_DIR, exist_ok=True)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

class GlobalLogger:
    """全局单例日志类，控制台+文件输出，自动捕获print"""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: str = "logs", log_name: str = ""):
        """初始化：只执行一次"""
        if GlobalLogger._initialized:
            return
        GlobalLogger._initialized = True

        # 日志目录
        self.log_dir = log_dir

        # 日志文件名（按日期）
        if log_name == "":
            log_name = f"run_{datetime.now().strftime('%Y%m%d')}.log"
        self.log_path = os.path.join(log_dir, log_name)

        # 配置 logger
        self.logger = logging.getLogger("GlobalLogger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # 清空旧handler

        # 格式
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. 文件输出
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 2. 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 3. 重定向 print 到日志
        sys.stdout = self
        sys.stderr = self

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

logger = GlobalLogger(log_dir=LOGS_DIR) 
