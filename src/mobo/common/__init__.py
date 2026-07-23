"""公共基础设施：集中式路径解析与全局日志。"""

from .logging import GlobalLogger, logger
from .paths import (
    DATA_DIR,
    KEY_FILE_DIR,
    LOGS_DIR,
    MODELS_DIR,
    PACKAGE_DIR,
    PROJECT_DIR,
    TEST_DIR,
    model_family_dir,
)

__all__ = [
    "GlobalLogger",
    "logger",
    "PACKAGE_DIR",
    "PROJECT_DIR",
    "DATA_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "TEST_DIR",
    "KEY_FILE_DIR",
    "model_family_dir",
]
