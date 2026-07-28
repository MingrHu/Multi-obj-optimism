"""集中式路径解析。

统一管理项目根目录与数据子目录，替代旧代码中散落的 ``../../data/...``
相对路径以及 macOS/Windows 硬编码绝对路径。所有路径均以 :class:`pathlib.Path`
表示，并支持通过环境变量覆盖，便于在打包部署或数据外置时使用。

环境变量：
- ``MOBO_PROJECT_DIR``：覆盖自动推导的项目根目录。
- ``MOBO_DATA_DIR``：覆盖数据目录（默认 ``<project>/data``）。
"""

from __future__ import annotations

import os
from pathlib import Path

# 当前包目录 src/mobo
PACKAGE_DIR = Path(__file__).resolve().parent.parent


def _discover_project_dir() -> Path:
    """向上查找项目根目录。

    以包含 ``pyproject.toml`` 或 ``data`` 目录的最近祖先作为项目根，比硬编码
    向上数层更稳健；若均未找到则回退到 ``src`` 的上一级目录。

    :return: 项目根目录
    """
    env = os.environ.get("MOBO_PROJECT_DIR")
    if env:
        return Path(env).expanduser().resolve()

    for candidate in [PACKAGE_DIR, *PACKAGE_DIR.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "data").is_dir():
            return candidate

    # 回退：src/mobo -> src -> 项目根
    return PACKAGE_DIR.parents[1]


PROJECT_DIR = _discover_project_dir()


def _resolve_data_dir() -> Path:
    """解析数据目录，优先使用环境变量 ``MOBO_DATA_DIR``。"""
    env = os.environ.get("MOBO_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return PROJECT_DIR / "data"


DATA_DIR = _resolve_data_dir()
LOGS_DIR = PROJECT_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"
TEST_DIR = DATA_DIR / "TEST"
KEY_FILE_DIR = DATA_DIR / "keyfile"


def model_family_dir(family: str) -> Path:
    """返回指定模型族的目录（如 ``PRG`` / ``DNN``）。

    :param family: 模型族名称
    :return: ``MODELS_DIR/<family>`` 路径
    """
    return MODELS_DIR / family


__all__ = [
    "PACKAGE_DIR",
    "PROJECT_DIR",
    "DATA_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "TEST_DIR",
    "KEY_FILE_DIR",
    "model_family_dir",
]
