"""DEFORM 自动化子包：采样、KEY/DB 处理、求解调度与任务服务。

平台边界：底层通过子进程驱动 Windows 平台的 ``DEF_PRE_64.exe`` /
``DEF_ARM_CTL.COM``，仅能在装有 DEFORM 的 Windows 环境真实运行。
"""

from .config import DeformConfig
from .service import (
    CreateSmpGenTask,
    InitExecutionTask,
    QueryExecutionStatus,
    RunExecutionStep,
    RunExtractData,
)

__all__ = [
    "DeformConfig",
    "CreateSmpGenTask",
    "InitExecutionTask",
    "QueryExecutionStatus",
    "RunExecutionStep",
    "RunExtractData",
]
