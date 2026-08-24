"""DEFORM 自动化子包：采样、KEY/DB 处理、求解调度与任务服务。

模块划分：
- :mod:`~mobo.automation.config`：DEFORM 关键字/对象/目标函数映射；
- :mod:`~mobo.automation.sampling`：LHS / 全因子采样（纯逻辑）；
- :mod:`~mobo.automation.keyfile`：KEY 文件文本处理（纯逻辑）；
- :mod:`~mobo.automation.solver`：DEFORM 子进程驱动（KEY↔DB、求解调度）；
- :mod:`~mobo.automation.extract`：结果数据集提取编排；
- :mod:`~mobo.automation.pipeline`：任务类 :class:`ForgingTask` 与状态 :class:`TaskStatus`；
- :mod:`~mobo.automation.service`：任务级服务接口。

平台边界：:mod:`~mobo.automation.solver` 通过子进程驱动 Windows 平台的
``DEF_PRE_64.exe`` / ``DEF_ARM_CTL.COM``，仅能在装有 DEFORM 的 Windows 环境真实运行。
"""

from .config import DeformConfig
from .pipeline import ForgingTask, TaskStatus, generate_sample_file
from .service import (
    align_result_db_dirs,
    create_sampling_task,
    init_execution_task,
    query_execution_status,
    run_execution_step,
    run_extract_data,
)
from .multi_operation import MultiOperationTask, generate_multi_operation_samples
from .multi_operation_service import (
    create_multi_operation_sampling_task,
    init_multi_operation_task,
    query_multi_operation_status,
    run_multi_operation_extract,
    run_multi_operation_task,
)
from .task_collection import (
    TASK_COLLECTION,
    TC4_RING_MULTI_TASK_1,
    RING_7050_SINGLE_TASK_1,
    MultiOperationTaskDefinition,
    SingleOperationTaskDefinition,
    TargetDefinition,
    get_task_definition,
    get_multi_operation_task_definition,
    get_single_operation_task_definition,
)

__all__ = [
    "DeformConfig",
    "ForgingTask",
    "TaskStatus",
    "generate_sample_file",
    "create_sampling_task",
    "init_execution_task",
    "query_execution_status",
    "run_execution_step",
    "run_extract_data",
    "align_result_db_dirs",
    "MultiOperationTask",
    "generate_multi_operation_samples",
    "create_multi_operation_sampling_task",
    "init_multi_operation_task",
    "query_multi_operation_status",
    "run_multi_operation_extract",
    "run_multi_operation_task",
    "MultiOperationTaskDefinition",
    "SingleOperationTaskDefinition",
    "TargetDefinition",
    "TASK_COLLECTION",
    "TC4_RING_MULTI_TASK_1",
    "RING_7050_SINGLE_TASK_1",
    "get_task_definition",
    "get_multi_operation_task_definition",
    "get_single_operation_task_definition",
]
