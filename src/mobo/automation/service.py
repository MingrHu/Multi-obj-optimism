"""DEFORM 自动化服务层。

面向任务的服务接口：抽样、初始化执行任务、逐阶段推进、查询状态、提取数据。
通过任务字典管理 :class:`~mobo.automation.pipeline.ForgingTask` 实例，返回统一的
``{"task_id", "status", "message"}`` 结构（``status`` 取 ``"success"``/``"failed"``，
查询接口返回底层 :class:`TaskStatus` 的整数值字符串，与旧实现语义一致）。
"""

from __future__ import annotations

import os
from typing import Dict, List

from mobo.common.logging import logger
from .pipeline import ForgingTask, generate_sample_file

# 任务管理字典：task_id -> 任务实例
_execution_tasks: Dict[str, ForgingTask] = {}
_sampling_done: Dict[str, str] = {}

# 初始化执行任务所需的路径键
_REQUIRED_PATH_KEYS = (
    "smp_file",
    "std_key_file",
    "temp_key_path",
    "res_db_path",
    "res_key_path",
    "res_txt_path",
)


def _result(task_id: str, ok: bool, message: str) -> Dict[str, str]:
    """构造统一的服务返回结构。"""
    return {"task_id": task_id, "status": "success" if ok else "failed", "message": message}


def create_sampling_task(
    task_id: str,
    save_dir: str,
    method: str,
    param_ranges: Dict[str, tuple[float, float]],
    n_samples: int = 0,
    level_nums: List[int] = [],
) -> Dict[str, str]:
    """创建并执行抽样任务。

    :param task_id: 任务 ID
    :param save_dir: 样本保存目录
    :param method: 采样方法 ``"lhs"`` / ``"full"``
    :param param_ranges: 参数区间字典
    :param n_samples: LHS 样本数
    :param level_nums: 全因子各参数水平数
    :return: 统一返回结构；``n_samples==0`` 时返回空字典（与旧行为一致）
    """
    if n_samples == 0:
        return {}
    try:
        out_path = generate_sample_file(method, param_ranges, save_dir, n_samples, level_nums)
        _sampling_done[task_id] = out_path
        return _result(task_id, True, f"成功使用 {method} 方法生成样本")
    except Exception as exc:
        logger.error(f"抽样任务创建失败：{exc}")
        return _result(task_id, False, f"使用 {method} 方法生成样本失败")


def init_execution_task(
    task_id: str,
    paths_config: Dict[str, str],
    param_table: List[List[str]],
    target_table: List[List[str]],
    in_progress: List[bool],
    max_step: int,
) -> Dict[str, str]:
    """初始化执行任务：校验路径、创建目录、构建任务并启动生成 KEY 阶段。

    :param task_id: 任务 ID
    :param paths_config: 路径配置（需含 ``_REQUIRED_PATH_KEYS`` 全部键）
    :param param_table: 工艺参数固定表头 2×n
    :param target_table: 目标固定表头 2×m
    :param in_progress: 每个目标是否全过程提取
    :param max_step: KEY 求解最大步数
    :return: 统一返回结构
    """
    try:
        if any(not paths_config.get(k) for k in _REQUIRED_PATH_KEYS):
            return _result(task_id, False, "未指定样本、模板 KEY、临时/结果路径等必填项")

        for path in paths_config.values():
            target_dir = path if not os.path.splitext(path)[1] else os.path.dirname(path)
            os.makedirs(target_dir, exist_ok=True)

        task = _execution_tasks.get(
            task_id,
            ForgingTask(
                sample_file=paths_config["smp_file"],
                template_key=paths_config["std_key_file"],
                temp_key_dir=paths_config["temp_key_path"],
                result_db_dir=paths_config["res_db_path"],
                result_key_dir=paths_config["res_key_path"],
                result_txt_dir=paths_config["res_txt_path"],
                param_table=param_table,
                target_table=target_table,
                in_progress=in_progress,
                max_step=max_step,
            ),
        )
        _execution_tasks[task_id] = task
        task.generate_keys()
        return _result(task_id, True, "执行任务初始化成功")
    except Exception as exc:
        logger.error(f"执行任务初始化失败：{exc}")
        return _result(task_id, False, f"执行任务初始化失败：{exc}")


def run_execution_step(task_id: str) -> Dict[str, str]:
    """推进执行任务的求解阶段。

    :param task_id: 任务 ID
    :return: 统一返回结构
    """
    if task_id not in _execution_tasks:
        return _result(task_id, False, "执行任务不存在")
    _execution_tasks[task_id].run_solver()
    return _result(task_id, True, "计算任务开始运行")


def query_execution_status(task_id: str) -> Dict[str, str]:
    """查询执行任务状态（``status`` 为底层 TaskStatus 整数值字符串）。

    :param task_id: 任务 ID
    :return: ``{"task_id", "status", "message"}``
    """
    if task_id not in _execution_tasks:
        return _result(task_id, False, "执行任务不存在")
    status = _execution_tasks[task_id].status
    return {
        "task_id": task_id,
        "status": f"{int(status)}",
        "message": f"执行任务状态：{status.name}",
    }


def run_extract_data(task_id: str) -> Dict[str, str]:
    """推进执行任务的数据提取阶段。

    :param task_id: 任务 ID
    :return: 统一返回结构
    """
    if task_id not in _execution_tasks:
        return _result(task_id, False, "执行任务不存在")
    _execution_tasks[task_id].extract()
    return _result(task_id, True, "开始提取数据")


__all__ = [
    "create_sampling_task",
    "init_execution_task",
    "run_execution_step",
    "query_execution_status",
    "run_extract_data",
]
