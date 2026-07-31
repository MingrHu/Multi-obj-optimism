"""DEFORM 自动化服务层。

面向任务的服务接口：抽样、初始化执行任务、逐阶段推进、查询状态、提取数据。
任务的必要输入（路径、参数表、目标表等）在 :func:`init_execution_task` 时落盘到
``TASKS_DIR/<task_id>/state.json``；之后的 :func:`run_execution_step` /
:func:`run_extract_data` / :func:`query_execution_status` 仅凭 ``task_id`` 即可继续，
:class:`~mobo.automation.pipeline.ForgingTask` 会按保存的目录约定从磁盘重建，
无需重新传入参数（支持跨进程续跑）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from mobo.common import task_store
from mobo.common.logging import logger
from .pipeline import ForgingTask, generate_sample_file

_KIND = "automation"

# 初始化执行任务所需的路径键
_REQUIRED_PATH_KEYS = (
    "smp_file",
    "std_key_file",
    "temp_key_path",
    "res_db_path",
    "res_key_path",
    "res_txt_path",
)

# 执行任务续跑所需的参数键（三路解析：记录 > 传入 > 报错）
_REQUIRED_EXEC_KEYS = (
    "paths_config",
    "param_table",
    "target_table",
    "in_progress",
    "max_step",
)


def _result(task_id: str, ok: bool, message: str) -> Dict[str, str]:
    """构造统一的服务返回结构。"""
    return {"task_id": task_id, "status": "success" if ok else "failed", "message": message}


def _rebuild_task(task_id: str, provided: Optional[Dict[str, Any]] = None) -> ForgingTask:
    """按参数从磁盘目录重建 ForgingTask（中间产物由确定性文件名推导）。

    参数走三路解析：优先用任务记录里的 req，缺失时用本次传入值并回填记录，
    两者都没有则报错。
    """
    req = task_store.resolve_req(task_id, _KIND, provided or {}, _REQUIRED_EXEC_KEYS)
    paths = req["paths_config"]
    task = ForgingTask(
        sample_file=paths["smp_file"],
        template_key=paths["std_key_file"],
        temp_key_dir=paths["temp_key_path"],
        result_db_dir=paths["res_db_path"],
        result_key_dir=paths["res_key_path"],
        result_txt_dir=paths["res_txt_path"],
        param_table=[list(row) for row in req["param_table"]],
        target_table=[list(row) for row in req["target_table"]],
        in_progress=list(req["in_progress"]),
        max_step=req["max_step"],
        process_info_file=paths["process_info_file"]
    )
    # 从临时 KEY 目录恢复已生成的输入 KEY（文件名确定，直接扫目录）
    temp_dir = paths["temp_key_path"]
    if os.path.isdir(temp_dir):
        task.key_files = sorted(
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".KEY")
        )
    return task


def create_sampling_task(
    task_id: str,
    save_dir: str,
    method: str,
    param_ranges: Dict[str, tuple[float, float]],
    n_samples: int = 0,
    level_nums: List[int] = [],
) -> Dict[str, str]:
    """创建并执行抽样任务，结果落盘到 state.json。"""
    if n_samples == 0:
        return {}
    try:
        out_path = generate_sample_file(task_id, method, param_ranges, save_dir, n_samples, level_nums)
        task_store.init_state(task_id, _KIND, {
            "sampling": {"method": method, "save_dir": save_dir,
                         "n_samples": n_samples, "level_nums": level_nums},
        })
        task_store.update(task_id, stage="sampling", status="finished",
                          data={"sample_file": out_path})
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
    """初始化执行任务：校验路径、落盘输入参数、构建任务并生成 KEY 文件。"""
    try:
        if any(not paths_config.get(k) for k in _REQUIRED_PATH_KEYS):
            return _result(task_id, False, "未指定样本、模板 KEY、临时/结果路径等必填项")

        for path in paths_config.values():
            target_dir = path if not os.path.splitext(path)[1] else os.path.dirname(path)
            os.makedirs(target_dir, exist_ok=True)

        # 三路解析并落盘续跑所需的全部输入参数（记录已有则沿用，缺失则回填）
        task = _rebuild_task(task_id, {
            "paths_config": paths_config,
            "param_table": param_table,
            "target_table": target_table,
            "in_progress": in_progress,
            "max_step": max_step,
        })
        task.generate_keys()
        task_store.update(task_id, stage="generate_keys", status="finished",
                          data={"key_file_count": len(task.key_files)})
        return _result(task_id, True, "执行任务初始化成功，KEY 文件生成完成")
    except Exception as exc:
        logger.error(f"执行任务初始化失败：{exc}")
        if task_store.exists(task_id):
            task_store.update(task_id, stage="generate_keys", status="failed")
        return _result(task_id, False, f"执行任务初始化失败：{exc}")


def run_execution_step(task_id: str, **overrides: Any) -> Dict[str, str]:
    """推进求解阶段。

    优先从任务记录续跑；记录缺失的参数可用 ``overrides`` 补齐（会回填记录），
    记录与传入都没有则报错。
    """
    try:
        task = _rebuild_task(task_id, overrides)
        task.run_solver()
        task_store.update(task_id, stage="run_solver", status="finished",
                          data={"db_file_count": len(task.db_files)})
        return _result(task_id, True, "计算任务运行完成")
    except Exception as exc:
        logger.error(f"求解运行失败：{exc}")
        if task_store.exists(task_id):
            task_store.update(task_id, stage="run_solver", status="failed")
        return _result(task_id, False, f"求解运行失败：{exc}")


def query_execution_status(task_id: str) -> Dict[str, str]:
    """查询执行任务状态（从 state.json 读取阶段与状态）。"""
    state = task_store.load(task_id)
    if state is None:
        return _result(task_id, False, "执行任务不存在")
    return {
        "task_id": task_id,
        "status": state.get("status", "unknown"),
        "message": f"当前阶段：{state.get('stage')}",
    }


def run_extract_data(task_id: str, **overrides: Any) -> Dict[str, str]:
    """推进数据提取阶段。

    优先从任务记录续跑；记录缺失的参数可用 ``overrides`` 补齐（会回填记录），
    记录与传入都没有则报错。
    """
    try:
        task = _rebuild_task(task_id, overrides)
        task.prepare_db_files()  # 按结果 DB 目录约定重建 db_files
        task.extract()
        task_store.update(task_id, stage="extract", status="finished",
                          data={"result_dir": task.result_txt_dir})
        return _result(task_id, True, "数据提取完成")
    except Exception as exc:
        logger.error(f"数据提取失败：{exc}")
        if task_store.exists(task_id):
            task_store.update(task_id, stage="extract", status="failed")
        return _result(task_id, False, f"数据提取失败：{exc}")


__all__ = [
    "create_sampling_task",
    "init_execution_task",
    "run_execution_step",
    "query_execution_status",
    "run_extract_data",
]
