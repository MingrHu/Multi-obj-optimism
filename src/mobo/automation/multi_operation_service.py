"""多工步自动化的任务级入口。"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional, Sequence

from mobo.common import task_store
from mobo.common.logging import logger
from mobo.common.paths import task_dir

from .multi_operation import MultiOperationTask, Operation, generate_multi_operation_samples
from .incremental import IncrementalDataset

_KIND = "automation_multi_operation"


def _range_suffix(sample_start: int, sample_end: int | None) -> str:
    """返回多工步分片文件名后缀；完整范围保持历史文件名。"""
    if sample_start == 0 and sample_end is None:
        return ""
    return f"_{sample_start}_{sample_end if sample_end is not None else 'end'}"


def _workflow_state_file(
    task_id: str, sample_start: int = 0, sample_end: int | None = None
) -> str:
    suffix = _range_suffix(sample_start, sample_end)
    return str(task_dir(task_id) / f"multi_operation_state{suffix}.json")


def create_multi_operation_sampling_task(task_id: str, operations: Sequence[Operation],
                                         save_dir: str, method: str = "lhs",
                                         n_samples: int = 0,
                                         level_nums: Sequence[int] = ()) -> Dict[str, Any]:
    """生成多工步联合样本并记录任务输入。"""
    req = {"operations": list(operations), "save_dir": save_dir, "method": method,
           "n_samples": n_samples, "level_nums": list(level_nums)}
    task_store.init_state(task_id, _KIND, req)
    sample_file = generate_multi_operation_samples(
        task_id, operations, save_dir, method, n_samples, level_nums
    )
    return task_store.update(task_id, stage="sampled", status="running",
                             data={"sample_file": sample_file})


def init_multi_operation_task(task_id: str, sample_file: str,
                              operations: Sequence[Operation], work_dir: str,
                              max_parallel_samples: int = 24,
                              keep_checkpoints: bool = True,
                              dry_run: bool = False,
                              incremental: bool = False,
                              sample_start: int = 0,
                              sample_end: int | None = None) -> Dict[str, Any]:
    """初始化多工步任务，并为完整采样 TXT 的指定行范围预生成 KEY。"""
    req = {"sample_file": os.path.abspath(sample_file), "operations": list(operations),
           "work_dir": os.path.abspath(work_dir),
           "max_parallel_samples": max_parallel_samples,
           "keep_checkpoints": keep_checkpoints, "dry_run": dry_run,
           "incremental": incremental,
           "sample_start": sample_start, "sample_end": sample_end,
           "state_file": _workflow_state_file(task_id, sample_start, sample_end)}
    task_store.init_state(task_id, _KIND, req)
    task_store.update(task_id, req=req)
    task = MultiOperationTask(
        task_id=task_id,
        sample_file=req["sample_file"],
        operations=req["operations"],
        work_dir=req["work_dir"],
        max_parallel_samples=req["max_parallel_samples"],
        keep_checkpoints=req["keep_checkpoints"],
        dry_run=req["dry_run"],
        state_file=req["state_file"],
        sample_start=req["sample_start"],
        sample_end=req["sample_end"],
    )
    key_files = task.prepare_parameterized_keys()
    return task_store.update(task_id, stage="initialized", status="running",
                             data={"workflow_state": task.state_file,
                                   "key_file_count": len(key_files)})


def _rebuild(task_id: str, provided: Optional[Dict[str, Any]] = None) -> MultiOperationTask:
    required = ["sample_file", "operations", "work_dir"]
    req = task_store.resolve_req(task_id, _KIND, provided or {}, required)
    sample_start = int(req.get("sample_start", 0))
    sample_end_value = req.get("sample_end")
    sample_end = int(sample_end_value) if sample_end_value is not None else None
    state_file = _workflow_state_file(task_id, sample_start, sample_end)
    if req.get("state_file") != state_file:
        task_store.update(
            task_id,
            req={"state_file": state_file},
            data={"workflow_state": state_file},
        )
    return MultiOperationTask(
        task_id=task_id, sample_file=req["sample_file"], operations=req["operations"],
        work_dir=req["work_dir"],
        max_parallel_samples=req.get("max_parallel_samples", 1),
        keep_checkpoints=req.get("keep_checkpoints", True),
        dry_run=req.get("dry_run", False),
        state_file=state_file,
        sample_start=sample_start,
        sample_end=sample_end,
    )


def run_multi_operation_task(task_id: str, **provided: Any) -> Dict[str, Any]:
    """仅凭 task_id 从磁盘重建并运行或续跑多工步任务。"""
    # 1 加载任务信息 可断点续跑
    task = _rebuild(task_id, provided)
    state = task_store.load(task_id) or {}
    req = state.get("req") or {}

    # 2 是否在计算完成DB后立马提取数据
    incremental_enabled = bool(req.get("incremental"))
    incremental_state_file = ""
    incremental_output_file = ""
    if incremental_enabled:
        from .task_collection import get_multi_operation_task_definition

        # 2.1 实时写入数据集初始化
        definition = get_multi_operation_task_definition(task_id)
        requested_start = int(req.get("sample_start", 0))
        requested_end_value = req.get("sample_end")
        requested_end = (
            int(requested_end_value) if requested_end_value is not None else None
        )
        suffix = _range_suffix(requested_start, requested_end)
        incremental_state_file = str(
            task_dir(task_id) / f"incremental_dataset{suffix}.json"
        )
        incremental_output_file = str(
            definition.workspace / "results"
            / f"{task_id}_incremental_result{suffix}.txt"
        )
        dataset = IncrementalDataset(
            incremental_state_file,
            incremental_output_file,
        )

        # 2.2 注册样本实时更新回调
        def on_sample_completed(sample_index: int) -> None:
            if dataset.is_completed(sample_index):
                logger.info(f"样本 {sample_index} 增量提取已完成，本次跳过")
                return
            dataset.mark_started(sample_index)
            logger.info(
                f"样本 {sample_index} 增量提取开始: state={incremental_state_file}, "
                f"output={incremental_output_file}"
            )
            try:
                row = definition.extract_sample_row(task, sample_index)
                dataset.commit(sample_index, row)
                logger.info(
                    f"样本 {sample_index} 增量提取完成: columns={len(row)}, "
                    f"output={incremental_output_file}"
                )
            except Exception as exc:
                traceback_text = traceback.format_exc()
                dataset.mark_failed(
                    sample_index,
                    str(exc),
                    error_type=type(exc).__name__,
                    traceback_text=traceback_text,
                )
                logger.error(
                    f"样本 {sample_index} 增量提取失败: "
                    f"{type(exc).__name__}: {exc}\n{traceback_text}"
                )

        task.on_sample_completed = on_sample_completed

    # 3 运行或续跑任务
    task_store.update(task_id, stage="solving", status="running")
    result = task.run()
    status = "finished" if result["status"] == "completed" else "failed"
    return task_store.update(task_id, stage="completed" if status == "finished" else "failed",
                             status=status, data={"workflow": result,
                                                 "workflow_state": task.state_file,
                                                 "result_db_files": task.result_db_files(),
                                                 "incremental_state_file":
                                                     incremental_state_file,
                                                 "incremental_output_file":
                                                     incremental_output_file})


def query_multi_operation_status(task_id: str) -> Optional[Dict[str, Any]]:
    """返回任务级状态，并在可用时附带逐样本、逐工步状态。"""
    state = task_store.load(task_id)
    if state is None:
        return None
    path = (state.get("data") or {}).get("workflow_state")
    if path and os.path.exists(path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            state = dict(state)
            state["workflow"] = json.load(f)
    return state


def run_multi_operation_extract(task_id: str, result_dir: str | None = None) -> Dict[str, Any]:
    """仅凭 task_id 恢复多工步任务并生成无表头结果数据集。"""
    from .task_collection import get_multi_operation_task_definition

    task = _rebuild(task_id)
    definition = get_multi_operation_task_definition(task_id)
    result_file = definition.extract_dataset(task, result_dir=result_dir)
    return task_store.update(
        task_id, stage="extract", status="finished",
        data={"result_file": result_file},
    )


__all__ = [
    "create_multi_operation_sampling_task", "init_multi_operation_task",
    "run_multi_operation_task", "query_multi_operation_status",
    "run_multi_operation_extract",
]
