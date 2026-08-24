"""多工步自动化的任务级入口。"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional, Sequence

from mobo.common import task_store
from mobo.common.paths import task_dir

from .multi_operation import MultiOperationTask, Operation, generate_multi_operation_samples

_KIND = "automation_multi_operation"


def _workflow_state_file(task_id: str) -> str:
    return str(task_dir(task_id) / "multi_operation_state.json")


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
                              dry_run: bool = False) -> Dict[str, Any]:
    """初始化多工步计算任务；全部续跑参数写入 state.json。"""
    req = {"sample_file": os.path.abspath(sample_file), "operations": list(operations),
           "work_dir": os.path.abspath(work_dir),
           "max_parallel_samples": max_parallel_samples,
           "keep_checkpoints": keep_checkpoints, "dry_run": dry_run,
           "state_file": _workflow_state_file(task_id)}
    task_store.init_state(task_id, _KIND, req)
    task_store.update(task_id, req=req)
    task = MultiOperationTask(task_id, **req)
    return task_store.update(task_id, stage="initialized", status="running",
                             data={"workflow_state": task.state_file})


def _rebuild(task_id: str, provided: Optional[Dict[str, Any]] = None) -> MultiOperationTask:
    required = ["sample_file", "operations", "work_dir"]
    req = task_store.resolve_req(task_id, _KIND, provided or {}, required)
    state_file = _workflow_state_file(task_id)
    legacy_state = os.path.join(req["work_dir"], "multi_operation_state.json")
    if not os.path.exists(state_file) and os.path.exists(legacy_state):
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        shutil.move(legacy_state, state_file)
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
    )


def run_multi_operation_task(task_id: str, **provided: Any) -> Dict[str, Any]:
    """仅凭 task_id 从磁盘重建并运行或续跑多工步任务。"""
    task = _rebuild(task_id, provided)
    task_store.update(task_id, stage="solving", status="running")
    result = task.run()
    status = "finished" if result["status"] == "completed" else "failed"
    return task_store.update(task_id, stage="completed" if status == "finished" else "failed",
                             status=status, data={"workflow": result,
                                                 "workflow_state": task.state_file,
                                                 "result_db_files": task.result_db_files()})


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
    from .task_collection import get_task_definition

    task = _rebuild(task_id)
    definition = get_task_definition(task_id)
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
