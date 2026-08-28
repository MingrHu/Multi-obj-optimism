"""多目标优化服务层。

按 ``interface_protocol.md`` 的协议，对外提供以 ``task_id``（opt_ 前缀）为主键的优化
接口：优化前落盘请求参数（含代理模型 ``model_id`` 溯源），优化后把响应（输出文件路径、
任务信息）写入 ``TASKS_DIR/<task_id>/state.json``。

历史 :func:`~mobo.optimization.ga.run.NSGA2_run` 与
:func:`~mobo.optimization.rl.run.train_and_optimize` 保留为无参演示实现。协议请求完整时，
本层改用参数化 NSGA-II 编排器；旧的简化调用仍兼容原演示入口。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from mobo.common import task_store
from mobo.common.logging import logger
from mobo.common.paths import DATA_DIR
from mobo.optimization.ga.parameterized import run_parameterized_nsga2
from mobo.optimization.ga.run import NSGA2_run
from mobo.optimization.rl.parameterized import run_parameterized_rl
from mobo.optimization.rl.run import train_and_optimize

_KIND = "optimization"

# 优化续跑所需的参数键（三路解析：记录 > 传入 > 报错）
_REQUIRED_OPT_KEYS = ("optimizer",)

_PARAMETERIZED_NSGA2_KEYS = (
    "model_id", "objective_names", "input_var_count", "all_var_list",
    "decision_var_indices", "decision_var_names", "decision_bounds",
    "constraints", "objective_config", "optimizer_config", "output_config",
)


def _resolve_model_dir(resolved: Dict[str, Any]) -> str:
    model_id = resolved["model_id"]
    model_state = task_store.load(model_id)
    if model_state is None or model_state.get("kind") != "surrogate":
        raise ValueError(f"代理模型任务不存在：{model_id}")
    if model_state.get("status") != "finished":
        raise ValueError(f"代理模型任务尚未完成：{model_id}")
    training_request = model_state.get("req") or {}
    if training_request.get("vars_out") != resolved["all_var_list"]:
        raise ValueError("all_var_list 与 model_id 训练时记录的变量顺序不一致")
    if training_request.get("n_vars") != resolved["input_var_count"]:
        raise ValueError("input_var_count 与 model_id 训练时记录不一致")
    model_dir = (model_state.get("data") or {}).get("model_dir")
    if not model_dir:
        raise ValueError(f"代理模型任务缺少 model_dir：{model_id}")
    return model_dir


def _new_task_id() -> str:
    return "opt_" + datetime.now().strftime("%Y%m%d_%H%M_%S%f")[:-3]


def run_optimization(
    req: Optional[Dict[str, Any]] = None,
    optimizer: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """运行多目标优化并把请求/响应落盘到 state.json。

    输入参数走三路解析：优先用 ``task_id`` 记录里的 req，缺失时用本次传入值并
    回填记录，两者都没有则报错。

    :param req: 协议输入参数（model_id/objective_names/constraints 等，用于溯源）
    :param optimizer: ``"nsga2"``（GA）或 ``"rl"``（PPO）
    :param task_id: 复用已有 task_id（None 则新建）
    :return: 协议 resp 字典（code/msg/task_id/data）
    """
    task_id = task_id or _new_task_id()
    try:
        resolved = task_store.resolve_req(
            task_id, _KIND, {**(req or {}), "optimizer": optimizer}, _REQUIRED_OPT_KEYS
        )
    except ValueError as exc:
        return {"code": 1, "msg": f"续跑参数缺失：{exc}", "task_id": task_id, "data": {}}

    optimizer = resolved["optimizer"]
    task_store.update(task_id, stage="optimize", status="running")

    try:
        started = time.time()
        constraint_check = None
        if optimizer in {"nsga2", "rl"} and all(
            key in resolved for key in _PARAMETERIZED_NSGA2_KEYS
        ):
            model_dir = _resolve_model_dir(resolved)
            configured_path = resolved["output_config"].get("pareto_txt_path")
            filename = "pareto_solutions.tsv" if optimizer == "nsga2" else "rl_solutions.tsv"
            default_path = str(Path(task_store.state_path(task_id)).parent / filename)
            runner = run_parameterized_nsga2 if optimizer == "nsga2" else run_parameterized_rl
            result = runner(
                resolved,
                model_dir=model_dir,
                output_path=configured_path or default_path,
            )
            file_resource = {"solution_txt_path": result["solution_txt_path"]}
            constraint_check = {
                "all_solution_feasible": result["all_solution_feasible"],
                "solution_count": result["solution_count"],
            }
        elif optimizer == "nsga2":
            NSGA2_run()
            file_resource = {
                "solution_txt_path": str(DATA_DIR / "pareto_solutions.txt"),
                "pareto_front_png": str(DATA_DIR / "pareto_front.png"),
            }
        elif optimizer == "rl":
            train_and_optimize()
            file_resource = {"solution_txt_path": str(DATA_DIR / "rl_solutions_sb3.txt")}
        else:
            raise ValueError(f"不支持的 optimizer：{optimizer}")
        cost = round(time.time() - started, 2)

        data = {
            "task_info": {
                "model_id": resolved.get("model_id"),
                "optimizer": optimizer,
                "mode": resolved.get("mode"),
                "decision_var_names": resolved.get("decision_var_names"),
                "objective_names": resolved.get("objective_names"),
                "result_columns": result.get("columns") if constraint_check is not None else None,
                "total_generation": (resolved.get("optimizer_config") or {}).get("n_gen"),
                "pop_size": (resolved.get("optimizer_config") or {}).get("pop_size"),
                "run_time_sec": cost,
            },
            "file_resource": file_resource,
        }
        if constraint_check is not None:
            data["constraint_check"] = constraint_check
        task_store.update(task_id, stage="optimize", status="finished", data=data)
        return {"code": 0, "msg": "优化计算完成", "task_id": task_id, "data": data}
    except Exception as exc:
        logger.error(f"多目标优化失败：{exc}")
        task_store.update(task_id, stage="optimize", status="failed")
        return {"code": 1, "msg": f"优化失败：{exc}", "task_id": task_id, "data": {}}


def query_optimization_status(task_id: str) -> Dict[str, Any]:
    """查询多目标优化任务状态（从 state.json 读取）。"""
    state = task_store.load(task_id)
    if state is None:
        return {"code": 1, "msg": "优化任务不存在", "task_id": task_id, "data": {}}
    return {"code": 0, "msg": "ok", "task_id": task_id,
            "data": {"status": state.get("status"), "stage": state.get("stage"),
                     **(state.get("data") or {})}}


__all__ = ["run_optimization", "query_optimization_status"]
