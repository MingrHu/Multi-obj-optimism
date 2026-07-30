"""多目标优化服务层。

按 ``interface_protocol.md`` 的协议，对外提供以 ``task_id``（opt_ 前缀）为主键的优化
接口：优化前落盘请求参数（含代理模型 ``model_id`` 溯源），优化后把响应（输出文件路径、
任务信息）写入 ``TASKS_DIR/<task_id>/state.json``。

底层 :func:`~mobo.optimization.ga.run.NSGA2_run` 与
:func:`~mobo.optimization.rl.run.train_and_optimize` 目前为无参演示实现（超参硬编码），
本层负责状态记录与结果落盘，不改动其函数体；参数化优化留待协议进一步细化。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

from mobo.common import task_store
from mobo.common.logging import logger
from mobo.common.paths import DATA_DIR
from mobo.optimization.ga.run import NSGA2_run
from mobo.optimization.rl.run import train_and_optimize

_KIND = "optimization"


def _new_task_id() -> str:
    return "opt_" + datetime.now().strftime("%Y%m%d_%H%M_%S%f")[:-3]


def run_optimization(
    req: Optional[Dict[str, Any]] = None,
    optimizer: str = "nsga2",
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """运行多目标优化并把请求/响应落盘到 state.json。

    :param req: 协议输入参数（model_id/objective_names/constraints 等，用于溯源）
    :param optimizer: ``"nsga2"``（GA）或 ``"rl"``（PPO）
    :param task_id: 复用已有 task_id（None 则新建）
    :return: 协议 resp 字典（code/msg/task_id/data）
    """
    req = req or {}
    task_id = task_id or _new_task_id()

    task_store.init_state(task_id, _KIND, {"optimizer": optimizer, **req})
    task_store.update(task_id, stage="optimize", status="running")

    try:
        started = time.time()
        if optimizer == "nsga2":
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
            "task_info": {"model_id": req.get("model_id"), "optimizer": optimizer,
                          "run_time_sec": cost},
            "file_resource": file_resource,
        }
        task_store.update(task_id, stage="optimize", status="finished", data=data)
        return {"code": 0, "msg": "多目标优化计算完成", "task_id": task_id, "data": data}
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
