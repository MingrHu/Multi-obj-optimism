"""对外协议门面。

公开函数接受 JSON 字符串或字典，始终返回可 JSON 序列化的协议字典。这里不启动
HTTP 服务；需要 REST/RPC 时可让任意传输框架直接调用这些函数。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mobo.optimization.service import query_optimization_status
from mobo.optimization.service import run_optimization as _run_optimization
from mobo.surrogate.service import query_model_status
from mobo.surrogate.service import train_surrogate as _train_surrogate

from .validation import (
    ApiValidationError,
    normalize_optimization_request,
    normalize_surrogate_request,
    validate_task_id,
)


def train_surrogate(request: Mapping[str, Any] | str) -> dict[str, Any]:
    """校验并同步训练一个代理模型。"""
    try:
        req = normalize_surrogate_request(request)
    except ApiValidationError as exc:
        return {"code": 1, "msg": f"请求参数错误：{exc}", "model_id": None, "data": {}}
    return _train_surrogate(**req)


def run_optimization(request: Mapping[str, Any] | str) -> dict[str, Any]:
    """校验并同步执行一次参数化 NSGA-II 优化。"""
    try:
        normalized = normalize_optimization_request(request)
    except ApiValidationError as exc:
        return {"code": 1, "msg": f"请求参数错误：{exc}", "task_id": None, "data": {}}

    task_id = normalized.pop("task_id", None)
    optimizer = normalized.pop("optimizer")
    return _run_optimization(normalized, optimizer=optimizer, task_id=task_id)


def query_task(task_id: str) -> dict[str, Any]:
    """按 ID 前缀查询代理训练或优化任务。"""
    try:
        if isinstance(task_id, str) and task_id.startswith("tr_"):
            return query_model_status(validate_task_id(task_id, "task_id", "tr_"))
        if isinstance(task_id, str) and task_id.startswith("opt_"):
            return query_optimization_status(validate_task_id(task_id, "task_id", "opt_"))
        raise ApiValidationError("无法识别任务 ID；仅支持 tr_ 或 opt_ 前缀")
    except ApiValidationError as exc:
        return {"code": 1, "msg": str(exc), "data": {}}


__all__ = ["train_surrogate", "run_optimization", "query_task"]
