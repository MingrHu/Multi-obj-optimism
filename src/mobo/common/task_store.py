"""任务状态持久化。

统一把任务的关键信息（输入参数、阶段状态、结果）以 JSON 落盘到
``TASKS_DIR/<task_id>/``。默认文件为 ``state.json``，需要隔离分片的流程可指定命名状态
文件。三类流程（automation / surrogate / optimization）共用本模块。

state.json 结构（约定，字段按流程可选）::

    {
        "task_id": "...",
        "kind": "automation" | "surrogate" | "optimization",
        "created_at": "YYYY-MM-DD HH:MM:SS",
        "updated_at": "YYYY-MM-DD HH:MM:SS",
        "status": "running" | "finished" | "failed",
        "stage": "<当前阶段名>",
        "req": { ... },        # 原始输入参数（用于续跑）
        "data": { ... },       # 阶段产物 / 结果（含文件路径、指标等）
        "history": [ ... ]     # 完整的阶段/状态转移记录（只追加，不覆盖）
    }
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from mobo.common.paths import task_dir

_STATE_FILE = "state.json"
_TASK_ROOT: ContextVar[Path | None] = ContextVar("mobo_task_root", default=None)


def _task_directory(task_id: str) -> Path:
    root = _TASK_ROOT.get()
    return root / task_id if root is not None else task_dir(task_id)


@contextmanager
def task_workspace(path: str | os.PathLike[str]):
    """Route task states into one DOE run without changing legacy CLI callers."""
    token = _TASK_ROOT.set(Path(path))
    try:
        yield
    finally:
        _TASK_ROOT.reset(token)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def state_path(task_id: str, state_name: str = _STATE_FILE) -> str:
    """返回任务状态文件的完整路径。"""
    if os.path.basename(state_name) != state_name:
        raise ValueError(f"状态文件名不能包含目录: {state_name}")
    return os.path.join(str(_task_directory(task_id)), state_name)


def exists(task_id: str, state_name: str = _STATE_FILE) -> bool:
    """判断任务是否已有持久化状态。"""
    return os.path.exists(state_path(task_id, state_name))


def load(task_id: str, state_name: str = _STATE_FILE) -> Optional[Dict[str, Any]]:
    """读取任务状态；不存在返回 None。"""
    path = state_path(task_id, state_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(state: Dict[str, Any], state_name: str = _STATE_FILE) -> str:
    """原子写入任务状态并刷新 ``updated_at``。

    :param state: 含 ``task_id`` 的完整状态字典
    :param state_name: 状态文件名，默认 ``state.json``
    :return: 状态文件路径
    """
    task_id = state["task_id"]
    directory = str(_task_directory(task_id))
    os.makedirs(directory, exist_ok=True)
    state["updated_at"] = _now()

    fd, tmp = tempfile.mkstemp(prefix=".state_", suffix=".json", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, state_path(task_id, state_name))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return state_path(task_id, state_name)


def init_state(task_id: str, kind: str, req: Dict[str, Any],
               state_name: str = _STATE_FILE) -> Dict[str, Any]:
    """创建并落盘一个新任务状态（已存在则原样读回）。

    :param task_id: 任务 ID
    :param kind: 流程类型 automation/surrogate/optimization
    :param req: 原始输入参数（用于续跑）
    :param state_name: 状态文件名，默认 ``state.json``
    :return: 状态字典
    """
    current = load(task_id, state_name)
    if current is not None:
        return current
    now = _now()
    state = {
        "task_id": task_id,
        "kind": kind,
        "created_at": now,
        "updated_at": now,
        "status": "running",
        "stage": "init",
        "req": req,
        "data": {},
        "history": [{"stage": "init", "status": "running", "at": now}],
    }
    save(state, state_name)
    return state


def update(task_id: str, *, state_name: str = _STATE_FILE,
           **fields: Any) -> Dict[str, Any]:
    """更新任务状态的顶层字段（``data``/``req`` 做浅合并）并落盘。
    每次更新都会把本次的 ``stage``/``status`` 作为一条记录追加到 ``history``

    :param task_id: 任务 ID
    :param state_name: 状态文件名，默认 ``state.json``
    :param fields: 待更新字段，如 ``status`` / ``stage`` / ``data`` / ``req``
    :return: 更新后的状态字典
    :raises FileNotFoundError: 任务状态不存在
    """
    state = load(task_id, state_name)
    if state is None:
        raise FileNotFoundError(f"任务状态不存在：{task_id}")
    for key, value in fields.items():
        if key in ("data", "req") and isinstance(value, dict):
            merged = dict(state.get(key) or {})
            merged.update(value)
            state[key] = merged
        else:
            state[key] = value
    # 追加一条转移记录
    if "stage" in fields or "status" in fields:
        history = list(state.get("history") or [])
        history.append({
            "stage": state.get("stage"),
            "status": state.get("status"),
            "at": _now(),
        })
        state["history"] = history
    save(state, state_name)
    return state


def resolve_req(task_id: str, kind: str, provided: Dict[str, Any],
                required: Iterable[str], state_name: str = _STATE_FILE) -> Dict[str, Any]:
    """三路解析续跑所需参数：优先用任务记录，其次用传入参数，否则报错。

    合并规则：任务记录 ``req`` 里已有的键沿用记录值；记录没有的传入键采用传入值
    并回填记录（保证记录完整）。``required`` 中的键若在记录与传入里都缺失则报错。
    非 required 的传入键（如溯源用的 ``model_id``）也会一并回填/返回。
    不存在的任务会用可用的传入参数初始化。

    :param task_id: 任务 ID
    :param kind: 流程类型（任务不存在时用于初始化）
    :param provided: 本次调用传入的参数（值为 None 视为未提供）
    :param required: 续跑必需的参数键
    :param state_name: 状态文件名，默认 ``state.json``
    :return: 合并后的完整 req 字典（记录值优先）
    :raises ValueError: 某个必需参数在记录与传入中都缺失
    """
    provided = {k: v for k, v in (provided or {}).items() if v is not None}
    state = load(task_id, state_name)
    stored = dict(state.get("req") or {}) if state is not None else {}

    # 记录优先合并；记录缺失的传入键需要回填
    resolved = {**provided, **stored}
    backfill = {k: v for k, v in provided.items() if k not in stored}

    missing = [k for k in required if k not in resolved]
    if missing:
        raise ValueError(f"续跑缺少必要参数（记录与传入均无）：{', '.join(missing)}")

    if state is None:
        init_state(task_id, kind, resolved, state_name)
    elif backfill:
        update(task_id, state_name=state_name, req=backfill)
    return resolved


__all__ = [
    "state_path",
    "exists",
    "load",
    "save",
    "init_state",
    "update",
    "resolve_req",
    "task_workspace",
]
