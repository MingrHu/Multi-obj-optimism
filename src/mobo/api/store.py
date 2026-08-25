"""DOE 聚合任务的 JSON 持久化。"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mobo.common.paths import DOE_TASKS_DIR
from mobo.common.paths import task_dir as legacy_task_dir

from .errors import ConflictError, NotFoundError

_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
# 可重入锁允许 update_section 在持锁状态下继续调用 update 和 load
_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def validate_id(doe_id: Any) -> str:
    # 严格限制目录名称 防止路径穿越和不同平台下的非法文件名
    if not isinstance(doe_id, str) or not _ID.fullmatch(doe_id):
        raise ValueError("id 只能包含字母、数字、下划线、短横线，且长度为 1-128")
    return doe_id


def task_dir(doe_id: str) -> Path:
    return DOE_TASKS_DIR / validate_id(doe_id)


def _state_file(doe_id: str) -> Path:
    return task_dir(doe_id) / "doe.json"


def _write(state: dict[str, Any]) -> None:
    directory = task_dir(state["id"])
    directory.mkdir(parents=True, exist_ok=True)
    # 先写同目录临时文件再原子替换 避免进程异常留下半截 JSON
    fd, temporary = tempfile.mkstemp(prefix=".doe_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, _state_file(state["id"]))
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def create(payload: dict[str, Any]) -> dict[str, Any]:
    # 调用方未指定标识时生成短 UUID 保证目录名稳定且足够唯一
    doe_id = validate_id(payload.get("id") or f"doe_{uuid.uuid4().hex[:16]}")
    with _LOCK:
        if _state_file(doe_id).exists():
            raise ConflictError(f"DOE 任务已存在：{doe_id}")
        now = _now()
        # 顶层状态用于快速查询 子流程详情分别保存在 sample training optimization
        state = {
            "id": doe_id,
            "name": payload.get("name", doe_id),
            "description": payload.get("description", ""),
            "metadata": payload.get("metadata", {}),
            "status": "created", "stage": "created", "progress": 0,
            "created_at": now, "updated_at": now, "sample": {},
            "training": {
                "status": "not_started", "stage": "not_started",
                "progress": 0, "models": [],
            },
            "optimization": {"status": "not_started"},
        }
        _write(state)
        # 创建固定子目录使同一 DOE 的样本 模型和优化结果互不混放
        for name in ("samples", "models", "training", "optimization"):
            (task_dir(doe_id) / name).mkdir(exist_ok=True)
        return state


def load(doe_id: str) -> dict[str, Any]:
    path = _state_file(doe_id)
    if not path.is_file():
        raise NotFoundError(f"DOE 任务不存在：{doe_id}")
    with _LOCK, path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def update(doe_id: str, **fields: Any) -> dict[str, Any]:
    with _LOCK:
        state = load(doe_id)
        state.update(fields)
        state["updated_at"] = _now()
        _write(state)
        return state


def update_section(doe_id: str, section: str, **fields: Any) -> dict[str, Any]:
    with _LOCK:
        state = load(doe_id)
        # 子区块采用浅合并 保留本次请求未更新的模型列表和历史结果字段
        value = dict(state.get(section) or {})
        value.update(fields)
        return update(doe_id, **{section: value})


def list_all() -> list[dict[str, Any]]:
    if not DOE_TASKS_DIR.is_dir():
        return []
    states = []
    for path in sorted(DOE_TASKS_DIR.glob("*/doe.json")):
        try:
            states.append(load(path.parent.name))
        # 单个损坏或正在被外部占用的状态文件不应阻断整个列表查询
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return states


def delete(doe_id: str) -> None:
    with _LOCK:
        state = load(doe_id)
        # 先清理旧算法服务目录 再删除 DOE 聚合目录 避免残留关联模型和优化记录
        _delete_legacy_artifacts(state)
        shutil.rmtree(task_dir(doe_id))


def reset_training(doe_id: str) -> None:
    with _LOCK:
        state = load(doe_id)
        # 模型快照和旧训练任务同时删除 采样文件及优化记录保持不变
        for model in state.get("training", {}).get("models", []):
            _remove_legacy_task(model.get("model_id"))
        directory = task_dir(doe_id)
        for name in ("models", "training"):
            target = directory / name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
        update_section(doe_id, "training", status="not_started", stage="not_started",
                       progress=0, models=[], error=None)


def _delete_legacy_artifacts(state: dict[str, Any]) -> None:
    for model in state.get("training", {}).get("models", []):
        _remove_legacy_task(model.get("model_id"))
    result = state.get("optimization", {}).get("result", {})
    _remove_legacy_task(result.get("optimization_id"))


def _remove_legacy_task(task_id: Any) -> None:
    if isinstance(task_id, str) and _ID.fullmatch(task_id):
        path = legacy_task_dir(task_id)
        if path.is_dir():
            shutil.rmtree(path)


__all__ = [
    "create", "delete", "list_all", "load", "reset_training", "task_dir",
    "update", "update_section", "validate_id",
]
