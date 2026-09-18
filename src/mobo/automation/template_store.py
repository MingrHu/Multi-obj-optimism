"""Persistent user-defined DEFORM task templates and structural validation."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from mobo.common.logging import logger
from mobo.common.paths import AUTOMATION_TEMPLATES_DIR
from mobo.extraction import registry as extraction_registry
from mobo.replacement import registry as replacement_registry

from .config import DeformConfig
from .task_collection import (
    TASK_COLLECTION,
    MultiOperationTaskDefinition,
    SingleOperationTaskDefinition,
    TargetDefinition,
)

TaskDefinition = MultiOperationTaskDefinition | SingleOperationTaskDefinition
_SCHEMA_VERSION = 1
_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def _template_path(task_id: str) -> Path:
    if not _TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError("任务 ID 只能包含英文、数字、点、下划线和短横线")
    return AUTOMATION_TEMPLATES_DIR / f"{task_id}.json"


def _target_to_dict(target: TargetDefinition) -> dict[str, Any]:
    return {
        "output_name": target.output_name,
        "target_name": target.target_name,
        "object_name": target.object_name,
        "operation_indices": list(target.operation_indices),
        "select_component": target.select_component,
        "in_progress": target.in_progress,
        "workpiece_type": target.workpiece_type,
        "description": target.description,
        "verified": target.verified,
    }


def definition_to_dict(definition: TaskDefinition) -> dict[str, Any]:
    """Convert a task definition into the versioned on-disk representation."""
    common = {
        "schema_version": _SCHEMA_VERSION,
        "task_id": definition.task_id,
        "name": definition.name,
        "workspace": str(definition.workspace),
        "targets": [_target_to_dict(target) for target in definition.targets],
    }
    if isinstance(definition, MultiOperationTaskDefinition):
        return common | {
            "kind": "multi",
            "operations": definition.operation_configs(),
        }
    return common | {
        "kind": "single",
        "template_key": definition.template_key,
        "parameters": [dict(parameter) for parameter in definition.parameters],
    }


def _targets_from_dict(items: list[dict[str, Any]]) -> tuple[TargetDefinition, ...]:
    return tuple(
        TargetDefinition(
            output_name=str(item["output_name"]),
            target_name=str(item["target_name"]),
            object_name=str(item["object_name"]),
            operation_indices=tuple(int(value) for value in item["operation_indices"]),
            select_component=(
                None if item.get("select_component") is None
                else int(item["select_component"])
            ),
            in_progress=bool(item.get("in_progress", False)),
            workpiece_type=str(item.get("workpiece_type") or "generic"),
            description=str(item.get("description") or ""),
            verified=bool(item.get("verified", True)),
        )
        for item in items
    )


def definition_from_dict(payload: dict[str, Any]) -> TaskDefinition:
    """Restore a task definition from persisted JSON and validate its schema."""
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"不支持的任务模板版本: {payload.get('schema_version')}")
    task_id = str(payload["task_id"])
    name = str(payload["name"])
    workspace = Path(str(payload["workspace"]))
    targets = _targets_from_dict(list(payload.get("targets") or []))
    if payload.get("kind") == "single":
        return SingleOperationTaskDefinition(
            task_id=task_id,
            name=name,
            workspace=workspace,
            template_key=str(payload["template_key"]),
            parameters=tuple(dict(item) for item in payload.get("parameters") or []),
            targets=targets,
        )
    if payload.get("kind") == "multi":
        return MultiOperationTaskDefinition(
            task_id=task_id,
            name=name,
            workspace=workspace,
            operations=tuple(dict(item) for item in payload.get("operations") or []),
            targets=targets,
        )
    raise ValueError(f"不支持的任务模板类型: {payload.get('kind')}")


def _validate_key_file(path_value: str, label: str, *, transition: bool = False) -> None:
    path = Path(path_value)
    if path.suffix.lower() != ".key":
        raise ValueError(f"{label}必须是 .KEY 文件")
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{label}不是可读取的 UTF-8 KEY 文件: {path}") from exc
    if not text.strip():
        raise ValueError(f"{label}不能为空: {path}")
    if transition and (
        "Data for Object # 1" not in text or "Inter-Object Data" not in text
    ):
        raise ValueError(f"{label}缺少多工步换模所需的对象或接触数据段")


def _parameter_objects(parameter: dict[str, Any]) -> list[str]:
    if parameter.get("objects"):
        return [str(value) for value in parameter["objects"]]
    if parameter.get("object"):
        return [str(parameter["object"])]
    return []


def _validate_parameters(parameters: list[dict[str, Any]], label: str) -> None:
    for index, parameter in enumerate(parameters, 1):
        name = str(parameter.get("name") or "")
        if replacement_registry.resolve(name) is None:
            raise ValueError(f"{label}参数 {index} 使用了未注册的参数类型: {name}")
        objects = _parameter_objects(parameter)
        if not objects:
            raise ValueError(f"{label}参数 {index} 缺少作用对象")
        invalid_objects = [
            value for value in objects
            if DeformConfig.get_object_id(value) is None
            and not (name == "ring_die_temperature" and value == "ring_dies")
        ]
        if invalid_objects:
            raise ValueError(f"{label}参数 {index} 包含未知对象: {', '.join(invalid_objects)}")
        value_range = parameter.get("range")
        if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
            raise ValueError(f"{label}参数 {index} 必须提供下限和上限")
        low, high = (float(value_range[0]), float(value_range[1]))
        if low > high:
            raise ValueError(f"{label}参数 {index} 下限不能大于上限")


def _validate_targets(targets: tuple[TargetDefinition, ...], operation_count: int) -> None:
    if not targets:
        raise ValueError("至少需要配置一个输出目标")
    output_names = [target.output_name for target in targets]
    if any(not name for name in output_names):
        raise ValueError("输出列名不能为空")
    if len(set(output_names)) != len(output_names):
        raise ValueError("输出列名不能重复")
    for target in targets:
        if DeformConfig.get_object_id(target.object_name) is None:
            raise ValueError(f"输出目标 {target.output_name} 使用了未知对象: {target.object_name}")
        if not target.operation_indices or any(
            index < 1 or index > operation_count for index in target.operation_indices
        ):
            raise ValueError(f"输出目标 {target.output_name} 引用了不存在的工步")
        try:
            extraction_registry.resolve(target.workpiece_type, target.target_name)
        except KeyError as exc:
            raise ValueError(
                f"输出目标 {target.output_name} 没有可用的提取能力: "
                f"{target.workpiece_type}/{target.target_name}"
            ) from exc


def validate_template(definition: TaskDefinition, *, require_files: bool = True) -> None:
    """Validate a user template before persistence or execution."""
    _template_path(definition.task_id)
    if not definition.name.strip():
        raise ValueError("任务名称不能为空")
    if not str(definition.workspace).strip():
        raise ValueError("任务工作区不能为空")
    if isinstance(definition, SingleOperationTaskDefinition):
        if require_files:
            _validate_key_file(definition.template_key, "单工步 KEY 模板")
        _validate_parameters(list(definition.parameters), "单工步")
        _validate_targets(definition.targets, 1)
        return
    if len(definition.operations) < 2:
        raise ValueError("多工步任务至少需要两个工步")
    total_parameters = 0
    for index, operation in enumerate(definition.operations, 1):
        if require_files:
            _validate_key_file(
                str(operation.get("template_key") or ""),
                f"工步 {index} KEY 模板",
                transition=index > 1,
            )
        parameters = list(operation.get("parameters") or [])
        total_parameters += len(parameters)
        _validate_parameters(parameters, f"工步 {index}")
        offset = operation.get("position_offset")
        if offset is not None and (not isinstance(offset, (list, tuple)) or len(offset) != 3):
            raise ValueError(f"工步 {index} 的位置偏移必须包含 X、Y、Z 三个数值")
    if total_parameters == 0:
        raise ValueError("至少需要配置一个输入参数")
    _validate_targets(definition.targets, len(definition.operations))


def list_templates(*, multi: bool | None = None) -> list[TaskDefinition]:
    """List valid persisted user templates, optionally filtered by task kind."""
    if not AUTOMATION_TEMPLATES_DIR.is_dir():
        return []
    definitions = []
    for path in sorted(AUTOMATION_TEMPLATES_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            definition = definition_from_dict(payload)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            logger.warning(f"忽略无效用户任务模板 {path}: {exc}")
            continue
        if multi is None or isinstance(definition, MultiOperationTaskDefinition) == multi:
            definitions.append(definition)
    return definitions


def load_template(task_id: str) -> TaskDefinition:
    """Load one persisted user template by its unique task ID."""
    path = _template_path(task_id)
    if not path.is_file():
        raise FileNotFoundError(f"用户任务模板不存在: {task_id}")
    return definition_from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_template(definition: TaskDefinition, *, overwrite: bool = False) -> str:
    """Validate and atomically persist a user template."""
    validate_template(definition)
    if definition.task_id in TASK_COLLECTION:
        raise ValueError("内置任务模板 ID 不可覆盖，请使用新的任务 ID")
    builtin_name_owner = next(
        (item.task_id for item in TASK_COLLECTION.values() if item.name == definition.name),
        None,
    )
    if builtin_name_owner is not None:
        raise ValueError("内置任务模板名称不可重复，请使用新的任务名称")
    path = _template_path(definition.task_id)
    if path.exists() and not overwrite:
        raise FileExistsError(f"任务模板 ID 已存在: {definition.task_id}")
    duplicate_name = next(
        (
            item.task_id for item in list_templates()
            if item.name == definition.name and item.task_id != definition.task_id
        ),
        None,
    )
    if duplicate_name is not None:
        raise ValueError(f"任务模板名称已存在: {definition.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = definition_to_dict(definition)
    fd, temporary = tempfile.mkstemp(
        prefix=".template_", suffix=".json", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return str(path)


def delete_template(task_id: str) -> None:
    """Delete one user template; built-in task IDs can never be removed."""
    if task_id in TASK_COLLECTION:
        raise ValueError("内置任务模板不能删除")
    path = _template_path(task_id)
    if not path.is_file():
        raise FileNotFoundError(f"用户任务模板不存在: {task_id}")
    path.unlink()


__all__ = [
    "TaskDefinition",
    "definition_from_dict",
    "definition_to_dict",
    "validate_template",
    "list_templates",
    "load_template",
    "save_template",
    "delete_template",
]
