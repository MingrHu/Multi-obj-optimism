"""外部接口请求的解析与校验。

这里使用标准库实现轻量校验，避免为了几个同步 Python 接口引入 Web 框架或
数据校验框架。返回值均为普通字典，可以直接交给现有 service 层。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mobo.common.paths import DATA_DIR


class ApiValidationError(ValueError):
    """请求不符合对外协议。"""


# 历史训练函数当前实际使用的固定参数。底层虽然接收 model_par，但尚未消费它。
# 对外接口只接受这些真实值，避免调用方误以为自定义超参数已经生效。
FIXED_MODEL_PARAMS: dict[int, dict[str, Any]] = {
    0: {"degree": 2},
    1: {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
    2: {"n_estimators": 300, "n_jobs": -1},
    3: {"alpha": 0.1, "n_restarts_optimizer": 20},
    4: {"epochs": 1000, "batch_size": 16, "verbose": 1, "patience": 50},
}

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _as_object(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ApiValidationError(f"不是合法的 JSON：{exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ApiValidationError("请求必须是 JSON 对象或字典")
    return dict(payload)


def _required(req: dict[str, Any], name: str) -> Any:
    value = req.get(name)
    if value is None:
        raise ApiValidationError(f"缺少必要参数：{name}")
    return value


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiValidationError(f"{name} 必须是整数")
    if minimum is not None and value < minimum:
        raise ApiValidationError(f"{name} 必须大于等于 {minimum}")
    return value


def _name_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise ApiValidationError(f"{name} 必须是非空字符串数组")
    if len(set(value)) != len(value):
        raise ApiValidationError(f"{name} 不能包含重复名称")
    return list(value)


def validate_task_id(value: Any, name: str, prefix: str) -> str:
    """校验外部可传入的任务 ID，阻止路径穿越和错误流程前缀。"""
    if not isinstance(value, str) or not value.startswith(prefix) or not _ID_PATTERN.fullmatch(value):
        raise ApiValidationError(f"{name} 必须以 {prefix} 开头，且只能包含字母、数字、_、-")
    return value


def normalize_surrogate_request(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """解析代理模型训练请求，并兼容仓库历史字段别名。"""
    req = _as_object(payload)
    model_index = _integer(_required(req, "model_index"), "model_index")
    if model_index not in FIXED_MODEL_PARAMS:
        raise ApiValidationError("model_index 仅支持 0(PRG)/1(SVR)/2(RF)/3(KM)/4(DNN)")

    data_file = Path(str(_required(req, "data_file"))).expanduser().resolve()
    if not data_file.is_file():
        raise ApiValidationError(f"data_file 不存在或不是文件：{data_file}")

    raw_names = req.get("all_var_list", req.get("vars_out"))
    all_var_list = _name_list(raw_names, "all_var_list")
    raw_count = req.get("input_var_count", req.get("n_vars"))
    input_var_count = _integer(raw_count, "input_var_count", minimum=1)
    if input_var_count >= len(all_var_list):
        raise ApiValidationError("input_var_count 必须小于 all_var_list 长度，且至少保留一个目标列")

    supplied_params = req.get("params", req.get("biz_params", {}))
    if not isinstance(supplied_params, Mapping):
        raise ApiValidationError("params 必须是 JSON 对象")
    actual_params = FIXED_MODEL_PARAMS[model_index]
    unknown = set(supplied_params) - set(actual_params)
    if unknown:
        raise ApiValidationError(f"当前模型不支持参数：{', '.join(sorted(unknown))}")
    ineffective = {
        key: value for key, value in supplied_params.items()
        if value != actual_params[key]
    }
    if ineffective:
        detail = ", ".join(f"{key}={value!r}" for key, value in actual_params.items())
        raise ApiValidationError(
            f"当前历史训练实现尚不支持自定义超参数；实际固定配置为：{detail}"
        )

    normalized = {
        "data_file": str(data_file),
        "vars_out": all_var_list,
        "n_vars": input_var_count,
        "model_index": model_index,
        "biz_params": dict(actual_params),
    }
    if req.get("model_id") is not None:
        normalized["model_id"] = validate_task_id(req["model_id"], "model_id", "tr_")
    return normalized


def normalize_optimization_request(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """解析参数化 NSGA-II 请求。"""
    req = _as_object(payload)
    model_id = validate_task_id(_required(req, "model_id"), "model_id", "tr_")

    all_var_list = _name_list(_required(req, "all_var_list"), "all_var_list")
    input_var_count = _integer(
        _required(req, "input_var_count"), "input_var_count", minimum=1
    )
    if input_var_count >= len(all_var_list):
        raise ApiValidationError("input_var_count 必须小于 all_var_list 长度")
    input_names = all_var_list[:input_var_count]
    output_names = all_var_list[input_var_count:]

    objective_names = _name_list(_required(req, "objective_names"), "objective_names")
    missing_objectives = set(objective_names) - set(output_names)
    if missing_objectives:
        raise ApiValidationError(
            "objective_names 必须来自 all_var_list 的输出列："
            + ", ".join(sorted(missing_objectives))
        )

    raw_indices = _required(req, "decision_var_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ApiValidationError("decision_var_indices 必须是非空整数数组")
    indices = [_integer(v, "decision_var_indices[]", minimum=0) for v in raw_indices]
    if len(set(indices)) != len(indices) or any(i >= input_var_count for i in indices):
        raise ApiValidationError("decision_var_indices 存在重复值或超出输入列范围")

    expected_decision_names = [input_names[i] for i in indices]
    decision_names = req.get("decision_var_names", expected_decision_names)
    if decision_names != expected_decision_names:
        raise ApiValidationError(
            f"decision_var_names 必须与下标对应：{expected_decision_names}"
        )

    raw_bounds = _required(req, "decision_bounds")
    if not isinstance(raw_bounds, list) or len(raw_bounds) != len(indices):
        raise ApiValidationError("decision_bounds 数量必须与 decision_var_indices 一致")
    bounds = []
    for pos, bound in enumerate(raw_bounds):
        if not isinstance(bound, Mapping) or "lower" not in bound or "upper" not in bound:
            raise ApiValidationError(f"decision_bounds[{pos}] 必须包含 lower 和 upper")
        lower, upper = bound["lower"], bound["upper"]
        if isinstance(lower, bool) or isinstance(upper, bool) or not all(
            isinstance(v, (int, float)) for v in (lower, upper)
        ):
            raise ApiValidationError(f"decision_bounds[{pos}] 的上下限必须是数值")
        if lower >= upper:
            raise ApiValidationError(f"decision_bounds[{pos}] 必须满足 lower < upper")
        bounds.append({"lower": float(lower), "upper": float(upper), "desc": bound.get("desc")})

    raw_config = req.get("objective_config") or [
        {"name": name, "minimize": True} for name in objective_names
    ]
    if not isinstance(raw_config, list) or len(raw_config) != len(objective_names):
        raise ApiValidationError("objective_config 数量必须与 objective_names 一致")
    objective_config = []
    for pos, config in enumerate(raw_config):
        if not isinstance(config, Mapping):
            raise ApiValidationError(f"objective_config[{pos}] 必须是对象")
        name = config.get("name")
        minimize = config.get("minimize", True)
        if name != objective_names[pos] or not isinstance(minimize, bool):
            raise ApiValidationError("objective_config 必须按 objective_names 顺序配置 name/minimize")
        objective_config.append({"name": name, "minimize": minimize})

    constraints = []
    raw_constraints = req.get("constraints", [])
    if not isinstance(raw_constraints, list):
        raise ApiValidationError("constraints 必须是数组")
    for pos, constraint in enumerate(raw_constraints):
        if not isinstance(constraint, Mapping):
            raise ApiValidationError(f"constraints[{pos}] 必须是对象")
        target = constraint.get("target_obj")
        kind = constraint.get("constraint_kind")
        value = constraint.get("limit_value")
        if target not in objective_names:
            raise ApiValidationError(f"constraints[{pos}].target_obj 不在 objective_names 中")
        if kind not in ("upper", "lower"):
            raise ApiValidationError(f"constraints[{pos}].constraint_kind 仅支持 upper/lower")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ApiValidationError(f"constraints[{pos}].limit_value 必须是数值")
        constraints.append({
            "target_obj": target,
            "constraint_kind": kind,
            "limit_value": float(value),
        })

    raw_optimizer = req.get("optimizer_config", {})
    if not isinstance(raw_optimizer, Mapping):
        raise ApiValidationError("optimizer_config 必须是对象")
    defaults = {
        "pop_size": 100,
        "n_offsprings": 100,
        "eliminate_duplicates": True,
        "n_gen": 200,
        "seed": 42,
    }
    unknown = set(raw_optimizer) - set(defaults)
    if unknown:
        raise ApiValidationError(f"optimizer_config 不支持参数：{', '.join(sorted(unknown))}")
    optimizer_config = {**defaults, **raw_optimizer}
    for key in ("pop_size", "n_offsprings", "n_gen"):
        optimizer_config[key] = _integer(optimizer_config[key], f"optimizer_config.{key}", minimum=1)
    optimizer_config["seed"] = _integer(optimizer_config["seed"], "optimizer_config.seed")
    if not isinstance(optimizer_config["eliminate_duplicates"], bool):
        raise ApiValidationError("optimizer_config.eliminate_duplicates 必须是布尔值")

    output_config = req.get("output_config", {})
    if not isinstance(output_config, Mapping):
        raise ApiValidationError("output_config 必须是对象")
    unknown_output = set(output_config) - {"pareto_txt_path"}
    if unknown_output:
        raise ApiValidationError(f"output_config 不支持参数：{', '.join(sorted(unknown_output))}")
    normalized_output = dict(output_config)
    if normalized_output.get("pareto_txt_path") is not None:
        output_path = Path(str(normalized_output["pareto_txt_path"])).expanduser().resolve()
        data_directory = DATA_DIR.resolve()
        if not output_path.is_relative_to(data_directory):
            raise ApiValidationError(
                f"output_config.pareto_txt_path 必须位于数据目录内：{data_directory}"
            )
        normalized_output["pareto_txt_path"] = str(output_path)

    normalized = {
        "model_id": model_id,
        "objective_names": objective_names,
        "input_var_count": input_var_count,
        "all_var_list": all_var_list,
        "decision_var_indices": indices,
        "decision_var_names": expected_decision_names,
        "decision_bounds": bounds,
        "constraints": constraints,
        "objective_config": objective_config,
        "optimizer_config": optimizer_config,
        "output_config": normalized_output,
        "optimizer": "nsga2",
    }
    if req.get("task_id") is not None:
        normalized["task_id"] = validate_task_id(req["task_id"], "task_id", "opt_")
    return normalized


__all__ = [
    "ApiValidationError",
    "FIXED_MODEL_PARAMS",
    "validate_task_id",
    "normalize_surrogate_request",
    "normalize_optimization_request",
]
