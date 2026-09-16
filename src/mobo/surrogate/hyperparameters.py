"""代理模型超参数规范、默认值与校验。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MODEL_PARAMETER_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "PRG": {
        "degree": {"type": "integer", "default": 2, "minimum": 1, "maximum": 10,
                   "label": "多项式阶数"},
        "include_bias": {"type": "boolean", "default": False, "label": "包含偏置特征"},
        "fit_intercept": {"type": "boolean", "default": True, "label": "拟合截距"},
    },
    "SVR": {
        "kernel": {"type": "string", "default": "rbf",
                   "choices": ["linear", "poly", "rbf", "sigmoid"], "label": "核函数"},
        "C": {"type": "number", "default": 1.0, "exclusive_minimum": 0, "label": "惩罚系数 C"},
        "epsilon": {"type": "number", "default": 0.1, "minimum": 0, "label": "Epsilon"},
        "gamma": {"type": "number_or_string", "default": "scale",
                  "choices": ["scale", "auto"], "exclusive_minimum": 0, "label": "Gamma"},
        "degree": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10,
                   "label": "Poly 核阶数"},
        "coef0": {"type": "number", "default": 0.0, "label": "核函数常数项"},
        "shrinking": {"type": "boolean", "default": True, "label": "启用收缩启发式"},
        "tol": {"type": "number", "default": 0.001, "exclusive_minimum": 0, "label": "收敛容差"},
        "max_iter": {"type": "integer", "default": -1, "label": "最大迭代次数（-1 不限制）"},
    },
    "RF": {
        "n_estimators": {"type": "integer", "default": 300, "minimum": 1, "maximum": 5000,
                         "label": "决策树数量"},
        "criterion": {"type": "string", "default": "squared_error",
                      "choices": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                      "label": "划分准则"},
        "max_depth": {"type": "integer", "default": None, "nullable": True, "minimum": 1,
                      "label": "最大深度"},
        "min_samples_split": {"type": "integer_or_number", "default": 2,
                              "label": "节点最少拆分样本"},
        "min_samples_leaf": {"type": "integer_or_number", "default": 1,
                             "label": "叶节点最少样本"},
        "max_features": {"type": "number_or_string", "default": 1.0, "nullable": True,
                         "choices": ["sqrt", "log2"], "label": "每次划分最大特征"},
        "bootstrap": {"type": "boolean", "default": True, "label": "Bootstrap 抽样"},
        "max_samples": {"type": "integer_or_number", "default": None, "nullable": True,
                        "label": "Bootstrap 最大样本"},
        "random_state": {"type": "integer", "default": 42, "nullable": True,
                         "label": "随机种子"},
        "n_jobs": {"type": "integer", "default": -1, "label": "并行任务数"},
    },
    "KM": {
        "alpha": {"type": "number", "default": 0.1, "minimum": 0, "label": "观测噪声 Alpha"},
        "n_restarts_optimizer": {"type": "integer", "default": 20, "minimum": 0,
                                 "label": "优化器重启次数"},
        "normalize_y": {"type": "boolean", "default": False, "label": "标准化目标值"},
        "random_state": {"type": "integer", "default": 42, "nullable": True,
                         "label": "随机种子"},
        "constant_value": {"type": "number", "default": 1.0, "exclusive_minimum": 0,
                           "label": "常数核初值"},
        "constant_lower": {"type": "number", "default": 0.001, "exclusive_minimum": 0,
                           "label": "常数核下界"},
        "constant_upper": {"type": "number", "default": 1000000.0, "exclusive_minimum": 0,
                           "label": "常数核上界"},
        "length_scale": {"type": "number", "default": 1.0, "exclusive_minimum": 0,
                         "label": "RBF 长度尺度"},
        "length_scale_lower": {"type": "number", "default": 0.1, "exclusive_minimum": 0,
                               "label": "长度尺度下界"},
        "length_scale_upper": {"type": "number", "default": 10000.0, "exclusive_minimum": 0,
                               "label": "长度尺度上界"},
    },
    "DNN": {
        "hidden_layer_1": {"type": "integer", "default": 64, "minimum": 1, "maximum": 4096,
                           "label": "隐藏层 1 神经元"},
        "hidden_layer_2": {"type": "integer", "default": 32, "minimum": 1, "maximum": 4096,
                           "label": "隐藏层 2 神经元"},
        "hidden_layer_3": {"type": "integer", "default": 16, "minimum": 1, "maximum": 4096,
                           "label": "隐藏层 3 神经元"},
        "activation": {"type": "string", "default": "relu",
                       "choices": ["relu", "tanh", "sigmoid", "elu", "selu"],
                       "label": "激活函数"},
        "dropout_1": {"type": "number", "default": 0.2, "minimum": 0, "exclusive_maximum": 1,
                      "label": "隐藏层 1 Dropout"},
        "dropout_2": {"type": "number", "default": 0.1, "minimum": 0, "exclusive_maximum": 1,
                      "label": "隐藏层 2 Dropout"},
        "batch_normalization": {"type": "boolean", "default": True, "label": "批归一化"},
        "learning_rate": {"type": "number", "default": 0.001, "exclusive_minimum": 0,
                          "label": "学习率"},
        "epochs": {"type": "integer", "default": 1000, "minimum": 1, "maximum": 100000,
                   "label": "最大训练轮数"},
        "batch_size": {"type": "integer", "default": 16, "minimum": 1, "label": "批大小"},
        "verbose": {"type": "integer", "default": 1, "choices": [0, 1, 2], "label": "日志级别"},
        "patience": {"type": "integer", "default": 50, "minimum": 1, "label": "早停耐心轮数"},
        "reduce_lr_factor": {"type": "number", "default": 0.2, "exclusive_minimum": 0,
                             "exclusive_maximum": 1, "label": "学习率衰减系数"},
        "reduce_lr_patience": {"type": "integer", "default": 5, "minimum": 1,
                               "label": "学习率衰减耐心轮数"},
        "min_lr": {"type": "number", "default": 0.000001, "minimum": 0,
                   "label": "最小学习率"},
    },
}


def parameter_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    """返回可安全序列化的超参数元数据副本。"""
    return deepcopy(MODEL_PARAMETER_SCHEMAS)


def default_model_params(family: str) -> dict[str, Any]:
    schema = _schema(family)
    return {name: deepcopy(spec["default"]) for name, spec in schema.items()}


def normalize_model_params(family: str, values: Any) -> dict[str, Any]:
    """校验覆盖值并返回包含默认值的完整有效参数。"""
    schema = _schema(family)
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError(f"{family} 超参数必须是 JSON 对象")
    unknown = sorted(set(values) - set(schema))
    if unknown:
        raise ValueError(f"{family} 不支持的超参数：{', '.join(unknown)}")
    normalized = default_model_params(family)
    for name, value in values.items():
        normalized[name] = _validate_value(family, name, value, schema[name])
    _validate_relations(family, normalized)
    return normalized


def _schema(family: str) -> dict[str, dict[str, Any]]:
    try:
        return MODEL_PARAMETER_SCHEMAS[family]
    except KeyError as exc:
        raise ValueError(f"不支持的代理模型：{family}") from exc


def _validate_value(family: str, name: str, value: Any, spec: dict[str, Any]) -> Any:
    if value is None:
        if spec.get("nullable"):
            return None
        raise ValueError(f"{family}.{name} 不能为 null")
    kind = spec["type"]
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{family}.{name} 必须是布尔值")
        result = value
    elif kind == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{family}.{name} 必须是整数")
        result = value
    elif kind == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{family}.{name} 必须是数值")
        result = float(value)
    elif kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"{family}.{name} 必须是字符串")
        result = value
    elif kind == "integer_or_number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{family}.{name} 必须是整数或小数")
        result = value if isinstance(value, int) else float(value)
    elif kind == "number_or_string":
        if isinstance(value, str):
            result = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result = float(value)
        else:
            raise ValueError(f"{family}.{name} 必须是数值或字符串")
    else:
        raise ValueError(f"{family}.{name} 使用了未知参数类型")
    choices = spec.get("choices")
    if choices is not None and isinstance(result, str) and result not in choices:
        raise ValueError(f"{family}.{name} 仅支持：{', '.join(map(str, choices))}")
    if choices is not None and not isinstance(result, str) and kind in {"integer", "number"}:
        if result not in choices:
            raise ValueError(f"{family}.{name} 仅支持：{', '.join(map(str, choices))}")
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        _validate_range(family, name, result, spec)
    return result


def _validate_range(family: str, name: str, value: float, spec: dict[str, Any]) -> None:
    if "minimum" in spec and value < spec["minimum"]:
        raise ValueError(f"{family}.{name} 必须 >= {spec['minimum']}")
    if "maximum" in spec and value > spec["maximum"]:
        raise ValueError(f"{family}.{name} 必须 <= {spec['maximum']}")
    if "exclusive_minimum" in spec and value <= spec["exclusive_minimum"]:
        raise ValueError(f"{family}.{name} 必须 > {spec['exclusive_minimum']}")
    if "exclusive_maximum" in spec and value >= spec["exclusive_maximum"]:
        raise ValueError(f"{family}.{name} 必须 < {spec['exclusive_maximum']}")


def _validate_relations(family: str, params: dict[str, Any]) -> None:
    if family == "SVR" and params["max_iter"] != -1 and params["max_iter"] < 1:
        raise ValueError("SVR.max_iter 只能为 -1 或正整数")
    if family == "RF":
        _validate_sample_threshold("RF.min_samples_split", params["min_samples_split"], 2)
        _validate_sample_threshold("RF.min_samples_leaf", params["min_samples_leaf"], 1)
        if params["n_jobs"] == 0:
            raise ValueError("RF.n_jobs 不能为 0")
        _validate_fraction_or_count("RF.max_features", params["max_features"])
        _validate_fraction_or_count("RF.max_samples", params["max_samples"])
        if params["max_samples"] is not None and not params["bootstrap"]:
            raise ValueError("RF.max_samples 仅能在 bootstrap=true 时设置")
    if family == "KM":
        if params["constant_lower"] >= params["constant_upper"]:
            raise ValueError("KM.constant_lower 必须小于 constant_upper")
        if params["length_scale_lower"] >= params["length_scale_upper"]:
            raise ValueError("KM.length_scale_lower 必须小于 length_scale_upper")


def _validate_sample_threshold(name: str, value: int | float, integer_minimum: int) -> None:
    if isinstance(value, int):
        if value < integer_minimum:
            raise ValueError(f"{name} 的整数值必须 >= {integer_minimum}")
    elif not 0 < value <= 1:
        raise ValueError(f"{name} 的小数值必须在 (0, 1] 范围内")


def _validate_fraction_or_count(name: str, value: Any) -> None:
    if value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{name} 的整数值必须 >= 1")
    elif not 0 < value <= 1:
        raise ValueError(f"{name} 的小数值必须在 (0, 1] 范围内")


__all__ = [
    "MODEL_PARAMETER_SCHEMAS",
    "default_model_params",
    "normalize_model_params",
    "parameter_catalog",
]
