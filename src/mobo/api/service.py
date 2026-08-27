"""DOE API 的实际处理层，负责持久化并适配现有算法模块。"""

from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import store
from .errors import ApiError, ConflictError
from .runtime import registry

MODEL_FAMILIES = {0: "PRG", 1: "SVR", 2: "RF", 3: "KM", 4: "DNN"}


# 所有需要 DOE 标识的 POST 请求先经过同一入口校验
def _require_id(payload: dict[str, Any]) -> str:
    try:
        return store.validate_id(payload.get("id"))
    except ValueError as exc:
        raise ApiError(str(exc)) from exc


def add_doe(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload.get("metadata", {}), dict):
        raise ApiError("metadata 必须是 JSON 对象")
    return store.create(payload)


def list_doe() -> list[dict[str, Any]]:
    return store.list_all()


def delete_doe(payload: dict[str, Any]) -> str:
    doe_id = _require_id(payload)
    # 删除前必须保证后台任务已经退出 避免线程继续向已删除目录写文件
    if registry.running(doe_id, "training") or registry.running(doe_id, "optimization"):
        raise ConflictError("任务正在运行，请先中止")
    store.delete(doe_id)
    return doe_id


def generate_sample(payload: dict[str, Any]) -> dict[str, Any]:
    # 算法模块延迟导入 使列表查询等轻量接口无需加载数值计算依赖
    from mobo.automation.sampling import generate_samples

    doe_id = _require_id(payload)
    store.load(doe_id)
    method = payload.get("method", "lhs")
    ranges = _normalize_ranges(payload.get("param_ranges"))
    n_samples = payload.get("n_samples", 0)
    levels = payload.get("level_nums", [])
    if method == "lhs" and (not isinstance(n_samples, int) or n_samples < 1):
        raise ApiError("lhs 的 n_samples 必须是正整数")
    # 样本强制写入当前 DOE 的 samples 子目录 避免污染共享数据目录
    output = generate_samples(
        doe_id, method, ranges, str(store.task_dir(doe_id) / "samples"), n_samples, levels
    )
    sample = {
        "method": method, "param_ranges": ranges, "sample_file": output,
        "columns": list(ranges),
    }
    store.update(doe_id, status="ready", stage="sample_generated", progress=10)
    store.update_section(doe_id, "sample", **sample)
    return sample


def generate_training_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    # 合成数据仅用于接口联调和完整优化流程演示 不替代真实仿真数据
    import numpy as np

    doe_id = _require_id(payload)
    store.load(doe_id)
    ranges = _normalize_ranges(payload.get("param_ranges"))
    input_names = payload.get("input_names") or list(ranges)
    target_names = payload.get("target_names")
    n_samples = payload.get("n_samples", 100)
    seed = payload.get("seed", 42)
    noise_ratio = payload.get("noise_ratio", 0.0)
    _validate_dataset_request(input_names, target_names, ranges, n_samples, seed, noise_ratio)

    rng = np.random.default_rng(seed)
    lower = np.array([ranges[name][0] for name in input_names], dtype=float)
    upper = np.array([ranges[name][1] for name in input_names], dtype=float)
    # 先在单位区间生成可复现样本 再映射到每个工艺参数的真实范围
    normalized = rng.uniform(0.0, 1.0, size=(n_samples, len(input_names)))
    inputs = lower + normalized * (upper - lower)
    targets = _build_demo_targets(normalized, len(target_names), noise_ratio, rng) # type: ignore
    output = store.task_dir(doe_id) / "training" / "demo_training_dataset.tsv"
    np.savetxt(output, np.column_stack([inputs, targets]), delimiter="\t", fmt="%.8f")

    data = {
        "data_file": str(output), "all_var_list": [*input_names, *target_names], # type: ignore
        "input_var_count": len(input_names), "sample_count": n_samples,
        "input_names": input_names, "target_names": target_names,
    }
    store.update_section(doe_id, "training", dataset=data)
    return data


def _validate_dataset_request(input_names, target_names, ranges, n_samples, seed, noise_ratio):
    if not isinstance(input_names, list) or not all(isinstance(name, str) for name in input_names):
        raise ApiError("input_names 必须是字符串数组")
    if len(input_names) != len(set(input_names)) or set(input_names) != set(ranges):
        raise ApiError("input_names 必须与param_ranges包含相同参数")
    if not isinstance(target_names, list) or not target_names:
        raise ApiError("target_names 必须是非空字符串数组")
    if not all(isinstance(name, str) and name for name in target_names):
        raise ApiError("target_names 必须是非空字符串数组")
    if len(set([*input_names, *target_names])) != len(input_names) + len(target_names):
        raise ApiError("输入变量和目标变量名称不能重复")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 10:
        raise ApiError("n_samples 必须是大于等于10的整数")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ApiError("seed 必须是整数")
    if isinstance(noise_ratio, bool) or not isinstance(noise_ratio, (int, float)):
        raise ApiError("noise_ratio 必须是非负数值")
    if noise_ratio < 0:
        raise ApiError("noise_ratio 必须是非负数值")


def _build_demo_targets(normalized, target_count, noise_ratio, rng):
    import numpy as np

    # 不同目标使用交替方向权重和不同中心点 使多目标优化具有可观察的权衡关系
    columns = []
    dimensions = normalized.shape[1]
    for target_index in range(target_count):
        signs = np.where((np.arange(dimensions) + target_index) % 2 == 0, 1.0, -1.0)
        weights = signs * (np.arange(dimensions, dtype=float) + 1.0)
        center = 0.25 + 0.15 * (target_index % 3)
        values = 100.0 * (target_index + 1)
        values = values + 20.0 * (normalized @ weights)
        values = values + 80.0 * np.square(normalized - center).sum(axis=1)
        scale = max(float(np.std(values)), 1.0)
        values = values + rng.normal(0.0, noise_ratio * scale, size=len(normalized))
        columns.append(values)
    return np.column_stack(columns)


def _normalize_ranges(value: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(value, dict) or not value:
        raise ApiError("param_ranges 必须是非空对象")
    result = {}
    for name, bounds in value.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ApiError(f"param_ranges.{name} 必须是 [lower, upper]")
        lower, upper = bounds
        if isinstance(lower, bool) or not isinstance(lower, (int, float)) or lower >= upper:
            raise ApiError(f"param_ranges.{name} 必须满足数值 lower < upper")
        result[str(name)] = (float(lower), float(upper))
    return result


def get_training_progress(doe_id: str) -> dict[str, Any]:
    state = store.load(store.validate_id(doe_id))
    training = state.get("training") or {}
    return {
        "id": doe_id,
        "status": training.get("status", "not_started"),
        "stage": training.get("stage", "not_started"),
        "progress": training.get("progress", 0),
        "models": training.get("models", []),
        "error": training.get("error"),
        "updated_at": state["updated_at"],
    }


def delete_training(payload: dict[str, Any]) -> str:
    doe_id = _require_id(payload)
    if registry.running(doe_id, "training"):
        raise ConflictError("训练正在运行，请先中止")
    store.reset_training(doe_id)
    store.update(doe_id, status="ready", stage="training_deleted", progress=10)
    return doe_id


def stop_training(payload: dict[str, Any]) -> bool:
    doe_id = _require_id(payload)
    store.load(doe_id)
    accepted = registry.stop(doe_id, "training")
    if accepted:
        store.update(doe_id, status="stopping", stage="training_stopping")
        store.update_section(doe_id, "training", status="stopping", stage="stopping")
    return accepted


def start_training(payload: dict[str, Any]) -> str:
    doe_id = _require_id(payload)
    store.load(doe_id)
    if registry.running(doe_id, "training"):
        raise ConflictError("training 已在运行")
    # 参数在启动线程前完成校验 防止无效请求进入后台后才暴露错误
    request = _normalize_training(payload)
    store.update(doe_id, status="running", stage="training", progress=15)
    store.update_section(
        doe_id, "training", status="queued", stage="queued", progress=0,
        models=[], request=request, error=None,
    )
    # HTTP 请求立即返回 后台线程负责训练 评价和持续更新进度
    registry.start(doe_id, "training", lambda cancel: _run_training(doe_id, request, cancel))
    return doe_id


def _normalize_training(payload: dict[str, Any]) -> dict[str, Any]:
    names = payload.get("all_var_list")
    count = payload.get("input_var_count")
    data_file = Path(str(payload.get("data_file", ""))).expanduser().resolve()
    if not data_file.is_file():
        raise ApiError(f"data_file 不存在：{data_file}")
    if not isinstance(names, list) or not all(isinstance(x, str) and x for x in names):
        raise ApiError("all_var_list 必须是非空字符串数组")
    if not isinstance(count, int) or not 0 < count < len(names):
        raise ApiError("input_var_count 必须大于 0 且小于变量总数")
    # 默认训练四种传统模型 DNN 计算成本高 因此需要调用方显式选择
    models = payload.get("models") or [{"model_index": index} for index in range(4)]
    for model in models:
        if not isinstance(model, dict) or model.get("model_index") not in MODEL_FAMILIES:
            raise ApiError("models.model_index 仅支持 0-4")
    return {
        "data_file": str(data_file), "all_var_list": names, "input_var_count": count,
        "models": models, "evaluation": payload.get("evaluation", {"enabled": True}),
    }


def _run_training(doe_id: str, request: dict[str, Any], cancel: threading.Event) -> None:
    try:
        # 先生成可推理的模型快照 再执行交叉验证并写入综合评分
        records = _train_models(doe_id, request, cancel)
        if cancel.is_set():
            _mark_cancelled(doe_id, "training")
            return
        records = _evaluate_models(doe_id, request, records, cancel)
        store.update_section(
            doe_id, "training", status="finished", stage="finished",
            progress=100, models=records,
        )
        store.update(doe_id, status="finished", stage="training_finished", progress=100)
    except Exception as exc:
        if cancel.is_set():
            _mark_cancelled(doe_id, "training")
        else:
            store.update_section(
                doe_id, "training", status="failed", stage="failed", error=str(exc)
            )
            store.update(doe_id, status="failed", stage="training_failed")


def _train_models(doe_id, request, cancel) -> list[dict[str, Any]]:
    # 复用已有代理模型服务 保持底层模型训练逻辑和历史产物格式不变
    from mobo.surrogate.service import train_surrogate

    records = []
    models = request["models"]
    for position, config in enumerate(models):
        if cancel.is_set():
            break
        index = config["model_index"]
        # 模型标识包含 DOE 和模型类型 末尾随机段防止重复训练覆盖旧任务
        model_id = f"tr_{doe_id}_{index}_{uuid.uuid4().hex[:6]}"
        response = train_surrogate(
            request["data_file"], request["all_var_list"], request["input_var_count"],
            index, config.get("params", {}), model_id,
        )
        if response["code"] != 0:
            raise RuntimeError(response["msg"])
        records.append(_snapshot_model(doe_id, response))
        training_progress = round(75 * (position + 1) / len(models))
        store.update(doe_id, progress=15 + round(60 * (position + 1) / len(models)))
        store.update_section(
            doe_id, "training", status="running", stage="training",
            progress=training_progress, models=records,
        )
    return records


def _snapshot_model(doe_id: str, response: dict[str, Any]) -> dict[str, Any]:
    data = response["data"]
    destination = store.task_dir(doe_id) / "models" / response["model_id"]
    # 底层先写共享模型目录 此处复制到 DOE 专属目录形成稳定快照
    shutil.copytree(data["model_dir"], destination, dirs_exist_ok=True)
    return {
        "model_id": response["model_id"], "model_index": data["model_index"],
        "model_family": data["model_family"], "model_dir": str(destination),
        "target_names": data["target_names"], "train_cost_sec": data["train_cost_sec"],
        "score": None, "evaluation": [],
    }


def _evaluate_models(doe_id, request, records, cancel):
    config = request.get("evaluation") or {}
    if not config.get("enabled", True) or cancel.is_set():
        return records
    from mobo.surrogate.evaluate import SurrogateModelEvaluator

    # 每个模型族独立交叉验证 目标评分取平均后用于自动选择最佳模型
    store.update_section(doe_id, "training", status="running", stage="evaluating", progress=75)
    evaluator = SurrogateModelEvaluator(
        request["data_file"], request["all_var_list"], request["input_var_count"],
        n_splits=config.get("n_splits", 5), random_state=config.get("random_state", 42),
    )
    for position, record in enumerate(records):
        if cancel.is_set():
            break
        summaries = evaluator.evaluate(models=[record["model_family"]])
        serialized = [asdict(item) for item in summaries]
        scores = [item["score"] for item in serialized if item["score"] is not None]
        record["evaluation"] = serialized
        record["score"] = sum(scores) / len(scores) if scores else None
        store.update(doe_id, progress=80 + round(20 * (position + 1) / len(records)))
        store.update_section(
            doe_id, "training", stage="evaluating",
            progress=75 + round(25 * (position + 1) / len(records)), models=records,
        )
    return records


def _mark_cancelled(doe_id: str, section: str) -> None:
    store.update_section(doe_id, section, status="stopped", stage="stopped")
    store.update(doe_id, status="stopped", stage=f"{section}_stopped")


def start_inference(payload: dict[str, Any]) -> dict[str, Any]:
    # 推理相关依赖按请求加载 避免服务启动时加载全部模型框架
    import joblib
    import numpy as np
    from mobo.optimization.ga.run import _load_model

    doe_id = _require_id(payload)
    record = _select_model(store.load(doe_id), payload.get("model_id"))
    state = store.load(doe_id)
    rows = _normalize_inference_inputs(payload.get("inputs"), state)
    fields = _normalize_fields(payload.get("fields"), record["target_names"])
    array = np.asarray(rows, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    scalers = joblib.load(Path(record["model_dir"]) / f"{record['target_names'][0]}_scalers.pkl")
    # 输入和输出必须复用训练时保存的标准化器 保证预测尺度正确
    scaled = scalers["scaler_X"].transform(array)
    predictions = []
    for index, target in enumerate(record["target_names"]):
        model = _load_model(record["model_dir"], target)
        values = np.asarray(model.predict(scaled)).reshape(-1, 1) # type: ignore
        predictions.append(scalers[f"scaler_y_{index}"].inverse_transform(values).ravel())
    matrix = np.column_stack(predictions)
    all_results = {
        target: matrix[:, index].tolist()
        for index, target in enumerate(record["target_names"])
    }
    store.update_section(
        doe_id, "inference", model_id=record["model_id"],
        columns=record["target_names"], values=all_results,
    )
    return {field: all_results[field] for field in fields}


def _normalize_inference_inputs(value: Any, state: dict[str, Any]) -> list[Any]:
    if isinstance(value, dict):
        request = state.get("training", {}).get("request") or {}
        all_names = request.get("all_var_list") or []
        count = request.get("input_var_count", 0)
        input_names = all_names[:count]
        if set(value) != set(input_names):
            raise ApiError(f"inputs 字段必须为：{input_names}")
        columns = [value[name] for name in input_names]
        if not columns or not all(isinstance(column, list) and column for column in columns):
            raise ApiError("inputs 的每个字段必须是非空数组")
        if len({len(column) for column in columns}) != 1:
            raise ApiError("inputs 的字段数组长度必须一致")
        return [list(row) for row in zip(*columns)]
    if not isinstance(value, list) or not value:
        raise ApiError("inputs 必须是非空二维数值数组或字段数组对象")
    return value


def _normalize_fields(value: Any, available: list[str]) -> list[str]:
    fields = available if value is None else value
    if not isinstance(fields, list) or not fields:
        raise ApiError("fields 必须是非空字符串数组")
    if not all(isinstance(field, str) and field for field in fields):
        raise ApiError("fields 必须是非空字符串数组")
    if len(fields) != len(set(fields)):
        raise ApiError("fields 不能包含重复字段")
    unknown = [field for field in fields if field not in available]
    if unknown:
        raise ApiError(f"请求字段不存在：{unknown}；可用字段：{available}")
    return fields


def get_data(payload: dict[str, Any]) -> dict[str, list[Any]]:
    doe_id = _require_id(payload)
    state = store.load(doe_id)
    if "fields" not in payload:
        raise ApiError("fields 为必填的非空字符串数组")
    data_type = payload.get("data_type")
    if data_type == "sample":
        section = state.get("sample") or {}
        return _read_tabular_fields(section.get("sample_file"), section.get("columns"), payload)
    if data_type == "dataset":
        section = (state.get("training") or {}).get("dataset") or {}
        return _read_tabular_fields(section.get("data_file"), section.get("all_var_list"), payload)
    if data_type == "optimization":
        section = (state.get("optimization") or {}).get("result") or {}
        task_info = section.get("task_info") or {}
        resources = section.get("file_resource") or {}
        return _read_tabular_fields(
            resources.get("solution_txt_path"), task_info.get("result_columns"), payload
        )
    if data_type == "inference":
        section = state.get("inference") or {}
        values = section.get("values")
        if not isinstance(values, dict):
            raise ConflictError("尚无可获取的推理结果")
        fields = _normalize_fields(payload.get("fields"), list(values))
        return {field: values[field] for field in fields}
    raise ApiError("data_type 仅支持 sample、dataset、optimization、inference")


def _read_tabular_fields(
    file_name: Any, columns: Any, payload: dict[str, Any]
) -> dict[str, list[Any]]:
    if not isinstance(file_name, str) or not Path(file_name).is_file():
        raise ConflictError("对应数据尚未生成或结果文件不存在")
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ConflictError("服务端未记录对应文件的字段顺序")
    fields = _normalize_fields(payload.get("fields"), columns)
    selected = {field: [] for field in fields}
    indices = {field: columns.index(field) for field in fields}
    with Path(file_name).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            cells = line.rstrip("\r\n").split("\t")
            if cells == [""]:
                continue
            if len(cells) != len(columns):
                raise ApiError(f"结果文件第 {line_number} 行列数与字段记录不一致", 500, 500)
            for field, index in indices.items():
                selected[field].append(_parse_cell(cells[index]))
    return selected


def _parse_cell(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def _select_model(state: dict[str, Any], model_id: str | None) -> dict[str, Any]:
    models = state.get("training", {}).get("models", [])
    if model_id:
        models = [item for item in models if item["model_id"] == model_id]
    if not models:
        raise ApiError("没有可用的已训练代理模型", status=409, code=409)
    # 未指定模型时选择平均评分最高者 未评价模型使用最低优先级
    return max(models, key=lambda item: item.get("score") if item.get("score") is not None else -1e99)


def start_optimization(payload: dict[str, Any]) -> str:
    doe_id = _require_id(payload)
    state = store.load(doe_id)
    if registry.running(doe_id, "optimization"):
        raise ConflictError("optimization 已在运行")
    # 归一化算法别名并绑定指定模型或当前评分最高的模型
    request = _normalize_optimization(payload, state)
    store.update(doe_id, status="running", stage="optimization", progress=5)
    store.update_section(doe_id, "optimization", status="queued", request=request, error=None)
    registry.start(doe_id, "optimization", lambda cancel: _run_optimization(doe_id, request, cancel))
    return doe_id


def _normalize_optimization(payload, state) -> dict[str, Any]:
    # 单目标和多目标都由参数化 NSGA2 承载 强化学习别名单独路由到 RL
    aliases = {
        "nsga2": "nsga2", "single": "nsga2", "single_objective": "nsga2",
        "multi_objective": "nsga2", "rl": "rl", "reinforcement_learning": "rl",
    }
    algorithm = payload.get("algorithm", payload.get("optimizer", "nsga2"))
    if algorithm not in aliases:
        raise ApiError(f"不支持的优化算法：{algorithm}")
    request = {key: value for key, value in payload.items() if key not in {"id", "algorithm"}}
    request["model_id"] = _select_model(state, payload.get("model_id"))["model_id"]
    request["optimizer"] = aliases[algorithm]
    # 调用方可只传核心目标和范围 其余常用优化参数由服务补全
    if request["optimizer"] == "nsga2" and request.get("objective_names"):
        names = request["objective_names"]
        request.setdefault("constraints", [])
        request.setdefault("objective_config", [
            {"name": name, "minimize": True} for name in names
        ])
        request.setdefault("optimizer_config", {
            "pop_size": 100, "n_offsprings": 100, "eliminate_duplicates": True,
            "n_gen": 200, "seed": 42,
        })
        request.setdefault("output_config", {})
    return request


def _run_optimization(doe_id: str, request: dict[str, Any], cancel: threading.Event) -> None:
    # 复用现有优化服务并通过独立优化标识保留底层状态记录
    from mobo.optimization.service import run_optimization

    try:
        optimizer = request.pop("optimizer")
        task_id = f"opt_{doe_id}_{uuid.uuid4().hex[:6]}"
        response = run_optimization(request, optimizer=optimizer, task_id=task_id)
        if cancel.is_set():
            _mark_cancelled(doe_id, "optimization")
        elif response["code"] != 0:
            raise RuntimeError(response["msg"])
        else:
            result = _collect_optimization_result(doe_id, response)
            store.update_section(doe_id, "optimization", status="finished", result=result)
            store.update(doe_id, status="finished", stage="optimization_finished", progress=100)
    except Exception as exc:
        if cancel.is_set():
            _mark_cancelled(doe_id, "optimization")
        else:
            store.update_section(doe_id, "optimization", status="failed", error=str(exc))
            store.update(doe_id, status="failed", stage="optimization_failed")


def _collect_optimization_result(doe_id, response) -> dict[str, Any]:
    data = response["data"]
    resources = dict(data.get("file_resource") or {})
    output_dir = store.task_dir(doe_id) / "optimization"
    # 将算法输出文件复制到 DOE 专属目录 返回路径不再依赖共享输出位置
    for key, source in list(resources.items()):
        path = Path(source)
        if path.is_file():
            destination = output_dir / path.name
            shutil.copy2(path, destination)
            resources[key] = str(destination)
    return {"optimization_id": response["task_id"], **data, "file_resource": resources}


def stop_optimization(payload: dict[str, Any]) -> bool:
    doe_id = _require_id(payload)
    store.load(doe_id)
    accepted = registry.stop(doe_id, "optimization")
    if accepted:
        store.update(doe_id, status="stopping", stage="optimization_stopping")
        store.update_section(doe_id, "optimization", status="stopping")
    return accepted


def get_optimization(doe_id: str) -> dict[str, Any]:
    state = store.load(store.validate_id(doe_id))
    return {"id": doe_id, **state["optimization"]}


__all__ = [
    "add_doe", "delete_doe", "delete_training", "generate_sample",
    "generate_training_dataset", "get_data", "get_optimization", "get_training_progress", "list_doe",
    "start_inference", "start_optimization", "start_training",
    "stop_optimization", "stop_training",
]
