"""DEFORM 多工步样本生成、连续换模求解与磁盘恢复。"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import pandas as pd

from .keyfile import apply_parameters, write_parameterized_key
from .sampling import generate_full_factorial, generate_lhs, save_samples
from .solver import db_to_key, key_to_db, run_key_actions, solve_db_sync

Operation = Dict[str, Any]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parameters(operation: Operation) -> List[Dict[str, Any]]:
    params = list(operation.get("parameters") or [])
    for item in params:
        if "name" not in item or not (item.get("object") or item.get("objects")):
            raise ValueError("每个工步参数必须包含 name，以及 object 或 objects")
    return params


def _parameter_objects(parameter: Dict[str, Any]) -> List[str]:
    """返回一个采样参数需要同步作用的对象名。"""
    if parameter.get("objects"):
        return [str(value) for value in parameter["objects"]]
    return [str(parameter["object"])]


def _expanded_parameter_values(
    parameters: Sequence[Dict[str, Any]], values: Sequence[str]
) -> tuple[List[str], List[str], List[str]]:
    """把单个采样列展开为一个或多个 KEY 对象替换请求。"""
    names: List[str] = []
    objects: List[str] = []
    expanded_values: List[str] = []
    for parameter, value in zip(parameters, values, strict=True):
        for object_name in _parameter_objects(parameter):
            names.append(str(parameter["name"]))
            objects.append(object_name)
            expanded_values.append(str(value))
    return names, objects, expanded_values


def generate_multi_operation_samples(task_id: str, operations: Sequence[Operation],
                                     save_dir: str, method: str = "lhs",
                                     n_samples: int = 0,
                                     level_nums: Sequence[int] = ()) -> str:
    """把各工步参数展平后联合采样，列顺序为工步顺序和参数声明顺序。"""
    ranges: Dict[str, tuple[float, float]] = {}
    for op_index, operation in enumerate(operations, 1):
        op_name = str(operation.get("name") or f"operation_{op_index}")
        for param_index, param in enumerate(_parameters(operation), 1):
            if "range" not in param:
                raise ValueError(f"{op_name} 的参数 {param['name']} 缺少 range")
            column = f"op{op_index}:{param['name']}:{param_index}"
            low, high = param["range"]
            ranges[column] = (float(low), float(high))
    if not ranges:
        raise ValueError("至少需要声明一个多工步采样参数")
    if method == "lhs":
        frame = generate_lhs(n_samples, ranges)
        tag = "lhs"
    elif method == "full":
        if not level_nums:
            raise ValueError("full 采样必须提供 level_nums")
        frame = generate_full_factorial(ranges, level_nums)
        tag = "fullfactorial"
    else:
        raise ValueError(f"不支持的采样方法: {method}")
    path = save_samples(task_id, frame, tag, save_dir)
    return path


def split_operation_key(template_path: str, output_dir: str) -> Dict[str, str]:
    """拆出后续工步所需的控制、模具和接触 KEY 片段，不复制 Object 1 网格。"""
    with open(template_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    object_marks: Dict[int, int] = {}
    inter_mark = None
    for index, line in enumerate(lines):
        match = re.match(r"^\*\s+Data for Object #\s*(\d+)", line)
        if match:
            object_marks[int(match.group(1))] = index
        elif re.match(r"^\*\s+Inter-Object Data", line):
            inter_mark = index
    if 1 not in object_marks or inter_mark is None:
        raise ValueError(f"无法识别 KEY 的对象分段: {template_path}")
    diegeo = next((i for i in range(object_marks[1], inter_mark)
                   if re.match(r"^DIEGEO\s+1(?:\s|$)", lines[i])), None)
    if diegeo is None:
        raise ValueError(f"未找到 Object 1 的 DIEGEO: {template_path}")
    os.makedirs(output_dir, exist_ok=True)
    chunks: Dict[str, List[str]] = {
        "simulation": lines[:object_marks[1]],
        "object1_control": lines[object_marks[1]:diegeo],
        "inter_object": lines[inter_mark:],
    }
    ndtmp = next((line for line in lines[diegeo:inter_mark]
                  if re.match(r"^NDTMP\s+1\s+0(?:\s|$)", line)), None)
    if ndtmp is not None:
        chunks["object1_temperature"] = [ndtmp]
    ids = sorted(object_marks)
    for object_id in ids:
        if object_id == 1:
            continue
        following = [object_marks[i] for i in ids if i > object_id]
        end = min(following) if following else inter_mark
        chunks[f"object{object_id}"] = lines[object_marks[object_id]:end]
    paths: Dict[str, str] = {}
    for name, chunk in chunks.items():
        path = os.path.join(output_dir, f"{name}.KEY")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(chunk)
        paths[name] = path
    return paths


def _center(key_path: str) -> tuple[float, float, float]:
    with open(key_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = line.split()
            if len(tokens) >= 5 and tokens[0] == "CNTRAX" and tokens[1] == "1":
                return tuple(float(value) for value in tokens[2:5])  # type: ignore
    raise ValueError(f"KEY 中未找到 Object 1 的 CNTRAX: {key_path}")


def _prepare_transition_simulation(
    path: str, *, inherit_materials: bool, enable_grain: bool
) -> None:
    """准备换模控制片段，同时保留前序 DB 中的材料与晶粒历史。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if inherit_materials:
        # DBREAD 已带入前序材料和晶粒状态，后续模板不得再次覆盖材料段。
        material_mark = next(
            (index for index, line in enumerate(lines)
             if "Property Data of Material" in line),
            None,
        )
        if material_mark is not None:
            lines = lines[:max(0, material_mark - 1)]
    if enable_grain:
        # TRANS 最后一列是 Grain 开关，仅修改临时换模片段，不修改模板 KEY。
        found = False
        for index, line in enumerate(lines):
            fields = line.split()
            if fields and fields[0] == "TRANS":
                if len(fields) < 6:
                    raise ValueError(f"TRANS 格式异常: {path}")
                fields[-1] = "1"
                lines[index] = "TRANS        " + "       ".join(fields[1:]) + "\n"
                found = True
                break
        if not found:
            raise ValueError(f"未找到 Grain 模式控制 TRANS: {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _has_grain_state(key_path: str, object_id: str = "1") -> bool:
    """判断终态 KEY 是否包含指定工件的晶粒状态。"""
    with open(key_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 4 and fields[0] == "GRAIN" and fields[1] == object_id:
                return int(fields[2]) > 0 and int(fields[3]) > 0
    return False


def _atomic_json(path: str, value: Dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".multi_", suffix=".json", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class MultiOperationTask:
    """一个样本使用同一 DB 依次换模；不同样本可并行。"""

    def __init__(self, task_id: str, sample_file: str, operations: Sequence[Operation],
                 work_dir: str, max_parallel_samples: int = 1,
                 keep_checkpoints: bool = True, dry_run: bool = False,
                 state_file: str | None = None,
                 on_sample_completed: Callable[[int], None] | None = None) -> None:
        if len(operations) < 2:
            raise ValueError("多工步任务至少需要两个工步")
        self.task_id = task_id
        self.sample_file = os.path.abspath(sample_file)
        self.operations = [dict(item) for item in operations]
        self.work_dir = os.path.abspath(work_dir)
        self.max_parallel_samples = max(1, int(max_parallel_samples))
        self.keep_checkpoints = keep_checkpoints
        self.dry_run = dry_run
        self.on_sample_completed = on_sample_completed
        self.state_file = os.path.abspath(
            state_file or os.path.join(self.work_dir, "multi_operation_state.json")
        )
        self._lock = threading.Lock()
        self.samples = pd.read_csv(self.sample_file, sep="\t", header=None, dtype=str).values.tolist()
        expected = sum(len(_parameters(op)) for op in self.operations)
        if any(len(row) != expected for row in self.samples):
            raise ValueError(f"样本列数必须为 {expected}")
        os.makedirs(self.work_dir, exist_ok=True)
        self.state = self._load_or_init_state()

    def _load_or_init_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        samples = {}
        for sample_index in range(len(self.samples)):
            samples[str(sample_index)] = {
                "status": "pending", "current_operation": 1,
                "operations": {str(i): {"status": "pending", "phase": "pending",
                                              "attempts": 0, "error": ""}
                               for i in range(1, len(self.operations) + 1)},
            }
        state = {"version": 1, "task_id": self.task_id, "status": "pending",
                 "created_at": _now(), "updated_at": _now(), "samples": samples}
        _atomic_json(self.state_file, state)
        return state

    def _save(self) -> None:
        with self._lock:
            self.state["updated_at"] = _now()
            _atomic_json(self.state_file, self.state)

    def _set_operation(self, sample_index: int, operation_index: int, **fields: Any) -> None:
        with self._lock:
            item = self.state["samples"][str(sample_index)]
            item["current_operation"] = operation_index
            item["operations"][str(operation_index)].update(fields)
            self.state["updated_at"] = _now()
            _atomic_json(self.state_file, self.state)

    def _values(self, sample_index: int, operation_index: int) -> List[str]:
        start = sum(len(_parameters(op)) for op in self.operations[:operation_index - 1])
        count = len(_parameters(self.operations[operation_index - 1]))
        return [str(v) for v in self.samples[sample_index][start:start + count]]

    def prepare_parameterized_keys(self) -> List[str]:
        """批量生成各样本、各工步的参数化模板 KEY，不调用 DEFORM。"""
        generated: List[str] = []
        for sample_index in range(len(self.samples)):
            sample_dir = os.path.join(self.work_dir, str(sample_index))
            for operation_index, operation in enumerate(self.operations, 1):
                operation_dir = os.path.join(sample_dir, f"op{operation_index}")
                os.makedirs(operation_dir, exist_ok=True)
                output_path = os.path.join(operation_dir, "parameterized_template.KEY")
                params = _parameters(operation)
                if not params:
                    shutil.copy2(operation["template_key"], output_path)
                else:
                    names, objects, values = _expanded_parameter_values(
                        params, self._values(sample_index, operation_index)
                    )
                    write_parameterized_key(
                        operation["template_key"], output_path,
                        names, objects, values,
                    )
                generated.append(output_path)
        return generated

    def _prepare_first(self, sample_index: int, operation_dir: str, db_path: str) -> None:
        operation = self.operations[0]
        params = _parameters(operation)
        key_path = os.path.join(operation_dir, "operation.KEY")
        if params:
            names, objects, values = _expanded_parameter_values(
                params, self._values(sample_index, 1)
            )
            write_parameterized_key(
                operation["template_key"], key_path, names, objects, values
            )
        else:
            shutil.copy2(operation["template_key"], key_path)
        if self.dry_run:
            Path(db_path).touch()
        else:
            key_to_db(key_path, db_path)

    def _prepare_transition(self, sample_index: int, operation_index: int,
                            operation_dir: str, db_path: str, previous_terminal: str) -> None:
        operation = self.operations[operation_index - 1]
        params = _parameters(operation)
        values = self._values(sample_index, operation_index)
        names, objects, expanded_values = _expanded_parameter_values(params, values)
        parts = split_operation_key(operation["template_key"], operation_dir)
        for path in parts.values():
            with open(path, "r", encoding="utf-8") as f:
                rendered = apply_parameters(
                    f.readlines(), names, objects, expanded_values
                )
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(rendered)
        _prepare_transition_simulation(
            parts["simulation"],
            inherit_materials=bool(operation.get("inherit_materials", False)),
            enable_grain=bool(operation.get("enable_grain", False)),
        )
        offset = operation.get("position_offset")
        if offset is None:
            source = _center(previous_terminal)
            target = _center(operation["template_key"])
            offset = [target[i] - source[i] for i in range(3)]
        name = str(operation.get("name") or f"Operation {operation_index}")
        lines = [
            # 先恢复前序完整状态，再叠加下一工步控制并对齐工件。
            "DBREAD 0\n", db_path + "\n",
            "KFREAD 1\n", parts["simulation"] + "\n",
            "OBJPOS 1 1 " + " ".join(str(v) for v in offset) + " 0 0 0 0\n",
            # 工件只更新控制和工艺温度，几何及历史场继续来自 DB。
            "KFREAD 1\n", parts["object1_control"] + "\n",
        ]
        if "object1_temperature" in parts:
            lines.extend(["KFREAD 1\n", parts["object1_temperature"] + "\n"])
        object_ids = sorted(int(k[6:]) for k in parts if re.fullmatch(r"object\d+", k))
        # 下一工步模具使用模板几何；对象1的工件几何从未导入。
        for object_id in object_ids:
            lines.extend([f"OBJTYP {object_id} 0\n", "KFREAD 1\n",
                          parts[f"object{object_id}"] + "\n"])
        # 最后加载接触关系，并在同一 DB 中创建新工步。
        lines.extend([
            "KFREAD 1\n", parts["inter_object"] + "\n", "INICTC 1\n",
            "OPRNAM\n", name + "\n", "SIMNAM\n", name + "\n",
            f"CURSIM {operation_index} {operation_index} 0\n",
            f"GENDB 1 1 {operation_index}\n", db_path + "\n",
        ])
        action_path = os.path.join(operation_dir, "transition.KEY")
        with open(action_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        if not self.dry_run:
            run_key_actions(action_path)

    def _run_sample(self, sample_index: int) -> None:
        sample_dir = os.path.join(self.work_dir, str(sample_index))
        os.makedirs(sample_dir, exist_ok=True)
        db_path = os.path.join(sample_dir, "result.DB")
        sample_state = self.state["samples"][str(sample_index)]
        sample_state["status"] = "running"
        self._save()
        for operation_index in range(1, len(self.operations) + 1):
            operation = self.operations[operation_index - 1]
            op_state = sample_state["operations"][str(operation_index)]
            if op_state["status"] == "completed":
                continue
            operation_dir = os.path.join(sample_dir, f"op{operation_index}")
            os.makedirs(operation_dir, exist_ok=True)
            checkpoint = os.path.join(sample_dir, f"checkpoint_{operation_index}.DB")
            previous_checkpoint = os.path.join(
                sample_dir, f"checkpoint_{operation_index - 1}.DB"
            )
            previous_terminal = os.path.join(
                sample_dir, f"terminal_{operation_index - 1}.KEY"
            )
            try:
                # preparing 可安全重做；solving 阶段则复用已经生成的 DB 继续提交。
                phase = op_state.get("phase", "pending")
                if phase in {"pending", "preparing"} or (
                        phase == "failed" and op_state.get("failed_phase") == "preparing"):
                    self._set_operation(sample_index, operation_index, status="running",
                                        phase="preparing", error="",
                                        attempts=int(op_state.get("attempts", 0)) + 1)
                    if operation_index == 1:
                        self._prepare_first(sample_index, operation_dir, db_path)
                    else:
                        if self.keep_checkpoints and os.path.exists(previous_checkpoint):
                            shutil.copy2(previous_checkpoint, db_path)
                        self._prepare_transition(sample_index, operation_index, operation_dir,
                                                 db_path, previous_terminal)
                    self._set_operation(sample_index, operation_index, phase="prepared")
                self._set_operation(sample_index, operation_index, status="running", phase="solving")
                if not self.dry_run:
                    solve_db_sync(db_path)
                    terminal = os.path.join(sample_dir, f"terminal_{operation_index}.KEY")
                    db_to_key(db_path, terminal, "")
                    # 每个需要晶粒演化的阶段都立即校验，避免错误传播到后续工步。
                    next_operation = (
                        self.operations[operation_index]
                        if operation_index < len(self.operations) else None
                    )
                    grain_required = bool(operation.get("enable_grain")) or bool(
                        next_operation and next_operation.get("enable_grain")
                    )
                    if grain_required and not _has_grain_state(terminal):
                        raise RuntimeError(
                            f"工步 {operation_index} 终态 KEY 缺少工件 GRAIN 状态: {terminal}"
                        )
                else:
                    terminal = os.path.join(sample_dir, f"terminal_{operation_index}.KEY")
                    shutil.copy2(self.operations[operation_index - 1]["template_key"], terminal)
                if self.keep_checkpoints:
                    shutil.copy2(db_path, checkpoint)
                self._set_operation(sample_index, operation_index, status="completed",
                                    phase="completed", completed_at=_now(), error="",
                                    db_path=db_path, terminal_key=terminal,
                                    checkpoint=checkpoint if self.keep_checkpoints else "")
            except Exception as exc:
                current_phase = sample_state["operations"][str(operation_index)].get("phase")
                self._set_operation(sample_index, operation_index, status="failed", phase="failed",
                                    failed_phase=current_phase, error=str(exc))
                sample_state["status"] = "failed"
                self._save()
                raise
        sample_state["status"] = "completed"
        sample_state["db_path"] = db_path
        sample_state["final_key"] = os.path.join(
            sample_dir, f"terminal_{len(self.operations)}.KEY"
        )
        self._save()
        if self.on_sample_completed is not None:
            self.on_sample_completed(sample_index)

    def run(self) -> Dict[str, Any]:
        """运行或续跑全部样本，已完成工步不会重复执行。"""
        self.state["status"] = "running"
        self._save()
        # 一边计算一边生成
        if self.on_sample_completed is not None:
            for sample_index in range(len(self.samples)):
                if self.state["samples"][str(sample_index)]["status"] == "completed":
                    self.on_sample_completed(sample_index)
        pending = [i for i in range(len(self.samples))
                   if self.state["samples"][str(i)]["status"] != "completed"]
        errors = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_samples) as executor:
            futures = {executor.submit(self._run_sample, i): i for i in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append({"sample": futures[future], "error": str(exc)})
        self.state["status"] = "failed" if errors else "completed"
        self.state["errors"] = errors
        self._save()
        return self.state

    def result_db_files(self) -> List[str]:
        """返回已经完成的样本最终 DB，供后处理或目标提取使用。"""
        return [item["db_path"] for item in self.state["samples"].values()
                if item.get("status") == "completed" and item.get("db_path")]


__all__ = [
    "Operation", "generate_multi_operation_samples", "split_operation_key",
    "MultiOperationTask",
]
