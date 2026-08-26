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

from mobo.common.logging import logger

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
    """DEFORM 多工步连续换模批处理任务。

    同一个样本的各工步按顺序执行，前一工步 DB 会复制到下一工步目录以继承历史场；
    不同样本之间可并行。每个工步的 KEY、DB、日志和检查点集中在对应 ``op<n>`` 目录。
    每个工步完成后保存终态 KEY 和可选的 DB 检查点，任务状态按样本、工步和
    ``preparing/solving/completed`` 阶段原子落盘，进程中断后可据此继续运行。

    :param task_id: 任务唯一标识，同时写入多工步状态文件
    :param sample_file: 无表头、制表符分隔的联合样本文件；每行对应一个样本
    :param operations: 按执行顺序排列的工步配置。每项包含 ``template_key``，并可包含
        ``name``、``parameters``、``inherit_materials``、``enable_grain`` 和
        ``position_offset``
    :param work_dir: 运行产物根目录；每个样本使用 ``<work_dir>/<sample_index>/``
    :param max_parallel_samples: 最大并行样本数；同一样本内部的工步仍然串行
    :param keep_checkpoints: 是否在每个工步目录保存 ``checkpoint.DB``
    :param dry_run: 为 True 时不调用 DEFORM，只生成文件并推进状态，用于测试
    :param state_file: 逐样本、逐工步恢复状态文件；未指定时保存到工作目录
    :param on_sample_completed: 单个样本全部工步完成后的可选回调，参数为样本序号；
        增量数据集功能通过该回调接入

    :ivar samples: 从 ``sample_file`` 加载的联合样本二维列表
    :ivar state: 当前多工步状态，包括任务状态及每个样本、工步的阶段和产物路径
    """

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
                state = json.load(f)
            self._refresh_summary(state)
            _atomic_json(self.state_file, state)
            return state
        samples = {}
        for sample_index in range(len(self.samples)):
            samples[str(sample_index)] = {
                "status": "pending", "current_operation": 1,
                "operations": {str(i): {"status": "pending", "phase": "pending",
                                              "attempts": 0, "error": ""}
                               for i in range(1, len(self.operations) + 1)},
            }
        state = {"version": 2, "task_id": self.task_id, "status": "pending",
                 "created_at": _now(), "updated_at": _now(), "samples": samples}
        self._refresh_summary(state)
        _atomic_json(self.state_file, state)
        return state

    @staticmethod
    def _refresh_summary(state: Dict[str, Any]) -> None:
        """刷新与单工步进度文件一致的样本总数及完成数汇总。"""
        statuses = [item.get("status", "pending") for item in state["samples"].values()]
        state["total"] = len(statuses)
        state["completed"] = statuses.count("completed")
        state["remaining"] = state["total"] - state["completed"]
        state["running"] = statuses.count("running")
        state["failed"] = statuses.count("failed")
        state["pending"] = statuses.count("pending")

    def _save(self) -> None:
        with self._lock:
            self.state["updated_at"] = _now()
            self._refresh_summary(self.state)
            _atomic_json(self.state_file, self.state)

    def _set_operation(self, sample_index: int, operation_index: int, **fields: Any) -> None:
        with self._lock:
            item = self.state["samples"][str(sample_index)]
            item["current_operation"] = operation_index
            item["operations"][str(operation_index)].update(fields)
            self.state["updated_at"] = _now()
            self._refresh_summary(self.state)
            _atomic_json(self.state_file, self.state)

    def _values(self, sample_index: int, operation_index: int) -> List[str]:
        start = sum(len(_parameters(op)) for op in self.operations[:operation_index - 1])
        count = len(_parameters(self.operations[operation_index - 1]))
        return [str(v) for v in self.samples[sample_index][start:start + count]]

    def _operation_dir(self, sample_index: int, operation_index: int) -> str:
        return os.path.join(self.work_dir, str(sample_index), f"op{operation_index}")

    def _parameterized_key_path(self, sample_index: int, operation_index: int) -> str:
        template = Path(str(self.operations[operation_index - 1]["template_key"]))
        return os.path.join(
            self._operation_dir(sample_index, operation_index),
            f"{template.stem}_parameterized.KEY",
        )

    def prepare_parameterized_keys(self) -> List[str]:
        """补生成缺失的样本/工步参数化 KEY，已存在的文件直接复用。"""
        generated: List[str] = []
        created_count = 0
        reused_count = 0
        for sample_index in range(len(self.samples)):
            sample_dir = os.path.join(self.work_dir, str(sample_index))
            for operation_index, operation in enumerate(self.operations, 1):
                operation_dir = os.path.join(sample_dir, f"op{operation_index}")
                os.makedirs(operation_dir, exist_ok=True)
                output_path = self._parameterized_key_path(sample_index, operation_index)
                generated.append(output_path)
                if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                    reused_count += 1
                    continue
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
                created_count += 1
                logger.info(
                    f"样本 {sample_index + 1} 工步 {operation_index} 参数化 KEY 已保存: "
                    f"{output_path}"
                )
        logger.info(
            f"参数化 KEY 准备完成：总数 {len(generated)}，新生成 {created_count}，"
            f"复用 {reused_count}"
        )
        return generated

    def parameterized_key_files(self) -> List[str]:
        """返回全部参数化 KEY 的确定性路径，不执行文件生成。"""
        return [
            self._parameterized_key_path(sample_index, operation_index)
            for sample_index in range(len(self.samples))
            for operation_index in range(1, len(self.operations) + 1)
        ]

    def prepare_initial_db_files(self) -> List[str]:
        """串行把每个未完成样本的工步1参数化 KEY 转换为初始 DB"""
        # 1 生成参数化 KEY
        generated = self.parameterized_key_files()
        if any(not os.path.isfile(path) or os.path.getsize(path) == 0 for path in generated):
            generated = self.prepare_parameterized_keys()
        first_keys = generated[::len(self.operations)]
        db_files: List[str] = []
        for sample_index, key_path in enumerate(first_keys):
            sample_state = self.state["samples"][str(sample_index)]
            op_state = sample_state["operations"]["1"]
            operation_dir = self._operation_dir(sample_index, 1)
            os.makedirs(operation_dir, exist_ok=True)
            db_path = os.path.join(operation_dir, "result.DB")
            db_files.append(db_path)
            if (sample_state.get("status") == "completed"
                    or op_state.get("status") == "completed"
                    or os.path.exists(db_path)):
                continue
            self._set_operation(
                sample_index, 1, status="pending", phase="preparing", error="",
                attempts=int(op_state.get("attempts", 0)) + 1,
            )
            try:
                if self.dry_run:
                    Path(db_path).touch()
                else:
                    key_to_db(key_path, db_path)
                if not os.path.exists(db_path):
                    raise FileNotFoundError(
                        f"工步 1 KEY 转 DB 后未生成结果文件: {db_path}"
                    )
            except Exception as exc:
                self._set_operation(
                    sample_index, 1, status="failed", phase="failed",
                    failed_phase="preparing", error=str(exc),
                )
                sample_state["status"] = "failed"
                self.state["status"] = "failed"
                self._save()
                raise
            self._set_operation(
                sample_index, 1, status="pending", phase="prepared", error=""
            )
            logger.info(f"样本 {sample_index + 1} 工步 1 初始 DB 已生成: {db_path}")
        return db_files

    def _prepare_first(self, sample_index: int, db_path: str) -> None:
        """使用工步1参数化 KEY 直接生成初始 DB"""
        operation = self.operations[0]
        params = _parameters(operation)
        parameterized = self._parameterized_key_path(sample_index, 1)
        if not os.path.isfile(parameterized) or os.path.getsize(parameterized) == 0:
            if params:
                names, objects, values = _expanded_parameter_values(
                    params, self._values(sample_index, 1)
                )
                write_parameterized_key(
                    operation["template_key"], parameterized, names, objects, values
                )
            else:
                shutil.copy2(operation["template_key"], parameterized)
        if self.dry_run:
            Path(db_path).touch()
        else:
            key_to_db(parameterized, db_path)

    def _prepare_transition(self, sample_index: int, operation_index: int,
                            operation_dir: str, db_path: str, previous_terminal: str) -> None:
        """多工步替换模控制逻辑 保留前序 DB 的材料和晶粒状态"""
        operation = self.operations[operation_index - 1]
        params = _parameters(operation)
        values = self._values(sample_index, operation_index)
        names, objects, expanded_values = _expanded_parameter_values(params, values)
        parameterized = self._parameterized_key_path(sample_index, operation_index)

        if os.path.exists(parameterized):
            parts = split_operation_key(parameterized, operation_dir)
        else:
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
        """多工步核心处理函数逻辑"""
        sample_dir = os.path.join(self.work_dir, str(sample_index))
        os.makedirs(sample_dir, exist_ok=True)
        sample_state = self.state["samples"][str(sample_index)]
        sample_state["status"] = "running"
        # {
        #   "status": "running",
        #   "current_operation": 2,
        #   "operations": {
        #     "1": {"status": "completed"},
        #     "2": {"status": "running"},
        #     "3": {"status": "pending"}
        #   }
        # }
        self._save()
        logger.info(f"样本 {sample_index + 1} 开始运行")

        # 多工步串行执行
        for operation_index in range(1, len(self.operations) + 1):
            operation = self.operations[operation_index - 1]
            op_state = sample_state["operations"][str(operation_index)]
            # 完成的就跳过
            if op_state["status"] == "completed":
                continue
            logger.info(
                f"样本 {sample_index + 1} 工步 {operation_index}/{len(self.operations)} 开始"
            )
            # 一些必要路径初始化
            operation_dir = os.path.join(sample_dir, f"op{operation_index}")
            os.makedirs(operation_dir, exist_ok=True)
            db_path = os.path.join(operation_dir, "result.DB")
            terminal = os.path.join(operation_dir, "terminal.KEY")
            checkpoint = os.path.join(operation_dir, "checkpoint.DB")

            # 运行：1-从未开始 2-准备中 3-准备失败 4-不存在DB文件
            try:
                phase = op_state.get("phase", "pending")
                if phase in {"pending", "preparing"} or (phase == "failed" and op_state.get("failed_phase") == "preparing") or not os.path.exists(db_path):
                    self._set_operation(sample_index, operation_index, status="running",
                                        phase="preparing", error="",
                                        attempts=int(op_state.get("attempts", 0)) + 1)
                    if operation_index == 1:
                        self._prepare_first(sample_index, db_path)
                    else:
                        previous_state = sample_state["operations"][str(operation_index - 1)]
                        previous_dir = self._operation_dir(sample_index, operation_index - 1)
                        previous_terminal = str(
                            previous_state.get("terminal_key")
                            or os.path.join(previous_dir, "terminal.KEY")
                        )
                        previous_db_candidates = [
                            str(previous_state.get("checkpoint") or ""),
                            str(previous_state.get("db_path") or ""),
                            os.path.join(previous_dir, "checkpoint.DB"),
                            os.path.join(previous_dir, "result.DB"),
                        ]
                        previous_db = next(
                            (path for path in previous_db_candidates
                             if path and os.path.isfile(path)),
                            "",
                        )
                        if not previous_db:
                            raise FileNotFoundError(
                                f"工步 {operation_index} 缺少前一工步 DB"
                            )
                        shutil.copy2(previous_db, db_path)
                        # 执行换模逻辑 生成新的工步 DB
                        self._prepare_transition(sample_index, operation_index, operation_dir,
                                                 db_path, previous_terminal)
                    self._set_operation(sample_index, operation_index, phase="prepared")
                if not os.path.exists(db_path):
                    raise FileNotFoundError(
                        f"工步 {operation_index} 准备完成后未生成结果 DB: {db_path}"
                    )
                self._set_operation(sample_index, operation_index, status="running", phase="solving")

                # 非空跑情况下执行deform求解
                if not self.dry_run:
                    solve_db_sync(db_path)
                    # 终态 KEY 文件 用于结果获取以及后续工步换模
                    db_to_key(db_path, terminal, "")
                    # 下一个工步
                    next_operation = (
                        self.operations[operation_index] if operation_index < len(self.operations) else None
                    )
                    # 部分KEY文件缺失晶粒组织设置 必须在初始KEY文件手动设置 在后续工步中直接从前序DB中继承晶粒组织状态
                    # 后续阶段的模板KEY只提供模具几何、接触关系、运动停止条件等
                    grain_required = bool(operation.get("enable_grain")) or bool(
                        next_operation and next_operation.get("enable_grain")
                    )
                    if grain_required and not _has_grain_state(terminal):
                        raise RuntimeError(
                            f"工步 {operation_index} 终态 KEY 缺少工件 GRAIN 状态: {terminal}"
                        )
                else:
                    shutil.copy2(self.operations[operation_index - 1]["template_key"], terminal)
                if self.keep_checkpoints:
                    shutil.copy2(db_path, checkpoint)
                self._set_operation(sample_index, operation_index, status="completed",
                                    phase="completed", completed_at=_now(), error="",
                                    db_path=db_path, terminal_key=terminal,
                                    checkpoint=checkpoint if self.keep_checkpoints else "")
                logger.info(
                    f"样本 {sample_index + 1} 工步 {operation_index}/{len(self.operations)} 完成"
                )
            except Exception as exc:
                current_phase = sample_state["operations"][str(operation_index)].get("phase")
                self._set_operation(sample_index, operation_index, status="failed", phase="failed",
                                    failed_phase=current_phase, error=str(exc))
                sample_state["status"] = "failed"
                self._save()
                logger.error(
                    f"样本 {sample_index + 1} 工步 {operation_index} 失败: {exc}"
                )
                raise
        final_index = len(self.operations)
        final_state = sample_state["operations"][str(final_index)]
        sample_state["status"] = "completed"
        sample_state["db_path"] = str(
            final_state.get("db_path")
            or os.path.join(self._operation_dir(sample_index, final_index), "result.DB")
        )
        sample_state["final_key"] = str(
            final_state.get("terminal_key")
            or os.path.join(self._operation_dir(sample_index, final_index), "terminal.KEY")
        )
        self._save()
        logger.info(
            f"样本 {sample_index + 1} 全部工步完成；总体进度 "
            f"{self.state['completed']}/{self.state['total']}，"
            f"剩余 {self.state['remaining']}"
        )
        # 开始回调
        if self.on_sample_completed is not None:
            self.on_sample_completed(sample_index)

    def run(self) -> Dict[str, Any]:
        """运行或续跑全部样本，已完成工步不会重复执行"""
        # 1 准备DB
        self.prepare_initial_db_files()
        self.state["status"] = "running"
        self._save()
        logger.info(
            f"多工步任务开始：总样本 {self.state['total']}，"
            f"已完成 {self.state['completed']}，待完成 {self.state['remaining']}"
        )
        # 2 如果有回调 先调用已完成样本的回调
        if self.on_sample_completed is not None:
            for sample_index in range(len(self.samples)):
                if self.state["samples"][str(sample_index)]["status"] == "completed":
                    self.on_sample_completed(sample_index)
        # 3 线程池并行执行未完成样本
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
        logger.info(
            f"多工步任务结束：已完成 {self.state['completed']}/{self.state['total']}，"
            f"失败 {self.state['failed']}，剩余 {self.state['remaining']}"
        )
        return self.state

    def result_db_files(self) -> List[str]:
        """返回已经完成的样本最终 DB，供后处理或目标提取使用。"""
        return [item["db_path"] for item in self.state["samples"].values()
                if item.get("status") == "completed" and item.get("db_path")]


__all__ = [
    "Operation", "generate_multi_operation_samples", "split_operation_key",
    "MultiOperationTask",
]
