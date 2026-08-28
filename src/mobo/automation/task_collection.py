"""可直接导入的自动化任务定义集合。"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from mobo.common.paths import AUTO_MULTI_DIR, AUTO_SINGLE_DIR, KEY_FILE_DIR, task_dir
from mobo.common.logging import logger
from mobo.extraction import registry as extraction_registry

from .config import DeformConfig
from .extract import export_saved_step_keys
from .keyfile import read_key_frames
from .multi_operation import MultiOperationTask, Operation, generate_multi_operation_samples
from .pipeline import ForgingTask, generate_sample_file


@dataclass(frozen=True)
class TargetDefinition:
    """单工步或多工步任务的一个目标提取定义。"""

    output_name: str
    target_name: str
    object_name: str
    operation_indices: tuple[int, ...]
    select_component: int | None = None
    in_progress: bool = False
    workpiece_type: str = "generic"
    description: str = ""
    verified: bool = True


@dataclass(frozen=True)
class MultiOperationTaskDefinition:
    """可采样、预生成 KEY、构建和运行的多工步任务定义。"""

    task_id: str
    name: str
    workspace: Path
    operations: tuple[Operation, ...]
    targets: tuple[TargetDefinition, ...]

    @property
    def sample_dir(self) -> Path:
        return self.workspace / "samples"

    @property
    def run_dir(self) -> Path:
        return self.workspace / "runs"

    def operation_configs(self) -> list[Operation]:
        return copy.deepcopy(list(self.operations))

    def validate(self) -> None:
        if len(self.operations) < 2:
            raise ValueError(f"{self.name} 至少需要两个工步")
        for index, operation in enumerate(self.operations, 1):
            template = Path(str(operation["template_key"]))
            if not template.is_file():
                raise FileNotFoundError(f"工步 {index} 模板不存在: {template}")
            for parameter in operation.get("parameters", []):
                low, high = parameter["range"]
                if float(low) > float(high):
                    raise ValueError(f"工步 {index} 参数范围上下界颠倒: {parameter['name']}")

    def generate_samples(
        self,
        *,
        method: str = "lhs",
        n_samples: int = 0,
        level_nums: Sequence[int] = (),
        save_dir: str | os.PathLike[str] | None = None,
    ) -> str:
        self.validate()
        destination = Path(save_dir) if save_dir is not None else self.sample_dir
        return generate_multi_operation_samples(
            self.task_id, self.operation_configs(), str(destination),
            method=method, n_samples=n_samples, level_nums=level_nums,
        )

    def build(
        self,
        sample_file: str | os.PathLike[str],
        *,
        work_dir: str | os.PathLike[str] | None = None,
        max_parallel_samples: int = 1,
        keep_checkpoints: bool = True,
        dry_run: bool = False,
    ) -> MultiOperationTask:
        self.validate()
        return MultiOperationTask(
            self.task_id,
            str(sample_file),
            self.operation_configs(),
            str(Path(work_dir) if work_dir is not None else self.run_dir),
            max_parallel_samples=max_parallel_samples,
            keep_checkpoints=keep_checkpoints,
            dry_run=dry_run,
            state_file=str(task_dir(self.task_id) / "multi_operation_state.json"),
        )

    def prepare_keys(
        self,
        sample_file: str | os.PathLike[str],
        *,
        work_dir: str | os.PathLike[str] | None = None,
    ) -> list[str]:
        task = self.build(sample_file, work_dir=work_dir, dry_run=True)
        return task.prepare_parameterized_keys()

    def run(
        self,
        sample_file: str | os.PathLike[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.build(sample_file, **kwargs).run()

    def extract_targets(
        self,
        key_files_by_operation: Mapping[int, Sequence[str]],
    ) -> Dict[str, str]:
        """从已经导出的各工步 KEY 帧提取任务目标，不负责 DB→KEY。"""
        results: Dict[str, str] = {}
        for target in self.targets:
            key_files = [
                path
                for operation_index in target.operation_indices
                for path in key_files_by_operation.get(operation_index, ())
            ]
            if not key_files:
                raise ValueError(f"目标 {target.output_name} 没有可用 KEY 帧")
            spec = extraction_registry.resolve(
                target.workpiece_type, target.target_name
            )
            object_id = DeformConfig.get_object_id(target.object_name)
            if object_id is None:
                raise ValueError(f"未知对象: {target.object_name}")
            if spec.kind == "key_file":
                value = spec.fn(key_files[-1], object_id=int(object_id))
                results[target.output_name] = "{:.6f}".format(float(value))
            else:
                frames = read_key_frames(key_files)
                results[target.output_name] = spec.fn(
                    frames, object_id, target.in_progress, target.select_component
                )
        return results

    def extract_dataset(
        self,
        task: MultiOperationTask,
        *,
        result_dir: str | os.PathLike[str] | None = None,
    ) -> str:
        """从已完成样本的各工步终态 KEY 生成无结果数据集。"""
        destination = Path(result_dir) if result_dir is not None else self.workspace / "results"
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / (
            datetime.now().strftime("%Y-%m-%d %H_%M_%S") + "_result.txt"
        )
        with output.open("w", encoding="utf-8") as stream:
            for sample_index, sample in enumerate(task.samples):
                state = task.state["samples"][str(sample_index)]
                if state.get("status") != "completed":
                    continue
                key_files = self._completed_key_files(task, sample_index) # type: ignore
                try:
                    targets = self.extract_targets(key_files)
                    stream.write("\t".join(
                        [str(value) for value in sample] + list(targets.values())
                    ) + "\n")
                    stream.flush()
                except Exception as exc:
                    logger.error(f"样本 {sample_index} 数据提取失败: {exc}")
        return str(output)

    def extract_sample_row(
        self,
        task: MultiOperationTask,
        sample_index: int,
    ) -> list[str]:
        """从一个已完成多工步样本的终态 KEY 生成数据集行。"""
        state = task.state["samples"][str(sample_index)]
        if state.get("status") != "completed":
            raise ValueError(f"样本 {sample_index} 尚未完成")
        key_files = self._completed_key_files(task, sample_index)
        targets = self.extract_targets(key_files)
        return [str(value) for value in task.samples[sample_index]] + list(targets.values())

    def _completed_key_files(
        self, task: MultiOperationTask, sample_index: int
    ) -> Dict[int, list[str]]:
        """终态目标复用 terminal KEY；全过程目标导出检查点 DB 的真实保存步。"""
        state = task.state["samples"][str(sample_index)]
        progress_operations = {
            operation_index
            for target in self.targets if target.in_progress
            for operation_index in target.operation_indices
        }
        result: Dict[int, list[str]] = {}
        for operation_index in range(1, len(self.operations) + 1):
            operation_state = state["operations"][str(operation_index)]
            terminal = str(operation_state["terminal_key"])
            if operation_index not in progress_operations:
                result[operation_index] = [terminal]
                continue
            checkpoint = str(operation_state.get("checkpoint") or "")
            if not checkpoint or not Path(checkpoint).is_file():
                raise FileNotFoundError(
                    f"工步 {operation_index} 的全过程目标缺少检查点 DB"
                )
            output_dir = Path(terminal).parent / "extracted_steps"
            result[operation_index] = export_saved_step_keys(
                checkpoint, str(output_dir)
            )
        return result


@dataclass(frozen=True)
class SingleOperationTaskDefinition:
    """可采样、生成 KEY、求解和提取的单工步任务定义。"""

    task_id: str
    name: str
    workspace: Path
    template_key: str
    parameters: tuple[dict[str, Any], ...]
    targets: tuple[TargetDefinition, ...]

    def validate(self) -> None:
        template = Path(self.template_key)
        if not template.is_file():
            raise FileNotFoundError(f"模板不存在: {template}")
        for parameter in self.parameters:
            low, high = parameter["range"]
            if float(low) > float(high):
                raise ValueError(f"参数范围上下界颠倒: {parameter['name']}")
        if any(target.operation_indices != (1,) for target in self.targets):
            raise ValueError("单工步目标的 operation_indices 必须为 (1,)")

    def generate_samples(
        self, *, method: str = "lhs", n_samples: int = 0,
        level_nums: Sequence[int] = (),
        save_dir: str | os.PathLike[str] | None = None,
    ) -> str:
        self.validate()
        ranges: Dict[str, tuple[float, float]] = {
            str(parameter["name"]): (
                float(parameter["range"][0]), float(parameter["range"][1])
            )
            for parameter in self.parameters
        }
        destination = Path(save_dir) if save_dir is not None else self.workspace / "samples"
        return generate_sample_file(
            self.task_id, method, ranges, str(destination), n_samples, level_nums
        )

    def build(
        self, sample_file: str | os.PathLike[str], *,
        workspace: str | os.PathLike[str] | None = None,
        dry_run: bool = False, max_parallel: int = 24,
        incremental: bool = False,
    ) -> ForgingTask:
        self.validate()
        root = Path(workspace) if workspace is not None else self.workspace
        return ForgingTask(
            sample_file=str(sample_file), template_key=self.template_key,
            temp_key_dir=str(root / "input_keys"),
            result_db_dir=str(root / "db"),
            result_key_dir=str(root / "result_keys"),
            result_txt_dir=str(root / "results"),
            param_table=[
                [str(item["name"]) for item in self.parameters],
                [str(item["object"]) for item in self.parameters],
            ],
            target_table=[
                [item.target_name for item in self.targets],
                [item.object_name for item in self.targets],
                [item.select_component for item in self.targets],
            ],
            in_progress=[item.in_progress for item in self.targets],
            process_info_file=str(task_dir(self.task_id) / "process_info.json"),
            dry_run=dry_run, max_parallel=max_parallel,
            incremental=incremental,
            incremental_state_file=str(task_dir(self.task_id) / "incremental_dataset.json"),
            incremental_output_file=str(root / "results" / f"{self.task_id}_incremental_result.txt"),
        )

    def prepare_keys(
        self, sample_file: str | os.PathLike[str], *,
        workspace: str | os.PathLike[str] | None = None,
    ) -> list[str]:
        task = self.build(sample_file, workspace=workspace)
        task.generate_keys()
        return task.key_files

    def run(self, sample_file: str | os.PathLike[str], **kwargs: Any) -> ForgingTask:
        task = self.build(sample_file, **kwargs)
        task.generate_keys()
        task.run_solver()
        task.extract()
        return task


_TC4_TEMPLATE_DIR = KEY_FILE_DIR / "tc4_ring_multi_task_1"
_TC4_DIES = ("driving_roll", "pressure_roll", "axial_roll_1", "axial_roll_2")


def _tc4_variable_parameters() -> list[dict[str, Any]]:
    return [
        {
            "name": "workpiece_temperature",
            "object": "workpiece",
            "range": [800.0, 960.0],
        },
        {
            "name": "roll_tmp",
            "objects": list(_TC4_DIES),
            "range": [200.0, 350.0],
        },
        {
            "name": "pressure_roll_constant_speed",
            "object": "pressure_roll",
            "range": [0.1, 2.2],
        },
    ]


TC4_RING_MULTI_TASK_1 = MultiOperationTaskDefinition(
    task_id="tc4-ring-multi-task-1",
    name="TC4碾环多工步任务1",
    workspace=AUTO_MULTI_DIR / "tc4_ring_multi_task_1",
    operations=(
        {
            "name": "TC4固定工步1",
            "template_key": str(_TC4_TEMPLATE_DIR / "1.KEY"),
            "parameters": [],
        },
        {
            "name": "TC4可变工步2",
            "template_key": str(_TC4_TEMPLATE_DIR / "2.KEY"),
            "inherit_materials": True,
            "enable_grain": True,
            "parameters": _tc4_variable_parameters(),
        },
        {
            "name": "TC4可变工步3",
            "template_key": str(_TC4_TEMPLATE_DIR / "3.KEY"),
            "inherit_materials": True,
            "enable_grain": True,
            "parameters": _tc4_variable_parameters(),
        },
    ),
    targets=(
        TargetDefinition(
            "roundness_inner", "roundness_inner", "workpiece", (3,),
            workpiece_type="ring", description="最终工步内圈圆度",
        ),
        TargetDefinition(
            "roundness_outer", "roundness_outer", "workpiece", (3,),
            workpiece_type="ring", description="最终工步外圈圆度",
        ),
        TargetDefinition(
            "die_load_y", "load", "driving_roll", (1, 2, 3),
            select_component=1, in_progress=True,
            description="工步1至3驱动辊Y向最大绝对载荷",
        ),
        TargetDefinition(
            "effective_strain_std", "strain_std", "workpiece", (3,),
            description="最终工步单元等效应变标准差",
        ),
        TargetDefinition(
            "average_grain_size", "grain_morph", "workpiece", (3,),
            select_component=1,
            description="最终工步工件平均晶粒尺寸",
        ),
    ),
)


_7050_TEMPLATE_DIR = KEY_FILE_DIR / "7050_ring_single_task_1"

# 模板中的 GRNDAT 取自 RINGROLL.KEY，仅用于打通晶粒状态读写与求解验证；
# 在用于 7050 晶粒演化预测前，必须替换为经试验标定的 7050 材料模型参数。
RING_7050_SINGLE_TASK_1 = SingleOperationTaskDefinition(
    task_id="7050-ring-single-task-1",
    name="7050碾环单工步任务1",
    workspace=AUTO_SINGLE_DIR / "7050_ring_single_task_1",
    template_key=str(_7050_TEMPLATE_DIR / "7050.KEY"),
    parameters=(
        {"name": "workpiece_temperature", "object": "workpiece", "range": [320.0, 450.0]},
        {"name": "ring_die_temperature", "object": "ring_dies", "range": [150.0, 250.0]},
        {"name": "pressure_roll_constant_speed", "object": "pressure_roll", "range": [0.2, 2.0]},
    ),
    targets=(
        TargetDefinition("roundness_inner", "roundness_inner", "workpiece", (1,),
                         workpiece_type="ring", description="最终内圈圆度"),
        TargetDefinition("roundness_outer", "roundness_outer", "workpiece", (1,),
                         workpiece_type="ring", description="最终外圈圆度"),
        TargetDefinition("die_load_y", "load", "driving_roll", (1,),
                         select_component=1, in_progress=True, description="驱动辊Y向最大绝对载荷"),
        TargetDefinition("effective_strain_std", "strain_std", "workpiece", (1,),
                         description="最终单元等效应变标准差"),
        TargetDefinition("average_grain_size", "grain_morph", "workpiece", (1,),
                         select_component=1, description="最终平均晶粒尺寸"),
        TargetDefinition("material_fill", "material_fill", "workpiece", (1,),
                         workpiece_type="ring", description="材料填充性（待定义）", verified=False),
    ),
)


_GH4169_TEMPLATE_DIR = KEY_FILE_DIR / "gh4169_ring_single_task_1"

# 模板保留 GH4169 自带的 GRNDAT 晶粒模型，并启用 50 μm 初始晶粒状态。
GH4169_RING_SINGLE_TASK_1 = SingleOperationTaskDefinition(
    task_id="gh4169-ring-single-task-1",
    name="GH4169碾环单工步任务1",
    workspace=AUTO_SINGLE_DIR / "gh4169_ring_single_task_1",
    template_key=str(_GH4169_TEMPLATE_DIR / "GH4169.KEY"),
    parameters=(
        {"name": "workpiece_temperature", "object": "workpiece", "range": [1100.0, 1150.0]},
        {"name": "ring_die_temperature", "object": "ring_dies", "range": [250.0, 350.0]},
        {"name": "pressure_roll_profile_peak_speed", "object": "pressure_roll", "range": [0.1, 2.5]},
    ),
    targets=(
        TargetDefinition("roundness_inner", "roundness_inner", "workpiece", (1,),
                         workpiece_type="ring", description="最终内圈圆度"),
        TargetDefinition("roundness_outer", "roundness_outer", "workpiece", (1,),
                         workpiece_type="ring", description="最终外圈圆度"),
        TargetDefinition("die_load_y", "load", "driving_roll", (1,),
                         select_component=1, in_progress=True, description="驱动辊Y向最大绝对载荷"),
        TargetDefinition("effective_strain_std", "strain_std", "workpiece", (1,),
                         description="最终单元等效应变标准差"),
        TargetDefinition("average_grain_size", "grain_morph", "workpiece", (1,),
                         select_component=1, description="最终平均晶粒尺寸"),
        TargetDefinition("material_fill", "material_fill", "workpiece", (1,),
                         workpiece_type="ring", description="材料填充性（待定义）", verified=False),
    ),
)


TASK_COLLECTION: Dict[str, MultiOperationTaskDefinition | SingleOperationTaskDefinition] = {
    TC4_RING_MULTI_TASK_1.task_id: TC4_RING_MULTI_TASK_1,
    RING_7050_SINGLE_TASK_1.task_id: RING_7050_SINGLE_TASK_1,
    GH4169_RING_SINGLE_TASK_1.task_id: GH4169_RING_SINGLE_TASK_1,
}


def get_task_definition(task_id: str) -> MultiOperationTaskDefinition | SingleOperationTaskDefinition:
    try:
        return TASK_COLLECTION[task_id]
    except KeyError as exc:
        raise KeyError(f"未注册的自动化任务: {task_id}") from exc


def get_multi_operation_task_definition(task_id: str) -> MultiOperationTaskDefinition:
    """获取多工步任务定义，并在误传单工步任务时立即报错。"""
    definition = get_task_definition(task_id)
    if not isinstance(definition, MultiOperationTaskDefinition):
        raise TypeError(f"任务不是多工步任务: {task_id}")
    return definition


def get_single_operation_task_definition(task_id: str) -> SingleOperationTaskDefinition:
    """获取单工步任务定义，并在误传多工步任务时立即报错。"""
    definition = get_task_definition(task_id)
    if not isinstance(definition, SingleOperationTaskDefinition):
        raise TypeError(f"任务不是单工步任务: {task_id}")
    return definition


__all__ = [
    "MultiOperationTaskDefinition",
    "SingleOperationTaskDefinition",
    "TargetDefinition",
    "TASK_COLLECTION",
    "TC4_RING_MULTI_TASK_1",
    "RING_7050_SINGLE_TASK_1",
    "GH4169_RING_SINGLE_TASK_1",
    "get_task_definition",
    "get_multi_operation_task_definition",
    "get_single_operation_task_definition",
]
