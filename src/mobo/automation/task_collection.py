"""可直接导入的自动化任务定义集合。"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from mobo.common.paths import AUTO_MULTI_DIR, KEY_FILE_DIR, task_dir
from mobo.common.logging import logger
from mobo.extraction import registry as extraction_registry

from .config import DeformConfig
from .keyfile import read_key_frames
from .multi_operation import MultiOperationTask, Operation, generate_multi_operation_samples


@dataclass(frozen=True)
class TargetDefinition:
    """多工步任务的一个目标提取定义。"""

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
                key_files = {
                    operation_index: [
                        state["operations"][str(operation_index)]["terminal_key"]
                    ]
                    for operation_index in range(1, len(self.operations) + 1)
                }
                try:
                    targets = self.extract_targets(key_files)
                    stream.write("\t".join(
                        [str(value) for value in sample] + list(targets.values())
                    ) + "\n")
                    stream.flush()
                except Exception as exc:
                    logger.error(f"样本 {sample_index} 数据提取失败: {exc}")
        return str(output)


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


TASK_COLLECTION: Dict[str, MultiOperationTaskDefinition] = {
    TC4_RING_MULTI_TASK_1.task_id: TC4_RING_MULTI_TASK_1,
}


def get_task_definition(task_id: str) -> MultiOperationTaskDefinition:
    try:
        return TASK_COLLECTION[task_id]
    except KeyError as exc:
        raise KeyError(f"未注册的自动化任务: {task_id}") from exc


__all__ = [
    "MultiOperationTaskDefinition",
    "TargetDefinition",
    "TASK_COLLECTION",
    "TC4_RING_MULTI_TASK_1",
    "get_task_definition",
]
