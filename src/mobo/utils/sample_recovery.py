"""从单工步任务生成的输入 KEY 反向恢复采样文件。"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MovctlBlock:
    """一个 KEY MOVCTL 头及其控制点速度。"""

    object_id: str
    mode: str
    header_last_value: Decimal
    point_speeds: tuple[Decimal, ...]


@dataclass(frozen=True)
class KeyParameterData:
    """反向恢复所需的 KEY 参数索引。"""

    keyword_values: Mapping[tuple[str, str], tuple[Decimal, ...]]
    movctl_blocks: tuple[MovctlBlock, ...]


@dataclass(frozen=True)
class RecoveryContext:
    """当前参数的任务绑定信息。"""

    name: str
    object_name: str
    object_id: str | None
    key_path: Path


ParameterExtractor = Callable[[KeyParameterData, RecoveryContext], Decimal]


class ParameterRecoveryRegistry:
    """按任务参数名注册 KEY 反向提取能力。"""

    def __init__(self) -> None:
        self._extractors: dict[str, ParameterExtractor] = {}

    def register(self, names: str | Sequence[str], extractor: ParameterExtractor) -> None:
        parameter_names = (names,) if isinstance(names, str) else tuple(names)
        for name in parameter_names:
            if name in self._extractors:
                raise ValueError(f"参数反向提取器已注册: {name}")
            self._extractors[name] = extractor

    def resolve(self, name: str) -> ParameterExtractor | None:
        """返回参数反向提取器；未注册时返回 ``None``。"""
        return self._extractors.get(name)


def _decimal(token: str, *, label: str, key_path: Path) -> Decimal:
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise ValueError(f"{key_path}: {label} 不是有效数值: {token!r}") from exc
    if not value.is_finite():
        raise ValueError(f"{key_path}: {label} 必须是有限数值: {token!r}")
    return value


def _decimal_text(value: Decimal) -> str:
    """与正常采样文件一致，固定保留两位小数。"""
    return f"{value:.2f}"


def _one_value(values: Sequence[Decimal], context: RecoveryContext, label: str) -> Decimal:
    if not values:
        raise ValueError(f"{context.key_path}: 未找到 {label}")
    unique = set(values)
    if len(unique) != 1:
        rendered = ", ".join(sorted(_decimal_text(value) for value in unique))
        raise ValueError(f"{context.key_path}: {label} 的多处值不一致: {rendered}")
    return values[0]


def _parse_key(key_path: Path) -> KeyParameterData:
    keyword_values: dict[tuple[str, str], list[Decimal]] = {}
    movctl_blocks: list[MovctlBlock] = []
    with key_path.open("r", encoding="utf-8", errors="strict") as stream:
        iterator = iter(enumerate(stream, 1))
        for line_number, line in iterator:
            tokens = line.split()
            if len(tokens) < 2:
                continue
            keyword, object_id = tokens[0], tokens[1]
            if keyword in {"REFTMP", "NDTMP", "ANGMOV"} and len(tokens) >= 3:
                keyword_values.setdefault((keyword, object_id), []).append(
                    _decimal(
                        tokens[-1], label=f"第 {line_number} 行 {keyword}", key_path=key_path
                    )
                )
            if keyword != "MOVCTL" or len(tokens) < 4:
                continue

            header_last_value = _decimal(
                tokens[-1], label=f"第 {line_number} 行 MOVCTL 末值", key_path=key_path
            )
            point_speeds: list[Decimal] = []
            if tokens[3] == "2":
                try:
                    point_count = int(tokens[-1])
                except ValueError as exc:
                    raise ValueError(
                        f"{key_path}: 第 {line_number} 行 MOVCTL 控制点数量无效"
                    ) from exc
                for _ in range(point_count):
                    try:
                        point_line_number, point_line = next(iterator)
                    except StopIteration as exc:
                        raise ValueError(f"{key_path}: MOVCTL 控制点块不完整") from exc
                    point_tokens = point_line.split()
                    if len(point_tokens) < 2:
                        raise ValueError(
                            f"{key_path}: 第 {point_line_number} 行 MOVCTL 控制点格式无效"
                        )
                    point_speeds.append(_decimal(
                        point_tokens[-1],
                        label=f"第 {point_line_number} 行 MOVCTL 速度",
                        key_path=key_path,
                    ))
            movctl_blocks.append(MovctlBlock(
                object_id=object_id,
                mode=tokens[3],
                header_last_value=header_last_value,
                point_speeds=tuple(point_speeds),
            ))
    return KeyParameterData(
        keyword_values={key: tuple(values) for key, values in keyword_values.items()},
        movctl_blocks=tuple(movctl_blocks),
    )


def _keyword_extractor(keyword: str) -> ParameterExtractor:
    def extract(data: KeyParameterData, context: RecoveryContext) -> Decimal:
        if context.object_id is None:
            raise ValueError(
                f"{context.key_path}: 参数 {context.name} 的对象 {context.object_name!r} "
                "没有 KEY 对象编号"
            )
        return _one_value(
            data.keyword_values.get((keyword, context.object_id), ()),
            context,
            f"对象 {context.object_id} 的 {keyword}",
        )

    return extract


def _movctl_blocks(
    data: KeyParameterData, context: RecoveryContext, *, mode: str | None = None
) -> list[MovctlBlock]:
    if context.object_id is None:
        raise ValueError(
            f"{context.key_path}: 参数 {context.name} 的对象 {context.object_name!r} "
            "没有 KEY 对象编号"
        )
    blocks = [
        block for block in data.movctl_blocks
        if block.object_id == context.object_id and (mode is None or block.mode == mode)
    ]
    if not blocks:
        mode_text = "" if mode is None else f"、模式 {mode}"
        raise ValueError(
            f"{context.key_path}: 未找到对象 {context.object_id}{mode_text} 的 MOVCTL"
        )
    return blocks


def _movctl_header(data: KeyParameterData, context: RecoveryContext) -> Decimal:
    return _one_value(
        [block.header_last_value for block in _movctl_blocks(data, context)],
        context,
        f"对象 {context.object_id} 的 MOVCTL 末值",
    )


def _constant_speed(data: KeyParameterData, context: RecoveryContext) -> Decimal:
    return _one_value(
        [block.header_last_value for block in _movctl_blocks(data, context, mode="0")],
        context,
        f"对象 {context.object_id} 的 MOVCTL 常速",
    ).copy_abs()


def _profile_abs_speeds(data: KeyParameterData, context: RecoveryContext) -> list[Decimal]:
    speeds = [
        speed.copy_abs()
        for block in _movctl_blocks(data, context, mode="2")
        for speed in block.point_speeds
    ]
    if not speeds:
        raise ValueError(f"{context.key_path}: 对象 {context.object_id} 的 MOVCTL 没有控制点")
    return speeds


def _profile_peak_speed(data: KeyParameterData, context: RecoveryContext) -> Decimal:
    nonzero = [speed for speed in _profile_abs_speeds(data, context) if speed != 0]
    return _one_value(nonzero, context, f"对象 {context.object_id} 的非零 MOVCTL 速度绝对值")


def _profile_lower_speed(data: KeyParameterData, context: RecoveryContext) -> Decimal:
    return min(_profile_abs_speeds(data, context))


def _profile_upper_speed(data: KeyParameterData, context: RecoveryContext) -> Decimal:
    return max(_profile_abs_speeds(data, context))


def _ring_die_temperature(data: KeyParameterData, context: RecoveryContext) -> Decimal:
    values: list[Decimal] = []
    for object_id in ("2", "3", "4", "5"):
        values.extend(data.keyword_values.get(("REFTMP", object_id), ()))
    return _one_value(values, context, "对象 2～5 的 REFTMP")


registry = ParameterRecoveryRegistry()
registry.register(("roll_tmp", "workpiece_temperature"), _keyword_extractor("REFTMP"))
registry.register("ring_die_temperature", _ring_die_temperature)
registry.register("driving_roll_rad_speed", _keyword_extractor("ANGMOV"))
registry.register("temp", _keyword_extractor("NDTMP"))
registry.register("speed", _movctl_header)
registry.register("pressure_roll_constant_speed", _constant_speed)
registry.register("pressure_roll_profile_peak_speed", _profile_peak_speed)
registry.register("pressure_roll_speed_lower", _profile_lower_speed)
registry.register("pressure_roll_speed_upper", _profile_upper_speed)


def extract_sample_from_key(
    key_path: str | Path,
    parameters: Sequence[Mapping[str, Any]],
    object_ids: Mapping[str, str],
) -> list[Decimal]:
    """按任务参数定义的列顺序，从一个输入 KEY 恢复一行样本。"""
    path = Path(key_path)
    data = _parse_key(path)
    values: list[Decimal] = []
    for parameter in parameters:
        name = str(parameter["name"])
        object_name = str(parameter["object"])
        extractor = registry.resolve(name)
        if extractor is None:
            raise ValueError(f"{path}: 参数 {name} 尚未注册 KEY 反向提取器")
        context = RecoveryContext(
            name=name,
            object_name=object_name,
            object_id=object_ids.get(object_name),
            key_path=path,
        )
        value = extractor(data, context)
        lower, upper = (Decimal(str(item)) for item in parameter["range"])
        if not lower <= value <= upper:
            raise ValueError(
                f"{path}: {name}={_decimal_text(value)} 超出任务范围 [{lower}, {upper}]"
            )
        values.append(value)
    return values


def _sample_index(path: Path, prefix: str) -> int:
    match = re.fullmatch(re.escape(prefix) + r"(\d+)", path.stem, re.IGNORECASE)
    if match is None:
        raise ValueError(f"KEY 文件名不符合 {prefix}<样本序号>.KEY: {path.name}")
    return int(match.group(1))


def _ordered_key_files(key_dir: Path, prefix: str) -> list[Path]:
    indexed: list[tuple[int, Path]] = []
    for path in key_dir.iterdir():
        if not path.is_file() or path.suffix.upper() != ".KEY":
            continue
        try:
            indexed.append((_sample_index(path, prefix), path))
        except ValueError:
            continue
    if not indexed:
        raise FileNotFoundError(f"{key_dir}: 未找到 {prefix}<序号>.KEY")
    indexed.sort(key=lambda item: item[0])
    indices = [index for index, _ in indexed]
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(f"KEY 样本序号必须从 0 连续排列；实际为: {indices}")
    return [path for _, path in indexed]


def recover_samples(
    task_id: str,
    key_dir: str | Path,
    output_file: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """根据已注册单工步任务定义重建无表头 TSV 采样文件。"""
    from mobo.automation.config import DeformConfig
    from mobo.automation.task_collection import get_single_operation_task_definition

    task = get_single_operation_task_definition(task_id)
    task.validate()
    source = Path(key_dir)
    destination = Path(output_file)
    if not source.is_dir():
        raise NotADirectoryError(f"KEY 目录不存在: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在；如需覆盖请传入 overwrite=True: {destination}")

    prefix = Path(task.template_key).stem
    key_files = _ordered_key_files(source, prefix)
    rows = [
        "\t".join(_decimal_text(value) for value in extract_sample_from_key(
            path, task.parameters, DeformConfig.OBJ_DEF
        ))
        for path in key_files
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据单工步任务定义，从参数化输入 KEY 恢复无表头 TSV 样本文件"
    )
    parser.add_argument("task_id", help="已注册的单工步任务 ID")
    parser.add_argument("key_dir", type=Path, help="任务生成的输入 KEY 目录")
    parser.add_argument("output_file", type=Path, help="恢复后的 sample TXT 路径")
    parser.add_argument("--force", action="store_true", help="允许覆盖已经存在的输出文件")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """命令行入口。"""
    args = _parser().parse_args(argv)
    output = recover_samples(
        args.task_id, args.key_dir, args.output_file, overwrite=args.force
    )
    print(f"已恢复单工步样本文件: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
