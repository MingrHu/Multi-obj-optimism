"""DEFORM KEY 工艺参数替换原子能力。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .base import LineReplacement, ParameterBinding


def format_deform_float(value: str) -> str:
    """把数值格式化为 DEFORM 规范的 10 位尾数、3 位指数科学计数法。"""
    try:
        n = float(value)
    except (ValueError, TypeError):
        return str(value)

    if n == 0.0:
        return "0.0000000000E+000"

    s = "{0:.10e}".format(n).upper()
    if "E" not in s:
        return f"{s}E+000"

    mantissa, exp = s.split("E")
    if exp.startswith("-"):
        sign, digits = "-", exp[1:]
    elif exp.startswith("+"):
        sign, digits = "+", exp[1:]
    else:
        sign, digits = "+", exp

    digits = digits.lstrip("0") or "0"
    return f"{mantissa}E{sign}{digits.zfill(3)}"


def replace_keyword_last_value(
    line: str,
    binding: ParameterBinding,
    *,
    keyword: str,
) -> LineReplacement:
    """匹配关键字和对象 ID，并替换该行最后一个值。"""
    tokens = line.split()
    matched = len(tokens) >= 2 and tokens[0] == keyword and tokens[1] == binding.object_id
    if not matched:
        return LineReplacement(line, False)
    start = line.rfind(tokens[-1])
    text = line[:start] + format_deform_float(binding.value) + line[start + len(tokens[-1]):]
    return LineReplacement(text, True)


def replace_object_temperature(
    lines: Sequence[str], bindings: Sequence[ParameterBinding]
) -> list[str]:
    """同时设置对象的 REFTMP 和统一 NDTMP 默认温度。"""
    result = list(lines)
    for binding in bindings:
        if binding.name != "workpiece_temperature" or binding.object_id is None:
            continue
        for index, line in enumerate(result):
            tokens = line.split()
            if len(tokens) < 2 or tokens[1] != binding.object_id:
                continue
            if tokens[0] in {"REFTMP", "NDTMP"}:
                start = line.rfind(tokens[-1])
                result[index] = (
                    line[:start] + format_deform_float(binding.value)
                    + line[start + len(tokens[-1]):]
                )
    return result


def replace_movctl_constant_speed(
    line: str, binding: ParameterBinding
) -> LineReplacement:
    """仅在 MOVCTL 常速模式（Ftype=0）下替换行末速度。"""
    tokens = line.split()
    matched = (
        len(tokens) >= 4
        and tokens[0] == "MOVCTL"
        and tokens[1] == binding.object_id
        and tokens[3] == "0"
    )
    if not matched:
        return LineReplacement(line, False)
    return replace_keyword_last_value(line, binding, keyword="MOVCTL")


def collect_speed_scale_specs(
    bindings: Sequence[ParameterBinding],
) -> Dict[str, Tuple[float, float]]:
    """从一个样本中收集碾环速度缩放目标区间，按对象 ID 聚合。"""
    lowers: Dict[str, float] = {}
    uppers: Dict[str, float] = {}
    for binding in bindings:
        object_id = binding.object_id
        if object_id is None:
            continue
        try:
            magnitude = abs(float(binding.value))
        except (ValueError, TypeError):
            continue
        if binding.name == 'pressure_roll_speed_lower':
            lowers[object_id] = magnitude
        elif binding.name == 'pressure_roll_speed_upper':
            uppers[object_id] = magnitude

    specs: Dict[str, Tuple[float, float]] = {}
    for object_id in lowers.keys() & uppers.keys():
        low, high = lowers[object_id], uppers[object_id]
        specs[object_id] = (min(low, high), max(low, high))
    return specs


def parse_speed(line: str) -> Optional[float]:
    """解析控制点行的速度列。"""
    body = line[:-1] if line.endswith("\n") else line
    _, sep, tail = body.rpartition(" ")
    if not sep or not tail.strip():
        return None
    try:
        return float(tail)
    except ValueError:
        return None


def scale_abs(value: float, src_min: float, src_max: float,
              lower: float, upper: float) -> float:
    """把速度绝对值线性映射到目标范围并保留符号。"""
    sign = -1.0 if value < 0 else 1.0
    span = src_max - src_min
    if span <= 0:
        scaled_abs = (lower + upper) / 2.0
    else:
        ratio = (abs(value) - src_min) / span
        scaled_abs = lower + ratio * (upper - lower)
    return sign * scaled_abs


def scale_speed_line(line: str, src_min: float, src_max: float,
                     lower: float, upper: float) -> str:
    """缩放单个控制点行的速度列。"""
    speed = parse_speed(line)
    if speed is None:
        return line
    body, newline = (line[:-1], "\n") if line.endswith("\n") else (line, "")
    head, sep, _ = body.rpartition(" ")
    scaled = scale_abs(speed, src_min, src_max, lower, upper)
    return head + sep + format_deform_float(str(scaled)) + newline


def scale_movctl_block(lines: List[str], specs: Dict[str, Tuple[float, float]]) -> List[str]:
    """按对象对 MOVCTL 后的控制点速度块做等比例缩放。"""
    if not specs:
        return lines

    movctl_key = "MOVCTL"
    result = list(lines)
    for idx, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) < 2 or tokens[0] != movctl_key:
            continue
        object_id = tokens[1]
        if object_id not in specs:
            continue
        try:
            count = int(tokens[-1])
        except ValueError:
            continue
        if count <= 0:
            continue

        block_speeds = []
        for offset in range(1, count + 1):
            j = idx + offset
            if j >= len(result):
                break
            speed = parse_speed(result[j])
            if speed is not None:
                block_speeds.append(abs(speed))
        if not block_speeds:
            continue
        src_min, src_max = min(block_speeds), max(block_speeds)

        lower, upper = specs[object_id]
        for offset in range(1, count + 1):
            j = idx + offset
            if j >= len(result):
                break
            result[j] = scale_speed_line(result[j], src_min, src_max, lower, upper)
    return result


def replace_pressure_roll_speed_profile(
    lines: Sequence[str], bindings: Sequence[ParameterBinding]
) -> list[str]:
    """根据同一对象的速度上下界替换完整 MOVCTL 控制点块。"""
    specs = collect_speed_scale_specs(bindings)
    return scale_movctl_block(list(lines), specs)


__all__ = [
    "collect_speed_scale_specs",
    "format_deform_float",
    "parse_speed",
    "replace_keyword_last_value",
    "replace_movctl_constant_speed",
    "replace_object_temperature",
    "replace_pressure_roll_speed_profile",
    "scale_abs",
    "scale_movctl_block",
    "scale_speed_line",
]
