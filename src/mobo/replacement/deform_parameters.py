"""DEFORM KEY 工艺参数替换原子能力。"""

from __future__ import annotations

import math
import re
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
    """设置工件初始节点温度（含显式节点记录），保留 REFTMP。"""
    result = list(lines)
    for binding in bindings:
        if binding.name == "workpiece_temperature" and binding.object_id is not None:
            result = _replace_nodal_temperature(result, (binding.object_id,), binding.value)
    return result


def replace_ring_die_temperature(
    lines: Sequence[str], bindings: Sequence[ParameterBinding]
) -> list[str]:
    """用一个采样值同步设置对象 2～5 的实际初始节点温度。"""
    result = list(lines)
    binding = next(
        (item for item in bindings if item.name == "ring_die_temperature"),
        None,
    )
    if binding is None:
        return result
    return _replace_nodal_temperature(result, ("2", "3", "4", "5"), binding.value)


def _replace_token(line: str, index: int, value: str) -> str:
    """仅替换指定字段，保留该行其余字节（包括空白与换行）。"""
    token = list(re.finditer(r"\S+", line))[index]
    return line[:token.start()] + value + line[token.end():]


def _finite_value(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("工艺参数必须是有限数值")
    return number


def _replace_nodal_temperature(
    lines: Sequence[str], object_ids: Sequence[str], value: str,
) -> list[str]:
    _finite_value(value)
    rendered = list(lines)
    found: set[str] = set()
    for index, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) < 2 or tokens[0] != "NDTMP" or tokens[1] not in object_ids:
            continue
        if len(tokens) not in (4, 5):
            raise ValueError(f"不支持的 NDTMP 格式: {line.strip()}")
        count = int(tokens[2])
        columns = int(tokens[4]) if len(tokens) == 5 else 1
        if count < 0 or columns != 1:
            raise ValueError("初温替换只支持单列 NDTMP 温度数据")
        found.add(tokens[1])
        rendered[index] = _replace_token(line, 3, format_deform_float(value))
        for offset in range(1, count + 1):
            row = index + offset
            if row >= len(lines) or len(lines[row].split()) != 2:
                raise ValueError("NDTMP 节点温度记录不完整")
            int(lines[row].split()[0])
            rendered[row] = _replace_token(lines[row], 1, format_deform_float(value))
    missing = set(object_ids) - found
    if missing:
        raise ValueError(f"缺少对象 {', '.join(sorted(missing))} 的 NDTMP 初始温度")
    return rendered


def replace_ring_contact_parameters(
    lines: Sequence[str], bindings: Sequence[ParameterBinding],
) -> list[str]:
    """精确替换环件与四个模具的常量剪切因子/接触换热系数。"""
    result = list(lines)
    specs = {
        "ring_shear_friction_factor": ("FRCFAC", ("1", "0")),
        "ring_contact_heat_transfer": ("IHTCOF", ("0",)),
    }
    for binding in bindings:
        if binding.name not in specs:
            continue
        if binding.object_id != "1":
            raise ValueError(f"{binding.name} 必须绑定 workpiece（对象 1）")
        value = _finite_value(binding.value)
        if value <= 0 or (binding.name == "ring_shear_friction_factor" and value > 1):
            raise ValueError("剪切因子须在 (0, 1] 内，接触换热系数须为正数")
        keyword, mode = specs[binding.name]
        found: set[str] = set()
        for index, line in enumerate(result):
            tokens = line.split()
            if len(tokens) < 3 or tokens[0] != keyword or tokens[1] != "1" or tokens[2] not in {"2", "3", "4", "5"}:
                continue
            if len(tokens) != 4 + len(mode) or tuple(tokens[3:-1]) != mode:
                raise ValueError(f"{keyword} 对象对 1–{tokens[2]} 不是支持的常量模式")
            if tokens[2] in found:
                raise ValueError(f"{keyword} 对象对 1–{tokens[2]} 重复")
            found.add(tokens[2])
            result[index] = _replace_token(line, -1, format_deform_float(binding.value))
        if found != {"2", "3", "4", "5"}:
            raise ValueError(f"{keyword} 缺少环件与对象 2～5 的完整接触定义")
    return result


def replace_driving_roll_angular_speed(
    lines: Sequence[str], bindings: Sequence[ParameterBinding],
) -> list[str]:
    """只支持显式角速度控制的恒定 ANGMOV，不修改扭矩或函数表。"""
    result = list(lines)
    for binding in bindings:
        if binding.name != "driving_roll_rad_speed":
            continue
        _finite_value(binding.value)
        found = False
        for index, line in enumerate(result):
            tokens = line.split()
            if len(tokens) < 2 or tokens[0] != "ANGMOV" or tokens[1] != binding.object_id:
                continue
            if len(tokens) != 5 or tokens[2:4] != ["1", "0"] or found:
                raise ValueError("驱动辊角速度需要唯一的 ANGMOV 常角速度定义")
            found = True
            result[index] = _replace_token(line, 4, format_deform_float(binding.value))
        if not found:
            raise ValueError("未找到驱动辊的 ANGMOV 定义")
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


def replace_movctl_profile_peak_speed(
    lines: Sequence[str], bindings: Sequence[ParameterBinding]
) -> list[str]:
    """替换平台速度并反比例缩放时间，保持14.5 mm进给及原启停形状。"""
    result = list(lines)
    for binding in bindings:
        if binding.name != "pressure_roll_profile_peak_speed" or binding.object_id is None:
            continue
        magnitude = _finite_value(binding.value)
        if magnitude <= 0:
            raise ValueError("芯辊峰值速度必须为正数，才能完成14.5 mm行程")
        found = False
        for index, line in enumerate(result):
            tokens = line.split()
            if len(tokens) < 2 or tokens[0] != "MOVCTL" or tokens[1] != binding.object_id:
                continue
            if len(tokens) != 8 or tokens[3] != "2" or found:
                raise ValueError("芯辊峰值速度需要唯一的 MOVCTL 五点速度曲线")
            found = True
            count = int(tokens[-1])
            if count != 5:
                raise ValueError("芯辊峰值速度只支持五点速度曲线")
            nonzero = 0
            for offset in range(1, count + 1):
                point_index = index + offset
                if point_index >= len(result) or len(result[point_index].split()) != 2:
                    raise ValueError("MOVCTL 控制点记录不完整")
                speed = parse_speed(result[point_index])
                if speed is None or not math.isfinite(speed):
                    raise ValueError("MOVCTL 控制点速度无效")
                if speed == 0.0:
                    continue
                nonzero += 1
                value = -magnitude if speed < 0 else magnitude
                result[point_index] = _replace_token(
                    result[point_index], -1, format_deform_float(str(value))
                )
            if nonzero != 2:
                raise ValueError("MOVCTL 五点曲线应包含两个非零平台速度点")
        if not found:
            raise ValueError("未找到芯辊的 MOVCTL 速度曲线")
        result = _retime_ring_feed(result, binding)
    return result


def _retime_ring_feed(lines: Sequence[str], binding: ParameterBinding) -> list[str]:
    """只改进给时间、TMAX和必要的NSTEP，SMAX/DTMAX及其他内容不变。"""
    controls: dict[str, list[tuple[int, list[str]]]] = {
        name: [] for name in ("PDIE", "SMAX", "TMAX", "NSTEP", "MOVCTL", "STROKE")
    }
    for index, line in enumerate(lines):
        tokens = line.split()
        if not tokens or tokens[0] not in controls:
            continue
        if tokens[0] in {"MOVCTL", "STROKE"} and tokens[1:2] != [binding.object_id]:
            continue
        controls[tokens[0]].append((index, tokens))
    if any(len(items) != 1 for items in controls.values()):
        raise ValueError("等行程进给需要唯一的PDIE/SMAX/TMAX/NSTEP/MOVCTL/STROKE定义")
    if controls["PDIE"][0][1][1] != binding.object_id:
        raise ValueError("等行程进给要求压力辊是主模具PDIE")
    stroke_limit = [float(value) for value in controls["SMAX"][0][1][1:]]
    initial_stroke = [float(value) for value in controls["STROKE"][0][1][2:]]
    if stroke_limit != [0.0, 14.5, 0.0] or initial_stroke != [0.0, 0.0, 0.0]:
        raise ValueError("等行程进给要求Y向SMAX=14.5 mm且初始STROKE为零")
    profile_index, header = controls["MOVCTL"][0]
    if header[2:4] != ["1", "2"] or [float(value) for value in header[4:7]] != [0.0, 1.0, 0.0]:
        raise ValueError("等行程进给只支持Y正向的时间函数速度控制")
    points = [[float(value) for value in lines[profile_index + offset].split()]
              for offset in range(1, 6)]
    times = [point[0] for point in points]
    if (times[0] != 0 or not all(math.isfinite(time) for time in times)
            or any(right <= left for left, right in zip(times, times[1:], strict=False))
            or [point[1] != 0 for point in points] != [False, True, True, False, False]
            or points[1][1] != points[2][1]):
        raise ValueError("等行程进给需要从零开始、严格递增的五点启停平台曲线")
    area = sum((abs(left[1]) + abs(right[1])) * (right[0] - left[0]) / 2
               for left, right in zip(points, points[1:], strict=False))
    scale = 14.5 / area
    if math.isclose(scale, 1.0, rel_tol=1e-12):
        scale = 1.0
    time_index, time_tokens = controls["TMAX"][0]
    step_index, step_tokens = controls["NSTEP"][0]
    if len(time_tokens) != 2 or len(step_tokens) != 2:
        raise ValueError("等行程进给只支持标量TMAX和NSTEP")
    original_limit = _finite_value(time_tokens[1])
    original_steps = int(step_tokens[1])
    if original_limit <= 0 or original_steps <= 0:
        raise ValueError("等行程进给要求TMAX和NSTEP为正数")
    time_limit = max(original_limit, times[-1]) * scale
    if not math.isfinite(time_limit):
        raise ValueError("进给时间超出可表示范围")
    result = list(lines)
    for offset, time in enumerate(times, 1):
        if time * scale != time:
            result[profile_index + offset] = _replace_token(
                result[profile_index + offset], 0, format_deform_float(str(time * scale))
            )
    if time_limit != original_limit:
        result[time_index] = _replace_token(lines[time_index], 1, format_deform_float(str(time_limit)))
    required_steps = math.ceil(original_steps * max(1.0, time_limit / original_limit))
    if required_steps != original_steps:
        result[step_index] = _replace_token(lines[step_index], 1, str(required_steps))
    return result


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


def rescale_movctl_speed_profile(
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
    "replace_driving_roll_angular_speed",
    "replace_movctl_constant_speed",
    "replace_movctl_profile_peak_speed",
    "replace_object_temperature",
    "replace_ring_die_temperature",
    "replace_ring_contact_parameters",
    "rescale_movctl_speed_profile",
    "scale_abs",
    "scale_movctl_block",
    "scale_speed_line",
]
