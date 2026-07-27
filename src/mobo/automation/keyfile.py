"""KEY 文件文本处理。

负责 DEFORM KEY 文件的纯文本层面操作，不涉及子进程：

- :func:`format_deform_float`：把数值格式化为 DEFORM 要求的科学计数法；
- :func:`derive_output_path`：由源文件名派生输出路径；
- :func:`generate_key_files`：把工艺参数写入模板 KEY，批量生成输入 KEY；
- :func:`read_key_frames`：读取一组 KEY 文件的全部文本行。

KEY 文件本质是文本文件，目标行格式为：``<关键字> <对象ID> ... <参数值>``，
:func:`generate_key_files` 依据 :class:`~mobo.automation.config.DeformConfig` 的
关键字/对象映射定位并替换参数值。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from mobo.common.logging import logger
from .config import DeformConfig


def format_deform_float(value: str) -> str:
    """把数值格式化为 DEFORM 规范的 10 位尾数、3 位指数科学计数法。

    非数值输入原样返回；0 返回固定的 ``0.0000000000E+000``。

    :param value: 待格式化的字符串
    :return: DEFORM 科学计数法字符串
    """
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


def derive_output_path(source_file: str, save_dir: str, tag: str, file_type: str) -> str:
    """由源文件主名派生输出文件路径。

    规则：``<save_dir>/<源文件主名><tag>.<file_type>``。

    :param source_file: 源文件路径（取其无扩展名主名）
    :param save_dir: 保存目录
    :param tag: 附加标识（如序号）
    :param file_type: 目标扩展名（不含点，如 ``"KEY"`` / ``"DB"``）
    :return: 输出文件完整路径
    """
    stem = Path(source_file).stem
    return os.path.join(save_dir, f"{stem}{tag}.{file_type}")


def _apply_params_to_line(line: str, param_names: Sequence[str], object_names: Sequence[str],
                          values: Sequence[str]) -> str:
    """若某行匹配到 (关键字, 对象ID)，则用格式化后的参数值替换该行末尾值。

    :param line: KEY 文件的一行
    :param param_names: 工艺参数名序列（如 ``["temp", "speed"]``）
    :param object_names: 参数对应的对象名序列（如 ``["workpiece", "topdie"]``）
    :param values: 与参数一一对应的取值序列
    :return: 替换后的行（未匹配则原样返回）
    """
    tokens = line.split()
    if len(tokens) < 2:
        return line

    for pos, value in enumerate(values):
        param_name = param_names[pos]
        # 碾环 MOVCTL 速度上/下界：交给块级裁剪，不做单行末值替换
        if param_name == 'pressure_roll_speed_upper' or param_name == 'pressure_roll_speed_lower':
            continue
        key_var = DeformConfig.get_key_var(param_name)
        object_id = DeformConfig.get_object_id(object_names[pos])
        # 目标行：第一个 token 为关键字，第二个 token 为对象 ID
        if key_var == tokens[0] and tokens[1] == object_id:
            return line.replace(tokens[-1], format_deform_float(value))
    return line


def _collect_speed_clip_specs(
    param_names: Sequence[str],
    object_names: Sequence[str],
    values: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    """从一个样本中收集碾环速度裁剪区间，按对象 ID 聚合。

    仅当同一对象同时提供 ``*_speed_lower`` 与 ``*_speed_upper`` 时才生成裁剪区间。

    :param param_names: 参数名序列
    :param object_names: 参数对应对象名序列
    :param values: 参数取值序列
    :return: ``{对象ID: (lower, upper)}``（lower/upper 均取绝对值并保证 lower<=upper）
    """
    lowers: Dict[str, float] = {}
    uppers: Dict[str, float] = {}
    for pos, param_name in enumerate(param_names):
        object_id = DeformConfig.get_object_id(object_names[pos])
        if object_id is None:
            continue
        try:
            magnitude = abs(float(values[pos]))
        except (ValueError, TypeError):
            continue
        if param_name == 'pressure_roll_speed_lower':
            lowers[object_id] = magnitude
        elif param_name == 'pressure_roll_speed_upper':
            uppers[object_id] = magnitude

    specs: Dict[str, Tuple[float, float]] = {}
    for object_id in lowers.keys() & uppers.keys():
        low, high = lowers[object_id], uppers[object_id]
        specs[object_id] = (min(low, high), max(low, high))
    return specs


def _clip_abs(value: float, lower: float, upper: float) -> float:
    """把数值的绝对值裁剪到 ``[lower, upper]`` 区间，保留原符号。

    :param value: 原始速度值（可正可负）
    :param lower: 绝对值下界（非负）
    :param upper: 绝对值上界（非负）
    :return: 裁剪后的速度值
    """
    sign = -1.0 if value < 0 else 1.0
    clipped = min(max(abs(value), lower), upper)
    return sign * clipped


def _clip_speed_line(line: str, lower: float, upper: float) -> str:
    """裁剪单个控制点行的速度列（第 2 列），保留原时间列与行内空白格式。

    :param line: 形如 ``    <时间>    <速度>\\n`` 的控制点行
    :param lower: 速度绝对值下界
    :param upper: 速度绝对值上界
    :return: 速度被裁剪后的行；解析失败则原样返回
    """
    body, newline = (line[:-1], "\n") if line.endswith("\n") else (line, "")
    head, sep, tail = body.rpartition(" ")
    if not sep or not tail.strip():
        return line
    try:
        speed = float(tail)
    except ValueError:
        return line
    clipped = _clip_abs(speed, lower, upper)
    # 用格式化后的裁剪值替换速度列，保留原时间列与列间空白
    return head + sep + format_deform_float(str(clipped)) + newline


def _clip_movctl_block(lines: List[str], specs: Dict[str, Tuple[float, float]]) -> List[str]:
    """对 ``MOVCTL <对象ID> ... <m>`` 之后的 m 行控制点做速度裁剪。

    定位每个待裁剪对象的 ``MOVCTL`` 行，读取其行尾整数 ``m`` 作为控制点行数，
    再把紧随其后的 m 行速度按 ``specs`` 的区间裁剪。

    :param lines: KEY 文件全部文本行（会返回裁剪后的新列表，不原地修改入参）
    :param specs: ``{对象ID: (lower, upper)}``
    :return: 裁剪后的文本行列表
    """
    if not specs:
        return lines

    movctl_key = DeformConfig.get_key_var("speed")  # "MOVCTL"
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

        lower, upper = specs[object_id]
        for offset in range(1, count + 1):
            j = idx + offset
            if j >= len(result):
                break
            result[j] = _clip_speed_line(result[j], lower, upper)
    return result


def generate_key_files(template_path: str, param_table: List[List[str]], save_dir: str) -> List[str]:
    """把工艺参数写入模板 KEY，批量生成输入 KEY 文件。

    ``param_table`` 前两行为固定表头：第 0 行是参数名、第 1 行是对象名；从第 2 行起
    每行是一个样本的参数取值。为每个样本生成一个 KEY 文件。

    处理分两趟：先对每行做单值替换（:func:`_apply_params_to_line`），再对碾环
    ``MOVCTL`` 速度控制点做多行块裁剪（:func:`_clip_movctl_block`），后者由样本中的
    ``*_speed_lower`` / ``*_speed_upper`` 参数给出裁剪区间。

    :param template_path: 模板 KEY 文件路径
    :param param_table: 参数表 ``[[参数名...], [对象名...], [样本1值...], ...]``
    :param save_dir: 生成的 KEY 文件保存目录
    :return: 生成的 KEY 文件路径列表
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(template_path, "r", encoding="utf-8") as f:
        template_lines = f.readlines()

    param_names, object_names = param_table[0], param_table[1]
    generated: List[str] = []

    for sample_idx, values in enumerate(param_table[2:]):
        new_lines = [
            _apply_params_to_line(line, param_names, object_names, values)
            for line in template_lines
        ]
        # 碾环 MOVCTL 速度控制点块裁剪（无相关参数时零行为变化）
        clip_specs = _collect_speed_clip_specs(param_names, object_names, values)
        new_lines = _clip_movctl_block(new_lines, clip_specs)
        out_path = derive_output_path(template_path, save_dir, str(sample_idx), "KEY")
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        generated.append(out_path)
        logger.info(f"第 {sample_idx + 1} 个 KEY 文件已保存: {out_path}")

    return generated


def read_key_frames(key_files: Sequence[str]) -> List[List[str]]:
    """读取一组 KEY 文件的全部文本行。

    :param key_files: KEY 文件路径序列（通常是同一 DB 的各步导出）
    :return: 每个文件的行列表组成的列表
    """
    frames: List[List[str]] = []
    for key_file in key_files:
        with open(key_file, "r", encoding="utf-8") as f:
            frames.append(f.readlines())
    return frames


__all__ = [
    "format_deform_float",
    "derive_output_path",
    "generate_key_files",
    "read_key_frames",
]
