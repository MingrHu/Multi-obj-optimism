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
from typing import List, Sequence

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
        key_var = DeformConfig.get_key_var(param_names[pos])
        object_id = DeformConfig.get_object_id(object_names[pos])
        # 目标行：第一个 token 为关键字，第二个 token 为对象 ID
        if key_var == tokens[0] and tokens[1] == object_id:
            return line.replace(tokens[-1], format_deform_float(value))
    return line


def generate_key_files(template_path: str, param_table: List[List[str]], save_dir: str) -> List[str]:
    """把工艺参数写入模板 KEY，批量生成输入 KEY 文件。

    ``param_table`` 前两行为固定表头：第 0 行是参数名、第 1 行是对象名；从第 2 行起
    每行是一个样本的参数取值。为每个样本生成一个 KEY 文件。

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
