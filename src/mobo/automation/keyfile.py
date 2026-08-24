"""KEY 文件文本处理。

负责 DEFORM KEY 文件的纯文本层面操作，不涉及子进程：

- :func:`format_deform_float`：把数值格式化为 DEFORM 要求的科学计数法；
- :func:`derive_output_path`：由源文件名派生输出路径；
- :func:`generate_key_files`：把工艺参数写入模板 KEY，批量生成输入 KEY；
- :func:`read_key_frames`：读取一组 KEY 文件的全部文本行

KEY 文件本质是文本文件，目标行格式为：``<关键字> <对象ID> ... <参数值>``。
本模块只编排文件读写并把参数请求路由到 :mod:`mobo.replacement` 原子能力层。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, cast

from mobo.common.logging import logger
from mobo.replacement import ParameterBinding, registry as replacement_registry
from mobo.replacement.base import DocumentReplacer, LineReplacer
from mobo.replacement.deform_parameters import (
    collect_speed_scale_specs,
    format_deform_float,
)
from .config import DeformConfig


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
    """按注册表路由单行替换能力。"""
    return _apply_bindings_to_line(
        line, _parameter_bindings(param_names, object_names, values)
    )


def _apply_bindings_to_line(
    line: str, bindings: Sequence[ParameterBinding]
) -> str:
    """把预解析的参数请求路由到行级能力。"""
    for binding in bindings:
        spec = replacement_registry.resolve(binding.name)
        if spec is None or spec.kind != "line":
            continue
        replacer = cast(LineReplacer, spec.fn)
        result = replacer(line, binding)
        if result.matched:
            return result.text
    return line


def _parameter_bindings(
    param_names: Sequence[str],
    object_names: Sequence[str],
    values: Sequence[str],
) -> list[ParameterBinding]:
    """把三列表头和值组装成替换注册表的请求。"""
    return [
        ParameterBinding(
            name=param_names[pos],
            object_name=object_names[pos],
            object_id=DeformConfig.get_object_id(object_names[pos]),
            value=value,
        )
        for pos, value in enumerate(values)
    ]


def _collect_speed_scale_specs(
    param_names: Sequence[str],
    object_names: Sequence[str],
    values: Sequence[str],
) -> dict[str, tuple[float, float]]:
    """兼容旧私有接口，并把请求路由到速度区间原子能力。"""
    return collect_speed_scale_specs(_parameter_bindings(param_names, object_names, values))


def _apply_document_replacers(
    lines: Sequence[str], bindings: Sequence[ParameterBinding]
) -> List[str]:
    """按注册顺序调用文档级能力，同一能力只执行一次。"""
    result = list(lines)
    applied: set[str] = set()
    for binding in bindings:
        spec = replacement_registry.resolve(binding.name)
        if spec is None or spec.kind != "document" or spec.name in applied:
            continue
        replacer = cast(DocumentReplacer, spec.fn)
        result = replacer(result, bindings)
        applied.add(spec.name)
    return result


def generate_key_files(template_path: str, param_table: List[List[str]], save_dir: str) -> List[str]:
    """把工艺参数写入模板 KEY，批量生成输入 KEY 文件。

    ``param_table`` 前两行为固定表头：第 0 行是参数名、第 1 行是对象名；从第 2 行起
    每行是一个样本的参数取值。为每个样本生成一个 KEY 文件
    
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
        new_lines = apply_parameters(template_lines, param_names, object_names, values)
        out_path = derive_output_path(template_path, save_dir, str(sample_idx), "KEY")
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        generated.append(out_path)
        logger.info(f"第 {sample_idx + 1} 个 KEY 文件已保存: {out_path}")

    return generated


def apply_parameters(lines: Sequence[str], param_names: Sequence[str],
                     object_names: Sequence[str], values: Sequence[str]) -> List[str]:
    """把一组样本参数应用到 KEY 文本行并返回新列表。"""
    bindings = _parameter_bindings(param_names, object_names, values)
    rendered = [
        _apply_bindings_to_line(line, bindings)
        for line in lines
    ]
    return _apply_document_replacers(rendered, bindings)


def write_parameterized_key(template_path: str, output_path: str,
                            param_names: Sequence[str], object_names: Sequence[str],
                            values: Sequence[str]) -> str:
    """基于模板写出一个参数化 KEY，不修改模板文件。"""
    with open(template_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rendered = apply_parameters(lines, param_names, object_names, values)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(rendered)
    return output_path


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
    "apply_parameters",
    "write_parameterized_key",
    "generate_key_files",
    "read_key_frames",
]
