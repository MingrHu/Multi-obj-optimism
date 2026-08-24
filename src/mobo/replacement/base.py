"""工艺参数替换原子能力的公共类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence


ReplacementKind = Literal["line", "document"]


@dataclass(frozen=True)
class ParameterBinding:
    """一个样本中已经解析对象 ID 的工艺参数。

    :param name: 样本参数名，用于从注册表查找替换能力，如 ``roll_tmp``。
    :param object_name: 配置中的对象名称，如 ``workpiece``、``pressure_roll``。
    :param object_id: 对象名称解析得到的 KEY 对象编号；无法解析时为 ``None``。
    :param value: 当前样本给该参数提供的原始字符串值。
    """

    name: str                   # 参数名，也是替换能力的路由键
    object_name: str            # 业务对象名
    object_id: str | None       # KEY 文件中的对象 ID
    value: str                  # 当前样本参数值


@dataclass(frozen=True)
class LineReplacement:
    """单行替换能力的返回结果。

    :param text: 替换后的 KEY 文本行；未匹配时应返回原行。
    :param matched: 是否命中了关键字和对象 ID。命中后路由层停止尝试后续参数，
        用于保留“首个匹配参数优先”的历史语义。
    """

    text: str                   # 处理后的完整文本行
    matched: bool               # 当前能力是否确认处理了该行


# 行级能力：输入一行 KEY 和一个参数，返回匹配状态及处理后的行 比如单行替换
LineReplacer = Callable[[str, ParameterBinding], LineReplacement]

# 文档级能力：输入全部 KEY 行和全部参数，返回处理后的完整 KEY 行列表 比如块替换
DocumentReplacer = Callable[[Sequence[str], Sequence[ParameterBinding]], list[str]]


@dataclass(frozen=True)
class ReplacerSpec:
    """注册表返回的替换能力描述。

    :param name: 能力的唯一名称。多个参数可以指向同一名称的文档级能力，以保证
        一个样本中该能力只执行一次。
    :param fn: 实际执行替换的原子函数；签名由 ``kind`` 决定。
    :param kind: 能力类型，``line`` 表示单行替换，``document`` 表示整份 KEY 处理。
    """

    name: str                               # 原子能力名称
    fn: LineReplacer | DocumentReplacer     # 实际替换函数
    kind: ReplacementKind                   # "line" 或 "document"


__all__ = [
    "DocumentReplacer",
    "LineReplacer",
    "LineReplacement",
    "ParameterBinding",
    "ReplacementKind",
    "ReplacerSpec",
]
