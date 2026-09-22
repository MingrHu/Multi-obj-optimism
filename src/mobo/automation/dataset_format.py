"""数据集数值输出格式"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_DOWN, localcontext
from typing import Any, Sequence

_TWO_DECIMALS = Decimal("0.00")


def format_dataset_value(value: Any) -> str:
    """数值向零截断为两位小数 非数值与非有限值保持原文"""
    text = str(value)
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return text
    precision = max(
        28,
        len(number.as_tuple().digits) + abs(number.as_tuple().exponent) + 2,
    )
    with localcontext() as context:
        context.prec = precision
        truncated = number.quantize(_TWO_DECIMALS, rounding=ROUND_DOWN)
    if truncated == 0:
        truncated = abs(truncated)
    return format(truncated, ".2f")


def format_dataset_row(row: Sequence[Any]) -> list[str]:
    """格式化数据集一行的所有字段"""
    return [format_dataset_value(value) for value in row]


__all__ = ["format_dataset_row", "format_dataset_value"]
