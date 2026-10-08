"""数据集数值输出格式"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_DOWN, localcontext
from typing import Any, Sequence

DATASET_FIELD_WIDTH = 20
DATASET_DECIMAL_PLACES = 6


def format_dataset_value(value: Any, *, decimal_places: int = 2) -> str:
    """数值向零截断；默认保留历史两位格式，非数值与非有限值保持原文。"""
    if decimal_places < 0:
        raise ValueError("decimal_places 必须为非负整数")
    text = str(value)
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return text
    precision = max(
        28,
        len(number.as_tuple().digits) + abs(number.as_tuple().exponent) + decimal_places,
    )
    with localcontext() as context:
        context.prec = precision
        truncated = number.quantize(Decimal(1).scaleb(-decimal_places), rounding=ROUND_DOWN)
    if truncated == 0:
        truncated = abs(truncated)
    return format(truncated, f".{decimal_places}f")


def format_dataset_row(row: Sequence[Any]) -> list[str]:
    """无表头 TSV 的字段固定20字符右对齐、保留6位小数，缺失值不改变列序。"""
    fields = [
        format_dataset_value(value, decimal_places=DATASET_DECIMAL_PLACES)
        for value in row
    ]
    if any(len(field) > DATASET_FIELD_WIDTH for field in fields):
        raise ValueError(f"数据字段超过固定宽度 {DATASET_FIELD_WIDTH}，拒绝截断数值")
    return [field.rjust(DATASET_FIELD_WIDTH) for field in fields]


__all__ = ["DATASET_FIELD_WIDTH", "DATASET_DECIMAL_PLACES", "format_dataset_row", "format_dataset_value"]
