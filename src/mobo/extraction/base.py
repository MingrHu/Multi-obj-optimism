"""原子能力层的类型与协议定义。

统一描述「按工件类型分派的 KEY 文件目标提取器」，容纳两种异构调用约定：

- ``key_lines``：输入为 KEY 文件解析出的多帧文本行，签名
  ``(all_lines: list[list[str]], obj_id: str, in_progress: bool) -> str``，
  对应 :mod:`mobo.extraction.deform_targets` 中的 ``_extract*``。
- ``key_file``：输入为 KEY 文件路径，签名 ``(key_path, **kwargs) -> float``，
  对应 :func:`mobo.extraction.ring_roundness.extract_ring_roundness`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Protocol, runtime_checkable

# 提取器调用约定
ExtractorKind = Literal["key_lines", "key_file"]


@runtime_checkable
class KeyLinesExtractor(Protocol):
    """基于 KEY 文本行的提取器协议。"""

    def __call__(self, all_lines: List[List[str]], obj_id: str, in_progress: bool) -> str:
        ...


@runtime_checkable
class KeyFileExtractor(Protocol):
    """基于 KEY 文件路径的提取器协议。"""

    def __call__(self, key_path: Any, **kwargs: Any) -> float:
        ...


@dataclass(frozen=True)
class ExtractorSpec:
    """一个已注册的原子提取能力。

    :param fn: 提取函数本体
    :param kind: 调用约定（``key_lines`` / ``key_file``）
    :param workpiece_type: 工件类型（如 ``generic`` / ``ring``）
    :param target_name: 目标名称（如 ``stress`` / ``load`` / ``grain`` / ``roundness_inner``）
    :param description: 能力描述
    """

    fn: Callable[..., Any]
    kind: ExtractorKind
    workpiece_type: str
    target_name: str
    description: str = ""


__all__ = [
    "ExtractorKind",
    "KeyLinesExtractor",
    "KeyFileExtractor",
    "ExtractorSpec",
]
