"""工艺参数替换能力注册与路由。"""

from __future__ import annotations

from collections.abc import Iterable

from .base import DocumentReplacer, LineReplacer, ReplacementKind, ReplacerSpec


class ReplacementRegistry:
    """按工艺参数名解析原子替换能力。"""

    def __init__(self) -> None:
        self._specs: dict[str, ReplacerSpec] = {}

    def register_fn(
        self,
        parameter_names: str | Iterable[str],
        fn: LineReplacer | DocumentReplacer,
        *,
        kind: ReplacementKind,
        name: str | None = None,
    ) -> ReplacerSpec:
        names = (parameter_names,) if isinstance(parameter_names, str) else tuple(parameter_names)
        if not names:
            raise ValueError("parameter_names 不能为空")
        spec = ReplacerSpec(name or names[0], fn, kind)
        for parameter_name in names:
            self._specs[parameter_name] = spec
        return spec

    def resolve(self, parameter_name: str) -> ReplacerSpec | None:
        return self._specs.get(parameter_name)


__all__ = ["ReplacementRegistry"]
