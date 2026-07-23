"""原子能力层的注册表与分派器。

上游按 ``(workpiece_type, target_name)`` 请求提取能力，:meth:`ExtractorRegistry.resolve`
返回对应的 :class:`~mobo.extraction.base.ExtractorSpec`，调用方再依据 ``spec.kind``
选择相应的调用约定。当某工件类型缺少专属提取器时，支持回退到通用工件
（默认 ``generic``），实现「根据上游传入的工件类型选择抽取函数」的诉求。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from .base import ExtractorKind, ExtractorSpec

F = Callable[..., Any]


class ExtractorRegistry:
    """``(workpiece_type, target_name) -> ExtractorSpec`` 的注册表。"""

    def __init__(self) -> None:
        self._specs: Dict[Tuple[str, str], ExtractorSpec] = {}

    def register(
        self,
        workpiece_type: str,
        target_name: str,
        *,
        kind: ExtractorKind,
        description: str = "",
    ) -> Callable[[F], F]:
        """装饰器形式注册提取器。

        :param workpiece_type: 工件类型
        :param target_name: 目标名称
        :param kind: 调用约定
        :param description: 能力描述
        :return: 装饰器（原样返回被装饰函数）
        """

        def decorator(fn: F) -> F:
            self.register_fn(
                workpiece_type, target_name, fn, kind=kind, description=description
            )
            return fn

        return decorator

    def register_fn(
        self,
        workpiece_type: str,
        target_name: str,
        fn: F,
        *,
        kind: ExtractorKind,
        description: str = "",
    ) -> None:
        """直接注册一个提取函数。

        :param workpiece_type: 工件类型
        :param target_name: 目标名称
        :param fn: 提取函数
        :param kind: 调用约定
        :param description: 能力描述
        """
        key = (workpiece_type, target_name)
        self._specs[key] = ExtractorSpec(
            fn=fn,
            kind=kind,
            workpiece_type=workpiece_type,
            target_name=target_name,
            description=description,
        )

    def get(self, workpiece_type: str, target_name: str) -> ExtractorSpec:
        """精确获取提取器；不存在则抛 :class:`KeyError`。

        :param workpiece_type: 工件类型
        :param target_name: 目标名称
        :return: 对应的 :class:`ExtractorSpec`
        """
        key = (workpiece_type, target_name)
        if key not in self._specs:
            raise KeyError(
                f"未注册的提取器：workpiece_type={workpiece_type!r}, target_name={target_name!r}"
            )
        return self._specs[key]

    def resolve(
        self,
        workpiece_type: str,
        target_name: str,
        *,
        fallback_workpiece: str = "generic",
    ) -> ExtractorSpec:
        """解析提取器，支持回退到通用工件。

        先按 ``(workpiece_type, target_name)`` 精确查找；找不到且 ``fallback_workpiece``
        与之不同的，再按 ``(fallback_workpiece, target_name)`` 查找。

        :param workpiece_type: 工件类型
        :param target_name: 目标名称
        :param fallback_workpiece: 回退工件类型
        :return: 对应的 :class:`ExtractorSpec`
        """
        key = (workpiece_type, target_name)
        if key in self._specs:
            return self._specs[key]

        if fallback_workpiece != workpiece_type:
            fallback_key = (fallback_workpiece, target_name)
            if fallback_key in self._specs:
                return self._specs[fallback_key]

        raise KeyError(
            f"无法解析提取器：workpiece_type={workpiece_type!r}, "
            f"target_name={target_name!r}（回退 {fallback_workpiece!r} 亦未命中）"
        )

    def targets_for(self, workpiece_type: str) -> List[str]:
        """列出某工件类型已注册的所有目标名称。

        :param workpiece_type: 工件类型
        :return: 目标名称列表（按注册顺序）
        """
        return [t for (w, t) in self._specs if w == workpiece_type]

    def keys(self) -> List[Tuple[str, str]]:
        """列出所有已注册的 ``(workpiece_type, target_name)`` 键。"""
        return list(self._specs.keys())


# 模块级单例
registry = ExtractorRegistry()


__all__ = ["ExtractorRegistry", "registry"]
