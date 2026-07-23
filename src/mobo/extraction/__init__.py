"""原子能力层：按工件类型分派的 KEY 文件目标提取。

导入本包时会把内置提取器注册到模块级单例 :data:`registry`：

- ``("generic", "stress"/"load"/"grain")``：DEFORM 目标提取（``key_lines`` 约定）。
- ``("ring", "roundness_inner"/"roundness_outer")``：碾环圆度（``key_file`` 约定）。

新增提取器时，只需在此处或调用方通过 :meth:`ExtractorRegistry.register_fn`
注册，无需改动底层提取函数体。
"""

from functools import partial

from .base import ExtractorKind, ExtractorSpec, KeyFileExtractor, KeyLinesExtractor
from .deform_targets import (
    _extractGrainStdv,
    _extractMaxLoad,
    _extractMaxStress,
    calculate_von_mises,
)
from .registry import ExtractorRegistry, registry
from .ring_roundness import extract_ring_roundness

# ---- 注册内置提取器（通用工件的 DEFORM 目标提取，key_lines 约定）----
registry.register_fn("generic", "stress", _extractMaxStress, kind="key_lines",
                     description="模具/工件最大等效应力（Von Mises）")
registry.register_fn("generic", "load", _extractMaxLoad, kind="key_lines",
                     description="上模最大载荷 FORCE")
registry.register_fn("generic", "grain", _extractGrainStdv, kind="key_lines",
                     description="锻件晶粒尺寸标准差")

# ---- 注册碾环工件的圆度提取（key_file 约定）----
registry.register_fn(
    "ring", "roundness_inner",
    partial(extract_ring_roundness, which="inner"),
    kind="key_file", description="碾环内圈圆度（MZC/LSC）",
)
registry.register_fn(
    "ring", "roundness_outer",
    partial(extract_ring_roundness, which="outer"),
    kind="key_file", description="碾环外圈圆度（MZC/LSC）",
)


__all__ = [
    "ExtractorKind",
    "ExtractorSpec",
    "KeyFileExtractor",
    "KeyLinesExtractor",
    "ExtractorRegistry",
    "registry",
    "extract_ring_roundness",
    "calculate_von_mises",
]
