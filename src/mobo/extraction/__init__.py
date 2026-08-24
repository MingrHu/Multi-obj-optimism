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
    _extractEffectiveStrainStdv,
    _extractGrainMorph,
    _extractMaxLoad,
    _extractMaxStress,
    _extractMaterialFill,
    _extractUsrGrainStdv,
    calculate_von_mises,
)
from .registry import ExtractorRegistry, registry
from .ring_roundness import extract_ring_roundness

# ---- 注册内置提取器（通用工件的 DEFORM 目标提取，key_lines 约定）----
registry.register_fn("generic", "stress", _extractMaxStress, kind="key_lines",
                     description="模具/工件最大等效应力（Von Mises）")
registry.register_fn("generic", "load", _extractMaxLoad, kind="key_lines",
                     description="模具/工件 FORCE 分量绝对值最大值")
registry.register_fn("generic", "strain_std", _extractEffectiveStrainStdv,
                     kind="key_lines", description="单元等效应变标准差")
registry.register_fn("generic", "grain", _extractUsrGrainStdv, kind="key_lines",
                     description="自定义晶粒模型 USRELM 尺寸标准差")
registry.register_fn("generic", "grain_morph", _extractGrainMorph, kind="key_lines",
                     description="GRAIN 指定晶粒组织分量平均值")
registry.register_fn("ring", "material_fill", _extractMaterialFill, kind="key_lines",
                     description="碾环材料填充性（占位，尚未定义计算方法）")

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
