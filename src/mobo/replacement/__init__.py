"""工艺参数替换原子能力层。"""

from functools import partial

from .base import LineReplacement, ParameterBinding, ReplacerSpec
from .deform_parameters import (
    replace_keyword_last_value,
    replace_movctl_constant_speed,
    replace_object_temperature,
    replace_pressure_roll_speed_profile,
)
from .registry import ReplacementRegistry


registry = ReplacementRegistry()
registry.register_fn(
    "roll_tmp", partial(replace_keyword_last_value, keyword="REFTMP"),
    kind="line",
)
registry.register_fn(
    "driving_roll_rad_speed", partial(replace_keyword_last_value, keyword="ANGMOV"),
    kind="line",
)
registry.register_fn(
    "temp", partial(replace_keyword_last_value, keyword="NDTMP"),
    kind="line",
)
registry.register_fn(
    "speed", partial(replace_keyword_last_value, keyword="MOVCTL"),
    kind="line",
)
registry.register_fn(
    "pressure_roll_constant_speed",
    replace_movctl_constant_speed,
    kind="line",
)
registry.register_fn(
    "workpiece_temperature", replace_object_temperature,
    kind="document",
)
registry.register_fn(
    ("pressure_roll_speed_lower", "pressure_roll_speed_upper"),
    replace_pressure_roll_speed_profile,
    kind="document",
    name="pressure_roll_speed_profile",
)


__all__ = [
    "LineReplacement",
    "ParameterBinding",
    "ReplacementRegistry",
    "ReplacerSpec",
    "registry",
]
