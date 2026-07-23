"""ExtractorRegistry 注册/解析/回退测试。"""

import pytest

from mobo.extraction import registry
from mobo.extraction.base import ExtractorSpec
from mobo.extraction.registry import ExtractorRegistry


def test_builtin_registrations():
    keys = registry.keys()
    assert ("generic", "stress") in keys
    assert ("generic", "load") in keys
    assert ("generic", "grain") in keys
    assert ("ring", "roundness_inner") in keys
    assert ("ring", "roundness_outer") in keys


def test_get_returns_spec():
    spec = registry.get("generic", "stress")
    assert isinstance(spec, ExtractorSpec)
    assert spec.kind == "key_lines"
    assert spec.workpiece_type == "generic"


def test_get_missing_raises():
    with pytest.raises(KeyError):
        registry.get("nonexistent", "whatever")


def test_resolve_fallback_to_generic():
    # ring 未注册 stress，应回退到 generic
    spec = registry.resolve("ring", "stress")
    assert spec.workpiece_type == "generic"
    assert spec.target_name == "stress"


def test_resolve_no_fallback_hit_raises():
    with pytest.raises(KeyError):
        registry.resolve("ring", "does_not_exist")


def test_targets_for():
    assert set(registry.targets_for("generic")) >= {"stress", "load", "grain"}
    assert set(registry.targets_for("ring")) == {"roundness_inner", "roundness_outer"}


def test_register_decorator_and_register_fn():
    reg = ExtractorRegistry()

    @reg.register("blade", "twist", kind="key_lines", description="叶片扭转")
    def _twist(all_lines, obj, in_progress):
        return "0.0"

    assert reg.get("blade", "twist").fn is _twist
    assert reg.get("blade", "twist").description == "叶片扭转"

    reg.register_fn("blade", "camber", lambda p, **k: 1.0, kind="key_file")
    assert reg.get("blade", "camber").kind == "key_file"


def test_resolve_prefers_specific_over_fallback():
    reg = ExtractorRegistry()
    reg.register_fn("generic", "load", lambda *a, **k: "g", kind="key_lines")
    reg.register_fn("ring", "load", lambda *a, **k: "r", kind="key_lines")
    assert reg.resolve("ring", "load").workpiece_type == "ring"
