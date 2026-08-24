from mobo.replacement.base import LineReplacement
from mobo.replacement.registry import ReplacementRegistry


def test_registry_resolves_parameter_capability():
    registry = ReplacementRegistry()

    def fn(line, binding):
        return LineReplacement(line, False)

    spec = registry.register_fn("temperature", fn, kind="line")

    assert registry.resolve("temperature") is spec
    assert registry.resolve("missing") is None


def test_registry_groups_document_parameter_aliases():
    registry = ReplacementRegistry()

    def fn(lines, bindings):
        return list(lines)

    spec = registry.register_fn(
        ("speed_lower", "speed_upper"), fn,
        kind="document", name="speed_profile",
    )

    assert registry.resolve("speed_lower") is spec
    assert registry.resolve("speed_upper") is spec
    assert spec.name == "speed_profile"
