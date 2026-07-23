"""集成测试：用真实 RINGROLL.KEY 跑圆度提取并验证 registry 分派。"""

import pytest

pytestmark = pytest.mark.integration


def test_extract_ring_roundness_inner_outer(ringroll_key):
    from mobo.extraction.ring_roundness import extract_ring_roundness

    inner = extract_ring_roundness(ringroll_key, which="inner", samples=500)
    outer = extract_ring_roundness(ringroll_key, which="outer", samples=500)
    assert inner >= 0.0
    assert outer >= 0.0


def test_registry_dispatch_ring(ringroll_key):
    from mobo.extraction import registry

    spec = registry.resolve("ring", "roundness_inner")
    assert spec.kind == "key_file"
    value = spec.fn(ringroll_key, samples=500)
    assert value >= 0.0


def test_extract_ring_roundness_invalid_which(ringroll_key):
    from mobo.extraction.ring_roundness import extract_ring_roundness

    with pytest.raises(ValueError):
        extract_ring_roundness(ringroll_key, which="middle")
