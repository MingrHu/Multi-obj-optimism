"""DeformConfig 配置映射测试。"""

from mobo.automation.config import DeformConfig


def test_get_key_var():
    assert DeformConfig.get_key_var("temp") == "NDTMP"
    assert DeformConfig.get_key_var("speed") == "MOVCTL"
    assert DeformConfig.get_key_var("unknown") is None


def test_get_object_id():
    assert DeformConfig.get_object_id("workpiece") == "1"
    assert DeformConfig.get_object_id("topdie") == "2"
    assert DeformConfig.get_object_id("butdie") == "3"
    assert DeformConfig.get_object_id("nope") is None


def test_get_target_function_lines_adapter_delegates(monkeypatch):
    # 文本类目标：适配器只用 frames 调底层函数（应力/载荷/晶粒）
    import mobo.automation.config as config

    seen = {}

    def fake_stress(frames, obj_id, in_progress):
        seen.update(frames=frames, obj_id=obj_id, in_progress=in_progress)
        return "3.14"

    monkeypatch.setattr(config, "_extractMaxStress", fake_stress)
    # partial 已在 import 期绑定原函数，这里直接验证适配器语义
    result = config._lines_target(fake_stress, ["k0.KEY"], [["line"]], "1", True)
    assert result == "3.14"
    assert seen == {"frames": [["line"]], "obj_id": "1", "in_progress": True}


def test_get_target_function_roundness_adapter(monkeypatch):
    # 几何类目标：适配器用最终步 KEY 文件计算圆度，对象名已转 ID
    import mobo.automation.config as config

    calls = {}

    def fake_extract(key_path, *, which, object_id):
        calls.update(key_path=key_path, which=which, object_id=object_id)
        return 1.2345678

    monkeypatch.setattr(config, "extract_ring_roundness", fake_extract)
    result = config._roundness_target("inner", ["k0.KEY", "k1.KEY"], [["l"]], "1", False)
    assert result == "1.234568"  # 6 位小数
    assert calls == {"key_path": "k1.KEY", "which": "inner", "object_id": 1}


def test_target_function_registry_has_all():
    for name in ("stress", "load", "grain", "roundness_inner", "roundness_outer"):
        assert DeformConfig.get_target_function(name) is not None
    assert DeformConfig.get_target_function("missing") is None
