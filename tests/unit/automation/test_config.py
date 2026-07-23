"""DeformConfig 配置映射测试。"""

from mobo.automation.config import DeformConfig
from mobo.extraction.deform_targets import (
    _extractGrainStdv,
    _extractMaxLoad,
    _extractMaxStress,
)


def test_get_key_var():
    assert DeformConfig.get_key_var("temp") == "NDTMP"
    assert DeformConfig.get_key_var("speed") == "MOVCTL"
    assert DeformConfig.get_key_var("unknown") is None


def test_get_object_id():
    assert DeformConfig.get_object_id("workpiece") == "1"
    assert DeformConfig.get_object_id("topdie") == "2"
    assert DeformConfig.get_object_id("butdie") == "3"
    assert DeformConfig.get_object_id("nope") is None


def test_get_target_function_maps_to_extractors():
    assert DeformConfig.get_target_function("stress") is _extractMaxStress
    assert DeformConfig.get_target_function("load") is _extractMaxLoad
    assert DeformConfig.get_target_function("grain") is _extractGrainStdv
    assert DeformConfig.get_target_function("missing") is None
