"""代理模型超参数规范测试。"""

import pytest

from mobo.surrogate.hyperparameters import (
    default_model_params,
    normalize_model_params,
    parameter_catalog,
)


def test_defaults_cover_all_five_model_families():
    catalog = parameter_catalog()
    assert set(catalog) == {"PRG", "SVR", "RF", "KM", "DNN"}
    assert default_model_params("PRG")["degree"] == 2
    assert default_model_params("RF")["n_estimators"] == 300
    assert default_model_params("DNN")["epochs"] == 1000


def test_overrides_are_merged_without_dropping_defaults():
    params = normalize_model_params("SVR", {"C": 12.5, "kernel": "linear"})
    assert params["C"] == 12.5
    assert params["kernel"] == "linear"
    assert params["epsilon"] == 0.1
    assert params["gamma"] == "scale"


@pytest.mark.parametrize(
    ("family", "params", "message"),
    [
        ("PRG", {"degree": 0}, "degree"),
        ("SVR", {"kernel": "unknown"}, "kernel"),
        ("RF", {"n_jobs": 0}, "n_jobs"),
        ("RF", {"bootstrap": False, "max_samples": 0.5}, "bootstrap"),
        ("KM", {"length_scale_lower": 10, "length_scale_upper": 1}, "length_scale"),
        ("DNN", {"dropout_1": 1.0}, "dropout_1"),
    ],
)
def test_invalid_parameters_are_rejected(family, params, message):
    with pytest.raises(ValueError, match=message):
        normalize_model_params(family, params)


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="不支持的超参数"):
        normalize_model_params("RF", {"made_up": 1})
