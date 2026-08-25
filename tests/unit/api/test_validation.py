"""对外 JSON 协议的纯参数校验测试。"""

import json

import pytest

from mobo.api.validation import (
    ApiValidationError,
    normalize_optimization_request,
    normalize_surrogate_request,
)


def test_surrogate_accepts_json_and_normalizes_aliases(tmp_path):
    dataset = tmp_path / "samples.tsv"
    dataset.write_text("1\t2\n2\t3\n", encoding="utf-8")
    request = json.dumps({
        "data_file": str(dataset),
        "all_var_list": ["temperature", "grain"],
        "input_var_count": 1,
        "model_index": 0,
        "params": {"degree": 2},
    })

    normalized = normalize_surrogate_request(request)

    assert normalized["vars_out"] == ["temperature", "grain"]
    assert normalized["n_vars"] == 1
    assert normalized["biz_params"] == {"degree": 2}


def test_surrogate_rejects_hyperparameter_that_bottom_layer_ignores(tmp_path):
    dataset = tmp_path / "samples.tsv"
    dataset.write_text("1\t2\n", encoding="utf-8")
    with pytest.raises(ApiValidationError, match="尚不支持自定义超参数"):
        normalize_surrogate_request({
            "data_file": str(dataset),
            "all_var_list": ["x", "y"],
            "input_var_count": 1,
            "model_index": 4,
            "params": {"epochs": 300},
        })


def test_optimization_fills_defaults_and_checks_name_mapping():
    normalized = normalize_optimization_request({
        "model_id": "tr_1",
        "objective_names": ["grain", "load"],
        "input_var_count": 2,
        "all_var_list": ["temperature", "speed", "grain", "load"],
        "decision_var_indices": [1],
        "decision_var_names": ["speed"],
        "decision_bounds": [{"lower": 10, "upper": 50}],
        "constraints": [{
            "target_obj": "load",
            "constraint_kind": "upper",
            "limit_value": 330000,
        }],
    })

    assert normalized["optimizer_config"]["n_gen"] == 200
    assert normalized["objective_config"][0] == {"name": "grain", "minimize": True}
    assert normalized["decision_bounds"][0] == {
        "lower": 10.0, "upper": 50.0, "desc": None
    }


def test_optimization_rejects_decision_name_mismatch():
    with pytest.raises(ApiValidationError, match="必须与下标对应"):
        normalize_optimization_request({
            "model_id": "tr_1",
            "objective_names": ["grain"],
            "input_var_count": 1,
            "all_var_list": ["temperature", "grain"],
            "decision_var_indices": [0],
            "decision_var_names": ["wrong"],
            "decision_bounds": [{"lower": 800, "upper": 1000}],
        })


def test_optimization_rejects_path_traversal_id():
    with pytest.raises(ApiValidationError, match="只能包含"):
        normalize_optimization_request({"model_id": "tr_../../outside"})
