"""SurrogateOptimizationProblem 测试（用桩 scaler/model，纯 numpy，无需真实模型）。"""

import numpy as np
import pytest

from mobo.optimization.ga.problem import (
    ConstraintSpec,
    ObjectiveSpec,
    SurrogateOptimizationProblem,
)


def _make_problem(fake_scaler_cls, fake_model_cls, **kwargs):
    scalers = {
        "scaler_X": fake_scaler_cls(mean=[900.0, 500.0, 30.0], scale=[10.0, 100.0, 5.0]),
        "scaler_y_0": fake_scaler_cls(mean=[25.0], scale=[5.0]),
        "scaler_y_1": fake_scaler_cls(mean=[300000.0], scale=[50000.0]),
    }
    objectives = [
        ObjectiveSpec(name="grain", model=fake_model_cls([1.0, 0.0, 0.0]), y_index=0, minimize=True),
        ObjectiveSpec(name="load", model=fake_model_cls([0.0, 1.0, 0.0]), y_index=1, minimize=True),
    ]
    return SurrogateOptimizationProblem(objectives=objectives, scalers=scalers, **kwargs)


def test_empty_objectives_raises(fake_scaler_cls):
    with pytest.raises(ValueError):
        SurrogateOptimizationProblem(objectives=[], scalers={"scaler_X": fake_scaler_cls([0], [1])})


def test_missing_scaler_x_raises(fake_model_cls):
    with pytest.raises(KeyError):
        SurrogateOptimizationProblem(
            objectives=[ObjectiveSpec("g", fake_model_cls([1.0]), 0)],
            scalers={},
        )


def test_bounds_and_dims(fake_scaler_cls, fake_model_cls):
    prob = _make_problem(
        fake_scaler_cls, fake_model_cls,
        decision_var_indices=[0, 1, 2],
        bounds=[(875, 965), (300, 700), (10, 50)],
    )
    assert prob.n_var == 3
    assert prob.n_obj == 2
    assert prob.n_constr == 0
    np.testing.assert_allclose(prob.xl, [875, 300, 10])
    np.testing.assert_allclose(prob.xu, [965, 700, 50])


def test_decision_var_index_out_of_range(fake_scaler_cls, fake_model_cls):
    with pytest.raises(ValueError):
        _make_problem(fake_scaler_cls, fake_model_cls, decision_var_indices=[0, 9])


def test_duplicate_decision_var_index(fake_scaler_cls, fake_model_cls):
    with pytest.raises(ValueError):
        _make_problem(fake_scaler_cls, fake_model_cls, decision_var_indices=[0, 0])


def test_evaluate_objectives(fake_scaler_cls, fake_model_cls):
    prob = _make_problem(
        fake_scaler_cls, fake_model_cls,
        decision_var_indices=[0, 1, 2],
        bounds=[(875, 965), (300, 700), (10, 50)],
    )
    out = {}
    # 取基准点（scaler mean）附近，scaled=0 -> model 预测 0 -> 反标准化回 mean
    prob._evaluate(np.array([900.0, 500.0, 30.0]), out)
    assert "F" in out
    np.testing.assert_allclose(out["F"], [25.0, 300000.0], rtol=1e-6)


def test_evaluate_with_constraints(fake_scaler_cls, fake_model_cls):
    prob = _make_problem(
        fake_scaler_cls, fake_model_cls,
        decision_var_indices=[0, 1, 2],
        bounds=[(875, 965), (300, 700), (10, 50)],
        constraints=[ConstraintSpec(objective="grain", kind="upper", value=30)],
    )
    out = {}
    prob._evaluate(np.array([900.0, 500.0, 30.0]), out)
    assert "G" in out
    # grain=25 <= 30 -> 约束值 25-30 = -5 (满足)
    np.testing.assert_allclose(out["G"], [-5.0], rtol=1e-6)


def test_weighted_single_uses_standardized_directional_objectives(
    fake_scaler_cls, fake_model_cls
):
    problem = _make_problem(
        fake_scaler_cls, fake_model_cls,
        decision_var_indices=[0, 1, 2],
        bounds=[(875, 965), (300, 700), (10, 50)],
        objective_mode="single", objective_weights=[0.25, 0.75],
    )
    problem.objectives[1] = ObjectiveSpec(
        name="load", model=fake_model_cls([0.0, 1.0, 0.0]),
        y_index=1, minimize=False,
    )
    out = {}
    problem._evaluate(np.array([910.0, 600.0, 30.0]), out)

    assert problem.n_obj == 1
    # scaled predictions are grain=1 and load=1; maximizing load changes its sign
    np.testing.assert_allclose(out["F"], [-0.5], rtol=1e-6)


def test_bounds_from_scaler_std_default(fake_scaler_cls, fake_model_cls):
    """未提供 bounds 时按 mean ± std*scale 推导。"""
    prob = _make_problem(fake_scaler_cls, fake_model_cls, decision_var_indices=[0])
    # mean=900, scale=10, std=3 -> [870, 930]
    np.testing.assert_allclose(prob.xl, [870.0])
    np.testing.assert_allclose(prob.xu, [930.0])
