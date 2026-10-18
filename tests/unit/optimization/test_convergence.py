"""三种模式的过程观测 不改变优化计算"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mobo.optimization.convergence import (
    calculate_change_rate,
    ga_callback,
    read_curve,
    record_convergence,
    rl_callback,
)


def _algorithm(generation, f, feasible):
    values = {"F": np.asarray(f), "FEAS": np.asarray(feasible)}
    termination = SimpleNamespace(force_termination=False)
    termination.terminate = lambda: setattr(termination, "force_termination", True)
    return SimpleNamespace(
        n_gen=generation, pop=SimpleNamespace(get=values.get),
        termination=termination,
    )


def test_calculate_change_rate_matches_early_stopping_formula():
    assert calculate_change_rate([100, 99, 96]) == pytest.approx(0.04)
    assert calculate_change_rate([0, 0, 0]) == 0
    with pytest.raises(ValueError, match="values不能为空"):
        calculate_change_rate([])


def test_single_observes_feasible_historical_best(tmp_path):
    path = tmp_path / "convergence.json"
    objectives = [{"name": "a", "minimize": True}, {"name": "b", "minimize": False}]
    with record_convergence(path, "single", objectives):
        observe = ga_callback({"objective_config": objectives})
        f = [[100], [3]]
        observe(_algorithm(1, f, [False, False]))
        observe(_algorithm(2, f, [False, True]))
        observe(_algorithm(3, [[8]], [True]))
    curve = json.loads(path.read_text())
    assert curve["x"] == [1, 2, 3]
    assert curve["y"] == [None, 3, 3]
    assert not path.with_suffix(".json.tmp").exists()
    assert rl_callback() is None


def test_rl_records_completed_training_episode_rewards_only(tmp_path):
    path = tmp_path / "convergence.json"
    with record_convergence(path, "reinforcement_learning", []):
        callback = rl_callback()
        for step, reward, done in [(1, 2, False), (2, 4, True), (3, -2, True)]:
            callback.num_timesteps = step
            callback.locals = {"rewards": np.array([reward]), "dones": np.array([done])}
            assert callback._on_step()
    curve = json.loads(path.read_text())
    assert curve["x"] == [1, 2]
    assert curve["y"] == [6, 2]


def test_single_curve_stops_when_trailing_change_is_below_threshold(tmp_path):
    path = tmp_path / "single_early_stop.json"
    request = {"optimizer_config": {
        "early_stop_enabled": True, "early_stop_threshold": 0.05,
        "early_stop_window": 3, "early_stop_min_points": 4,
    }}
    with record_convergence(path, "single", []):
        observe = ga_callback(request)
        algorithms = []
        for generation, value in enumerate((10.0, 9.0, 8.1, 8.0, 7.95), start=1):
            algorithm = _algorithm(generation, [[value]], [True])
            algorithms.append(algorithm)
            observe(algorithm)
    assert not algorithms[-2].termination.force_termination
    assert algorithms[-1].termination.force_termination
    early = json.loads(path.read_text())["early_stopping"]
    assert early["stopped_early"] and early["stop_point"] == 5
    assert early["change_rate"] < 0.05


def test_multi_curve_uses_hypervolume_for_early_stopping(tmp_path):
    path = tmp_path / "multi_early_stop.json"
    problem = SimpleNamespace(
        objectives=[SimpleNamespace(name="a", minimize=True, y_index=0)],
        scalers={"scaler_y_0": SimpleNamespace(mean_=[0], scale_=[1])},
    )
    request = {"optimizer_config": {
        "early_stop_enabled": True, "early_stop_threshold": 0.05,
        "early_stop_window": 3, "early_stop_min_points": 3,
    }}
    with record_convergence(path, "multi", []):
        observe = ga_callback(request)
        algorithms = []
        for generation in range(1, 4):
            algorithm = _algorithm(generation, [[1.0]], [True])
            algorithm.problem = problem
            algorithms.append(algorithm)
            observe(algorithm)
    assert algorithms[-1].termination.force_termination
    assert json.loads(path.read_text())["early_stopping"]["stopped_early"]


def test_rl_curve_returns_false_to_stop_learning(tmp_path):
    path = tmp_path / "rl_early_stop.json"
    request = {"optimizer_config": {
        "early_stop_enabled": True, "early_stop_threshold": 0.05,
        "early_stop_window": 3, "early_stop_min_points": 3,
    }}
    with record_convergence(path, "reinforcement_learning", []):
        callback = rl_callback(request)
        decisions = []
        for step in range(1, 4):
            callback.num_timesteps = step
            callback.locals = {"rewards": np.array([1.0]), "dones": np.array([True])}
            decisions.append(callback._on_step())
    assert decisions == [True, True, False]
    assert json.loads(path.read_text())["early_stopping"]["stopped_early"]


def test_context_is_reset_on_failure_and_keeps_partial_curve(tmp_path):
    path = tmp_path / "convergence.json"
    with pytest.raises(RuntimeError), record_convergence(path, "single", []):
        ga_callback({})(_algorithm(1, [[2]], [True]))
        raise RuntimeError("stopped")
    assert json.loads(path.read_text())["y"] == [2]
    assert rl_callback() is None


def test_hypervolume_uses_fixed_training_scale_and_reference(tmp_path):
    path = tmp_path / "convergence.json"
    problem = SimpleNamespace(
        objectives=[SimpleNamespace(name="a", minimize=True, y_index=0),
                    SimpleNamespace(name="b", minimize=False, y_index=1)],
        scalers={"scaler_y_0": SimpleNamespace(mean_=[100], scale_=[10]),
                 "scaler_y_1": SimpleNamespace(mean_=[500], scale_=[100])},
    )
    with record_convergence(path, "multi", []):
        observe = ga_callback({})
        for generation, f, feasible in [
            (1, [[110, -600]], [False]),
            (2, [[110, -600]], [True]),
            (3, [[120, -800]], [True]),
            (4, [[1000, -1000]], [True]),
        ]:
            algorithm = _algorithm(generation, f, feasible)
            algorithm.problem = problem
            observe(algorithm)
    curve = json.loads(path.read_text())
    assert curve["x"] == [1, 2, 3, 4]
    assert curve["y"] == [0, 15, 14, 0]
    assert curve["metric"] == "hypervolume" and curve["direction"] == "max"
    assert curve["hv_config"] == {
        "objective_names": ["a", "b"], "offset": [100, -500],
        "scale": [10, 100], "reference_point": [4, 4],
    }


@pytest.mark.parametrize("mode", ["single", "multi", "reinforcement_learning"])
def test_read_legacy_curve_never_returns_multiple_lines(tmp_path, mode):
    path = tmp_path / "convergence.json"
    path.write_text(json.dumps({"x": [100, 200], "y": {
        "weighted_objective": [3, 2], "mean_episode_reward": [1, 2], "a": [4, 3],
    }}))
    curve = read_curve(path, mode, [])
    assert isinstance(curve["y"], list)
    if mode == "single":
        assert curve["x"] == [100, 200] and curve["y"] == [3, 2]
    elif mode == "reinforcement_learning":
        assert curve["x"] == [1, 2] and curve["y"] == [1, 2]
    else:
        assert curve["x"] == [] and curve["y"] == []
