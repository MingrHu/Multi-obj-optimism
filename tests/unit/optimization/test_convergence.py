"""三种模式的过程观测 不改变优化计算"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mobo.optimization.convergence import ga_callback, read_curve, record_convergence, rl_callback


def _algorithm(generation, f, feasible):
    values = {"F": np.asarray(f), "FEAS": np.asarray(feasible)}
    return SimpleNamespace(n_gen=generation, pop=SimpleNamespace(get=values.get))


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
