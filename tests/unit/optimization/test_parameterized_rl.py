"""通用代理模型 PPO 环境和无表头结果导出测试"""

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from mobo.optimization.rl.parameterized import SurrogatePPOEnv, run_parameterized_rl


def _artifacts(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    scaler_x = StandardScaler().fit(x)
    scalers = {"scaler_X": scaler_x}
    for index, (name, values) in enumerate((
        ("y1", 2.0 * x), ("limit", x),
    )):
        scaler_y = StandardScaler().fit(values)
        model = LinearRegression().fit(
            scaler_x.transform(x), scaler_y.transform(values).ravel()
        )
        joblib.dump(model, model_dir / f"{name}_model.pkl")
        scalers[f"scaler_y_{index}"] = scaler_y
    joblib.dump(scalers, model_dir / "y1_scalers.pkl")
    return model_dir


def _request():
    return {
        "mode": "single", "objective_names": ["y1"],
        "objective_config": [{"name": "y1", "minimize": True, "weight": 1.0}],
        "all_var_list": ["x", "y1", "limit"], "input_var_count": 1,
        "decision_var_indices": [0], "decision_var_names": ["x"],
        "decision_bounds": [{"lower": 0.0, "upper": 3.0}],
        "constraints": [{
            "target_obj": "limit", "constraint_kind": "upper", "limit_value": 1.5,
        }],
        "optimizer_config": {
            "total_timesteps": 2, "episode_steps": 3, "evaluation_episodes": 2,
            "learning_rate": 0.001, "constraint_penalty": 5.0,
            "seed": 42, "max_solutions": 10,
        },
    }


def test_parameterized_rl_environment_uses_requested_model_and_constraints(tmp_path):
    env = SurrogatePPOEnv(_request(), str(_artifacts(tmp_path)))
    observation, _ = env.reset(seed=2)
    observation, reward, terminated, truncated, info = env.step(np.zeros(1))

    assert observation.shape == (1,)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    assert set(info["objectives"]) == {"y1", "limit"}
    assert info["feasible"] == (info["objectives"]["limit"] <= 1.5)


def test_parameterized_rl_writes_headerless_tsv(monkeypatch, tmp_path):
    import stable_baselines3

    class FakePPO:
        def __init__(self, policy, env, **kwargs):
            self.env = env

        def learn(self, total_timesteps):
            return self

        def predict(self, observation, deterministic=True):
            return np.zeros_like(observation), None

    monkeypatch.setattr(stable_baselines3, "PPO", FakePPO)
    output = tmp_path / "rl.tsv"
    result = run_parameterized_rl(
        _request(), model_dir=str(_artifacts(tmp_path)), output_path=str(output)
    )

    assert result["columns"] == ["x", "y1", "feasible"]
    assert result["solution_count"] >= 1
    first_line = output.read_text(encoding="utf-8").splitlines()[0]
    assert len(first_line.split("\t")) == 3
    assert not first_line.startswith("x\t")
