"""ForgingEnv 测试（用桩 scaler/model，避免加载真实模型文件）。"""

import numpy as np
import pytest

from mobo.optimization.rl.env import ForgingEnv


@pytest.fixture
def stub_env(monkeypatch, fake_scaler_cls, fake_model_cls):
    """构造一个不加载真实模型的 ForgingEnv。"""

    def fake_load_resources(self):
        self.scalers = {
            "scaler_X": fake_scaler_cls(mean=[920.0, 500.0, 30.0], scale=[26.0, 116.0, 12.0]),
            "scaler_y_0": fake_scaler_cls(mean=[25.0], scale=[5.0]),
            "scaler_y_1": fake_scaler_cls(mean=[300000.0], scale=[50000.0]),
        }
        self.scaler_X = self.scalers["scaler_X"]
        self.models = {
            "grain": fake_model_cls([1.0, 0.0, 0.0]),
            "load": fake_model_cls([0.0, 1.0, 0.0]),
        }

    monkeypatch.setattr(ForgingEnv, "_load_resources", fake_load_resources)
    return ForgingEnv(model_family="PRG")


def test_spaces(stub_env):
    assert stub_env.action_space.shape == (3,)
    assert stub_env.observation_space.shape == (3,)


def test_reset_sets_state_and_weights(stub_env):
    obs, info = stub_env.reset(seed=0, options={"weights": [0.7, 0.3]})
    assert obs.shape == (3,)
    assert np.all(obs >= stub_env.low_bound) and np.all(obs <= stub_env.high_bound)
    np.testing.assert_allclose(stub_env.weights, [0.7, 0.3])


def test_predict_at_mean(stub_env):
    stub_env.reset(seed=0)
    # 在 scaler 均值处，scaled=0 -> 预测 0 -> 反标准化回 y 均值
    r = stub_env._predict(np.array([920.0, 500.0, 30.0], dtype=np.float32))
    assert r["grain"] == pytest.approx(25.0, rel=1e-4)
    assert r["load"] == pytest.approx(300000.0, rel=1e-4)


def test_step_returns_reward_and_info(stub_env):
    stub_env.reset(seed=1, options={"weights": [0.5, 0.5]})
    obs, reward, terminated, truncated, info = stub_env.step(np.array([0.1, -0.1, 0.2], dtype=np.float32))
    assert obs.shape == (3,)
    assert "grain" in info and "load" in info
    assert isinstance(float(reward), float)
    assert terminated is False and truncated is False
