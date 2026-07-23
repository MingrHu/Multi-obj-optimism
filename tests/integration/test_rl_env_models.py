"""集成测试：用真实 PRG 模型验证 ForgingEnv 预测可跑且确定。"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_forging_env_predict_with_real_models(prg_model_dir):
    from mobo.optimization.rl.env import ForgingEnv

    env = ForgingEnv(model_family="PRG")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3,)

    # 固定输入 -> 预测确定（同一输入两次结果一致）
    x = np.array([900.0, 500.0, 30.0], dtype=np.float32)
    r1 = env._predict(x)
    r2 = env._predict(x)
    assert r1["grain"] == pytest.approx(r2["grain"])
    assert r1["load"] == pytest.approx(r2["load"])
    # 数值应在合理物理范围内
    assert 0 < r1["grain"] < 200
    assert r1["load"] > 0


def test_forging_env_step(prg_model_dir):
    from mobo.optimization.rl.env import ForgingEnv

    env = ForgingEnv(model_family="PRG")
    env.reset(seed=1, options={"weights": [0.5, 0.5]})
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.1, -0.1, 0.2], dtype=np.float32)
    )
    assert "grain" in info and "load" in info
    assert not terminated and not truncated
