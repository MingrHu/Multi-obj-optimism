"""随机森林训练入口的固定并行参数测试。"""

import numpy as np

from mobo.surrogate import random_forest


class _IdentityScaler:
    def inverse_transform(self, values):
        return values


class _FakeRandomForest:
    kwargs = None

    def __init__(self, **kwargs):
        _FakeRandomForest.kwargs = kwargs

    def fit(self, x, y):
        return self

    def predict(self, x):
        return np.zeros(len(x))


def test_rf_run_uses_all_available_cpu_cores(monkeypatch):
    x = np.array([[0.0], [1.0]])
    y = np.array([[0.0], [1.0]])
    monkeypatch.setattr(random_forest, "load_and_preprocess_data", lambda *args: (x, y))
    monkeypatch.setattr(
        random_forest,
        "split_data_without_val",
        lambda *args: (
            x,
            x,
            [y.ravel()],
            [y.ravel()],
            {"scaler_y_0": _IdentityScaler()},
        ),
    )
    monkeypatch.setattr(random_forest, "RandomForestRegressor", _FakeRandomForest)
    monkeypatch.setattr(random_forest, "save_model", lambda *args: None)

    random_forest.rf_run("unused.tsv", ["x", "target"], 1)

    assert _FakeRandomForest.kwargs == {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
    }
