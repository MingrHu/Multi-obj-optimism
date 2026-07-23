"""pytest 公共 fixture 与 marker 说明。

marker：
- ``slow``：耗时用例（如 keras/DNN 训练），默认 ``-m "not slow"`` 跳过。
- ``deform``：依赖 Windows DEFORM 环境的用例，非 Windows 默认跳过。
- ``integration``：依赖仓库 ``data/`` 真实产物的集成用例。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from mobo.common.paths import KEY_FILE_DIR, MODELS_DIR, TEST_DIR


@pytest.fixture
def simulated_data_file():
    """示例数据集 ``data/TEST/simulated.txt``（3 输入 + grain + load）。"""
    path = TEST_DIR / "simulated.txt"
    if not path.exists():
        pytest.skip(f"示例数据集不存在：{path}")
    return str(path)


@pytest.fixture
def prg_model_dir():
    """PRG 代理模型目录 ``data/models/PRG``（缺失则跳过依赖它的用例）。"""
    d = MODELS_DIR / "PRG"
    if not (d / "grain_scalers.pkl").exists():
        pytest.skip(f"PRG 模型不存在：{d}")
    return d


@pytest.fixture
def ringroll_key():
    """碾环 KEY 文件 ``data/KEY_FILE/RINGROLL.KEY``（缺失则跳过）。"""
    path = KEY_FILE_DIR / "RINGROLL.KEY"
    if not path.exists():
        pytest.skip(f"KEY 文件不存在：{path}")
    return path


class _FakeScaler:
    """最简 StandardScaler 桩：mean_/scale_ 已知，transform/inverse_transform 线性。"""

    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) * self.scale_ + self.mean_


class _FakeModel:
    """线性预测桩：predict 返回输入各列的加权和。"""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights


@pytest.fixture
def fake_scaler_cls():
    return _FakeScaler


@pytest.fixture
def fake_model_cls():
    return _FakeModel
