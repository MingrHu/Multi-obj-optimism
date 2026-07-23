"""surrogate.common 纯逻辑测试（不训练模型）。"""

import time

import numpy as np
import pytest

from mobo.surrogate.common import (
    Time,
    evaluate_model,
    load_and_preprocess_data,
    normal_max_absolute_error,
    split_data_with_val,
    split_data_without_val,
)


def test_load_and_preprocess_data(simulated_data_file):
    X, Y = load_and_preprocess_data(simulated_data_file, ["a", "b", "c", "grain", "load"], 3)
    assert X.shape[1] == 3
    assert Y.shape[1] == 2
    assert X.shape[0] == Y.shape[0]


def test_load_and_preprocess_data_column_mismatch(simulated_data_file):
    with pytest.raises(ValueError):
        load_and_preprocess_data(simulated_data_file, ["a", "b"], 1)


def test_split_data_without_val_scalers():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    Y = rng.normal(size=(50, 2))
    Xtr, Xte, Ytr_list, Yte_list, scalers = split_data_without_val(X, Y)
    assert "scaler_X" in scalers
    assert "scaler_y_0" in scalers and "scaler_y_1" in scalers
    assert len(Ytr_list) == 2
    # 标准化后训练集近似零均值
    assert abs(float(np.mean(Xtr))) < 1e-6


def test_split_data_without_val_shape_mismatch():
    with pytest.raises(ValueError):
        split_data_without_val(np.zeros((10, 3)), np.zeros((9, 2)))


def test_split_data_with_val_type_check():
    with pytest.raises(TypeError):
        split_data_with_val([[1, 2]], np.zeros((1, 1)))


def test_split_data_with_val_three_way():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 4))
    Y = rng.normal(size=(100, 3))
    Xtr, Xval, Xte, Ytr, Yval, Yte, scalers = split_data_with_val(X, Y)
    assert Xtr.shape[0] + Xval.shape[0] + Xte.shape[0] == 100
    assert len(scalers) == 1 + 3  # scaler_X + 3 targets


def test_normal_max_absolute_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert normal_max_absolute_error(y_true, y_pred) == 0.0
    y_pred2 = np.array([1.0, 2.0, 5.0])
    assert normal_max_absolute_error(y_true, y_pred2) > 0


def test_evaluate_model_score():
    # 完美 R2 且最快时间 -> 分数最高
    s_best = evaluate_model(1.0, 1.0, 1.0, 10.0, 0.5, 0.5)
    s_worst = evaluate_model(0.0, 10.0, 1.0, 10.0, 0.5, 0.5)
    assert s_best > s_worst
    # T_max == T_min 时间项取 1.0
    assert evaluate_model(1.0, 5.0, 5.0, 5.0) == pytest.approx(1.0)


def test_time_utility():
    t = Time("blk").start()
    time.sleep(0.005)
    t.stop()
    assert t.get_duration("s") > 0
    assert t.get_duration("ms") == pytest.approx(t.get_duration("s") * 1000)
    with pytest.raises(ValueError):
        t.get_duration("bad")


def test_time_requires_start():
    with pytest.raises(RuntimeError):
        Time().stop()
