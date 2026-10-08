"""真实 DNN 子进程训练与多输出推理兼容性"""

import joblib
import numpy as np
import pytest
import threading

from mobo.surrogate.common import model_output_dir
from mobo.surrogate.dnn import dnn_run
from mobo.surrogate import dnn_process


@pytest.mark.slow
def test_parallel_dnn_models_keep_all_scalers(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 3))
    y = np.column_stack([x[:, 0] + x[:, 1], 10 * x[:, 2] + 100])
    dataset = tmp_path / "dataset.tsv"
    np.savetxt(dataset, np.column_stack([x, y]), delimiter="\t")
    models = tmp_path / "models"
    monkeypatch.setenv("MOBO_DNN_MAX_WORKERS", "2")
    monkeypatch.setattr(dnn_process, "worker_count", lambda targets: min(2, targets))
    with model_output_dir(models):
        dnn_run(str(dataset), ["X1", "X2", "X3", "Y1", "Y2"], 3, {"epochs": 2})

    from keras.models import load_model

    first_scalers = joblib.load(models / "Y1_scalers.pkl")
    for index, name in enumerate(["Y1", "Y2"]):
        scalers = joblib.load(models / f"{name}_scalers.pkl")
        assert set(scalers) == {"scaler_X", "scaler_y_0", "scaler_y_1"}
        np.testing.assert_allclose(scalers[f"scaler_y_{index}"].mean_,
                                   first_scalers[f"scaler_y_{index}"].mean_)
        model = load_model(models / f"{name}_model.keras")
        scaled = model(scalers["scaler_X"].transform(x[:2]), training=False).numpy()
        raw = scalers[f"scaler_y_{index}"].inverse_transform(scaled)
        assert raw.shape == (2, 1)
        assert np.isfinite(raw).all()


@pytest.mark.slow
def test_cancelling_parallel_dnn_reaps_real_children(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.tsv"
    np.savetxt(dataset, np.ones((50, 3)), delimiter="\t")
    cancel = threading.Event()
    children = []
    original_popen = dnn_process.subprocess.Popen

    def tracked_popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        if len(children) == 2:
            cancel.set()
        return child

    monkeypatch.setenv("MOBO_DNN_MAX_WORKERS", "2")
    monkeypatch.setattr(dnn_process, "worker_count", lambda targets: min(2, targets))
    monkeypatch.setattr(dnn_process.subprocess, "Popen", tracked_popen)
    with model_output_dir(tmp_path / "models"), dnn_process.training_control(cancel):
        with pytest.raises(RuntimeError, match="中止"):
            dnn_run(str(dataset), ["X", "Y1", "Y2"], 1)
    assert len(children) == 2
    assert all(child.poll() is not None for child in children)
