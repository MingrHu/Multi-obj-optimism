"""DOE Flask API 的请求、响应与落盘测试。"""

import threading
from pathlib import Path

import pytest

from mobo.api import store
from mobo.api.app import create_app


def _client(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DOE_TASKS_DIR", tmp_path / "doe_tasks")
    return create_app({"TESTING": True}).test_client()


def test_add_list_progress_and_delete(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    created = client.post("/api/v1/doe/add", json={"id": "doe_1", "name": "test"})
    assert created.status_code == 201
    assert created.json["data"]["id"] == "doe_1"
    assert (tmp_path / "doe_tasks" / "doe_1" / "doe.json").is_file()

    listed = client.get("/api/v1/doe/list")
    assert listed.json["data"]["total"] == 1
    progress = client.get("/api/hust/v1/doe/train/progress?id=doe_1")
    assert progress.json["data"]["status"] == "not_started"
    assert progress.json["data"]["stage"] == "not_started"
    assert progress.json["data"]["progress"] == 0
    assert "optimization" not in progress.json["data"]

    deleted = client.post("/api/v1/doe/delete", json={"id": "doe_1"})
    assert deleted.status_code == 200
    assert not (tmp_path / "doe_tasks" / "doe_1").exists()


def test_add_rejects_duplicate_and_path_traversal(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "same"})
    assert client.post("/api/v1/doe/add", json={"id": "same"}).status_code == 409
    response = client.post("/api/v1/doe/add", json={"id": "../../outside"})
    assert response.status_code == 400


def test_sample_generation_uses_doe_directory(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "sample_1"})
    response = client.post("/api/v1/hust/doe/sample/generate", json={
        "id": "sample_1", "method": "full",
        "param_ranges": {"temperature": [900, 1000], "speed": [10, 20]},
        "level_nums": [2, 2],
    })
    assert response.status_code == 200
    assert response.json["data"]["sample_file"].endswith("sample_1-fullfactorial.txt")
    assert (tmp_path / "doe_tasks" / "sample_1" / "samples").is_dir()


def test_training_dataset_generation_uses_doe_directory(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "dataset_1"})
    request = {
        "id": "dataset_1", "input_names": ["temperature", "speed"],
        "target_names": ["grain", "load"],
        "param_ranges": {"temperature": [900, 1100], "speed": [10, 50]},
        "n_samples": 12, "seed": 7,
    }
    response = client.post("/api/v1/hust/doe/dataset/generate", json=request)

    assert response.status_code == 200
    data = response.json["data"]
    dataset = Path(data["data_file"])
    assert dataset.parent == tmp_path / "doe_tasks" / "dataset_1" / "training"
    assert data["all_var_list"] == ["temperature", "speed", "grain", "load"]
    assert len(dataset.read_text(encoding="utf-8").splitlines()) == 12
    assert all(len(line.split("\t")) == 4 for line in dataset.read_text().splitlines())


def test_training_request_is_async(monkeypatch, tmp_path):
    from mobo.api import service

    client = _client(monkeypatch, tmp_path)
    dataset = tmp_path / "dataset.txt"
    dataset.write_text("1\t2\n2\t3\n", encoding="utf-8")
    client.post("/api/v1/doe/add", json={"id": "train_1"})
    captured = []
    monkeypatch.setattr(service.registry, "start", lambda *args: captured.append(args[:2]))

    response = client.post("/api/v1/hust/doe/train/startTrain", json={
        "id": "train_1", "data_file": str(dataset),
        "all_var_list": ["x", "y"], "input_var_count": 1,
        "models": [{"model_index": 2}],
    })
    assert response.status_code == 202
    assert captured == [("train_1", "training")]


def test_all_documented_routes_exist(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    routes = {rule.rule for rule in client.application.url_map.iter_rules()}
    assert {
        "/api/v1/doe/add", "/api/v1/doe/list", "/api/v1/doe/delete",
        "/api/v1/hust/doe/sample/generate", "/api/v1/hust/doe/dataset/generate",
        "/api/hust/v1/doe/train/progress",
        "/api/v1/hust/doe/train/delete", "/api/v1/hust/doe/train/stop",
        "/api/v1/hust/doe/train/startTrain",
        "/api/v1/hust/doe/inference/startInference",
        "/api/v1/hust/doe/optimize/start", "/api/v1/hust/doe/optimize/stop",
        "/api/v1/hust/doe/optimize/getById",
    } <= routes
    assert "/api/v1/hust/doe/train/progress" not in routes


def test_training_worker_snapshots_model(monkeypatch, tmp_path):
    from mobo.api import service
    from mobo.surrogate import service as surrogate_service

    _client(monkeypatch, tmp_path)
    store.create({"id": "worker_1"})
    source = tmp_path / "legacy_model"
    source.mkdir()
    (source / "y_model.pkl").write_bytes(b"model")
    (source / "y_scalers.pkl").write_bytes(b"scalers")

    def fake_train(*args):
        return {"code": 0, "model_id": args[-1], "data": {
            "model_dir": str(source), "model_index": 2, "model_family": "RF",
            "target_names": ["y"], "train_cost_sec": 0.1,
        }}

    monkeypatch.setattr(surrogate_service, "train_surrogate", fake_train)
    request = {
        "data_file": "unused", "all_var_list": ["x", "y"], "input_var_count": 1,
        "models": [{"model_index": 2}], "evaluation": {"enabled": False},
    }
    service._run_training("worker_1", request, threading.Event())

    training = store.load("worker_1")["training"]
    assert training["status"] == "finished"
    assert training["stage"] == "finished"
    assert training["progress"] == 100
    assert Path(training["models"][0]["model_dir"]).is_dir()


def test_inference_loads_best_scored_model(monkeypatch, tmp_path):
    import joblib
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    from mobo.api import service

    _client(monkeypatch, tmp_path)
    store.create({"id": "infer_1"})
    model_dir = store.task_dir("infer_1") / "models" / "best"
    model_dir.mkdir(parents=True)
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[10.0], [20.0], [30.0]])
    scaler_x, scaler_y = StandardScaler().fit(x), StandardScaler().fit(y)
    model = LinearRegression().fit(scaler_x.transform(x), scaler_y.transform(y).ravel())
    joblib.dump(model, model_dir / "target_model.pkl")
    joblib.dump({"scaler_X": scaler_x, "scaler_y_0": scaler_y},
                model_dir / "target_scalers.pkl")
    store.update_section("infer_1", "training", status="finished", models=[{
        "model_id": "best", "model_dir": str(model_dir), "target_names": ["target"],
        "score": 0.9,
    }])

    result = service.start_inference({"id": "infer_1", "inputs": [[1.5]]})
    assert result["model_id"] == "best"
    assert result["predictions"][0][0] == pytest.approx(25.0)
