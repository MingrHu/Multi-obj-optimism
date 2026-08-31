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
    assert set(created.json["data"]) == {
        "id", "name", "description", "metadata", "status", "stage",
        "progress", "created_at", "updated_at",
    }
    assert (tmp_path / "doe_tasks" / "doe_1" / "doe.json").is_file()

    listed = client.get("/api/v1/doe/list")
    assert listed.json["data"]["total"] == 1
    assert set(listed.json["data"]["items"][0]) == {
        "id", "name", "description", "metadata", "status", "stage",
        "progress", "created_at", "updated_at",
    }
    progress = client.get("/api/v1/hust/doe/train/progress?id=doe_1")
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
    data = response.json["data"]
    assert data["id"] == "sample_1"
    assert data["method"] == "full"
    assert data["sample_count"] == 4
    assert data["level_nums"] == [2, 2]
    assert data["resource_id"].startswith("tos-")
    assert "sample_file" not in data
    assert (tmp_path / "doe_tasks" / "sample_1" / "samples").is_dir()

    selected = client.get(
        "/api/v1/hust/doe/data/get",
        query_string=[("id", "sample_1"), ("resource_id", data["resource_id"]),
                      ("fields", "temperature")],
    )
    assert selected.status_code == 200
    assert selected.json["data"]["resource_type"] == "sample"
    assert sorted(selected.json["data"]["values"]["temperature"]) == [
        900.0, 900.0, 1000.0, 1000.0,
    ]


def test_lhs_response_reports_requested_and_actual_counts(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "lhs_1"})
    response = client.post("/api/v1/hust/doe/sample/generate", json={
        "id": "lhs_1", "method": "lhs",
        "param_ranges": {"temperature": [900, 1000], "speed": [10, 20]},
        "n_samples": 4,
    })

    assert response.status_code == 200
    data = response.json["data"]
    assert data["id"] == "lhs_1"
    assert data["n_samples"] == 4
    assert data["sample_count"] >= data["n_samples"]
    sample_file = store.load("lhs_1")["sample"]["sample_file"]
    lines = Path(sample_file).read_text(encoding="utf-8").splitlines()
    assert len(lines) == data["sample_count"]


def test_sample_generation_returns_documented_failures(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "invalid_sample"})
    invalid = client.post("/api/v1/hust/doe/sample/generate", json={
        "id": "invalid_sample", "method": "full",
        "param_ranges": {"x1": [0, 1], "x2": [10, 20]},
        "level_nums": [3],
    })
    missing = client.post("/api/v1/hust/doe/sample/generate", json={
        "id": "missing_doe", "method": "lhs",
        "param_ranges": {"x": [0, 1]}, "n_samples": 4,
    })

    assert invalid.status_code == 400
    assert invalid.json == {
        "code": 1,
        "message": "full 的 level_nums 必须是与 param_ranges 等长的正整数数组",
        "data": {},
    }
    assert missing.status_code == 404
    assert missing.json == {
        "code": 404, "message": "DOE 任务不存在：missing_doe", "data": {},
    }


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
    assert data["resource_id"].startswith("tos-")
    assert "data_file" not in data
    dataset = Path(store.load("dataset_1")["training"]["dataset"]["data_file"])
    assert dataset.parent == tmp_path / "doe_tasks" / "dataset_1" / "training"
    assert data["all_var_list"] == ["temperature", "speed", "grain", "load"]
    assert len(dataset.read_text(encoding="utf-8").splitlines()) == 12
    assert all(len(line.split("\t")) == 4 for line in dataset.read_text().splitlines())

    selected = client.get(
        "/api/v1/hust/doe/data/get",
        query_string=[("id", "dataset_1"), ("resource_id", data["resource_id"]),
                      ("fields", "speed"), ("fields", "grain")],
    )
    assert list(selected.json["data"]["values"]) == ["speed", "grain"]
    assert len(selected.json["data"]["values"]["speed"]) == 12


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
        "evaluation": {"enabled": False},
    })
    assert response.status_code == 202
    assert response.json["data"] == {
        "id": "train_1", "status": "queued", "stage": "queued", "progress": 0,
        "sample_count": 2, "input_names": ["x"], "target_names": ["y"],
        "models": ["RF"],
    }
    assert captured == [("train_1", "training")]


def test_training_accepts_inline_data_and_model_names(monkeypatch, tmp_path):
    from mobo.api import service

    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "inline_train"})
    captured = []
    monkeypatch.setattr(service.registry, "start", lambda *args: captured.append(args[:2]))
    response = client.post("/api/v1/hust/doe/train/startTrain", json={
        "id": "inline_train",
        "data_source": {
            "input_data": {"labels": ["x1", "x2"], "samples": [[1, 2], [2, 3], [3, 4]]},
            "output_data": {"labels": ["y"], "samples": [[3], [5], [7]]},
        },
        "models": [{"name": "RF", "params": {"n_estimators": 20, "n_jobs": 1}}],
        "evaluation": {"enabled": True, "method": "k_fold", "n_splits": 3},
    })

    assert response.status_code == 202
    assert response.json["data"]["sample_count"] == 3
    assert response.json["data"]["models"] == ["RF"]
    assert captured == [("inline_train", "training")]
    state = store.load("inline_train")["training"]
    dataset = Path(state["dataset"]["data_file"])
    assert dataset.parent == tmp_path / "doe_tasks" / "inline_train" / "training"
    assert dataset.read_text(encoding="utf-8").splitlines() == ["1\t2\t3", "2\t3\t5", "3\t4\t7"]
    assert state["request"]["models"] == [{
        "model_index": 2, "params": {"n_estimators": 20, "n_jobs": 1},
    }]


def test_training_rejects_mismatched_inline_rows(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "bad_inline"})
    response = client.post("/api/v1/hust/doe/train/startTrain", json={
        "id": "bad_inline",
        "data_source": {
            "input_data": {"labels": ["x"], "samples": [[1], [2]]},
            "output_data": {"labels": ["y"], "samples": [[3]]},
        },
        "models": [{"name": "RF"}],
        "evaluation": {"enabled": False},
    })

    assert response.status_code == 400
    assert response.json == {
        "code": 1, "message": "输入样本数量与输出样本数量必须一致", "data": {},
    }


def test_training_stop_uses_doe_id_and_reports_state(monkeypatch, tmp_path):
    from mobo.api import service

    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "stop_train"})
    store.update_section(
        "stop_train", "training", status="running", stage="training", progress=40,
    )
    monkeypatch.setattr(service.registry, "stop", lambda *args: True)

    response = client.post("/api/v1/hust/doe/train/stop", json={"id": "stop_train"})

    assert response.status_code == 200
    assert response.json == {
        "code": 0, "message": "已发送中止请求",
        "data": {
            "id": "stop_train", "accepted": True, "status": "stopping",
            "stage": "stopping", "progress": 40,
        },
    }


def test_training_delete_clears_files_and_records(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "delete_train"})
    task = store.task_dir("delete_train")
    (task / "training" / "dataset.tsv").write_text("1\t2\n", encoding="utf-8")
    (task / "models" / "snapshot").mkdir()
    store.update_section(
        "delete_train", "training", status="finished", stage="finished", progress=100,
        dataset={"data_file": "removed"}, request={"models": []}, models=[],
    )

    response = client.post("/api/v1/hust/doe/train/delete", json={"id": "delete_train"})

    assert response.status_code == 200
    assert response.json["data"] == {
        "id": "delete_train", "status": "not_started",
        "stage": "not_started", "progress": 0,
    }
    assert store.load("delete_train")["training"] == {
        "status": "not_started", "stage": "not_started",
        "progress": 0, "models": [], "error": None,
    }
    assert not any((task / "training").iterdir())
    assert not any((task / "models").iterdir())


def test_training_lifecycle_returns_documented_failures(monkeypatch, tmp_path):
    from mobo.api import service

    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "running_train"})
    monkeypatch.setattr(service.registry, "running", lambda *args: True)

    conflict = client.post("/api/v1/hust/doe/train/delete", json={"id": "running_train"})
    missing = client.get("/api/v1/hust/doe/train/progress?id=missing_train")
    invalid = client.post("/api/v1/hust/doe/train/stop", json={"TrainId": 10011})

    assert conflict.status_code == 409
    assert conflict.json["message"] == "训练正在运行，请先中止"
    assert missing.status_code == 404
    assert missing.json["message"] == "DOE 任务不存在：missing_train"
    assert invalid.status_code == 400
    assert "id 只能包含" in invalid.json["message"]


def test_progress_hides_internal_error_details(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "hidden_errors"})
    store.update_section(
        "hidden_errors", "training", status="failed",
        error=r"模型文件不存在 C:\\server\\private\\model.pkl",
    )
    store.update_section(
        "hidden_errors", "optimization", status="failed",
        error="结果文件不存在 /srv/private/result.tsv",
    )

    training = client.get(
        "/api/v1/hust/doe/train/progress?id=hidden_errors",
    ).json["data"]
    optimization = client.get(
        "/api/v1/hust/doe/optimize/getById?id=hidden_errors",
    ).json["data"]

    assert training["error"] == "代理模型训练失败，内部异常详情由服务端维护"
    assert optimization["error"] == "优化执行失败，内部异常详情由服务端维护"


def test_all_documented_routes_exist(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    routes = {rule.rule for rule in client.application.url_map.iter_rules()}
    assert {
        "/api/v1/doe/add", "/api/v1/doe/list", "/api/v1/doe/delete",
        "/api/v1/hust/doe/sample/generate", "/api/v1/hust/doe/dataset/generate",
        "/api/v1/hust/doe/data/get",
        "/api/v1/hust/doe/train/progress",
        "/api/v1/hust/doe/train/delete", "/api/v1/hust/doe/train/stop",
        "/api/v1/hust/doe/train/startTrain",
        "/api/v1/hust/doe/inference/startInference",
        "/api/v1/hust/doe/optimize/start", "/api/v1/hust/doe/optimize/stop",
        "/api/v1/hust/doe/optimize/getById",
    } <= routes
    assert "/api/hust/v1/doe/train/progress" not in routes
    assert client.post("/api/v1/hust/doe/data/get", json={}).status_code == 405


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
    public_progress = service.get_training_progress("worker_1")
    assert "model_dir" not in public_progress["models"][0]


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

    result = service.start_inference({
        "id": "infer_1", "inputs": [1.5], "fields": ["target"],
    })
    assert result["predictions"]["target"][0] == pytest.approx(25.0)
    assert result["resource_id"].startswith("tos-")
    assert store.load("infer_1")["inference"]["model_id"] == "best"


def test_inference_supports_named_inputs_and_selected_outputs(monkeypatch, tmp_path):
    import joblib
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    from mobo.api import service

    _client(monkeypatch, tmp_path)
    store.create({"id": "infer_fields"})
    model_dir = store.task_dir("infer_fields") / "models" / "best"
    model_dir.mkdir(parents=True)
    x = np.array([[0.0], [1.0], [2.0]])
    scaler_x = StandardScaler().fit(x)
    scalers = {"scaler_X": scaler_x}
    for index, (target, factor) in enumerate((("grain", 10.0), ("load", 20.0))):
        y = x * factor
        scaler_y = StandardScaler().fit(y)
        model = LinearRegression().fit(
            scaler_x.transform(x), scaler_y.transform(y).ravel()
        )
        joblib.dump(model, model_dir / f"{target}_model.pkl")
        scalers[f"scaler_y_{index}"] = scaler_y
    joblib.dump(scalers, model_dir / "grain_scalers.pkl")
    store.update_section("infer_fields", "training", status="finished", request={
        "all_var_list": ["temperature", "grain", "load"], "input_var_count": 1,
    }, models=[{
        "model_id": "best", "model_dir": str(model_dir),
        "target_names": ["grain", "load"], "score": 1.0,
    }])

    response = service.start_inference({
        "id": "infer_fields", "inputs": {"temperature": [0.5, 1.5]},
        "fields": ["load"],
    })
    assert response["predictions"] == {"load": pytest.approx([10.0, 30.0])}
    fetched = service.get_data({
        "id": "infer_fields", "resource_id": response["resource_id"], "fields": ["load"],
    })
    assert fetched["values"] == {"load": pytest.approx([10.0, 30.0])}


def test_inference_returns_documented_failures(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    store.create({"id": "bad_inference"})
    store.update_section("bad_inference", "training", request={
        "all_var_list": ["x1", "x2", "y"], "input_var_count": 2,
    }, models=[{
        "model_id": "selected", "model_dir": "unused",
        "target_names": ["y"], "score": 1.0,
    }])
    store.create({"id": "no_model"})

    invalid = client.post("/api/v1/hust/doe/inference/startInference", json={
        "id": "bad_inference", "inputs": [1.0],
    })
    unavailable = client.post("/api/v1/hust/doe/inference/startInference", json={
        "id": "no_model", "inputs": [1.0, 2.0],
    })
    missing = client.post("/api/v1/hust/doe/inference/startInference", json={
        "id": "missing_inference", "inputs": [1.0, 2.0],
    })

    assert invalid.status_code == 400
    assert invalid.json["message"] == "inputs[0] 的参数数量必须为 2"
    assert unavailable.status_code == 409
    assert unavailable.json["message"] == "没有可用的已训练代理模型"
    assert missing.status_code == 404
    assert missing.json["message"] == "DOE 任务不存在：missing_inference"


def _prepare_optimization_doe(doe_id):
    store.create({"id": doe_id})
    store.update_section(doe_id, "training", status="finished", request={
        "all_var_list": ["x1", "x2", "y1", "y2", "limit"],
        "input_var_count": 2,
    }, models=[{
        "model_id": f"model_{doe_id}", "model_dir": "unused",
        "target_names": ["y1", "y2", "limit"], "score": 1.0,
    }])


def test_optimization_accepts_multi_and_weighted_single(monkeypatch, tmp_path):
    from mobo.api import service

    client = _client(monkeypatch, tmp_path)
    captured = []
    monkeypatch.setattr(service.registry, "start", lambda *args: captured.append(args[:2]))
    _prepare_optimization_doe("multi_opt")
    multi = client.post("/api/v1/hust/doe/optimize/start", json={
        "id": "multi_opt", "mode": "multi",
        "objectives": [
            {"name": "y1", "direction": "min"},
            {"name": "y2", "direction": "max"},
        ],
        "constraints": [{"name": "limit", "lower": 1, "upper": 5}],
        "decision_variables": [
            {"name": "x1", "lower": 0, "upper": 1},
            {"name": "x2", "lower": 10, "upper": 20},
        ],
        "algorithm": {"name": "nsga2", "params": {"pop_size": 10, "n_gen": 2}},
    })
    _prepare_optimization_doe("single_opt")
    single = client.post("/api/v1/hust/doe/optimize/start", json={
        "id": "single_opt", "mode": "single",
        "objectives": [
            {"name": "y1", "direction": "min", "weight": 0.7},
            {"name": "y2", "direction": "max", "weight": 0.3},
        ],
        "decision_variables": [{"name": "x1", "lower": 0, "upper": 1}],
        "algorithm": {"name": "nsga2", "params": {}},
    })

    assert multi.status_code == 202
    assert multi.json["data"]["mode"] == "multi"
    assert multi.json["data"]["algorithm"] == "nsga2"
    request = store.load("multi_opt")["optimization"]["request"]
    assert request["decision_var_indices"] == [0, 1]
    assert request["constraints"] == [
        {"target_obj": "limit", "constraint_kind": "lower", "limit_value": 1.0},
        {"target_obj": "limit", "constraint_kind": "upper", "limit_value": 5.0},
    ]
    assert single.status_code == 202
    assert store.load("single_opt")["optimization"]["request"]["objective_config"][0]["weight"] == 0.7
    assert captured == [("multi_opt", "optimization"), ("single_opt", "optimization")]


def test_optimization_accepts_rl_and_rejects_unsupported_algorithm(monkeypatch, tmp_path):
    from mobo.api import service

    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(service.registry, "start", lambda *args: None)
    _prepare_optimization_doe("rl_opt")
    request = {
        "id": "rl_opt", "mode": "reinforcement_learning",
        "objectives": [
            {"name": "y1", "direction": "min", "weight": 0.6},
            {"name": "y2", "direction": "max", "weight": 0.4},
        ],
        "constraints": [{"name": "limit", "upper": 5}],
        "decision_variables": [{"name": "x1", "lower": 0, "upper": 1}],
        "algorithm": {"name": "ppo", "params": {"total_timesteps": 10}},
    }
    accepted = client.post("/api/v1/hust/doe/optimize/start", json=request)
    request["algorithm"] = {"name": "pso", "params": {}}
    rejected = client.post("/api/v1/hust/doe/optimize/start", json=request)

    assert accepted.status_code == 202
    assert accepted.json["data"]["mode"] == "reinforcement_learning"
    assert accepted.json["data"]["algorithm"] == "ppo"
    assert store.load("rl_opt")["optimization"]["request"]["optimizer"] == "rl"
    assert rejected.status_code == 400
    assert rejected.json["message"] == "reinforcement_learning模式当前仅支持ppo"


def test_stop_optimization_returns_current_state(monkeypatch, tmp_path):
    from mobo.api import service

    client = _client(monkeypatch, tmp_path)
    store.create({"id": "stop_opt"})
    store.update_section(
        "stop_opt", "optimization", status="running", stage="optimizing", progress=10,
    )
    monkeypatch.setattr(service.registry, "stop", lambda *args: True)

    response = client.post("/api/v1/hust/doe/optimize/stop", json={"id": "stop_opt"})

    assert response.status_code == 200
    assert response.json == {
        "code": 0, "message": "已发送中止请求", "data": {
            "id": "stop_opt", "accepted": True, "status": "stopping",
            "stage": "stopping", "progress": 10,
        },
    }


def test_get_optimization_returns_stable_fields(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    store.create({"id": "query_opt"})

    response = client.get(
        "/api/v1/hust/doe/optimize/getById", query_string={"id": "query_opt"},
    )

    assert response.status_code == 200
    assert response.json["data"] == {
        "id": "query_opt", "status": "not_started", "stage": "not_started",
        "progress": 0, "request": None, "result": None, "error": None,
        "updated_at": store.load("query_opt")["updated_at"],
    }


def test_optimization_control_returns_documented_failures(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    invalid = client.post("/api/v1/hust/doe/optimize/stop", json={})
    missing = client.get(
        "/api/v1/hust/doe/optimize/getById", query_string={"id": "missing_opt"},
    )

    assert invalid.status_code == 400
    assert "id 只能包含" in invalid.json["message"]
    assert missing.status_code == 404
    assert missing.json["message"] == "DOE 任务不存在：missing_opt"


def test_get_optimization_data_uses_recorded_columns(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    store.create({"id": "opt_data"})
    result_file = store.task_dir("opt_data") / "optimization" / "pareto.tsv"
    result_file.write_text("900\t10\t2.5\ttrue\n950\t20\t2.0\tfalse\n", encoding="utf-8")
    resource = store.register_resource(
        "opt_data", "optimization",
        ["temperature", "speed", "grain", "feasible"], path=str(result_file),
    )
    store.update_section("opt_data", "optimization", status="finished", result={
        "task_info": {
            "result_columns": ["temperature", "speed", "grain", "feasible"],
        },
        "file_resource": {"solution_txt_path": str(result_file)},
        **resource,
    })

    response = client.get(
        "/api/v1/hust/doe/data/get",
        query_string=[("id", "opt_data"), ("resource_id", resource["resource_id"]),
                      ("fields", "temperature"), ("fields", "feasible")],
    )
    assert response.status_code == 200
    assert response.json["data"]["values"] == {
        "temperature": [900.0, 950.0], "feasible": [True, False],
    }
    result = client.get(
        "/api/v1/hust/doe/optimize/getById", query_string={"id": "opt_data"},
    ).json["data"]["result"]
    assert result["resource_id"] == resource["resource_id"]
    assert "file_resource" not in result
    assert str(result_file) not in str(result)


def test_get_data_rejects_unknown_field(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "bad_field"})
    generated = client.post("/api/v1/hust/doe/sample/generate", json={
        "id": "bad_field", "method": "full",
        "param_ranges": {"temperature": [900, 1000]}, "level_nums": [2],
    })
    response = client.get("/api/v1/hust/doe/data/get", query_string={
        "id": "bad_field", "resource_id": generated.json["data"]["resource_id"],
        "fields": "speed",
    })
    assert response.status_code == 400
    assert "可用字段" in response.json["message"]


def test_get_data_rejects_unknown_resource(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.post("/api/v1/doe/add", json={"id": "missing_resource"})

    response = client.get("/api/v1/hust/doe/data/get", query_string={
        "id": "missing_resource", "resource_id": "tos-00000000000000000000",
        "fields": "temperature",
    })

    assert response.status_code == 404
    assert response.json == {
        "code": 404,
        "message": "数据资源不存在：tos-00000000000000000000",
        "data": {},
    }
