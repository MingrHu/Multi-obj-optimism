"""server_demo 的 request_id 幂等行为测试。"""

import json

import pytest

from mobo.api import server_demo


def test_same_request_id_returns_cached_result(monkeypatch, tmp_path):
    monkeypatch.setattr(server_demo, "REQUESTS_DIR", tmp_path / "requests")
    calls = []
    monkeypatch.setattr(
        server_demo,
        "train_models",
        lambda request: calls.append("train") or {"PRG": "tr_1"},
    )
    monkeypatch.setattr(
        server_demo,
        "select_best_model",
        lambda request, output_dir: calls.append("evaluate") or ("PRG", []),
    )
    monkeypatch.setattr(
        server_demo,
        "optimize",
        lambda request, model_id, output_dir: calls.append("optimize")
        or {"code": 0, "task_id": "opt_1"},
    )
    serialized = json.dumps({"request_id": "req_same", "value": 1})

    first = server_demo.run_service(serialized)
    second = server_demo.run_service(serialized)

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["model_ids"] == {"PRG": "tr_1"}
    assert second["optimization"]["task_id"] == "opt_1"
    assert calls == ["train", "evaluate", "optimize"]


def test_same_request_id_rejects_different_content(monkeypatch, tmp_path):
    monkeypatch.setattr(server_demo, "REQUESTS_DIR", tmp_path / "requests")
    monkeypatch.setattr(server_demo, "train_models", lambda request: {"PRG": "tr_1"})
    monkeypatch.setattr(
        server_demo,
        "select_best_model",
        lambda request, output_dir: ("PRG", []),
    )
    monkeypatch.setattr(
        server_demo,
        "optimize",
        lambda request, model_id, output_dir: {"code": 0, "task_id": "opt_1"},
    )
    server_demo.run_service(json.dumps({"request_id": "req_conflict", "value": 1}))

    with pytest.raises(ValueError, match="已绑定其他请求"):
        server_demo.run_service(json.dumps({"request_id": "req_conflict", "value": 2}))
