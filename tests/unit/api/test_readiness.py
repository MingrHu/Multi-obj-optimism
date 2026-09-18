"""API运行依赖预加载与健康检查测试"""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from mobo.api import readiness
from mobo.api.app import create_app


def test_runtime_dependencies_are_ready_before_health_check():
    response = create_app({"TESTING": True}).test_client().get("/health")

    assert response.status_code == 200
    dependencies = response.json["data"]["dependencies"]
    assert dependencies["ready"] is True
    assert set(dependencies["components"]) == {
        "numpy", "pandas", "pyDOE", "sampling", "surrogate_training",
        "surrogate_evaluation", "inference", "optimization_ga", "optimization_rl",
    }
    assert all(
        dependencies["components"][name] == "ready"
        for name in (
            "sampling", "surrogate_training", "surrogate_evaluation", "inference",
            "optimization_ga", "optimization_rl",
        )
    )
    assert dependencies["error"] is None


def test_dependency_failure_prevents_worker_start(monkeypatch):
    monkeypatch.setattr(
        readiness, "_STATE", {"ready": False, "components": {}, "error": None}
    )
    monkeypatch.setattr(
        readiness,
        "_load_components",
        lambda: (_ for _ in ()).throw(ImportError("numpy failed")),
    )

    with pytest.raises(RuntimeError, match="DOE运行依赖预加载失败"):
        readiness.load_runtime_dependencies()

    assert readiness.get_readiness() == {
        "ready": False,
        "components": {},
        "error": "ImportError: numpy failed",
    }


def test_concurrent_preload_only_imports_components_once(monkeypatch):
    calls = 0
    calls_lock = threading.Lock()

    def load_components():
        nonlocal calls
        with calls_lock:
            calls += 1
        return {"sampling": "ready"}

    monkeypatch.setattr(
        readiness, "_STATE", {"ready": False, "components": {}, "error": None}
    )
    monkeypatch.setattr(readiness, "_load_components", load_components)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: readiness.load_runtime_dependencies(), range(16)))

    assert calls == 1
    assert all(result["ready"] is True for result in results)


def test_health_returns_503_when_dependency_state_is_not_ready(monkeypatch):
    app = create_app({"TESTING": True})
    monkeypatch.setattr(
        readiness,
        "get_readiness",
        lambda: {"ready": False, "components": {}, "error": "not ready"},
    )

    response = app.test_client().get("/health")

    assert response.status_code == 503
    assert response.json["code"] == 1
    assert response.json["data"]["dependencies"]["ready"] is False
