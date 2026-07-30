"""任务状态持久化 (:mod:`mobo.common.task_store`) 测试。"""

import pytest

from mobo.common import task_store


@pytest.fixture(autouse=True)
def _tmp_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(task_store, "task_dir", lambda tid: tmp_path / "tasks" / tid)


def test_init_and_load():
    assert task_store.load("t1") is None
    assert not task_store.exists("t1")
    state = task_store.init_state("t1", "automation", {"foo": 1})
    assert state["task_id"] == "t1"
    assert state["kind"] == "automation"
    assert state["status"] == "running"
    assert task_store.exists("t1")
    assert task_store.load("t1")["req"] == {"foo": 1}


def test_init_idempotent():
    first = task_store.init_state("t1", "automation", {"foo": 1})
    second = task_store.init_state("t1", "automation", {"foo": 999})
    # 已存在则原样读回，不覆盖
    assert second["req"] == first["req"] == {"foo": 1}


def test_update_shallow_merge():
    task_store.init_state("t1", "surrogate", {"a": 1})
    task_store.update("t1", stage="train", status="finished", data={"r2": 0.9})
    task_store.update("t1", data={"rmse": 0.1})
    state = task_store.load("t1")
    assert state["stage"] == "train"
    assert state["status"] == "finished"
    # data 浅合并，两次 update 的键都在
    assert state["data"] == {"r2": 0.9, "rmse": 0.1}


def test_update_missing_raises():
    with pytest.raises(FileNotFoundError):
        task_store.update("nope", status="x")


def test_save_refreshes_updated_at():
    s = task_store.init_state("t1", "optimization", {})
    created = s["created_at"]
    task_store.update("t1", stage="optimize")
    assert task_store.load("t1")["created_at"] == created
    assert "updated_at" in task_store.load("t1")


def test_history_accumulates_not_overwrites():
    task_store.init_state("t1", "automation", {})
    task_store.update("t1", stage="generate_keys", status="finished")
    task_store.update("t1", stage="run_solver", status="finished")
    task_store.update("t1", data={"x": 1})  # 无 stage/status，不追加记录
    stages = [h["stage"] for h in task_store.load("t1")["history"]]
    assert stages == ["init", "generate_keys", "run_solver"]


def test_resolve_req_prefers_record():
    task_store.init_state("t1", "automation", {"a": 1})
    # 记录里已有 a，传入的 a 被忽略；b 记录没有则采用传入并回填
    resolved = task_store.resolve_req("t1", "automation", {"a": 999, "b": 2}, ["a", "b"])
    assert resolved == {"a": 1, "b": 2}
    assert task_store.load("t1")["req"] == {"a": 1, "b": 2}


def test_resolve_req_uses_provided_when_no_record():
    resolved = task_store.resolve_req("t1", "automation", {"a": 1, "b": 2}, ["a", "b"])
    assert resolved == {"a": 1, "b": 2}
    # 任务不存在时用传入参数初始化记录
    assert task_store.load("t1")["req"] == {"a": 1, "b": 2}


def test_resolve_req_missing_raises():
    task_store.init_state("t1", "automation", {"a": 1})
    with pytest.raises(ValueError):
        task_store.resolve_req("t1", "automation", {}, ["a", "b"])


def test_resolve_req_none_counts_as_missing():
    with pytest.raises(ValueError):
        task_store.resolve_req("t1", "automation", {"a": None}, ["a"])
