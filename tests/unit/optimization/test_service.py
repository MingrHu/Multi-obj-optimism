"""多目标优化服务层 (:mod:`mobo.optimization.service`) 测试。

打桩 NSGA2_run / train_and_optimize 避免真实优化，把任务目录重定向到 ``tmp_path``，
验证 optimizer 分派、model_id 溯源、resp 落盘与状态查询。
"""

import pytest

from mobo.common import task_store
from mobo.optimization import service


@pytest.fixture(autouse=True)
def _tmp_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(task_store, "task_dir", lambda tid: tmp_path / "tasks" / tid)


@pytest.fixture(autouse=True)
def _fake_optimizers(monkeypatch):
    calls = []
    monkeypatch.setattr(service, "NSGA2_run", lambda: calls.append("nsga2"))
    monkeypatch.setattr(service, "train_and_optimize", lambda: calls.append("rl"))
    return calls


def test_run_nsga2_persists_and_traces_model(_fake_optimizers):
    resp = service.run_optimization({"model_id": "tr_1"}, optimizer="nsga2", task_id="opt_1")
    assert resp["code"] == 0
    assert _fake_optimizers == ["nsga2"]
    # model_id 溯源与输出文件路径落盘
    assert resp["data"]["task_info"]["model_id"] == "tr_1"
    assert resp["data"]["file_resource"]["solution_txt_path"].endswith("pareto_solutions.txt")
    state = task_store.load("opt_1")
    assert state["kind"] == "optimization" and state["status"] == "finished"


def test_run_rl(_fake_optimizers):
    resp = service.run_optimization({}, optimizer="rl", task_id="opt_rl")
    assert resp["code"] == 0
    assert _fake_optimizers == ["rl"]
    assert resp["data"]["file_resource"]["solution_txt_path"].endswith("rl_solutions_sb3.txt")


def test_run_unknown_optimizer():
    resp = service.run_optimization({}, optimizer="bogus", task_id="opt_x")
    assert resp["code"] == 1
    assert task_store.load("opt_x")["status"] == "failed"


def test_run_failure_records_failed(monkeypatch):
    monkeypatch.setattr(service, "NSGA2_run", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = service.run_optimization({}, optimizer="nsga2", task_id="opt_bad")
    assert resp["code"] == 1
    assert task_store.load("opt_bad")["status"] == "failed"


def test_query_optimization_status(_fake_optimizers):
    service.run_optimization({"model_id": "tr_1"}, optimizer="nsga2", task_id="opt_q")
    resp = service.query_optimization_status("opt_q")
    assert resp["code"] == 0
    assert resp["data"]["status"] == "finished"
    assert service.query_optimization_status("nope")["code"] == 1
