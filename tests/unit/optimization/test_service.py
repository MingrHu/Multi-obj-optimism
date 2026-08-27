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


def test_run_resumes_from_record(_fake_optimizers):
    """先落盘 optimizer/model_id，再仅凭 task_id 续跑（不重传）。"""
    service.run_optimization({"model_id": "tr_1"}, optimizer="rl", task_id="opt_r")
    resp = service.run_optimization(task_id="opt_r")
    assert resp["code"] == 0
    # optimizer 从记录读取，仍走 rl 分支并保留 model_id 溯源
    assert _fake_optimizers == ["rl", "rl"]
    assert resp["data"]["task_info"]["model_id"] == "tr_1"


def test_run_missing_optimizer_reports_error():
    """无记录且未传 optimizer -> code 1。"""
    resp = service.run_optimization(task_id="opt_missing")
    assert resp["code"] == 1
    assert "缺失" in resp["msg"]


def test_run_record_takes_precedence(_fake_optimizers):
    """已有记录时，重传的 optimizer 不覆盖记录里的值。"""
    service.run_optimization({}, optimizer="nsga2", task_id="opt_pre")
    service.run_optimization(optimizer="rl", task_id="opt_pre")
    # 记录里的 optimizer 仍是 nsga2，两次都走 nsga2
    assert _fake_optimizers == ["nsga2", "nsga2"]


def test_parameterized_nsga2_uses_model_task(monkeypatch):
    task_store.init_state(
        "tr_model",
        "surrogate",
        {"vars_out": ["temperature", "grain"], "n_vars": 1},
    )
    task_store.update(
        "tr_model", status="finished", stage="train", data={"model_dir": "isolated/models"}
    )
    captured = {}

    def fake_parameterized(request, *, model_dir, output_path):
        captured.update(request=request, model_dir=model_dir, output_path=output_path)
        return {
            "solution_txt_path": output_path,
            "solution_count": 4,
            "all_solution_feasible": True,
            "columns": ["temperature", "grain", "feasible"],
        }

    monkeypatch.setattr(service, "run_parameterized_nsga2", fake_parameterized)
    request = {
        "model_id": "tr_model",
        "objective_names": ["grain"],
        "input_var_count": 1,
        "all_var_list": ["temperature", "grain"],
        "decision_var_indices": [0],
        "decision_var_names": ["temperature"],
        "decision_bounds": [{"lower": 800.0, "upper": 1000.0}],
        "constraints": [],
        "objective_config": [{"name": "grain", "minimize": True}],
        "optimizer_config": {"pop_size": 10, "n_gen": 2},
        "output_config": {},
    }

    response = service.run_optimization(request, optimizer="nsga2", task_id="opt_params")

    assert response["code"] == 0
    assert captured["model_dir"] == "isolated/models"
    assert captured["output_path"].endswith("pareto_solutions.tsv")
    assert response["data"]["constraint_check"]["solution_count"] == 4
    assert response["data"]["task_info"]["result_columns"] == [
        "temperature", "grain", "feasible"
    ]


def test_parameterized_nsga2_rejects_variable_order_mismatch(monkeypatch):
    task_store.init_state(
        "tr_order",
        "surrogate",
        {"vars_out": ["speed", "grain"], "n_vars": 1},
    )
    task_store.update(
        "tr_order", status="finished", stage="train", data={"model_dir": "models"}
    )
    monkeypatch.setattr(
        service,
        "run_parameterized_nsga2",
        lambda *args, **kwargs: pytest.fail("变量校验失败时不应启动优化"),
    )
    request = {
        "model_id": "tr_order",
        "objective_names": ["grain"],
        "input_var_count": 1,
        "all_var_list": ["temperature", "grain"],
        "decision_var_indices": [0],
        "decision_var_names": ["temperature"],
        "decision_bounds": [{"lower": 800.0, "upper": 1000.0}],
        "constraints": [],
        "objective_config": [{"name": "grain", "minimize": True}],
        "optimizer_config": {"pop_size": 10, "n_gen": 2},
        "output_config": {},
    }

    response = service.run_optimization(request, optimizer="nsga2", task_id="opt_order")

    assert response["code"] == 1
    assert "变量顺序不一致" in response["msg"]
