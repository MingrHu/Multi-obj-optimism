"""外部门面只返回协议字典，不向调用方抛出参数异常。"""

from mobo.api import facade


def test_invalid_json_becomes_protocol_error():
    response = facade.train_surrogate("{bad json")
    assert response["code"] == 1
    assert response["model_id"] is None
    assert "请求参数错误" in response["msg"]


def test_query_task_dispatches_by_prefix(monkeypatch):
    monkeypatch.setattr(
        facade, "query_model_status", lambda task_id: {"code": 0, "model_id": task_id}
    )
    monkeypatch.setattr(
        facade, "query_optimization_status", lambda task_id: {"code": 0, "task_id": task_id}
    )

    assert facade.query_task("tr_x")["model_id"] == "tr_x"
    assert facade.query_task("opt_x")["task_id"] == "opt_x"
    assert facade.query_task("unknown")["code"] == 1
    assert facade.query_task("tr_../../outside")["code"] == 1
