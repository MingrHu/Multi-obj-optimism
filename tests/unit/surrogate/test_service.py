"""代理模型服务层 (:mod:`mobo.surrogate.service`) 测试。

打桩 :class:`Doe_surrogateModel` 避免真实训练，把任务目录重定向到 ``tmp_path``，
验证 model_index 映射、biz_params 转换、resp 落盘与状态查询。
"""

import pytest

from mobo.common import task_store
from mobo.surrogate import service


@pytest.fixture(autouse=True)
def _tmp_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(task_store, "task_dir", lambda tid: tmp_path / "tasks" / tid)


class _FakeDoe:
    calls = []

    def __init__(self, data_file, vars_out, n_vars):
        self.args = (data_file, vars_out, n_vars)

    def train_save_model(self, which_model, model_par=None):
        _FakeDoe.calls.append((which_model, model_par or []))


@pytest.fixture(autouse=True)
def _fake_doe(monkeypatch):
    _FakeDoe.calls = []
    monkeypatch.setattr(service, "Doe_surrogateModel", _FakeDoe)


def test_train_prg_maps_index_and_params():
    resp = service.train_surrogate(
        "d.txt", ["1", "2", "grain", "load"], 2,
        model_index=0, biz_params={"degree": 3}, model_id="tr_x",
    )
    assert resp["code"] == 0
    assert resp["model_id"] == "tr_x"
    # 协议 model_index=0(PRG) -> which_model=2，degree 转成 ["3"]
    assert _FakeDoe.calls == [(2, ["3"])]
    assert resp["data"]["model_family"] == "PRG"
    # 目标名与保存路径落盘
    assert resp["data"]["target_names"] == ["grain", "load"]
    assert resp["data"]["model_save_paths"]["grain"].endswith("grain_model.pkl")


def test_train_dnn_uses_keras_ext_and_param_order():
    resp = service.train_surrogate(
        "d.txt", ["1", "res"], 1,
        model_index=4, biz_params={"epochs": 10, "batch_size": 8, "verbose": 0, "patience": 5},
        model_id="tr_dnn",
    )
    # model_index=4(DNN) -> which_model=1；参数按 epochs,batch_size,verbose,patience 顺序
    assert _FakeDoe.calls == [(1, ["10", "8", "0", "5"])]
    assert resp["data"]["model_save_paths"]["res"].endswith("res_model.keras")


def test_train_persists_state():
    service.train_surrogate("d.txt", ["1", "grain"], 1, model_index=2, model_id="tr_rf")
    state = task_store.load("tr_rf")
    assert state["kind"] == "surrogate"
    assert state["status"] == "finished"
    assert state["req"]["model_index"] == 2


def test_train_invalid_index():
    resp = service.train_surrogate("d.txt", ["1", "grain"], 1, model_index=9)
    assert resp["code"] == 1


def test_train_failure_records_failed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("train boom")

    monkeypatch.setattr(_FakeDoe, "train_save_model", boom)
    resp = service.train_surrogate("d.txt", ["1", "grain"], 1, model_index=0, model_id="tr_bad")
    assert resp["code"] == 1
    assert task_store.load("tr_bad")["status"] == "failed"


def test_query_model_status():
    service.train_surrogate("d.txt", ["1", "grain"], 1, model_index=0, model_id="tr_q")
    resp = service.query_model_status("tr_q")
    assert resp["code"] == 0
    assert resp["data"]["status"] == "finished"
    assert service.query_model_status("nope")["code"] == 1


def test_train_resumes_from_record():
    """先落盘参数，再仅凭 model_id 续跑（不重传参数）。"""
    service.train_surrogate("d.txt", ["1", "grain"], 1, model_index=0, model_id="tr_r")
    _FakeDoe.calls = []
    # 只传 model_id，参数从记录读取
    resp = service.train_surrogate(model_id="tr_r")
    assert resp["code"] == 0
    assert _FakeDoe.calls == [(2, [])]


def test_train_missing_params_reports_error():
    """无记录且未传必要参数 -> code 1。"""
    resp = service.train_surrogate(model_id="tr_missing")
    assert resp["code"] == 1
    assert "缺失" in resp["msg"]


def test_train_record_takes_precedence():
    """已有记录时，重传的参数不覆盖记录里的值。"""
    service.train_surrogate("d.txt", ["1", "grain"], 1, model_index=0, model_id="tr_p")
    service.train_surrogate("other.txt", model_index=4, model_id="tr_p")
    assert task_store.load("tr_p")["req"]["data_file"] == "d.txt"
    assert task_store.load("tr_p")["req"]["model_index"] == 0
