"""服务层 (:mod:`mobo.automation.service`) 测试。

打桩 :class:`ForgingTask` 与采样，并把任务状态目录重定向到 ``tmp_path``，
验证服务接口的返回结构、state.json 落盘与「仅凭 task_id 续跑」的行为。
"""

import pytest

from mobo.automation import service
from mobo.common import task_store


@pytest.fixture(autouse=True)
def _tmp_tasks(monkeypatch, tmp_path):
    """把 task_store 的任务目录重定向到 tmp_path，避免污染仓库 data/。"""
    monkeypatch.setattr(task_store, "task_dir", lambda tid: tmp_path / "tasks" / tid)


class _FakeTask:
    """记录各阶段调用的假 ForgingTask。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.key_files = ["k0.KEY"]
        self.db_files = ["d0.DB"]
        self.result_txt_dir = "res_txt"

    def generate_keys(self):
        self.calls.append("generate_keys")

    def prepare_db_files(self):
        self.calls.append("prepare_db_files")
        return []

    def run_solver(self):
        self.calls.append("run_solver")

    def extract(self):
        self.calls.append("extract")


def _paths(tmp_path):
    return {
        "smp_file": str(tmp_path / "smp.txt"),
        "std_key_file": str(tmp_path / "MODEL.KEY"),
        "temp_key_path": str(tmp_path / "temp_key"),
        "res_db_path": str(tmp_path / "res_db"),
        "res_key_path": str(tmp_path / "res_key"),
        "res_txt_path": str(tmp_path / "res_txt"),
    }


def test_create_sampling_task_zero_returns_empty():
    assert service.create_sampling_task("t1", "/tmp", "lhs", {"a": (0.0, 1.0)}, 0) == {}


def test_create_sampling_task_success(monkeypatch):
    monkeypatch.setattr(service, "generate_sample_file", lambda *a, **k: "/tmp/INlhs.txt")
    res = service.create_sampling_task("t1", "/tmp", "lhs", {"a": (0.0, 1.0)}, 10)
    assert res["status"] == "success"
    # 抽样结果落盘
    state = task_store.load("t1")
    assert state["data"]["sample_file"] == "/tmp/INlhs.txt"


def test_create_sampling_task_failure(monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad")

    monkeypatch.setattr(service, "generate_sample_file", boom)
    res = service.create_sampling_task("t1", "/tmp", "lhs", {"a": (0.0, 1.0)}, 10)
    assert res["status"] == "failed"


def test_init_execution_task_missing_paths():
    res = service.init_execution_task("t1", {}, [["temp"]], [["grain"]], [False], 10)
    assert res["status"] == "failed"


def test_init_execution_task_persists_req(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    res = service.init_execution_task(
        "t1", _paths(tmp_path), [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )
    assert res["status"] == "success"
    # 输入参数已落盘，供后续步骤续跑
    state = task_store.load("t1")
    assert state["req"]["max_step"] == 100
    assert state["req"]["param_table"] == [["temp"], ["workpiece"]]
    assert state["stage"] == "generate_keys" and state["status"] == "finished"


def test_run_execution_step_unknown_task():
    assert service.run_execution_step("nope")["status"] == "failed"


def test_run_and_extract_resume_from_disk(monkeypatch, tmp_path):
    """init 落盘后，run/extract 仅凭 task_id 从磁盘重建任务续跑。"""
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    service.init_execution_task(
        "t1", _paths(tmp_path), [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )

    # 只传 task_id，不再传任何参数
    res = service.run_execution_step("t1")
    assert res["status"] == "success"
    assert task_store.load("t1")["stage"] == "run_solver"

    res = service.run_extract_data("t1")
    assert res["status"] == "success"
    assert task_store.load("t1")["stage"] == "extract"


def test_query_execution_status_unknown():
    assert service.query_execution_status("nope")["status"] == "failed"


def test_query_execution_status_reads_state(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    service.init_execution_task(
        "t1", _paths(tmp_path), [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )
    res = service.query_execution_status("t1")
    assert res["status"] == "finished"
    assert "generate_keys" in res["message"]
