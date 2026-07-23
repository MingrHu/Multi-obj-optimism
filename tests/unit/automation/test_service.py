"""服务层 (:mod:`mobo.automation.service`) 测试。

打桩 :class:`ForgingTask` 与采样，验证服务接口的返回结构、任务字典管理与状态查询
（``query_execution_status`` 返回底层 TaskStatus 整数值字符串）。
"""

import pytest

from mobo.automation import service
from mobo.automation.pipeline import TaskStatus


@pytest.fixture(autouse=True)
def _clean_state():
    """每个用例前后清空模块级任务字典，避免相互污染。"""
    service._execution_tasks.clear()
    service._sampling_done.clear()
    yield
    service._execution_tasks.clear()
    service._sampling_done.clear()


class _FakeTask:
    """记录各阶段调用的假 ForgingTask。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.status = TaskStatus.DONE
        self.calls = []

    def generate_keys(self):
        self.calls.append("generate_keys")

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
    assert service._sampling_done["t1"] == "/tmp/INlhs.txt"


def test_create_sampling_task_failure(monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad")

    monkeypatch.setattr(service, "generate_sample_file", boom)
    res = service.create_sampling_task("t1", "/tmp", "lhs", {"a": (0.0, 1.0)}, 10)
    assert res["status"] == "failed"


def test_init_execution_task_missing_paths():
    res = service.init_execution_task("t1", {}, [["temp"]], [["grain"]], [False], 10)
    assert res["status"] == "failed"


def test_init_execution_task_success(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "ForgingTask", _FakeTask)
    res = service.init_execution_task(
        "t1", _paths(tmp_path), [["temp"], ["workpiece"]], [["grain"], ["workpiece"]], [False], 100
    )
    assert res["status"] == "success"
    assert "t1" in service._execution_tasks
    assert service._execution_tasks["t1"].calls == ["generate_keys"]


def test_run_execution_step_unknown_task():
    assert service.run_execution_step("nope")["status"] == "failed"


def test_run_execution_step_success():
    service._execution_tasks["t1"] = _FakeTask()
    res = service.run_execution_step("t1")
    assert res["status"] == "success"
    assert service._execution_tasks["t1"].calls == ["run_solver"]


def test_run_extract_data_success():
    service._execution_tasks["t1"] = _FakeTask()
    res = service.run_extract_data("t1")
    assert res["status"] == "success"
    assert service._execution_tasks["t1"].calls == ["extract"]


def test_query_execution_status_unknown():
    assert service.query_execution_status("nope")["status"] == "failed"


def test_query_execution_status_returns_int_string():
    task = _FakeTask()
    task.status = TaskStatus.DONE
    service._execution_tasks["t1"] = task
    assert service.query_execution_status("t1")["status"] == "0"

    task.status = TaskStatus.RUNNING
    assert service.query_execution_status("t1")["status"] == "1"

    task.status = TaskStatus.FAILED
    assert service.query_execution_status("t1")["status"] == "-1"
