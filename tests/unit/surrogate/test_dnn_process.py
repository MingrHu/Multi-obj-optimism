import json
import threading

import pytest

from mobo.surrogate import dnn_process
from mobo.surrogate.common import model_output_dir


class FakeProcess:
    def __init__(self, code=None):
        self.code = code
        self.terminated = False
        self.waited = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def wait(self, timeout=None):
        self.waited = True
        return self.code


def test_worker_count_limits_targets_and_cpu(monkeypatch):
    monkeypatch.delenv("MOBO_DNN_MAX_WORKERS", raising=False)
    monkeypatch.setattr(dnn_process.os, "cpu_count", lambda: 16)
    if hasattr(dnn_process.os, "sched_getaffinity"):
        monkeypatch.setattr(dnn_process.os, "sched_getaffinity", lambda _: set(range(16)))
    assert dnn_process.worker_count(12) == 8
    assert dnn_process.worker_count(2) == 2
    monkeypatch.setenv("MOBO_DNN_MAX_WORKERS", "1")
    assert dnn_process.worker_count(8) == 1
    monkeypatch.setenv("MOBO_DNN_MAX_WORKERS", "0")
    with pytest.raises(ValueError, match="正整数"):
        dnn_process.worker_count(8)


def test_train_passes_task_output_and_cancellation(monkeypatch, tmp_path):
    captured = {}
    cancel = threading.Event()

    def fake_run(spec_file, targets, limit, event):
        captured.update(json.loads(spec_file.read_text(encoding="utf-8")))
        assert targets == 2
        assert event is cancel

    monkeypatch.setattr(dnn_process, "_run_workers", fake_run)
    with model_output_dir(tmp_path / "isolated"), dnn_process.training_control(cancel):
        dnn_process.train_parallel(str(tmp_path / "data.tsv"), ["X", "Y1", "Y2"], 1, {"epochs": 7})
    assert captured["output_dir"] == str((tmp_path / "isolated").resolve())
    assert captured["params"] == {"epochs": 7}
    assert dnn_process._CANCEL.get() is None


def test_scheduler_launches_independent_targets_with_limited_threads(monkeypatch, tmp_path):
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess(0)

    monkeypatch.setattr(dnn_process.subprocess, "Popen", popen)
    dnn_process._run_workers(tmp_path / "spec.json", 3, 2, None)
    assert [command[-1] for command, _ in calls] == ["0", "1", "2"]
    assert all(kwargs["env"]["TF_NUM_INTRAOP_THREADS"] == "1" for _, kwargs in calls)
    assert all(command[2] == "mobo.surrogate.dnn_process" for command, _ in calls)


@pytest.mark.parametrize("fail", [False, True])
def test_cancel_or_failure_terminates_other_children(monkeypatch, tmp_path, fail):
    cancel = threading.Event()
    children = []

    def popen(command, **kwargs):
        process = FakeProcess(1 if fail and not children else None)
        children.append(process)
        if len(children) == 2 and not fail:
            cancel.set()
        return process

    monkeypatch.setattr(dnn_process.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="失败" if fail else "中止"):
        dnn_process._run_workers(tmp_path / "spec.json", 4, 2, cancel)
    assert len(children) == 2
    assert children[1].terminated
    assert all(process.waited for process in children)


def test_worker_keeps_full_column_order_and_target_index(monkeypatch, tmp_path):
    from mobo.surrogate import dnn

    spec_file = tmp_path / "spec.json"
    spec = {"file": "dataset.tsv", "vars_out": ["X", "A", "B"], "n_var": 1,
            "params": {"epochs": 5}, "output_dir": str(tmp_path / "models")}
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    called = []
    monkeypatch.setattr(dnn, "_train_outputs", lambda *args: called.append(args))
    dnn_process._worker_main(str(spec_file), 1)
    assert called == [("dataset.tsv", ["X", "A", "B"], 1, {"epochs": 5}, [1])]
