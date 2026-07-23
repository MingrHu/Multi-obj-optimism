"""DEFORM 子进程驱动 (:mod:`mobo.automation.solver`) 测试。

不依赖真实 DEFORM：通过打桩 :class:`subprocess.Popen` 与 ``time.sleep`` 验证命令串
构造与 :class:`DeformSolver` 的并发调度逻辑。
"""

import subprocess

import pytest

from mobo.automation import solver


def test_key_to_db_command_string(monkeypatch):
    captured = {}
    monkeypatch.setattr(solver, "_run_pre_with_commands", lambda cmd: captured.setdefault("cmd", cmd))
    solver.key_to_db("in.KEY", "out.DB")
    assert captured["cmd"] == "E\n2\n1\nin.KEY\nE\nE\n7\n2\nout.DB\nY\nE\nY\n"


def test_db_to_key_command_string(monkeypatch):
    captured = {}
    monkeypatch.setattr(solver, "_run_pre_with_commands", lambda cmd: captured.setdefault("cmd", cmd))
    solver.db_to_key("in.DB", "out.KEY", "5")
    assert captured["cmd"] == "E\n2\n2\nin.DB\n5\nE\nE\n8\nout.KEY\nE\nY\n"


def test_key_to_db_batch(monkeypatch):
    calls = []
    monkeypatch.setattr(solver, "key_to_db", lambda k, d: calls.append((k, d)))
    solver.key_to_db_batch(["a.KEY", "b.KEY"], ["a.DB", "b.DB"])
    assert calls == [("a.KEY", "a.DB"), ("b.KEY", "b.DB")]


class _FakeProcess:
    """最小化的假子进程：stdout 逐行返回后结束。"""

    def __init__(self, lines):
        self._lines = list(lines)
        self.stdout = self

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def poll(self):
        return None if self._lines else 0


def test_run_pre_with_commands_invokes_popen(monkeypatch, tmp_path):
    seen = {}

    def fake_popen(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return _FakeProcess(["log line 1\n"])

    monkeypatch.setattr(solver.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(solver, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(solver, "_OPERATION_LOG", str(tmp_path / "op.log"))

    solver._run_pre_with_commands("E\nY\n")
    assert solver.DEF_PRE_64 in seen["command"]
    assert seen["kwargs"]["shell"] is True
    assert (tmp_path / "op.log").read_text() == "log line 1\n"


def test_deform_solver_running_counter():
    s = solver.DeformSolver(max_parallel=4)
    assert s.running == 0
    s._running = 3
    assert s.running == 3


def test_deform_solver_feed_stdin(monkeypatch):
    monkeypatch.setattr(solver.time, "sleep", lambda *_: None)
    written = []

    class _P:
        class _Stdin:
            def write(self, c):
                written.append(c)

            def flush(self):
                pass

        stdin = _Stdin()

    solver.DeformSolver._feed_stdin(_P(), "payload")
    assert written == ["payload\n"]


def test_deform_solver_run_all_schedules_each(monkeypatch):
    monkeypatch.setattr(solver.time, "sleep", lambda *_: None)
    s = solver.DeformSolver(max_parallel=2)
    submitted = []
    monkeypatch.setattr(s, "submit", lambda target, work_dir: submitted.append((target, work_dir)))

    s.run_all(["/res/0/model.DB", "/res/1/model.DB"])
    assert len(submitted) == 2
    # solve_target 去掉扩展名，work_dir 为其所在目录
    targets = [t for t, _ in submitted]
    assert targets[0].endswith("model") and not targets[0].endswith(".DB")
