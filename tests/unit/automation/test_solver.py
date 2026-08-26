"""DEFORM 子进程驱动 (:mod:`mobo.automation.solver`) 测试。

不依赖真实 DEFORM：通过打桩 :class:`subprocess.Popen` 与 ``time.sleep`` 验证命令串
构造与 :class:`DeformSolver` 的并发调度逻辑。
"""

import threading
import time

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


def test_db_to_key_exports_are_serialized(monkeypatch):
    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    def fake_run(_command):
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with counter_lock:
            active -= 1

    monkeypatch.setattr(solver, "_run_pre_with_commands", fake_run)
    threads = [
        threading.Thread(target=solver.db_to_key, args=(f"{index}.DB", f"{index}.KEY"))
        for index in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active == 1


def test_query_db_steps_parses_actual_saved_steps(monkeypatch):
    captured = {}

    def fake_run(command):
        captured["command"] = command
        return "Step Numbers:\n     -1      1     59     60\n\n Step Number = 60 ?"

    monkeypatch.setattr(solver, "_run_pre_with_commands", fake_run)
    assert solver.query_db_steps("model.DB") == [-1, 1, 59, 60]
    assert captured["command"] == "E\n2\n2\nmodel.DB\n\nE\nE\nY\n"


def test_query_db_steps_rejects_unparseable_output(monkeypatch):
    monkeypatch.setattr(solver, "_run_pre_with_commands", lambda _command: "invalid")
    with pytest.raises(RuntimeError, match="解析数据库保存步号"):
        solver.query_db_steps("model.DB")


def test_run_key_actions_command_string(monkeypatch):
    captured = {}
    monkeypatch.setattr(solver, "_run_pre_with_commands", lambda cmd: captured.setdefault("cmd", cmd))
    solver.run_key_actions("transition.KEY")
    assert captured["cmd"] == "E\n2\n1\ntransition.KEY\nE\nE\nY\n"


@pytest.mark.parametrize("normal_marker", [
    "NORMAL STOP",
    "Simulation Module Indicates End of Simulation",
])
def test_solve_db_sync_accepts_deform_normal_end_markers(
    monkeypatch, tmp_path, normal_marker
):
    written = []

    class _Stdin:
        def write(self, value):
            written.append(value)

        def flush(self):
            pass

        def close(self):
            pass

    class _Process:
        stdin = _Stdin()

        def wait(self):
            return 0

    monkeypatch.setattr(solver.subprocess, "Popen", lambda *args, **kwargs: _Process())
    db_path = tmp_path / "sample.DB"
    (tmp_path / "sample.LOG").write_text(normal_marker + "\n", encoding="utf-8")
    (tmp_path / "FOR003").write_text("residue", encoding="utf-8")
    (tmp_path / "FOR003.LOCK").write_text("residue", encoding="utf-8")
    solver.solve_db_sync(str(db_path))
    assert written == ["sample\nB\n"]
    assert not (tmp_path / "FOR003").exists()
    assert not (tmp_path / "FOR003.LOCK").exists()


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

    output = solver._run_pre_with_commands("E\nY\n")
    assert solver.DEF_PRE_64 in seen["command"]
    assert seen["kwargs"]["shell"] is True
    assert (tmp_path / "op.log").read_text() == "log line 1\n"
    assert output == "log line 1\n"


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


def test_deform_solver_does_not_mark_failed_job_done(monkeypatch):
    s = solver.DeformSolver()
    s._running = 1
    started = []
    completed = []
    monkeypatch.setattr(s, "_mark_started", started.append)
    monkeypatch.setattr(s, "_mark_done", completed.append)
    monkeypatch.setattr(
        solver, "solve_db_sync",
        lambda _: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    s._solve_one("sample", "/work", db_key="/work/sample.DB")
    assert started == ["/work/sample.DB"]
    assert completed == []
    assert s.running == 0


def test_deform_solver_run_all_schedules_each(monkeypatch):
    monkeypatch.setattr(solver.time, "sleep", lambda *_: None)
    s = solver.DeformSolver(max_parallel=2)
    submitted = []
    monkeypatch.setattr(s, "submit", lambda target, work_dir, db_key="": submitted.append((target, work_dir)))

    s.run_all(["/res/0/model.DB", "/res/1/model.DB"])
    assert len(submitted) == 2
    # solve_target 去掉扩展名，work_dir 为其所在目录
    targets = [t for t, _ in submitted]
    assert targets[0].endswith("model") and not targets[0].endswith(".DB")


def test_progress_init_and_mark_done(tmp_path):
    pf = str(tmp_path / "process_info.json")
    s = solver.DeformSolver(process_info_file=pf)
    dbs = ["/res/0/m.DB", "/res/1/m.DB"]
    prog = s._init_progress(dbs)
    assert prog["total"] == 2 and prog["completed"] == 0
    assert prog["created_at"]  # 初始化即写入起始时间
    # 标记开始/完成后落盘、计数刷新、时间戳记录
    s._mark_started("/res/0/m.DB")
    s._mark_done("/res/0/m.DB")
    import json
    saved = json.loads((tmp_path / "process_info.json").read_text())
    assert saved["completed"] == 1
    item0 = next(it for it in saved["db_files"] if it["db_path"] == "/res/0/m.DB")
    assert item0["done"] and item0["started_at"] and item0["finished_at"]
    done = [it["db_path"] for it in saved["db_files"] if it["done"]]
    assert done == ["/res/0/m.DB"]


def test_created_at_stable_across_reinit(tmp_path):
    """created_at 只在首次初始化写入，续跑重新 init 不刷新。"""
    pf = str(tmp_path / "process_info.json")
    s = solver.DeformSolver(process_info_file=pf)
    dbs = ["/res/0/m.DB"]
    first = s._init_progress(dbs)["created_at"]
    second = s._init_progress(dbs)["created_at"]
    assert first == second


def test_pending_db_files_skips_completed(tmp_path):
    pf = str(tmp_path / "process_info.json")
    s = solver.DeformSolver(process_info_file=pf)
    dbs = ["/res/0/m.DB", "/res/1/m.DB", "/res/2/m.DB"]
    s._init_progress(dbs)
    s._mark_done("/res/1/m.DB")
    # 续跑时只返回未完成的
    assert s.pending_db_files(dbs) == ["/res/0/m.DB", "/res/2/m.DB"]


def test_init_progress_preserves_prior_done(tmp_path):
    pf = str(tmp_path / "process_info.json")
    s = solver.DeformSolver(process_info_file=pf)
    dbs = ["/res/0/m.DB", "/res/1/m.DB"]
    s._init_progress(dbs)
    s._mark_done("/res/0/m.DB")
    # 再次 init（模拟重启进入 run_all）应保留已完成标记
    prog = s._init_progress(dbs)
    assert prog["completed"] == 1


def test_run_all_resumes_only_pending(monkeypatch, tmp_path):
    monkeypatch.setattr(solver.time, "sleep", lambda *_: None)
    pf = str(tmp_path / "process_info.json")
    s = solver.DeformSolver(max_parallel=2, process_info_file=pf)
    dbs = ["/res/0/m.DB", "/res/1/m.DB"]
    s._init_progress(dbs)
    s._mark_done("/res/0/m.DB")  # 模拟中断前已完成 0

    submitted = []
    monkeypatch.setattr(s, "submit",
                        lambda target, work_dir, db_key="": submitted.append(db_key))
    s.run_all(dbs)
    # 只重新提交未完成的 1
    assert submitted == ["/res/1/m.DB"]


def test_progress_noop_without_file():
    """未配置 process_info_file 时进度操作为空操作，不报错。"""
    s = solver.DeformSolver()
    s._mark_done("/x.DB")  # 不应抛异常
    assert s.pending_db_files(["/x.DB"]) == ["/x.DB"]


def test_completed_callback_runs_for_resumed_db(monkeypatch, tmp_path):
    progress_file = str(tmp_path / "process_info.json")
    completed = []
    s = solver.DeformSolver(
        process_info_file=progress_file,
        on_completed=completed.append,
    )
    dbs = ["/res/0/m.DB"]
    s._init_progress(dbs)
    s._mark_done(dbs[0])
    monkeypatch.setattr(s, "submit", lambda *args, **kwargs: None)
    s.run_all(dbs)
    assert completed == [dbs[0]]
