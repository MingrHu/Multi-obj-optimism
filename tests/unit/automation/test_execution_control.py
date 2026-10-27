"""取消竞态、队列排空与中断数据库无损恢复。"""
import threading
from pathlib import Path

import pytest

from mobo.automation import solver
from mobo.automation.execution_control import BatchControl, BatchStopped, batch_scope, check_stopped


def test_cancel_before_process_registration_terminates_new_process(monkeypatch):
    from mobo.automation import execution_control as control_module
    class Process:
        pid = 123
        ended = False
        def poll(self):
            return 1 if self.ended else None
    process = Process()
    killed = []
    def terminate(item):
        killed.append(item.pid)
        item.ended = True
    monkeypatch.setattr(control_module, '_terminate_tree', terminate)
    control = BatchControl()
    control.request_stop()
    with pytest.raises(BatchStopped):
        control.register(process)
    assert killed == [123]


def test_stop_does_not_submit_remaining_queue_and_waits_for_worker(monkeypatch, tmp_path):
    control = BatchControl()
    calls = []
    finished = threading.Event()
    def solve(path):
        calls.append(path)
        control.request_stop()
        assert not finished.is_set()
        finished.set()
        check_stopped()
    monkeypatch.setattr(solver, 'solve_db_sync', solve)
    runner = solver.DeformSolver(max_parallel=1, process_info_file=str(tmp_path/'progress.json'))
    paths = [str(tmp_path/f'{i}.DB') for i in range(5)]
    with batch_scope(control), pytest.raises(BatchStopped):
        runner.run_all(paths)
    assert len(calls) == 1
    assert finished.is_set() and runner.running == 0
    assert runner.pending_db_files(paths) == paths
    assert runner._load_progress()["status"] == "stopped"
    assert all(row["stopped"] for row in runner._load_progress()["db_files"])


def test_recover_latest_step_on_copy_and_archive_original(monkeypatch, tmp_path):
    raw = tmp_path/'FOR003'
    raw.write_bytes(b'precious progress')
    db = tmp_path/'result.DB'
    db.write_bytes(b'older db')
    monkeypatch.setattr(solver, 'query_db_steps', lambda path: [-1, 60, 83] if Path(path).read_bytes() == b'precious progress' else [-84, -1, 60, 83])
    def actions(path):
        text = Path(path).read_text()
        assert 'DBREAD 0\n' in text and 'GENDB 1 1 0\n' in text
        assert raw.read_bytes() == b'precious progress'
        candidate = Path(text.splitlines()[1])
        candidate.write_bytes(b'restart latest')
    monkeypatch.setattr(solver, 'run_key_actions', actions)
    assert solver.recover_interrupted_db(str(db))
    assert db.read_bytes() == b'restart latest'
    assert next((tmp_path/'recovery').glob('*/FOR003')).read_bytes() == b'precious progress'
    assert next((tmp_path/'recovery').glob('*/result.DB')).read_bytes() == b'older db'


def test_bad_recovery_preserves_original_and_allows_key_rebuild(monkeypatch, tmp_path):
    (tmp_path/'FOR003').write_bytes(b'damaged')
    db = tmp_path/'result.DB'
    db.write_bytes(b'old')
    monkeypatch.setattr(solver, 'query_db_steps', lambda path: (_ for _ in ()).throw(RuntimeError('bad database')))
    assert not solver.recover_interrupted_db(str(db))
    assert not db.exists()
    assert next((tmp_path/'recovery').glob('*/FOR003')).read_bytes() == b'damaged'


def test_cannot_recover_database_owned_by_live_solver(monkeypatch, tmp_path):
    raw = tmp_path/'FOR003'
    raw.write_bytes(b'live')
    (tmp_path/'DEFORM_RUNNING_JOB_STATUS.TXT').write_text('processID=123\n')
    monkeypatch.setattr(solver, 'process_alive', lambda pid: True)
    with pytest.raises(RuntimeError, match='仍在使用'):
        solver.recover_interrupted_db(str(tmp_path/'result.DB'))
    assert raw.read_bytes() == b'live'
    assert not (tmp_path/'recovery').exists()


def test_cancel_during_recovery_keeps_original_untouched(monkeypatch, tmp_path):
    control = BatchControl()
    raw = tmp_path/'FOR003'
    raw.write_bytes(b'progress')
    monkeypatch.setattr(solver, 'query_db_steps', lambda path: [-1, 83])
    def cancel(path):
        control.request_stop()
        check_stopped()
    monkeypatch.setattr(solver, 'run_key_actions', cancel)
    with batch_scope(control), pytest.raises(BatchStopped):
        solver.recover_interrupted_db(str(tmp_path/'result.DB'))
    assert raw.read_bytes() == b'progress'
    assert not (tmp_path/'recovery').exists()
