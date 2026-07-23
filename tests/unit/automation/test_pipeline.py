"""Doe_execute / Doe_sample_generate 状态机测试（is_test 分支，不启子进程）。"""

import threading

import pytest

from mobo.automation import pipeline
from mobo.automation.pipeline import Doe_execute, Doe_sample_generate


@pytest.fixture(autouse=True)
def _instant_sleep(monkeypatch):
    """把 is_test 分支里的 time.sleep(10) 变为立即返回，加速状态机测试。"""
    monkeypatch.setattr(pipeline.time, "sleep", lambda *_: None)


def _join_worker_threads(before):
    """等待 generate/process/extract 启动的后台线程结束。"""
    for t in threading.enumerate():
        if t not in before and t is not threading.current_thread():
            t.join(timeout=5.0)


def _make_exec(tmp_path):
    smp = tmp_path / "smp.txt"
    smp.write_text("900 500 500 30\n910 480 480 28\n", encoding="utf-8")
    return Doe_execute(
        sample_file=str(smp),
        std_key_file=str(tmp_path / "model.key"),
        temp_key_path=str(tmp_path / "temp_key"),
        res_db_path=str(tmp_path / "res_db"),
        res_key_path=str(tmp_path / "res_key"),
        res_txt_path=str(tmp_path / "res_txt"),
        parmeter=[["temp"], ["workpiece"]],
        target_var=[["grain"], ["workpiece"]],
        is_inprogress=[False],
        max_step=10,
        is_test=True,
    )


def test_generate_key_file_state_machine(tmp_path):
    exc = _make_exec(tmp_path)
    assert exc.pre_status == pipeline.Task_Status_done  # 初始为 done
    before = set(threading.enumerate())
    exc.generate_key_file()
    _join_worker_threads(before)
    assert exc.tmp_key_file == ["MingrHu"]
    assert exc.pre_status == pipeline.Task_Status_done


def test_process_run_requires_done(tmp_path, monkeypatch):
    exc = _make_exec(tmp_path)
    exc.pre_status = pipeline.Task_Status_running  # 非 done
    errors = []
    monkeypatch.setattr(pipeline.logger, "error", lambda m: errors.append(m))
    exc.process_run()
    assert any("pre_status not done" in e for e in errors)


def test_sample_generate_unsupported_method(monkeypatch):
    errors = []
    monkeypatch.setattr(pipeline.logger, "error", lambda m: errors.append(m))
    Doe_sample_generate("bogus", {"t": (0.0, 1.0)}, "/tmp/whatever", 5)
    assert any("Unsupported sample method" in e for e in errors)
