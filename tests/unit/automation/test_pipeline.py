"""ForgingTask / TaskStatus 状态机测试 (:mod:`mobo.automation.pipeline`)。

用 ``dry_run=True`` 只推进状态、不真正调用 DEFORM，验证三阶段的状态转移与前置约束。
"""

import threading


from mobo.automation import pipeline
from mobo.automation.pipeline import ForgingTask, TaskStatus, generate_sample_file


def _join_worker_threads(before):
    """等待各阶段启动的后台线程结束。"""
    for t in threading.enumerate():
        if t not in before and t is not threading.current_thread():
            t.join(timeout=5.0)


def _make_task(tmp_path):
    smp = tmp_path / "smp.txt"
    smp.write_text("900 30\n910 28\n", encoding="utf-8")
    return ForgingTask(
        sample_file=str(smp),
        template_key=str(tmp_path / "model.key"),
        temp_key_dir=str(tmp_path / "temp_key"),
        result_db_dir=str(tmp_path / "res_db"),
        result_key_dir=str(tmp_path / "res_key"),
        result_txt_dir=str(tmp_path / "res_txt"),
        param_table=[["temp", "speed"], ["workpiece", "topdie"]],
        target_table=[["grain"], ["workpiece"]],
        in_progress=[False],
        dry_run=True,
    )


def test_status_enum_values():
    assert int(TaskStatus.DONE) == 0
    assert int(TaskStatus.RUNNING) == 1
    assert int(TaskStatus.FAILED) == -1


def test_initial_status_is_done(tmp_path):
    task = _make_task(tmp_path)
    assert task.status == TaskStatus.DONE


def test_load_samples_into_table(tmp_path):
    task = _make_task(tmp_path)
    task.load_samples_into_table()
    # 表头 2 行 + 2 个样本
    assert len(task.param_table) == 4
    assert task.param_table[2] == ["900", "30"]


def test_generate_keys_dry_run(tmp_path):
    task = _make_task(tmp_path)
    before = set(threading.enumerate())
    task.generate_keys()
    _join_worker_threads(before)
    assert task.status == TaskStatus.DONE
    assert task.key_files == ["<dry-run>"]


def test_full_pipeline_dry_run(tmp_path):
    task = _make_task(tmp_path)
    before = set(threading.enumerate())
    task.generate_keys()
    _join_worker_threads(before)
    assert task.status == TaskStatus.DONE

    task.run_solver()
    _join_worker_threads(before)
    assert task.status == TaskStatus.DONE

    task.extract()
    _join_worker_threads(before)
    assert task.status == TaskStatus.DONE


def test_stage_requires_previous_done(tmp_path, monkeypatch):
    task = _make_task(tmp_path)
    task.status = TaskStatus.RUNNING  # 模拟上一阶段未完成
    errors = []
    monkeypatch.setattr(pipeline.logger, "error", lambda m: errors.append(m))
    thread = task.run_solver()
    assert thread is None
    assert any("上一阶段未完成" in e for e in errors)


def test_run_async_marks_failed(tmp_path):
    task = _make_task(tmp_path)
    before = set(threading.enumerate())

    def boom():
        raise RuntimeError("boom")

    task._run_async("坏阶段", boom)
    _join_worker_threads(before)
    assert task.status == TaskStatus.FAILED


def test_generate_sample_file_delegates(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(
        pipeline, "generate_samples",
        lambda task_id, method, pr, sd, ns, ln: called.update(task_id=task_id, method=method, sd=sd) or "out.txt",
    )
    out = generate_sample_file("t1", "lhs", {"t": (0.0, 1.0)}, str(tmp_path), 5)
    assert out == "out.txt"
    assert called["method"] == "lhs"
    assert called["task_id"] == "t1"
