"""六参数模板经真实 UI 控件保存、抽样、生成 KEY 与启动流水线。"""

import re
from pathlib import Path

import pytest


@pytest.fixture
def page(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from work_platform.mobo_ui.app import AutomationPage, QThreadPool
    from mobo.automation import template_store
    from mobo.automation.task_collection import GH4169_RING_SINGLE_TASK_1

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(template_store, "AUTOMATION_TEMPLATES_DIR", tmp_path / "templates")
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    widget = AutomationPage(QThreadPool.globalInstance(), multi=False)
    widget.task_editor.load_definition(GH4169_RING_SINGLE_TASK_1)
    widget.task_editor.task_id.setText("gh4169-ui-test")
    widget.task_editor.task_name.setText("GH4169六参数验证")
    widget.workspace_path.setText(str(tmp_path / "run"))
    template = tmp_path / "ring.KEY"
    template.write_text("".join(
        [f"REFTMP {obj} 20\nNDTMP {obj} 0 150\n" for obj in range(1, 6)]
        + ["ANGMOV 2 1 0 1.2\nMOVCTL 3 1 2 0 1 0 5\n0 0\n10 0.6\n36.25 0.6\n46.25 0\n65 0\n"]
        + [f"FRCFAC 1 {obj} 1 0 0.7\nIHTCOF 1 {obj} 0 5\n" for obj in range(2, 6)]
    ))
    widget.template_path.setText(str(template))
    errors = []
    widget.show_error = errors.append

    def immediate(action, done, failed):
        try:
            result = action()
        except Exception as exc:
            failed(str(exc))
        else:
            done(result)

    widget.run_async = immediate
    yield widget, errors
    widget.timer.stop()
    widget.close()
    app.processEvents()


def test_edit_save_reload_sample_generate_and_run(page, monkeypatch, tmp_path):
    from mobo.automation.template_store import load_template
    from mobo.automation.pipeline import ForgingTask

    widget, errors = page
    assert widget.task_editor.parameter_table.rowCount() == 6
    # 固定角速度输入由同值上下限表达，其余五个参数仍抽样。
    widget.task_editor.parameter_table.item(3, 3).setText("1.75")
    widget.task_editor.parameter_table.item(3, 4).setText("1.75")
    widget.save_current_template()
    assert not errors
    assert load_template("gh4169-ui-test").parameters[3]["range"] == [1.75, 1.75]
    widget.load_definition()
    assert widget.task_editor.parameter_table.text(3, 3) == "1.75"
    widget.sample_count.setValue(2)
    widget.generate_samples()
    samples = Path(widget.sample_path.text()).read_text().splitlines()
    assert len(samples) == 2
    assert all(re.fullmatch(r"\d+\.\d{2}", value)
               for row in samples for value in row.split("\t"))
    assert all(row.split("\t")[3] == "1.75" for row in samples)
    widget.prepare_keys()
    keys = list((tmp_path / "run" / "input_keys").glob("*.KEY"))
    assert len(keys) == 2 and not errors
    text = keys[0].read_text()
    assert "REFTMP 1 20\n" in text
    assert "ANGMOV 2 1 0 1.7500000000E+000" in text
    assert "FRCFAC 1 5 1 0" in text

    calls = []
    original = ForgingTask.run_solver

    def checked_run(task):
        calls.append(task.param_table)
        return original(task)

    monkeypatch.setattr(ForgingTask, "run_solver", checked_run)
    widget.dry_run.setChecked(True)
    widget.run_calculation()
    assert len(calls) == 1 and len(calls[0][0]) == 6
    assert not errors and widget.stage_label.text() == "批处理已完成"


def test_old_samples_do_not_start_solver_or_report_success(page, monkeypatch, tmp_path):
    from mobo.automation.pipeline import ForgingTask

    widget, errors = page
    sample = tmp_path / "old.txt"
    sample.write_text("1125\t300\t1.2\n")
    widget.sample_path.setText(str(sample))
    widget.dry_run.setChecked(False)
    monkeypatch.setattr(ForgingTask, "run_solver", lambda task: pytest.fail("不应启动求解"))
    widget.run_calculation()
    assert errors and "列数" in errors[0]
    assert widget.stage_label.text() != "批处理已完成"


@pytest.mark.parametrize("first,last", [(1, 100), (101, 200)])
def test_range_controls_apply_to_keys_solver_and_extraction(page, monkeypatch, tmp_path, first, last):
    from mobo.automation.pipeline import ForgingTask
    from mobo.automation import task_collection

    widget, errors = page
    monkeypatch.setattr(task_collection, "task_dir", lambda tid: tmp_path / "state" / tid)
    sample = tmp_path / "300.tsv"
    sample.write_text("1125.00\t300.00\t1.25\t1.50\t0.50\t4.50\n" * 300)
    widget.sample_path.setText(str(sample))
    widget.limit_sample_range.setChecked(True)
    widget.sample_start.setValue(first)
    widget.sample_end.setValue(last)
    widget.prepare_keys()
    assert not errors
    files = list((tmp_path / "run" / "input_keys").glob("*.KEY"))
    assert {path.name for path in files} == {f"ring{i}.KEY" for i in range(first - 1, last)}

    solver_calls, extraction_calls = [], []
    original_solver, original_extract = ForgingTask.run_solver, ForgingTask.extract

    def solve(task):
        solver_calls.append(task.sample_indices)
        return original_solver(task)

    def extract(task):
        extraction_calls.append((task.sample_indices, task.db_files, task.result_txt_dir))
        return original_extract(task)

    monkeypatch.setattr(ForgingTask, "run_solver", solve)
    monkeypatch.setattr(ForgingTask, "extract", extract)
    widget.dry_run.setChecked(True)
    widget.run_calculation()
    widget.extract_results()
    assert not errors
    expected = list(range(first - 1, last))
    assert solver_calls == [expected]
    assert extraction_calls[0][0] == expected
    assert len(extraction_calls[0][1]) == 100
    assert f"range_{first - 1}_{last}" in extraction_calls[0][2]


@pytest.mark.parametrize("first,last", [(120, 100), (201, 301)])
def test_invalid_range_is_reported_before_starting_work(page, tmp_path, first, last):
    widget, errors = page
    sample = tmp_path / "300.tsv"
    sample.write_text("1125\t300\t1.25\t1.5\t0.5\t4.5\n" * 300)
    widget.sample_path.setText(str(sample))
    widget.limit_sample_range.setChecked(True)
    widget.sample_start.setValue(first)
    widget.sample_end.setValue(last)
    widget.run_async = lambda *args: pytest.fail("无效范围不应启动后台任务")
    widget.prepare_keys()
    assert errors
    assert not widget.busy


def test_progress_shows_source_sample_numbers_and_changes_with_range(page, monkeypatch, tmp_path):
    import json
    from mobo.automation import task_collection

    widget, errors = page
    monkeypatch.setattr(task_collection, "task_dir", lambda tid: tmp_path / "state" / tid)
    sample = tmp_path / "300.tsv"
    sample.write_text("1125\t300\t1.25\t1.5\t0.5\t4.5\n" * 300)
    widget.sample_path.setText(str(sample))
    widget.limit_sample_range.setChecked(True)
    widget.sample_end.setValue(200)
    widget.sample_start.setValue(101)
    task = widget.current_definition().build(
        sample, workspace=widget.workspace_path.text(), **widget._sample_range(),
    )
    path = Path(task.process_info_file)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"db_files": [
        {"db_path": str(tmp_path / "db" / "100" / "ring100.DB"), "done": True},
        {"db_path": str(tmp_path / "db" / "199" / "ring199.DB"), "done": False},
    ]}))
    widget.refresh_progress()
    assert widget.batch_table.rowCount() == 2
    assert widget.batch_table.item(0, 0).text() == "101"
    assert widget.batch_table.item(1, 0).text() == "200"
    widget.sample_end.setValue(300)
    widget.sample_start.setValue(201)
    assert widget.batch_table.rowCount() == 0
    assert not errors
