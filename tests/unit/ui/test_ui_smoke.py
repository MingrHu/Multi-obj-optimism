from __future__ import annotations

import importlib
import time

import pytest


def test_pyside_application_module_imports_when_gui_extra_is_installed():
    pytest.importorskip("PySide6")
    module = importlib.import_module("UI.mobo_ui.app")
    assert callable(module.main)


def test_path_picker_keeps_a_readable_browse_button(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    picker = module.PathPicker("训练数据")
    directory_picker = module.PathPicker("工作区", directory=True)

    assert picker.button.text() == "选择文件"
    assert directory_picker.button.text() == "选择文件夹"
    assert picker.button.minimumWidth() >= 104
    app.processEvents()


def test_connection_check_is_signal_driven_and_bounded(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtTest import QSignalSpy, QTest
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    window = module.MainWindow()
    window.api_url.setText("http://127.0.0.1:1")
    spy = QSignalSpy(window.connection_checked)
    started = time.monotonic()

    window.check_connection()
    while spy.count() == 0 and time.monotonic() - started < 4:
        QTest.qWait(20)

    assert spy.count() == 1
    assert time.monotonic() - started < 4
    assert window.connect_button.isEnabled()
    assert window.connection.text() == "●  未连接"
    window.close()
    app.processEvents()


def test_results_page_maps_any_columns_to_2d_or_3d_axes(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.ResultsPage()
    headers = [f"process_{index}" for index in range(11)] + ["load", "grain", "roundness"]
    page.set_data(headers, [[float(index) for index in range(len(headers))]])

    assert page.x_field.count() == len(headers)
    assert page.y_field.count() == len(headers)
    assert page.z_field.count() == len(headers)
    page.x_field.setCurrentText("process_8")
    page.y_field.setCurrentText("roundness")
    page.update_chart()
    assert "roundness / process_8" == page.chart.chart().title()

    page.style.setCurrentIndex(page.style.findData("scatter3d"))
    assert page.z_field.isVisibleTo(page)
    assert page.scatter3d is None
    app.processEvents()


def test_results_page_keeps_nan_column_in_table_but_not_axis_picker(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.ResultsPage()
    page.set_data(["load", "grain", "empty_target"], [[100, 20, float("nan")]])

    assert page.table.columnCount() == 3
    assert page.x_field.count() == 2
    assert page.x_field.findText("empty_target") == -1
    assert "empty_target" in page.chart_hint.text()
    app.processEvents()


def test_model_schema_uses_explicit_column_roles_and_populates_optimization(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    pool = module.QThreadPool.globalInstance()
    model_page = module.ModelPage(pool, lambda: None)
    model_page.headers = ["load", "workpiece_temperature", "roundness"]
    model_page.rows = [[100, 900, 0.3], [120, 1100, 0.2]]
    model_page.field_roles = ["output", "input", "output"]

    schema = model_page._current_schema()
    assert schema["inputs"] == [{
        "name": "workpiece_temperature", "lower": 900.0, "upper": 1100.0,
    }]
    assert schema["outputs"] == ["load", "roundness"]

    optimization_page = module.OptimizationPage(pool, lambda: None)
    optimization_page.set_schema(schema)
    assert optimization_page.variables.rows() == [["workpiece_temperature", "900", "1100"]]
    assert [row[0] for row in optimization_page.objectives.rows()] == ["load", "roundness"]
    app.processEvents()


def test_field_editor_returns_renamed_fields_and_roles(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QTableWidgetItem

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    dialog = module.FieldEditorDialog(
        ["字段 1", "字段 2", "字段 3"], ["input", "input", "output"]
    )
    dialog.table.setItem(0, 1, QTableWidgetItem("工件温度"))
    dialog.role_boxes[1].setCurrentIndex(dialog.role_boxes[1].findData("output"))

    names, roles = dialog.values()
    assert names == ["工件温度", "字段 2", "字段 3"]
    assert roles == ["input", "output", "output"]
    app.processEvents()


def test_completed_training_and_optimization_switch_to_restart_actions(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    pool = module.QThreadPool.globalInstance()
    model_page = module.ModelPage(pool, lambda: None)
    model_page.update_progress({
        "status": "finished",
        "stage": "finished",
        "progress": 100,
        "models": [{"model_family": "PRG", "score": 0.86}],
    })
    optimization_page = module.OptimizationPage(pool, lambda: None)
    optimization_page.update_progress({
        "status": "finished",
        "stage": "finished",
        "progress": 100,
        "result": {"columns": []},
    })

    assert model_page.start_button.text() == "重新训练"
    assert "无需重复训练" in model_page.training_record_hint.text()
    assert model_page.stage_label.text() == "阶段：已完成"
    assert "T" not in model_page.updated_label.text()
    assert optimization_page.start_button.text() == "重新优化"
    assert optimization_page.stage_label.text() == "阶段：已完成"
    app.processEvents()


def test_doe_lists_wait_for_manual_selection_and_optimization_does_not_navigate(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtTest import QSignalSpy
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    pool = module.QThreadPool.globalInstance()
    items = {"items": [{"id": "done-1", "name": "已完成任务"}]}
    model_page = module.ModelPage(pool, lambda: None)
    optimization_page = module.OptimizationPage(pool, lambda: None)
    model_page._does_loaded(items)
    optimization_page._does_loaded(items)
    result_spy = QSignalSpy(optimization_page.result_loaded)
    optimization_page.update_progress({
        "status": "finished",
        "stage": "finished",
        "progress": 100,
        "result": {"columns": ["temperature", "load"], "resource_id": "tos-result"},
    })

    assert model_page.doe_id.currentIndex() == -1
    assert optimization_page.doe_id.currentIndex() == -1
    assert not model_page.doe_id.isEditable()
    assert not optimization_page.doe_id.isEditable()
    assert optimization_page.doe_id.placeholderText() == "请选择已有 DOE 任务"
    assert result_spy.count() == 0
    app.processEvents()


def test_optimization_selection_loads_persisted_training_schema(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])

    class FakeClient:
        def training_progress(self, doe_id):
            assert doe_id == "trained-1"
            return {
                "input_names": ["工件温度", "压下速度"],
                "target_names": ["载荷", "晶粒尺寸", "圆度"],
                "input_bounds": [
                    {"name": "工件温度", "lower": 900, "upper": 1100},
                    {"name": "压下速度", "lower": 10, "upper": 50},
                ],
            }

        def optimization_progress(self, doe_id):
            return {"status": "not_started", "stage": "not_started", "progress": 0}

    page = module.OptimizationPage(module.QThreadPool.globalInstance(), FakeClient)
    page.run_async = lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    page._does_loaded({"items": [{"id": "trained-1", "name": "唯一任务"}]})
    assert page.doe_id.currentIndex() == -1
    page.doe_id.setCurrentIndex(0)
    page.load_selected_doe()

    assert page.doe_id.currentText() == "唯一任务"
    assert page.doe_id.currentData() == "trained-1"
    assert page.variables.rows() == [["工件温度", "900", "1100"], ["压下速度", "10", "50"]]
    assert [row[0] for row in page.objectives.rows()] == ["载荷", "晶粒尺寸", "圆度"]
    app.processEvents()


def test_model_page_saves_field_definition_without_starting_training(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    captured = {}

    class FakeClient:
        def save_training_dataset(self, payload):
            captured.update(payload)
            return {
                "sample_count": 2,
                "input_names": ["工件温度"],
                "target_names": ["载荷"],
            }

    page = module.ModelPage(module.QThreadPool.globalInstance(), FakeClient)
    page.run_async = lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    page.doe_id.addItem("任务 A", "doe-a")
    page.doe_id.setCurrentIndex(0)
    page.headers = ["工件温度", "载荷"]
    page.rows = [[900, 8.2], [1100, 7.9]]
    page.field_roles = ["input", "output"]
    page._mark_configuration_dirty()
    page.status_loading = True
    page._refresh_training_actions()
    assert page.save_button.isEnabled()
    assert not page.start_button.isEnabled()
    page.save_configuration()

    assert captured["id"] == "doe-a"
    assert captured["data_source"] == {
        "input_data": {"labels": ["工件温度"], "samples": [[900], [1100]]},
        "output_data": {"labels": ["载荷"], "samples": [[8.2], [7.9]]},
    }
    assert page.save_button.text() == "已保存"
    assert page.save_button.isEnabled()
    assert "2 行数据" in page.save_hint.text()
    app.processEvents()


def test_model_page_uploads_selected_file_to_current_doe(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    captured = {}

    class FakeClient:
        def save_training_dataset(self, payload):
            captured.update(payload)
            return {
                "resource_id": "tos-uploaded",
                "sample_count": 2,
                "input_names": ["字段 1"],
                "target_names": ["字段 2"],
            }

    source = tmp_path / "训练数据.txt"
    source.write_text("900\t8.2\n1100\t7.9\n", encoding="utf-8")
    page = module.ModelPage(module.QThreadPool.globalInstance(), FakeClient)
    page.run_async = lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    page.doe_id.addItem("任务 A", "doe-a")
    page.doe_id.setCurrentIndex(0)
    page.input_count.setValue(1)

    page.load_dataset(str(source))

    assert captured["id"] == "doe-a"
    assert captured["data_source"]["source_name"] == "训练数据.txt"
    assert captured["data_source"]["input_data"]["samples"] == [[900.0], [1100.0]]
    assert captured["data_source"]["output_data"]["samples"] == [[8.2], [7.9]]
    assert page.loaded_dataset_key == "doe-a:tos-uploaded"
    assert page.save_button.text() == "已上传"
    assert "已上传并绑定" in page.save_hint.text()
    app.processEvents()


def test_model_page_resubmits_latest_file_after_duplicate_change(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    requests = []
    queued = []

    class FakeClient:
        def save_training_dataset(self, payload):
            requests.append(payload)
            return {
                "resource_id": f"tos-{len(requests)}",
                "sample_count": 2,
                "input_names": ["字段 1"],
                "target_names": ["字段 2"],
            }

    source = tmp_path / "重复选择.txt"
    source.write_text("900\t8.2\n1100\t7.9\n", encoding="utf-8")
    page = module.ModelPage(module.QThreadPool.globalInstance(), FakeClient)
    page.run_async = lambda fn, done=None, failed=None: queued.append((fn, done, failed))
    page.doe_id.addItem("任务 A", "doe-a")
    page.doe_id.setCurrentIndex(0)

    page.load_dataset(str(source))
    page.load_dataset(str(source))
    assert page.configuration_saving
    assert page.pending_configuration_save
    assert not page.save_button.isEnabled()

    action, done, _failed = queued.pop(0)
    done(action())
    assert page.configuration_saving
    assert len(queued) == 1

    action, done, _failed = queued.pop(0)
    done(action())
    assert len(requests) == 2
    assert not page.configuration_saving
    assert not page.configuration_dirty
    assert page.save_button.isEnabled()
    assert page.save_button.text() == "已上传"
    app.processEvents()


def test_new_doe_refreshes_and_selects_the_created_task(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])

    class FakeClient:
        def list_doe(self):
            return {"items": [{"id": "new-doe-id", "name": "新任务"}]}

        def training_progress(self, doe_id):
            assert doe_id == "new-doe-id"
            return {"status": "not_started", "stage": "not_started", "progress": 0}

    page = module.ModelPage(module.QThreadPool.globalInstance(), FakeClient)
    page.run_async = lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    page._created_doe({"id": "new-doe-id", "name": "新任务"})

    assert page.doe_id.currentText() == "新任务"
    assert page.doe_id.currentData() == "new-doe-id"
    assert not page.status_loading
    app.processEvents()


def test_saved_doe_restores_dataset_preview_and_field_roles(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])

    class FakeClient:
        def training_progress(self, doe_id):
            assert doe_id == "saved-doe"
            return {
                "status": "not_started",
                "stage": "not_started",
                "progress": 0,
                "input_names": ["温度", "速度"],
                "target_names": ["载荷"],
                "input_bounds": [],
                "dataset": {
                    "resource_id": "tos-saved",
                    "columns": ["温度", "速度", "载荷"],
                    "sample_count": 2,
                    "source_name": "训练数据.txt",
                },
            }

        def get_data(self, doe_id, resource_id, fields):
            assert (doe_id, resource_id) == ("saved-doe", "tos-saved")
            assert fields == ["温度", "速度", "载荷"]
            return {"values": {"温度": [900, 1000], "速度": [10, 20], "载荷": [8.2, 7.9]}}

    page = module.ModelPage(module.QThreadPool.globalInstance(), FakeClient)
    page.run_async = lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    page.doe_id.addItem("已保存任务", "saved-doe")
    page.doe_id.setCurrentIndex(0)
    page._doe_selected()

    assert page.headers == ["温度", "速度", "载荷"]
    assert page.rows == [[900, 10, 8.2], [1000, 20, 7.9]]
    assert page.field_roles == ["input", "input", "output"]
    assert page.input_count.value() == 2
    assert "训练数据.txt" in page.dataset.text()
    assert page.preview.rowCount() == 2
    assert page.save_button.text() == "已保存"
    assert "正在读取" not in page.training_record_hint.text()
    app.processEvents()


def test_repeated_doe_refresh_and_status_requests_are_coalesced(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    queued = []
    page = module.ModelPage(module.QThreadPool.globalInstance(), lambda: None)
    page.run_async = lambda fn, done=None, failed=None: queued.append((fn, done, failed))

    page.refresh_does()
    page.refresh_does()
    page.refresh_does()
    assert len(queued) == 1
    assert not page.refresh_doe_button.isEnabled()
    queued.pop(0)[1]({"items": []})
    assert page.refresh_doe_button.isEnabled()

    page.doe_id.addItem("任务 A", "doe-a")
    page.doe_id.addItem("任务 B", "doe-b")
    page.doe_id.setCurrentIndex(0)
    page._doe_selected()
    assert len(queued) == 1
    page.doe_id.setCurrentIndex(1)
    page._doe_selected()
    assert len(queued) == 2
    queued.pop(0)[1]({"status": "not_started", "stage": "not_started", "progress": 0})
    assert len(queued) == 1
    queued.pop(0)[1]({"status": "not_started", "stage": "not_started", "progress": 0})
    assert not page.status_loading
    assert "正在读取" not in page.training_record_hint.text()

    optimization_page = module.OptimizationPage(
        module.QThreadPool.globalInstance(), lambda: None
    )
    optimization_page.run_async = (
        lambda fn, done=None, failed=None: queued.append((fn, done, failed))
    )
    optimization_page.refresh_does()
    optimization_page.refresh_does()
    assert len(queued) == 1
    queued.pop(0)[1]({"items": []})
    assert optimization_page.refresh_doe_button.isEnabled()

    results_page = module.ResultsPage(
        module.QThreadPool.globalInstance(), lambda: None
    )
    results_page.run_async = (
        lambda fn, done=None, failed=None: queued.append((fn, done, failed))
    )
    results_page.refresh_tasks()
    results_page.refresh_tasks()
    assert len(queued) == 1
    queued.pop(0)[1]({"items": []})
    assert results_page.refresh_tasks_button.isEnabled()
    app.processEvents()


def test_async_mixin_keeps_worker_until_gui_callback(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.ModelPage(module.QThreadPool.globalInstance(), lambda: None)
    completed = []
    loop = QEventLoop()

    page.run_async(lambda: "完成", lambda value: (completed.append(value), loop.quit()))
    assert len(page._active_workers) == 1
    QTimer.singleShot(3000, loop.quit)
    loop.exec()

    assert completed == ["完成"]
    assert not page._active_workers
    app.processEvents()


def test_lhs_ui_can_generate_exact_count_without_boundaries(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    captured = {}

    class Definition:
        parameters = ({"name": "温度"}, {"name": "速度"})

        def generate_samples(self, **kwargs):
            captured.update(kwargs)
            output = tmp_path / "exact-lhs.txt"
            output.write_text("1\t2\n3\t4\n", encoding="utf-8")
            return str(output)

    page = module.AutomationPage(module.QThreadPool.globalInstance(), multi=False)
    page.current_definition = lambda: Definition()
    page.run_async = lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    page.workspace_path.setText(str(tmp_path))
    page.sample_count.setValue(2)
    page.include_boundaries.setChecked(False)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)

    page.generate_samples()

    assert captured["method"] == "lhs"
    assert captured["n_samples"] == 2
    assert captured["include_boundaries"] is False
    assert page.sample_path.text().endswith("exact-lhs.txt")
    app.processEvents()


def test_model_page_status_and_dataset_requests_have_timeouts(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.ModelPage(module.QThreadPool.globalInstance(), lambda: None)
    page.doe_id.addItem("任务 A", "doe-a")
    page.doe_id.setCurrentIndex(0)
    page.selected_doe_id = "doe-a"
    page.selection_revision = 3
    page.status_request_id = 7
    page.status_request_in_flight = True
    page.status_loading = True

    page._progress_timed_out("doe-a", 3, 7)
    assert not page.status_request_in_flight
    assert not page.status_loading
    assert "超时" in page.training_record_hint.text()

    page.dataset_request_key = "doe-a:tos-1"
    page._saved_dataset_timed_out("doe-a", 3, "doe-a:tos-1")
    assert not page.dataset_request_key
    assert "超时" in page.data_quality.text()
    app.processEvents()


def test_optimization_tables_use_structured_editors_and_live_validation(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.OptimizationPage(module.QThreadPool.globalInstance(), lambda: None)
    page.set_schema({
        "inputs": [{"name": "temperature", "lower": 900, "upper": 1100}],
        "outputs": ["load", "grain"],
    })
    page.mode.setCurrentIndex(page.mode.findData("single"))

    assert isinstance(page.objectives.itemDelegateForColumn(1), module.ChoiceDelegate)
    assert isinstance(page.objectives.itemDelegateForColumn(2), module.NumericDelegate)
    assert isinstance(page.variables.itemDelegateForColumn(1), module.NumericDelegate)
    assert [row[2] for row in page.objectives.rows()] == ["0.5", "0.5"]
    assert page.validate_configuration()

    page.variables.item(0, 2).setText("800")
    assert not page.validate_configuration()
    assert "下限小于上限" in page.validation_hint.text()
    app.processEvents()


def test_wide_tables_are_scrollable_resizable_and_rows_can_be_deleted(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QHeaderView

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    table = module.DataTable()
    table.enable_wide_columns()
    table.set_data([f"field_{index}" for index in range(14)], [[index for index in range(14)]])
    page = module.OptimizationPage(module.QThreadPool.globalInstance(), lambda: None)
    page.objectives.set_data(["名称", "方向", "权重"], [["load", "min", "1"]])
    before = page.objectives.rowCount()
    page.objectives.selectRow(0)
    page._delete_selected_rows(page.objectives)

    assert table.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
    assert sum(table.columnWidth(index) for index in range(table.columnCount())) > table.viewport().width()
    assert page.objectives.rowCount() == before - 1
    app.processEvents()


def test_results_page_loads_a_completed_backend_task_directly(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCharts import QChartView
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])

    class FakeClient:
        def list_doe(self):
            return {"items": [
                {"id": "done-1", "name": "已完成任务", "stage": "optimization_finished"},
                {"id": "running-1", "name": "运行中任务", "stage": "optimization_running"},
            ]}

        def optimization_progress(self, doe_id):
            assert doe_id == "done-1"
            return {
                "status": "finished",
                "result": {"columns": ["temperature", "load"], "resource_id": "tos-result"},
                "history": [
                    {
                        "run_id": "run-old", "status": "finished",
                        "request": {"optimizer": "nsga2", "requested_mode": "multi"},
                        "result": {
                            "columns": ["temperature", "load"], "resource_id": "tos-old",
                            "constraint_check": {"solution_count": 1},
                            "task_info": {"run_time_sec": 1.5},
                        },
                        "updated_at": "2026-09-08T10:00:00+08:00",
                    },
                    {
                        "run_id": "run-new", "status": "finished",
                        "request": {"optimizer": "nsga2", "requested_mode": "multi"},
                        "result": {
                            "columns": ["temperature", "load"], "resource_id": "tos-result",
                            "constraint_check": {"solution_count": 2},
                            "task_info": {"run_time_sec": 2.5},
                        },
                        "updated_at": "2026-09-09T10:00:00+08:00",
                    },
                ],
            }

        def get_data(self, doe_id, resource_id, fields):
            assert doe_id == "done-1"
            if resource_id == "tos-old":
                return {"values": {"temperature": [850], "load": [9.1]}}
            assert resource_id == "tos-result"
            return {"values": {"temperature": [900, 1000], "load": [8.2, 7.9]}}

    page = module.ResultsPage(module.QThreadPool.globalInstance(), FakeClient)
    page.run_async = lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    page._tasks_loaded(FakeClient().list_doe())
    assert page.result_task.count() == 1
    assert page.result_task.currentIndex() == -1
    page.result_task.setCurrentIndex(0)
    assert page.result_task.currentText() == "已完成任务"
    assert page.result_task.currentData() == "done-1"
    page.load_selected_task()

    assert page.headers == ["temperature", "load"]
    assert page.table.rowCount() == 2
    assert "已加载任务 done-1" in page.task_hint.text()
    assert page.history_table.rowCount() == 2
    assert page.result_run.count() == 2
    page.result_run.setCurrentIndex(page.result_run.findData("run-old"))
    page.load_selected_run()
    assert page.table.rowCount() == 1
    assert "run-old" in page.task_hint.text()
    assert page.chart.rubberBand() == QChartView.RubberBand.RectangleRubberBand
    app.processEvents()


def test_stale_doe_responses_do_not_overwrite_the_current_selection(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    pool = module.QThreadPool.globalInstance()
    model_page = module.ModelPage(pool, lambda: None)
    model_page.doe_id.addItem("任务 A", "doe-a")
    model_page.doe_id.addItem("任务 B", "doe-b")
    model_page.doe_id.setCurrentIndex(1)
    model_page._apply_progress(
        "doe-a", model_page.selection_revision, {"status": "finished", "models": [{"score": 1}]}
    )
    assert model_page.training_state == "not_started"

    optimization_page = module.OptimizationPage(pool, lambda: None)
    optimization_page.doe_id.addItem("任务 A", "doe-a")
    optimization_page.doe_id.addItem("任务 B", "doe-b")
    optimization_page.doe_id.setCurrentIndex(1)
    optimization_page.selection_revision = 2
    stale_schema = {
        "input_names": ["旧输入"],
        "target_names": ["旧目标"],
        "input_bounds": [{"name": "旧输入", "lower": 0, "upper": 1}],
    }
    optimization_page._training_schema_loaded("doe-a", 1, stale_schema)
    assert optimization_page.variables.rowCount() == 0
    assert optimization_page.objectives.rowCount() == 0
    app.processEvents()


def test_training_and_optimization_buttons_follow_runtime_state(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    pool = module.QThreadPool.globalInstance()
    model_page = module.ModelPage(pool, lambda: None)
    assert not model_page.start_button.isEnabled()
    assert not model_page.stop_button.isEnabled()
    model_page.training_state = "running"
    model_page._refresh_training_actions()
    assert model_page.stop_button.isEnabled()
    model_page.update_progress({"status": "finished", "models": []})
    assert not model_page.stop_button.isEnabled()

    optimization_page = module.OptimizationPage(pool, lambda: None)
    optimization_page.doe_id.addItem("任务", "doe-1")
    optimization_page.doe_id.setCurrentIndex(0)
    optimization_page.set_schema({
        "inputs": [{"name": "temperature", "lower": 900, "upper": 1100}],
        "outputs": ["load"],
    })
    optimization_page.update_progress({"status": "not_started", "progress": 0})
    assert optimization_page.start_button.isEnabled()
    assert not optimization_page.stop_button.isEnabled()
    optimization_page.update_progress({"status": "running", "progress": 10})
    assert not optimization_page.start_button.isEnabled()
    assert optimization_page.stop_button.isEnabled()
    app.processEvents()


def test_results_use_virtual_table_and_downsample_large_charts(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCharts import QLineSeries
    from PySide6.QtWidgets import QApplication, QTableView

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.ResultsPage()
    rows = [[index, index * 2, index * 3] for index in range(50_000)]
    page.set_data(["x", "y", "z"], rows)
    series = page.chart.chart().series()[0]

    assert isinstance(page.table, QTableView)
    assert isinstance(page.table.model(), module.ResultTableModel)
    assert page.table.rowCount() == 50_000
    assert isinstance(series, QLineSeries)
    assert series.count() == module.MAX_2D_PLOT_POINTS
    assert "均匀抽样" in page.plot_status.text()
    points, total = page._numeric_points_3d()
    assert total == 50_000
    assert len(points) == module.MAX_3D_PLOT_POINTS
    app.processEvents()


def test_ga_and_ppo_have_separate_parameter_panels_and_payloads(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    captured = []

    class FakeClient:
        def start_optimization(self, payload):
            captured.append(payload)
            return {}

    page = module.OptimizationPage(module.QThreadPool.globalInstance(), FakeClient)
    page.run_async = lambda fn, done=None, failed=None: captured.append(fn())
    page.doe_id.addItem("任务", "doe-1")
    page.doe_id.setCurrentIndex(0)
    page.set_schema({
        "inputs": [{"name": "temperature", "lower": 900, "upper": 1100}],
        "outputs": ["load"],
    })
    page.ga_population.setValue(80)
    page.ga_offspring.setValue(30)
    page.ga_generations.setValue(60)
    page.start_optimization()
    ga_payload = captured[0]
    assert ga_payload["algorithm"]["params"]["pop_size"] == 80
    assert ga_payload["algorithm"]["params"]["n_offsprings"] == 30

    captured.clear()
    page.optimization_state = "not_started"
    page.mode.setCurrentIndex(page.mode.findData("reinforcement_learning"))
    page.ppo_timesteps.setValue(1234)
    page.ppo_episode_steps.setValue(55)
    page.start_optimization()
    ppo_payload = captured[0]
    assert page.parameter_stack.currentIndex() == 1
    assert ppo_payload["algorithm"]["params"]["total_timesteps"] == 1234
    assert ppo_payload["algorithm"]["params"]["episode_steps"] == 55
    assert "pop_size" not in ppo_payload["algorithm"]["params"]
    app.processEvents()


def test_doe_refresh_failures_are_visible_inline(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("UI.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    pool = module.QThreadPool.globalInstance()
    model_page = module.ModelPage(pool, lambda: None)
    optimization_page = module.OptimizationPage(pool, lambda: None)
    model_page._doe_refresh_failed("连接超时")
    optimization_page._doe_refresh_failed("连接超时")

    assert "连接超时" in model_page.training_record_hint.text()
    assert "重试" in model_page.training_record_hint.text()
    assert "连接超时" in optimization_page.schema_hint.text()
    assert "重试" in optimization_page.schema_hint.text()
    app.processEvents()
