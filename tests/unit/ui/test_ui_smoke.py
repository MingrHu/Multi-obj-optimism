from __future__ import annotations

import importlib
import time
import warnings
from pathlib import Path

import pytest


def test_pyside_application_module_imports_when_gui_extra_is_installed():
    pytest.importorskip("PySide6")
    module = importlib.import_module("work_platform.mobo_ui.app")
    assert callable(module.main)


def test_ui_assets_resolve_when_desktop_entry_imports_top_level_package(monkeypatch):
    pytest.importorskip("PySide6")
    work_platform = Path(__file__).resolve().parents[3] / "work_platform"
    monkeypatch.syspath_prepend(str(work_platform))
    theme = importlib.import_module("mobo_ui.theme")

    assert Path(theme.ui_asset_path("hust_mark_name.png")).is_file()
    assert Path(theme.ui_asset_path("spin_up.svg")).is_file()
    assert Path(theme.ui_asset_path("spin_up_light.svg")).is_file()


def test_path_picker_keeps_a_readable_browse_button(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    picker = module.PathPicker("训练数据")
    directory_picker = module.PathPicker("工作区", directory=True)

    assert picker.button.text() == "选择文件"
    assert directory_picker.button.text() == "选择文件夹"
    assert picker.button.minimumWidth() >= 104
    app.processEvents()


def test_sidebar_groups_pages_without_numeric_prefixes_and_collapses(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    window = module.MainWindow()

    assert [button.text() for button in window.nav_group_buttons] == [
        "工作台", "优化模块"
    ]
    assert [button.text() for button in window.nav_buttons] == [
        "工作台概览",
        "单工步计算",
        "多工步批处理",
        "代理模型",
        "优化中心",
        "结果分析",
    ]
    assert all(button.isChecked() for button in window.nav_group_buttons)
    assert all(not content.isHidden() for content in window.nav_group_contents)

    window.nav_group_buttons[0].click()
    assert window.nav_group_contents[0].isHidden()
    assert window.nav_group_buttons[0].arrowType() == Qt.ArrowType.RightArrow
    window.nav_group_buttons[0].click()
    assert not window.nav_group_contents[0].isHidden()
    assert window.nav_group_buttons[0].arrowType() == Qt.ArrowType.DownArrow
    window.nav_buttons[1].click()
    assert window.stack.currentIndex() == 1
    window.close()
    app.processEvents()


def test_connection_check_is_signal_driven_and_bounded(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtTest import QSignalSpy, QTest
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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


def test_display_mode_switch_updates_widgets_and_all_chart_engines(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from matplotlib.colors import to_hex
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    module.configure_application(app)
    window = module.MainWindow()
    window.show()
    publication_index = window.display_mode_selector.findData("publication")

    window.display_mode_selector.setCurrentIndex(publication_index)
    app.processEvents()

    assert app.property("displayMode") == "publication"
    assert not window.hust_brand.isHidden()
    assert window.hust_brand.pixmap() is not None
    assert not window.hust_brand.pixmap().isNull()
    assert window.hust_brand.width() == 250
    assert "#e4e6e8" in app.styleSheet()
    assert "spin_up.svg" in app.styleSheet()
    assert "border-left: 3px solid #005b96" in app.styleSheet()

    model_page = window.pages[3]
    model_page.update_score_chart(
        [{"model_family": "PRG", "score": 0.91}]
    )
    assert model_page.score_chart.chart().backgroundBrush().color().name() == "#f2f3f4"
    assert not model_page.score_chart.chart().isBackgroundVisible()
    assert model_page.score_chart.chart().plotAreaBackgroundBrush().color().name() == "#ffffff"

    results_page = window.pages[5]
    results_page.set_data(
        ["温度", "载荷", "晶粒尺寸"],
        [[900.0, 100.0, 48.2], [1000.0, 120.0, 49.1]],
    )
    results_page.update_chart()
    assert results_page.chart.chart().backgroundBrush().color().name() == "#f2f3f4"
    assert not results_page.chart.chart().isBackgroundVisible()
    assert results_page.chart.chart().plotAreaBackgroundBrush().color().name() == "#ffffff"
    results_page.style.setCurrentIndex(results_page.style.findData("scatter3d"))
    results_page.update_chart()
    assert to_hex(results_page.scatter3d.figure.get_facecolor()) == "#ffffff"
    assert to_hex(results_page.scatter3d.axes.get_facecolor()) == "#ffffff"

    window.apply_display_mode("engineering")
    app.processEvents()
    assert window.hust_brand.isHidden()
    assert "spin_up_light.svg" in app.styleSheet()
    assert results_page.chart.chart().backgroundBrush().color().name() == "#132139"
    assert to_hex(results_page.scatter3d.figure.get_facecolor()) == "#0d1829"
    window.close()
    app.processEvents()


def test_results_page_maps_any_columns_to_2d_or_3d_axes(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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
    page.update_chart()
    assert isinstance(page.scatter3d, module.Scatter3DCanvas)
    assert page.chart_stack.currentWidget() is page.scatter3d
    assert page.scatter3d.axes.get_xlabel() == "process_8"
    assert page.scatter3d.axes.get_ylabel() == "roundness"
    assert page.scatter3d.axes.get_zlabel() == "process_2"
    original_x_limits = page.scatter3d.axes.get_xlim3d()
    page.zoom_chart(1.25)
    zoomed_x_limits = page.scatter3d.axes.get_xlim3d()
    assert zoomed_x_limits[1] - zoomed_x_limits[0] < original_x_limits[1] - original_x_limits[0]
    page.reset_chart_zoom()
    assert page.scatter3d.axes.get_xlim3d() == pytest.approx(original_x_limits)
    page.open_chart_window()
    assert len(page.chart_windows) == 1
    assert page.chart_windows[0].findChild(module.Scatter3DCanvas) is not None
    page.chart_windows[0].close()
    app.processEvents()


def test_results_3d_chart_renders_chinese_labels_without_layout_warnings(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.ResultsPage()
    page.set_data(
        ["驱动辊载荷 N", "等效应变标准差", "平均晶粒尺寸 μm"],
        [[index, index / 10, 48 + index / 100] for index in range(20)],
    )
    page.style.setCurrentIndex(page.style.findData("scatter3d"))
    supported_fonts = {
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
    }
    if module.matplotlib_cjk_font().get_name() not in supported_fonts:
        pytest.skip("No supported CJK font is installed")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        page.update_chart()
        page.scatter3d.draw()
        app.processEvents()

    messages = [str(item.message) for item in caught]
    assert not [message for message in messages if "Glyph" in message]
    assert not [message for message in messages if "Tight layout" in message]
    assert page.scatter3d.font_properties.get_name() in supported_fonts


def test_results_page_keeps_nan_column_in_table_but_not_axis_picker(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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


def test_running_training_uses_active_button_and_model_hint(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.ModelPage(module.QThreadPool.globalInstance(), lambda: None)
    page.update_progress({
        "status": "running",
        "stage": "evaluating",
        "progress": 95,
        "dataset": {"resource_id": "dataset-1"},
        "models": [],
        "current_model": "DNN",
        "current_model_index": 5,
        "total_models": 5,
    })

    assert page.start_button.text() == "正在训练…"
    assert not page.start_button.isEnabled()
    assert page.training_record_hint.text() == "正在进行交叉验证：DNN（5/5）。"
    assert "尚未开始训练" not in page.training_record_hint.text()
    app.processEvents()


def test_doe_lists_wait_for_manual_selection_and_optimization_does_not_navigate(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtTest import QSignalSpy
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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


def test_model_page_submits_explicit_hyperparameter_overrides(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    captured = {}

    class FakeClient:
        def start_training(self, payload):
            captured.update(payload)
            return {"status": "queued"}

        def training_progress(self, doe_id):
            assert doe_id == "doe-a"
            return {"status": "queued", "stage": "queued", "progress": 0}

    page = module.ModelPage(module.QThreadPool.globalInstance(), FakeClient)
    page.run_async = lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    page._hyperparameters_loaded({
        "models": {
            "RF": {
                "n_estimators": {
                    "type": "integer", "default": 300, "minimum": 1,
                    "label": "决策树数量",
                },
                "max_depth": {
                    "type": "integer", "default": None, "nullable": True,
                    "minimum": 1, "label": "最大深度",
                },
            }
        }
    })
    page.doe_id.addItem("任务 A", "doe-a")
    page.doe_id.setCurrentIndex(0)
    page.headers = ["温度", "载荷"]
    page.rows = [[900, 8.2], [1000, 8.0], [1100, 7.9]]
    page.field_roles = ["input", "output"]
    for name, checkbox in page.models.items():
        checkbox.setChecked(name == "RF")
    page.model_params["RF"] = {"n_estimators": 120, "max_depth": 8}
    page._refresh_model_parameter_buttons()

    page.start_training()

    assert captured["models"] == [{
        "name": "RF", "params": {"n_estimators": 120, "max_depth": 8}
    }]
    assert page.model_param_buttons["RF"].text() == "参数（2 项）"
    assert page.model_param_buttons["RF"].isEnabled()
    app.processEvents()


def test_hyperparameter_dialog_blank_values_mean_defaults(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    dialog = module.ModelHyperparameterDialog(
        "SVR",
        {
            "kernel": {
                "type": "string", "default": "rbf", "choices": ["linear", "rbf"],
                "label": "核函数",
            },
            "C": {"type": "number", "default": 1.0, "exclusive_minimum": 0, "label": "C"},
        },
        {},
    )
    assert dialog._collect_values() == {}
    dialog.editors["C"].setText("12.5")
    assert dialog._collect_values() == {"C": 12.5}
    app.processEvents()


def test_hyperparameter_dialog_accepts_numeric_or_named_union_value(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QLineEdit

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    dialog = module.ModelHyperparameterDialog(
        "SVR",
        {
            "gamma": {
                "type": "number_or_string",
                "default": "scale",
                "choices": ["scale", "auto"],
                "exclusive_minimum": 0,
                "label": "Gamma",
            },
        },
        {},
    )
    assert isinstance(dialog.editors["gamma"], QLineEdit)
    dialog.editors["gamma"].setText("0.125")
    assert dialog._collect_values() == {"gamma": 0.125}
    dialog.editors["gamma"].setText("auto")
    assert dialog._collect_values() == {"gamma": "auto"}
    app.processEvents()


def test_model_parameter_button_can_retry_after_catalog_failure(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])

    class FakeClient:
        def training_hyperparameters(self):
            return {
                "models": {
                    "RF": {
                        "n_estimators": {
                            "type": "integer",
                            "default": 300,
                            "minimum": 1,
                            "label": "决策树数量",
                        }
                    }
                }
            }

    page = module.ModelPage(module.QThreadPool.globalInstance(), FakeClient)
    page._hyperparameters_failed("404 NOT FOUND")
    assert page.model_param_buttons["RF"].isEnabled()
    assert page.model_param_buttons["RF"].text() == "参数（重试）"

    class FakeDialog:
        configured_values = {}

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return module.QDialog.DialogCode.Rejected

    monkeypatch.setattr(module, "ModelHyperparameterDialog", FakeDialog)
    page.run_async = lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    page.configure_model("RF")
    assert "RF" in page.hyperparameter_catalog
    assert page.model_param_buttons["RF"].text() == "参数（默认）"
    app.processEvents()


def test_model_page_uploads_selected_file_to_current_doe(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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


def test_model_page_can_delete_selected_doe_after_confirmation(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    deleted = []

    class FakeClient:
        def delete_doe(self, doe_id):
            deleted.append(doe_id)
            return {"value": doe_id}

        def list_doe(self):
            return {"items": [{"id": "doe-keep", "name": "保留任务"}]}

    client = FakeClient()
    page = module.ModelPage(
        module.QThreadPool.globalInstance(), lambda: client
    )
    page.run_async = (
        lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    )
    page.doe_id.addItem("待删除任务", "doe-delete")
    page.doe_id.setCurrentIndex(0)
    page.status_loading = False
    page.training_state = "running"
    page._refresh_training_actions()
    assert not page.delete_doe_button.isEnabled()
    page.training_state = "not_started"
    page._refresh_training_actions()
    assert page.delete_doe_button.isEnabled()
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    page.delete_current_doe()

    assert deleted == ["doe-delete"]
    assert page.doe_id.count() == 1
    assert page.doe_id.itemData(0) == "doe-keep"
    assert page.doe_id.currentIndex() == -1
    assert not page.delete_doe_button.isEnabled()
    assert page.training_state == "not_started"
    app.processEvents()


def test_async_mixin_keeps_worker_until_gui_callback(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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


def test_single_operation_editor_builds_definition_from_ui(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.task_editor")
    tasks = importlib.import_module("mobo.automation.task_collection")
    app = QApplication.instance() or QApplication([])
    editor = module.TaskDefinitionEditor(multi=False)
    editor.load_definition(tasks.RING_7050_SINGLE_TASK_1)

    editor.task_id.setText("custom-single")
    editor.task_name.setText("自定义单工步")
    editor.parameter_table.item(0, 3).setText("300")
    editor.parameter_table.item(0, 4).setText("480")
    definition = editor.build_definition(
        tasks.RING_7050_SINGLE_TASK_1,
        workspace=str(tmp_path / "single"),
        template_key=str(tmp_path / "template.KEY"),
    )

    assert definition.task_id == "custom-single"
    assert definition.name == "自定义单工步"
    assert definition.workspace == tmp_path / "single"
    assert definition.template_key == str(tmp_path / "template.KEY")
    assert definition.parameters[0]["range"] == [300.0, 480.0]
    assert definition.targets == tasks.RING_7050_SINGLE_TASK_1.targets
    app.processEvents()


def test_multi_operation_editor_applies_operation_and_parameter_rows(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.task_editor")
    tasks = importlib.import_module("mobo.automation.task_collection")
    app = QApplication.instance() or QApplication([])
    editor = module.TaskDefinitionEditor(multi=True)
    editor.load_definition(tasks.TC4_RING_MULTI_TASK_1)

    editor.task_id.setText("custom-multi")
    editor.operation_table.item(0, 1).setText(str(tmp_path / "first.KEY"))
    editor.parameter_table.item(0, 3).setText("810")
    editor.parameter_table.item(0, 4).setText("955")
    definition = editor.build_definition(
        tasks.TC4_RING_MULTI_TASK_1,
        workspace=str(tmp_path / "multi"),
    )

    assert definition.task_id == "custom-multi"
    assert definition.workspace == tmp_path / "multi"
    assert definition.operations[0]["template_key"] == str(tmp_path / "first.KEY")
    assert definition.operations[1]["parameters"][0]["range"] == [810.0, 955.0]
    assert definition.operations[1]["inherit_materials"] is True
    assert definition.operations[1]["enable_grain"] is True
    app.processEvents()


def test_task_editor_rejects_unknown_parameter_type(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.task_editor")
    tasks = importlib.import_module("mobo.automation.task_collection")
    app = QApplication.instance() or QApplication([])
    editor = module.TaskDefinitionEditor(multi=False)
    editor.load_definition(tasks.RING_7050_SINGLE_TASK_1)
    editor.parameter_table.cellWidget(0, 1).setCurrentText("not_registered")

    with pytest.raises(ValueError, match="未注册的参数类型"):
        editor.build_definition(
            tasks.RING_7050_SINGLE_TASK_1,
            workspace=str(tmp_path),
            template_key=str(tmp_path / "template.KEY"),
        )
    app.processEvents()


def test_automation_page_reports_invalid_task_definition_without_starting(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.AutomationPage(module.QThreadPool.globalInstance(), multi=False)
    page.task_editor.parameter_table.cellWidget(0, 1).setCurrentText("not_registered")
    errors = []
    page.show_error = errors.append
    page.run_async = lambda *args, **kwargs: pytest.fail("invalid definition started work")

    page.generate_samples()

    assert errors and "未注册的参数类型" in errors[0]
    assert not page.busy
    app.processEvents()


def test_automation_page_can_create_and_delete_user_template(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox

    store = importlib.import_module("mobo.automation.template_store")
    monkeypatch.setattr(store, "AUTOMATION_TEMPLATES_DIR", tmp_path / "templates")
    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.AutomationPage(module.QThreadPool.globalInstance(), multi=False)
    key_file = tmp_path / "custom.KEY"
    key_file.write_text("REFTMP 1 300\n", encoding="utf-8")
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args: QMessageBox.StandardButton.Yes,
    )

    page.new_template()
    page.task_editor.task_id.setText("ui-custom-single")
    page.task_editor.task_name.setText("界面自定义单工步")
    page.template_path.setText(str(key_file))
    page.workspace_path.setText(str(tmp_path / "workspace"))
    page.task_editor.target_table.item(0, 0).setText("stress_result")
    page.save_current_template()

    assert store.load_template("ui-custom-single").name == "界面自定义单工步"
    assert page.definition.currentData() == {
        "source": "custom",
        "task_id": "ui-custom-single",
    }
    assert page.delete_template_button.isEnabled()

    page.delete_current_template()

    with pytest.raises(FileNotFoundError):
        store.load_template("ui-custom-single")
    app.processEvents()


def test_model_page_status_and_dataset_requests_have_timeouts(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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


def test_model_score_chart_shows_values_and_full_score_tooltip(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.ModelPage(module.QThreadPool.globalInstance(), lambda: None)
    page.update_score_chart([
        {"model_family": "PRG", "score": 0.860123},
        {"model_family": "SVR", "score": 0.842567},
    ])

    series = page.score_chart.chart().series()[0]
    values = series.barSets()[0]
    assert series.isLabelsVisible()
    assert series.labelsPrecision() == 3
    assert series.labelsPosition() == module.QAbstractBarSeries.LabelsPosition.LabelsInsideEnd
    values.hovered.emit(True, 0)
    assert page.score_detail.text() == "当前模型  PRG：0.860123"
    values.hovered.emit(False, 0)
    assert "悬停或点击" in page.score_detail.text()
    page.resize(1100, 760)
    page.show()
    app.processEvents()
    hover_position = page.score_chart.chart().mapToPosition(
        QPointF(0.5, 0.4), series
    )
    page._score_chart_mouse_moved(hover_position)
    app.processEvents()
    assert page.score_detail.text() == "当前模型  PRG：0.860123"
    values.clicked.emit(1)
    assert page.score_detail.text() == "已选择  SVR：0.842567"
    values.hovered.emit(False, 1)
    assert page.score_detail.text() == "已选择  SVR：0.842567"
    app.processEvents()


def test_optimization_tables_use_structured_editors_and_live_validation(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    assert page.headers == ["temperature", "load"]
    assert page.table.rowCount() == 2
    assert "已加载任务 done-1" in page.task_hint.text()
    assert len(page.history_rows) == 2
    assert page.history_run_ids == ["run-new", "run-old"]
    assert not hasattr(page, "history_panel")
    assert page.open_history_button.isEnabled()
    page.open_history_window()
    assert len(page.history_windows) == 1
    assert page.history_windows[0].isVisible()
    history_window_table = page.history_windows[0].findChild(module.DataTable)
    assert history_window_table is not None
    assert history_window_table.rowCount() == 2
    history_window_table.selectRow(1)
    history_load_button = next(
        button for button in page.history_windows[0].findChildren(module.QPushButton)
        if button.text() == "加载选中版本"
    )
    assert history_load_button.isEnabled()
    history_load_button.click()
    assert page.table.rowCount() == 1
    assert "run-old" in page.task_hint.text()
    assert page.chart.rubberBand() == QChartView.RubberBand.RectangleRubberBand
    app.processEvents()


def test_results_auto_load_latest_completed_run_while_reoptimization_is_active(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])

    class FakeClient:
        def optimization_progress(self, doe_id):
            assert doe_id == "doe-1"
            return {
                "status": "running",
                "current_run_id": "run-new",
                "result": None,
                "history": [
                    {
                        "run_id": "run-old",
                        "status": "finished",
                        "request": {"optimizer": "nsga2", "requested_mode": "multi"},
                        "result": {
                            "columns": ["temperature", "load"],
                            "resource_id": "tos-old",
                        },
                    },
                    {
                        "run_id": "run-new",
                        "status": "running",
                        "request": {"optimizer": "nsga2", "requested_mode": "multi"},
                        "result": None,
                    },
                ],
            }

        def get_data(self, doe_id, resource_id, fields):
            assert (doe_id, resource_id, fields) == (
                "doe-1", "tos-old", ["temperature", "load"]
            )
            return {"values": {"temperature": [900], "load": [8.2]}}

    page = module.ResultsPage(module.QThreadPool.globalInstance(), FakeClient)
    page.run_async = lambda fn, done=None, failed=None: (done or (lambda _value: None))(fn())
    page._tasks_loaded({
        "items": [{
            "id": "doe-1",
            "name": "任务一",
            "has_optimization_result": True,
        }]
    })
    assert page.result_task.currentIndex() == -1

    page.result_task.setCurrentIndex(0)

    assert len(page.history_rows) == 2
    assert page.selected_run_id == "run-old"
    assert page.headers == ["temperature", "load"]
    assert page.table.rowCount() == 1
    assert "run-old" in page.task_hint.text()
    app.processEvents()


def test_stale_doe_responses_do_not_overwrite_the_current_selection(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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

    module = importlib.import_module("work_platform.mobo_ui.app")
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
    optimization_page.current_training_run_id = "train-1"
    optimization_page.set_schema({
        "inputs": [{"name": "temperature", "lower": 900, "upper": 1100}],
        "outputs": ["load"],
    })
    optimization_page.update_progress({"status": "not_started", "progress": 0})
    assert optimization_page.start_button.isEnabled()
    assert not optimization_page.stop_button.isEnabled()
    optimization_page.update_progress({"status": "running", "progress": 10})
    assert not optimization_page.start_button.isEnabled()
    assert optimization_page.start_button.text() == "正在优化…"
    assert optimization_page.stop_button.isEnabled()
    optimization_page.update_progress({"status": "queued", "progress": 0})
    assert optimization_page.start_button.text() == "正在优化…"
    app.processEvents()


def test_training_history_is_selectable_in_model_and_optimization_pages(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    history = [
        {
            "run_id": "train-old", "status": "finished",
            "updated_at": "2026-09-24T10:00:00+08:00",
            "models": [{"model_id": "rf-old", "model_family": "RF", "score": 0.8}],
        },
        {
            "run_id": "train-new", "status": "finished",
            "updated_at": "2026-09-24T11:00:00+08:00",
            "models": [{"model_id": "rf-new", "model_family": "RF", "score": 0.9}],
        },
    ]
    pool = module.QThreadPool.globalInstance()
    model_page = module.ModelPage(pool, lambda: None)
    model_page.update_progress({
        "status": "finished", "stage": "finished", "progress": 100,
        "current_run_id": "train-new", "selected_run_id": "train-new",
        "history": history, "models": history[-1]["models"],
    })
    assert model_page.training_run.count() == 2
    assert model_page.training_run.currentData() == "train-new"

    optimization_page = module.OptimizationPage(pool, lambda: None)
    optimization_page.doe_id.addItem("Task", "doe-1")
    optimization_page.doe_id.setCurrentIndex(0)
    optimization_page._training_schema_loaded("doe-1", 0, {
        "selected_run_id": "train-new", "current_run_id": "train-new",
        "history": history,
        "input_names": ["x"], "target_names": ["y"],
        "input_bounds": [{"name": "x", "lower": 0.0, "upper": 1.0}],
    })
    assert optimization_page.training_run.count() == 2
    assert optimization_page.current_training_run_id == "train-new"
    app.processEvents()


def test_results_use_virtual_table_and_downsample_large_charts(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCharts import QScatterSeries, QValueAxis
    from PySide6.QtWidgets import QApplication, QTableView

    module = importlib.import_module("work_platform.mobo_ui.app")
    app = QApplication.instance() or QApplication([])
    page = module.ResultsPage()
    rows = [[index, index * 2, index * 3] for index in range(50_000)]
    page.set_data(["x", "y", "z"], rows)
    series = page.chart.chart().series()[0]

    assert isinstance(page.table, QTableView)
    assert isinstance(page.table.model(), module.ResultTableModel)
    assert page.table.rowCount() == 50_000
    assert isinstance(series, QScatterSeries)
    assert series.count() == module.MAX_2D_PLOT_POINTS
    assert not page.chart.chart().legend().isVisible()
    assert "均匀抽样" in page.plot_status.text()
    axes = [axis for axis in page.chart.chart().axes() if isinstance(axis, QValueAxis)]
    assert [axis.titleText() for axis in axes] == ["x", "y"]
    assert page.open_history_button.text() == "查看运行历史"
    page.chart.double_clicked.emit()
    assert len(page.chart_windows) == 1
    assert page.chart_windows[0].isVisible()
    page.chart_windows[0].close()
    points, total = page._numeric_points_3d()
    assert total == 50_000
    assert len(points) == module.MAX_3D_PLOT_POINTS
    page.style.setCurrentIndex(page.style.findData("scatter3d"))
    page.update_chart()
    assert isinstance(page.scatter3d, module.Scatter3DCanvas)
    plotted = page.scatter3d.axes.collections[0]._offsets3d
    assert len(plotted[0]) == module.MAX_3D_PLOT_POINTS
    app.processEvents()


def test_ga_and_ppo_have_separate_parameter_panels_and_payloads(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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
    assert page.parameter_stack.parent() is page.algorithm_parameter_dialog
    assert page.ga_population.minimumWidth() >= 300
    assert "已设置" in page.algorithm_parameter_button.text()
    assert "NSGA-II" in page.algorithm_parameter_hint.text()
    page.start_optimization()
    ga_payload = captured[0]
    assert page.optimization_state == "queued"
    assert page.start_button.text() == "正在优化…"
    assert not page.start_button.isEnabled()
    assert ga_payload["algorithm"]["params"]["pop_size"] == 80
    assert ga_payload["algorithm"]["params"]["n_offsprings"] == 30

    captured.clear()
    page.optimization_state = "not_started"
    page.mode.setCurrentIndex(page.mode.findData("reinforcement_learning"))
    assert page.parameter_stack.currentIndex() == 1
    assert "PPO" in page.algorithm_parameter_hint.text()
    page.ppo_timesteps.setValue(1234)
    page.ppo_episode_steps.setValue(55)
    assert "已设置" in page.algorithm_parameter_button.text()
    page.start_optimization()
    ppo_payload = captured[0]
    assert ppo_payload["algorithm"]["params"]["total_timesteps"] == 1234
    assert ppo_payload["algorithm"]["params"]["episode_steps"] == 55
    assert "pop_size" not in ppo_payload["algorithm"]["params"]
    page._reset_current_algorithm_parameters()
    assert page.ppo_timesteps.value() == 20000
    assert page.ppo_learning_rate.value() == pytest.approx(0.001)
    assert "默认" in page.algorithm_parameter_button.text()

    def reject_after_edit(_dialog):
        page.ppo_timesteps.setValue(999)
        return module.QDialog.DialogCode.Rejected

    monkeypatch.setattr(module.AlgorithmParameterDialog, "exec", reject_after_edit)
    page.open_algorithm_parameters()
    assert page.ppo_timesteps.value() == 20000
    assert "默认" in page.algorithm_parameter_button.text()
    app.processEvents()


def test_doe_refresh_failures_are_visible_inline(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    module = importlib.import_module("work_platform.mobo_ui.app")
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
