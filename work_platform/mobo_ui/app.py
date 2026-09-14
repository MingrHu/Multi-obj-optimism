"""Native PySide6 workbench for sampling, DEFORM automation and optimization."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCharts import (
    QAbstractBarSeries,
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QLineSeries,
    QScatterSeries,
    QValueAxis,
)
from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QColor, QCursor, QFont, QFontDatabase, QPainter, QPalette
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QStyledItemDelegate,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTableView,
    QTabWidget,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from matplotlib import font_manager
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .core import (
    ApiClient,
    finite_float,
    format_timestamp,
    numeric_field_names,
    prepare_numeric_table,
    read_tabular_file,
    rows_by_fields,
    stage_text,
    status_view,
)


APP_STYLE = """
* { font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 14px; }
QMainWindow, QWidget#Root { background: #0b1220; color: #dce7f5; }
QFrame#TopBar { background: #101a2d; border-bottom: 1px solid #24344d; }
QFrame#Sidebar { background: #0e1728; border-right: 1px solid #24344d; }
QFrame#Card { background: #111d31; border: 1px solid #243650; border-radius: 12px; }
QFrame#StatCard { background: #13233a; border: 1px solid #284666; border-radius: 10px; }
QLabel#Title { font-size: 26px; font-weight: 700; color: #f4f8ff; }
QLabel#Subtitle { color: #8fa3bd; font-size: 13px; }
QLabel#Section { font-size: 17px; font-weight: 600; color: #eff6ff; }
QLabel#Metric { font-size: 25px; font-weight: 700; color: #69d6df; }
QLabel#Caption { color: #93a6bf; font-size: 12px; }
QLabel#StatusPill { padding: 5px 10px; border-radius: 9px; font-weight: 600; }
QPushButton, QToolButton { background: #172740; border: 1px solid #304864; border-radius: 7px;
  color: #e7f0fb; padding: 7px 13px; min-height: 20px; }
QPushButton:hover, QToolButton:hover { background: #1d3352; border-color: #42b9c7; }
QPushButton:pressed { background: #132239; }
QPushButton:disabled { color: #607188; background: #111a29; border-color: #1c2a3c; }
QPushButton#Primary { background: #1b8190; border-color: #35a9b6; color: white; font-weight: 600; }
QPushButton#Primary:hover { background: #2397a5; }
QPushButton#Danger { color: #ffb6b3; border-color: #7f4448; }
QPushButton#Nav { border: 0; background: transparent; text-align: left; padding: 11px 15px;
  color: #a9bad0; border-radius: 8px; }
QPushButton#Nav:hover { background: #14243b; color: #f6fbff; }
QPushButton#Nav:checked { background: #17364d; color: #71dbe2; border-left: 3px solid #5bd0d8; }
QToolButton#NavGroup { border: 0; background: transparent; color: #edf5ff; padding: 9px 8px;
  font-weight: 700; text-align: left; }
QToolButton#NavGroup:hover { background: #14243b; color: #71dbe2; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget, QTableView { background: #0d1829; color: #deebf8;
  border: 1px solid #2c4059; border-radius: 6px; padding: 6px; selection-background-color: #256d82; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus { border-color: #43bdc8; }
QComboBox::drop-down { border: 0; width: 25px; }
QComboBox#TaskSelector { background: #1b2635; color: #d8e0e9; border-color: #465466; }
QComboBox#TaskSelector QAbstractItemView { background: #202b3a; color: #d8e0e9;
  selection-background-color: #586575; selection-color: #ffffff; }
QHeaderView::section { background: #17263b; color: #9fb2c9; border: 0; border-right: 1px solid #263a53;
  border-bottom: 1px solid #263a53; padding: 8px; font-weight: 600; }
QTableWidget, QTableView { gridline-color: #21334a; alternate-background-color: #101d30; }
QProgressBar { background: #0d1828; border: 1px solid #2a4059; border-radius: 5px; height: 10px;
  text-align: center; color: transparent; }
QProgressBar::chunk { background: #4ac7d1; border-radius: 4px; }
QTabWidget::pane { border: 1px solid #263a52; border-radius: 7px; top: -1px; }
QTabBar::tab { background: #111d30; color: #91a5bd; padding: 9px 18px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #62d4dc; border-bottom-color: #62d4dc; }
QScrollBar:vertical { background: #0d1725; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #2a425d; border-radius: 5px; min-height: 30px; }
QStatusBar { background: #0e1727; color: #91a3b8; }
"""


def card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)
    if title:
        label = QLabel(title)
        label.setObjectName("Section")
        layout.addWidget(label)
    return frame, layout


def page_header(title: str, subtitle: str) -> QVBoxLayout:
    layout = QVBoxLayout()
    heading = QLabel(title)
    heading.setObjectName("Title")
    detail = QLabel(subtitle)
    detail.setObjectName("Subtitle")
    detail.setWordWrap(True)
    layout.addWidget(heading)
    layout.addWidget(detail)
    layout.setSpacing(3)
    return layout


def set_status(label: QLabel, state: str | None) -> None:
    view = status_view(state)
    colors = {
        "neutral": ("#24344b", "#adbed2"),
        "info": ("#173d55", "#73d9e1"),
        "warning": ("#503e1c", "#ffd47c"),
        "success": ("#153e35", "#75e1bd"),
        "danger": ("#4d262d", "#ffaaa8"),
    }
    bg, fg = colors[view.tone]
    label.setText(f"●  {view.text}")
    label.setStyleSheet(f"background:{bg}; color:{fg}; padding:5px 10px; border-radius:9px;")


class WorkerSignals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as exc:  # worker boundary: present a useful message to the operator
            try:
                self.signals.failed.emit(str(exc))
            except RuntimeError:
                pass  # the window may have closed while a request was finishing
        else:
            try:
                self.signals.completed.emit(result)
            except RuntimeError:
                pass


class AsyncMixin:
    pool: QThreadPool

    def run_async(
        self,
        fn: Callable[[], Any],
        done: Callable[[Any], None] | None = None,
        failed: Callable[[str], None] | None = None,
    ) -> None:
        worker = Worker(fn)
        # A queued Qt signal is discarded if its sender is destroyed before the
        # GUI thread handles it. Keep each runnable alive until its terminal
        # callback has actually executed.
        worker.setAutoDelete(False)
        active_workers = getattr(self, "_active_workers", None)
        if active_workers is None:
            active_workers = set()
            self._active_workers = active_workers
        active_workers.add(worker)

        def completed(result: Any) -> None:
            try:
                (done or (lambda _result: None))(result)
            finally:
                active_workers.discard(worker)

        def failed_callback(message: str) -> None:
            try:
                (failed or self.show_error)(message)
            finally:
                active_workers.discard(worker)

        worker.signals.completed.connect(completed)
        worker.signals.failed.connect(failed_callback)
        self.pool.start(worker)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "操作未完成", message)


class PathPicker(QWidget):
    changed = Signal(str)

    def __init__(self, placeholder: str, *, directory: bool = False):
        super().__init__()
        self.directory = directory
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed)
        self.button = QToolButton()
        self.button.setText("选择文件夹" if directory else "选择文件")
        self.button.setMinimumWidth(104)
        self.button.clicked.connect(self.browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def browse(self) -> None:
        if self.directory:
            selected = QFileDialog.getExistingDirectory(self, "选择文件夹", self.edit.text())
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self, "选择文件", self.edit.text(), "数据与 DEFORM 文件 (*.txt *.tsv *.csv *.KEY);;所有文件 (*)"
            )
        if selected:
            self.edit.setText(selected)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value)


class DataTable(QTableWidget):
    def __init__(self):
        super().__init__()
        self._wide_columns = False
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setMinimumHeight(190)

    def enable_wide_columns(self) -> None:
        """Keep wide data readable with draggable columns and horizontal scrolling."""
        self._wide_columns = True
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(70)
        header.setDefaultSectionSize(120)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._resize_wide_columns()

    def _resize_wide_columns(self) -> None:
        if not self._wide_columns or not self.columnCount():
            return
        self.resizeColumnsToContents()
        for column in range(self.columnCount()):
            self.setColumnWidth(column, min(240, max(90, self.columnWidth(column))))

    def set_data(self, headers: list[str], rows: list[list[Any]]) -> None:
        self.clear()
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row[: len(headers)]):
                item = QTableWidgetItem(str(value))
                if isinstance(value, (int, float)):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.setItem(row_index, column_index, item)
        self._resize_wide_columns()

    def rows(self) -> list[list[str]]:
        return [
            [self.item(r, c).text() if self.item(r, c) else "" for c in range(self.columnCount())]
            for r in range(self.rowCount())
        ]


class ResultTableModel(QAbstractTableModel):
    """Expose result rows lazily so large datasets do not allocate cell widgets."""

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.headers: list[str] = []
        self.rows: list[list[Any]] = []

    def set_data(self, headers: list[str], rows: list[list[Any]]) -> None:
        self.beginResetModel()
        self.headers = headers
        self.rows = rows
        self.endResetModel()

    def rowCount(self, _parent: QModelIndex | None = None) -> int:
        return len(self.rows)

    def columnCount(self, _parent: QModelIndex | None = None) -> int:
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self.rows):
            return None
        row = self.rows[index.row()]
        value = row[index.column()] if index.column() < len(row) else ""
        if role == Qt.ItemDataRole.DisplayRole:
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and section < len(self.headers):
            return self.headers[section]
        return section + 1


class VirtualDataTable(QTableView):
    """Read-only, resizable result table backed by a virtual Qt item model."""

    def __init__(self):
        super().__init__()
        self.table_model = ResultTableModel(self)
        self.setModel(self.table_model)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(70)
        header.setDefaultSectionSize(120)
        self.setMinimumHeight(190)

    def set_data(self, headers: list[str], rows: list[list[Any]]) -> None:
        self.table_model.set_data(headers, rows)
        for column, name in enumerate(headers):
            self.setColumnWidth(column, min(240, max(90, len(str(name)) * 15)))

    def rowCount(self) -> int:
        return self.table_model.rowCount()

    def columnCount(self) -> int:
        return self.table_model.columnCount()


MAX_2D_PLOT_POINTS = 2000
MAX_3D_PLOT_POINTS = 1000


def evenly_sample(values: list[Any], limit: int) -> list[Any]:
    """Retain the first/last observations and evenly sample the interior."""
    if len(values) <= limit:
        return values
    if limit <= 1:
        return values[:limit]
    last = len(values) - 1
    return [values[round(index * last / (limit - 1))] for index in range(limit)]


class ZoomableChartView(QChartView):
    """A 2D chart view with wheel zoom and rectangle rubber-band zoom."""

    double_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setRubberBand(QChartView.RubberBand.RectangleRubberBand)

    def wheelEvent(self, event) -> None:
        self.chart().zoom(1.2 if event.angleDelta().y() > 0 else 0.8)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self.double_clicked.emit()
        event.accept()


@lru_cache(maxsize=1)
def matplotlib_cjk_font() -> font_manager.FontProperties:
    """Select an installed font that contains Simplified Chinese glyphs."""
    for family in (
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
    ):
        properties = font_manager.FontProperties(family=family)
        try:
            font_manager.findfont(properties, fallback_to_default=False)
        except ValueError:
            continue
        return properties
    return font_manager.FontProperties(family="DejaVu Sans")


class Scatter3DCanvas(FigureCanvasQTAgg):
    """Driver-independent, reusable 3D scatter canvas embedded as a Qt widget."""

    def __init__(self, parent: QWidget | None = None):
        self.figure = Figure(facecolor="#111d31")
        self.figure.subplots_adjust(left=0.02, right=0.96, bottom=0.02, top=0.98)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.font_properties = matplotlib_cjk_font()
        self._base_limits: tuple[tuple[float, float], ...] | None = None
        self._base_view = (24.0, -58.0)

    def set_points(
        self,
        points: list[tuple[float, float, float]],
        labels: tuple[str, str, str],
    ) -> None:
        self.axes.clear()
        xs, ys, zs = zip(*points, strict=True)
        self.axes.scatter(
            xs,
            ys,
            zs,
            s=16,
            c="#51d1da",
            edgecolors="#b8f5f7",
            linewidths=0.25,
            depthshade=False,
        )
        self.axes.set_xlabel(
            labels[0], color="#a9bad0", labelpad=7, fontproperties=self.font_properties
        )
        self.axes.set_ylabel(
            labels[1], color="#a9bad0", labelpad=7, fontproperties=self.font_properties
        )
        self.axes.set_zlabel(
            labels[2], color="#a9bad0", labelpad=7, fontproperties=self.font_properties
        )
        self.axes.tick_params(colors="#9fb1c7", labelsize=8)
        self.axes.set_facecolor("#0d1829")
        self.axes.set_box_aspect((1.0, 1.0, 0.82), zoom=0.78)
        self.axes.grid(True, color="#23364e", linewidth=0.6)
        for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
            axis.pane.set_facecolor("#0d1829")
            axis.pane.set_edgecolor("#23364e")
        self.axes.view_init(elev=self._base_view[0], azim=self._base_view[1])
        self._base_limits = (
            self.axes.get_xlim3d(),
            self.axes.get_ylim3d(),
            self.axes.get_zlim3d(),
        )
        self.draw_idle()

    def zoom_by(self, factor: float) -> None:
        if factor <= 0:
            return
        for getter, setter in (
            (self.axes.get_xlim3d, self.axes.set_xlim3d),
            (self.axes.get_ylim3d, self.axes.set_ylim3d),
            (self.axes.get_zlim3d, self.axes.set_zlim3d),
        ):
            low, high = getter()
            center = (low + high) / 2
            half_span = (high - low) / (2 * factor)
            setter(center - half_span, center + half_span)
        self.draw_idle()

    def wheelEvent(self, event) -> None:
        self.zoom_by(1.2 if event.angleDelta().y() > 0 else 0.8)
        event.accept()

    def reset_view(self) -> None:
        if self._base_limits is not None:
            self.axes.set_xlim3d(*self._base_limits[0])
            self.axes.set_ylim3d(*self._base_limits[1])
            self.axes.set_zlim3d(*self._base_limits[2])
        self.axes.view_init(elev=self._base_view[0], azim=self._base_view[1])
        self.draw_idle()


class ChoiceDelegate(QStyledItemDelegate):
    """Constrain a table cell to one of a small set of protocol values."""

    def __init__(self, choices: list[tuple[str, str]], parent: QWidget | None = None):
        super().__init__(parent)
        self.choices = choices

    def createEditor(self, parent, _option, _index):
        editor = QComboBox(parent)
        for label, value in self.choices:
            editor.addItem(label, value)
        return editor

    def setEditorData(self, editor, index) -> None:
        position = editor.findData(index.data(Qt.ItemDataRole.EditRole))
        editor.setCurrentIndex(max(0, position))

    def setModelData(self, editor, model, index) -> None:
        model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)


class NumericDelegate(QStyledItemDelegate):
    """Use a bounded floating-point editor instead of unrestricted text."""

    def __init__(self, lower: float, upper: float, parent: QWidget | None = None):
        super().__init__(parent)
        self.lower = lower
        self.upper = upper

    def createEditor(self, parent, _option, _index):
        editor = QDoubleSpinBox(parent)
        editor.setDecimals(8)
        editor.setRange(self.lower, self.upper)
        editor.setKeyboardTracking(False)
        return editor

    def setEditorData(self, editor, index) -> None:
        try:
            editor.setValue(float(index.data(Qt.ItemDataRole.EditRole)))
        except (TypeError, ValueError):
            editor.setValue(0.0)

    def setModelData(self, editor, model, index) -> None:
        model.setData(index, f"{editor.value():.10g}", Qt.ItemDataRole.EditRole)


class FieldEditorDialog(QDialog):
    """Edit imported field names and assign each column a model role."""

    def __init__(self, headers: list[str], roles: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("定义训练字段")
        self.resize(620, 520)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "字段名称会原样传给后端，并作为后续推理和优化的变量名称。每列必须指定为输入参数或输出目标。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("Subtitle")
        layout.addWidget(hint)
        self.table = QTableWidget(len(headers), 3)
        self.table.setHorizontalHeaderLabels(["原始列", "字段名称", "字段角色"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.role_boxes: list[QComboBox] = []
        for index, name in enumerate(headers):
            position = QTableWidgetItem(f"第 {index + 1} 列")
            position.setFlags(position.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(index, 0, position)
            self.table.setItem(index, 1, QTableWidgetItem(name))
            role = QComboBox()
            role.addItem("输入参数", "input")
            role.addItem("输出目标", "output")
            role.setCurrentIndex(max(0, role.findData(roles[index])))
            self.table.setCellWidget(index, 2, role)
            self.role_boxes.append(role)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self) -> None:
        names, roles = self.values()
        if any(not name for name in names):
            QMessageBox.warning(self, "字段定义不完整", "字段名称不能为空。")
            return
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "字段名称重复", "字段名称必须唯一。")
            return
        if "input" not in roles or "output" not in roles:
            QMessageBox.warning(self, "字段角色不完整", "至少需要一个输入参数和一个输出目标。")
            return
        self.accept()

    def values(self) -> tuple[list[str], list[str]]:
        names = [
            self.table.item(row, 1).text().strip() if self.table.item(row, 1) else ""
            for row in range(self.table.rowCount())
        ]
        roles = [str(box.currentData()) for box in self.role_boxes]
        return names, roles


class DashboardPage(QWidget, AsyncMixin):
    navigate = Signal(int)

    def __init__(self, pool: QThreadPool):
        super().__init__()
        self.pool = pool
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)
        layout.addLayout(page_header("工程工作台", "从样本设计到代理模型与多目标优化，在一个工作区中完成。"))

        metrics = QHBoxLayout()
        for value, label in (
            ("2 种", "DEFORM 批处理方式"),
            ("5 个", "代理模型"),
            ("3 种", "优化模式"),
            ("3 类", "结果图表"),
        ):
            frame = QFrame()
            frame.setObjectName("StatCard")
            inner = QVBoxLayout(frame)
            metric = QLabel(value)
            metric.setObjectName("Metric")
            caption = QLabel(label)
            caption.setObjectName("Caption")
            inner.addWidget(metric)
            inner.addWidget(caption)
            metrics.addWidget(frame)
        layout.addLayout(metrics)

        flow, flow_layout = card("推荐工作流")
        row = QHBoxLayout()
        steps = [
            ("01", "设计样本", "设置参数边界并生成 LHS / 全因子样本", 1),
            ("02", "批量求解", "选择单工步或多工步，监控每个样本", 2),
            ("03", "训练模型", "对比 PRG、SVR、RF、KM 与 DNN 评分", 3),
            ("04", "优化分析", "运行 NSGA-II / PPO 并交互查看结果", 4),
        ]
        for index, name, detail, target in steps:
            step = QFrame()
            step.setObjectName("StatCard")
            box = QVBoxLayout(step)
            number = QLabel(index)
            number.setStyleSheet("color:#5ed1da;font-size:19px;font-weight:700")
            title = QLabel(name)
            title.setObjectName("Section")
            desc = QLabel(detail)
            desc.setObjectName("Caption")
            desc.setWordWrap(True)
            open_button = QPushButton("打开")
            open_button.clicked.connect(partial(self.navigate.emit, target))
            box.addWidget(number)
            box.addWidget(title)
            box.addWidget(desc)
            box.addStretch()
            box.addWidget(open_button)
            row.addWidget(step)
        flow_layout.addLayout(row)
        layout.addWidget(flow)

        note, note_layout = card("运行提示")
        note_label = QLabel(
            "真实 DEFORM 求解仅在已安装 DEFORM 的 Windows 计算机上可用。首次配置建议先启用“演练模式”，"
            "核对样本、KEY 文件和任务拆分，再启动实际批处理。"
        )
        note_label.setWordWrap(True)
        note_label.setObjectName("Subtitle")
        note_layout.addWidget(note_label)
        layout.addWidget(note)
        layout.addStretch()


class AutomationPage(QWidget, AsyncMixin):
    result_ready = Signal(str)

    def __init__(self, pool: QThreadPool, *, multi: bool):
        super().__init__()
        self.pool = pool
        self.multi = multi
        self.busy = False
        self.last_sample_file = ""
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.refresh_progress)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(16)
        title = "多工步批处理" if multi else "单工步计算"
        detail = "统一采样、换模、检查点恢复与分片批处理。" if multi else "生成参数化 KEY，批量求解并增量提取目标数据。"
        outer.addLayout(page_header(title, detail))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 8, 4)
        layout.setSpacing(16)

        setup, setup_layout = card("1 · 任务与路径")
        grid = QGridLayout()
        self.definition = QComboBox()
        self.definition.currentIndexChanged.connect(self.load_definition)
        grid.addWidget(QLabel("任务模板"), 0, 0)
        grid.addWidget(self.definition, 0, 1, 1, 3)
        self.template_path = PathPicker("DEFORM 模板 KEY 文件")
        grid.addWidget(QLabel("模板 KEY"), 1, 0)
        grid.addWidget(self.template_path, 1, 1, 1, 3)
        self.sample_path = PathPicker("样本 TSV 文件；也可在下方生成")
        grid.addWidget(QLabel("样本文件"), 2, 0)
        grid.addWidget(self.sample_path, 2, 1, 1, 3)
        self.workspace_path = PathPicker("任务输出工作区", directory=True)
        grid.addWidget(QLabel("工作区"), 3, 0)
        grid.addWidget(self.workspace_path, 3, 1, 1, 3)
        setup_layout.addLayout(grid)
        layout.addWidget(setup)

        sample, sample_layout = card("2 · 抽样设计")
        controls = QHBoxLayout()
        self.method = QComboBox()
        self.method.addItem("拉丁超立方（LHS）", "lhs")
        self.method.addItem("全因子（Full Factorial）", "full")
        self.method.currentIndexChanged.connect(self._sampling_method_changed)
        self.sample_count = QSpinBox()
        self.sample_count.setRange(1, 100000)
        self.sample_count.setValue(24)
        self.include_boundaries = QCheckBox("追加边界组合")
        self.include_boundaries.setChecked(False)
        self.include_boundaries.setToolTip(
            "未勾选时，LHS 严格生成指定数量的样本；勾选后追加所有参数上下界组合并去重。"
        )
        self.sample_button = QPushButton("生成样本")
        self.sample_button.setObjectName("Primary")
        self.sample_button.clicked.connect(self.generate_samples)
        controls.addWidget(QLabel("方法"))
        controls.addWidget(self.method, 1)
        controls.addWidget(QLabel("样本数 / 每列水平"))
        controls.addWidget(self.sample_count)
        controls.addWidget(self.include_boundaries)
        controls.addWidget(self.sample_button)
        sample_layout.addLayout(controls)
        self.parameter_table = DataTable()
        self.parameter_table.setMinimumHeight(150)
        sample_layout.addWidget(self.parameter_table)
        layout.addWidget(sample)

        execution, execution_layout = card("3 · 执行与进度")
        options = QHBoxLayout()
        self.parallel = QSpinBox()
        self.parallel.setRange(1, 64)
        self.parallel.setValue(4)
        self.dry_run = QCheckBox("演练模式（不调用 DEFORM）")
        self.incremental = QCheckBox("边求解边提取")
        self.incremental.setChecked(True)
        self.keep_checkpoints = QCheckBox("保留工步检查点")
        self.keep_checkpoints.setChecked(True)
        options.addWidget(QLabel("并行数"))
        options.addWidget(self.parallel)
        options.addWidget(self.dry_run)
        options.addWidget(self.incremental)
        if multi:
            options.addWidget(self.keep_checkpoints)
        options.addStretch()
        execution_layout.addLayout(options)
        status_row = QHBoxLayout()
        self.status_label = QLabel()
        set_status(self.status_label, "not_started")
        self.stage_label = QLabel("等待配置")
        self.stage_label.setObjectName("Subtitle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        status_row.addWidget(self.status_label)
        status_row.addWidget(self.stage_label)
        status_row.addWidget(self.progress, 1)
        execution_layout.addLayout(status_row)
        action_row = QHBoxLayout()
        self.prepare_button = QPushButton("仅生成 KEY")
        self.prepare_button.clicked.connect(self.prepare_keys)
        self.run_button = QPushButton("启动 / 继续运算")
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self.run_calculation)
        self.extract_button = QPushButton("提取结果")
        self.extract_button.clicked.connect(self.extract_results)
        action_row.addWidget(self.prepare_button)
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.extract_button)
        action_row.addStretch()
        execution_layout.addLayout(action_row)
        self.batch_table = DataTable()
        self.batch_table.setMinimumHeight(175)
        self.batch_table.set_data(["样本", "状态", "当前工步", "更新时间"], [])
        execution_layout.addWidget(self.batch_table)
        layout.addWidget(execution)
        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self.populate_definitions()

    def _sampling_method_changed(self) -> None:
        is_lhs = self.method.currentData() == "lhs"
        self.include_boundaries.setEnabled(is_lhs)
        self.include_boundaries.setVisible(is_lhs)

    def populate_definitions(self) -> None:
        from mobo.automation.task_collection import TASK_COLLECTION, MultiOperationTaskDefinition

        self.definition.blockSignals(True)
        for task_id, definition in TASK_COLLECTION.items():
            if isinstance(definition, MultiOperationTaskDefinition) == self.multi:
                self.definition.addItem(definition.name, task_id)
        self.definition.blockSignals(False)
        self.load_definition()

    def current_definition(self):
        from mobo.automation.task_collection import get_task_definition

        task_id = self.definition.currentData()
        if not task_id:
            raise ValueError("没有可用的任务模板")
        definition = get_task_definition(str(task_id))
        selected_template = self.template_path.text() if hasattr(self, "template_path") else ""
        if not selected_template:
            return definition
        if self.multi:
            operations = definition.operation_configs()
            operations[0]["template_key"] = selected_template
            return replace(definition, operations=tuple(operations))
        return replace(definition, template_key=selected_template)

    def load_definition(self) -> None:
        if self.definition.count() == 0:
            return
        from mobo.automation.task_collection import get_task_definition

        definition = get_task_definition(str(self.definition.currentData()))
        if self.multi:
            operations = definition.operations
            first_template = str(operations[0]["template_key"])
            rows = []
            for op_index, operation in enumerate(operations, 1):
                for parameter in operation.get("parameters", []):
                    rows.append([f"工步 {op_index}", parameter["name"], *parameter["range"]])
            headers = ["工步", "参数", "下限", "上限"]
            sample_dir = definition.sample_dir
            workspace = definition.run_dir
        else:
            first_template = definition.template_key
            rows = [[item["name"], item["object"], *item["range"]] for item in definition.parameters]
            headers = ["参数", "作用对象", "下限", "上限"]
            sample_dir = definition.workspace / "samples"
            workspace = definition.workspace
        self.template_path.setText(first_template)
        self.workspace_path.setText(str(workspace))
        self.parameter_table.set_data(headers, rows)
        existing = sorted(Path(sample_dir).glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True) if Path(sample_dir).exists() else []
        if existing:
            self.sample_path.setText(str(existing[0]))
        self.refresh_progress()

    def set_busy(self, busy: bool, stage: str = "") -> None:
        self.busy = busy
        for button in (self.sample_button, self.prepare_button, self.run_button, self.extract_button):
            button.setEnabled(not busy)
        if busy:
            set_status(self.status_label, "running")
            self.stage_label.setText(stage)
            self.progress.setRange(0, 0)
            self.timer.start()
        else:
            self.progress.setRange(0, 100)
            self.timer.stop()

    def generate_samples(self) -> None:
        definition = self.current_definition()
        method = self.method.currentData()
        count = self.sample_count.value()
        include_boundaries = self.include_boundaries.isChecked()
        destination = Path(self.workspace_path.text()) / "samples"
        self.set_busy(True, "正在生成样本")

        def action():
            variable_count = sum(len(op.get("parameters", [])) for op in definition.operations) if self.multi else len(definition.parameters)
            levels = [count] * variable_count if method == "full" else []
            return definition.generate_samples(
                method=method,
                n_samples=count,
                level_nums=levels,
                include_boundaries=include_boundaries,
                save_dir=destination,
            )

        self.run_async(action, self._samples_done, self._operation_failed)

    def _samples_done(self, path: str) -> None:
        self.set_busy(False)
        self.sample_path.setText(path)
        set_status(self.status_label, "finished")
        self.stage_label.setText("样本已生成")
        self.progress.setValue(100)
        actual_count = sum(
            1 for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        QMessageBox.information(
            self, "抽样完成", f"已生成 {actual_count} 个样本：\n{path}"
        )

    def prepare_keys(self) -> None:
        sample = self.require_sample()
        if not sample:
            return
        definition = self.current_definition()
        workspace = self.workspace_path.text()
        self.set_busy(True, "正在生成参数化 KEY")

        def action():
            if self.multi:
                return definition.prepare_keys(sample, work_dir=workspace)
            return definition.prepare_keys(sample, workspace=workspace)

        self.run_async(action, self._keys_done, self._operation_failed)

    def _keys_done(self, paths: list[str]) -> None:
        self.set_busy(False)
        set_status(self.status_label, "finished")
        self.stage_label.setText(f"已生成 {len(paths)} 个 KEY 文件")
        self.progress.setValue(100)

    def run_calculation(self) -> None:
        sample = self.require_sample()
        if not sample:
            return
        definition = self.current_definition()
        workspace = self.workspace_path.text()
        self.set_busy(True, "正在执行批处理")

        def action():
            if self.multi:
                task = definition.build(
                    sample,
                    work_dir=workspace,
                    max_parallel_samples=self.parallel.value(),
                    keep_checkpoints=self.keep_checkpoints.isChecked(),
                    dry_run=self.dry_run.isChecked(),
                )
                task.prepare_parameterized_keys()
                result = task.run()
                return {"result": result, "task": task, "multi": True}
            task = definition.build(
                sample,
                workspace=workspace,
                dry_run=self.dry_run.isChecked(),
                max_parallel=self.parallel.value(),
                incremental=self.incremental.isChecked(),
            )
            task.generate_keys()
            task.run_solver()
            if not self.dry_run.isChecked():
                task.extract()
            return {"result": {"status": "completed"}, "task": task, "multi": False}

        self.run_async(action, self._run_done, self._operation_failed)

    def _run_done(self, output: dict[str, Any]) -> None:
        self.set_busy(False)
        result = output["result"]
        completed = result.get("status") == "completed"
        set_status(self.status_label, "finished" if completed else "failed")
        self.stage_label.setText("批处理已完成" if completed else "批处理未完整完成")
        self.progress.setValue(100 if completed else 0)
        self.refresh_progress(output.get("task"))

    def extract_results(self) -> None:
        sample = self.require_sample()
        if not sample:
            return
        definition = self.current_definition()
        workspace = self.workspace_path.text()
        self.set_busy(True, "正在提取结果")

        def action():
            if self.multi:
                task = definition.build(sample, work_dir=workspace, dry_run=self.dry_run.isChecked())
                result = definition.extract_dataset(task, result_dir=Path(workspace) / "results")
            else:
                task = definition.build(sample, workspace=workspace, dry_run=self.dry_run.isChecked())
                task.load_samples_into_table()
                task.prepare_db_files()
                task.extract()
                candidates = sorted((Path(workspace) / "results").glob("*.txt"), key=lambda p: p.stat().st_mtime)
                result = str(candidates[-1]) if candidates else str(Path(workspace) / "results")
            return result

        self.run_async(action, self._extract_done, self._operation_failed)

    def _extract_done(self, path: str) -> None:
        self.set_busy(False)
        set_status(self.status_label, "finished")
        self.stage_label.setText("结果数据已生成")
        self.result_ready.emit(path)
        QMessageBox.information(self, "提取完成", f"结果位置：\n{path}")

    def _operation_failed(self, message: str) -> None:
        self.set_busy(False)
        set_status(self.status_label, "failed")
        self.stage_label.setText(message)
        self.show_error(message)

    def require_sample(self) -> str | None:
        path = self.sample_path.text()
        if not path or not Path(path).is_file():
            self.show_error("请先选择或生成有效的样本文件")
            return None
        if not self.workspace_path.text():
            self.show_error("请选择任务工作区")
            return None
        return path

    def refresh_progress(self, active_task: Any = None) -> None:
        try:
            definition = self.current_definition()
        except Exception:
            return
        rows: list[list[Any]] = []
        if self.multi:
            state_path = Path(os.environ.get("MOBO_DATA_DIR", "data")) / "tasks" / definition.task_id / "multi_operation_state.json"
            if active_task is not None:
                state = getattr(active_task, "state", {})
            elif state_path.is_file():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    state = {}
            else:
                state = {}
            samples = state.get("samples", {}) if isinstance(state, dict) else {}
            for sample_id, sample in sorted(samples.items(), key=lambda item: int(item[0])):
                operations = sample.get("operations", {})
                finished_ops = sum(1 for op in operations.values() if op.get("status") == "completed")
                rows.append([
                    int(sample_id) + 1,
                    sample.get("status", "pending"),
                    f"{finished_ops}/{len(operations)}",
                    format_timestamp(sample.get("updated_at")),
                ])
            if rows and self.busy:
                completed = sum(1 for row in rows if row[1] == "completed")
                self.progress.setRange(0, len(rows))
                self.progress.setValue(completed)
        else:
            path = Path(os.environ.get("MOBO_DATA_DIR", "data")) / "tasks" / definition.task_id / "process_info.json"
            if path.is_file():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raw = {}
                entries = raw.get("process_info", raw) if isinstance(raw, dict) else {}
                iterable = entries.items() if isinstance(entries, dict) else enumerate(entries)
                for sample_id, info in iterable:
                    value = info if isinstance(info, dict) else {"status": info}
                    rows.append([
                        sample_id,
                        value.get("status", "pending"),
                        "单工步",
                        format_timestamp(value.get("updated_at")),
                    ])
        self.batch_table.set_data(["样本", "状态", "当前工步", "更新时间"], rows)


class ModelPage(QWidget, AsyncMixin):
    schema_ready = Signal(object)

    def __init__(self, pool: QThreadPool, client_provider: Callable[[], ApiClient]):
        super().__init__()
        self.pool = pool
        self.client_provider = client_provider
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.poll)
        self.headers: list[str] = []
        self.rows: list[list[Any]] = []
        self.run_records: dict[str, dict[str, Any]] = {}
        self.field_roles: list[str] = []
        self.has_completed_training = False
        self.training_state = "not_started"
        self.selection_revision = 0
        self.configuration_dirty = False
        self.configuration_saving = False
        self.configuration_request_id = 0
        self.pending_configuration_save = False
        self.status_loading = False
        self.pending_doe_id = ""
        self.selected_doe_id = ""
        self.doe_refresh_in_flight = False
        self.doe_create_in_flight = False
        self.status_request_in_flight = False
        self.status_request_id = 0
        self.dataset_request_key = ""
        self.loaded_dataset_key = ""
        self.dataset_source_name = ""
        self.score_entries: list[tuple[str, float]] = []
        self.selected_score_index: int | None = None
        self.hovered_score_index: int | None = None
        self.score_series: QBarSeries | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setMinimumSize(850, 720)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        layout.addLayout(page_header("代理模型工作台", "选择训练数据与模型族，后台训练并对比交叉验证评分。"))

        split = QHBoxLayout()
        config, config_layout = card("训练配置")
        self.doe_id = QComboBox()
        self.doe_id.setObjectName("TaskSelector")
        self.doe_id.setEditable(False)
        self.doe_id.setPlaceholderText("请选择已有 DOE 任务")
        self.doe_id.activated.connect(lambda _index: self._doe_selected())
        self.refresh_doe_button = QPushButton("刷新 DOE")
        self.refresh_doe_button.clicked.connect(self.refresh_does)
        self.create_doe_button = QPushButton("新建 DOE")
        self.create_doe_button.clicked.connect(self.create_doe)
        doe_row = QHBoxLayout()
        doe_row.addWidget(QLabel("DOE"))
        doe_row.addWidget(self.doe_id, 1)
        doe_row.addWidget(self.create_doe_button)
        doe_row.addWidget(self.refresh_doe_button)
        config_layout.addLayout(doe_row)
        self.dataset = PathPicker("TXT / TSV / CSV 训练数据")
        self.dataset.changed.connect(self.load_dataset)
        config_layout.addWidget(QLabel("训练数据文件（支持 TXT / TSV / CSV）"))
        config_layout.addWidget(self.dataset)
        dataset_actions = QHBoxLayout()
        dataset_hint = QLabel(
            "TXT/TSV 可无表头；选择本地文件后会立即上传并绑定到当前 DOE。"
            "演示数据由当前 DOE 后端生成。"
        )
        dataset_hint.setObjectName("Caption")
        dataset_hint.setWordWrap(True)
        dataset_actions.addWidget(dataset_hint, 1)
        self.demo_button = QPushButton("生成并载入演示数据")
        self.demo_button.setMinimumWidth(176)
        self.demo_button.clicked.connect(self.load_demo_dataset)
        dataset_actions.addWidget(self.demo_button)
        config_layout.addLayout(dataset_actions)
        self.data_quality = QLabel("尚未加载训练数据")
        self.data_quality.setObjectName("Caption")
        self.data_quality.setWordWrap(True)
        config_layout.addWidget(self.data_quality)
        input_row = QHBoxLayout()
        self.input_count = QSpinBox()
        self.input_count.setRange(1, 999)
        self.input_count.setValue(2)
        self.input_count.setToolTip("修改后默认把前 N 列设为输入参数；可用“编辑字段”逐列指定。")
        self.input_count.valueChanged.connect(self._reset_default_roles)
        self.folds = QSpinBox()
        self.folds.setRange(2, 20)
        self.folds.setValue(5)
        input_row.addWidget(QLabel("默认输入列数"))
        input_row.addWidget(self.input_count)
        input_row.addSpacing(12)
        input_row.addWidget(QLabel("交叉验证折数"))
        input_row.addWidget(self.folds)
        self.edit_fields_button = QPushButton("编辑字段名称与角色")
        self.edit_fields_button.clicked.connect(self.edit_fields)
        input_row.addWidget(self.edit_fields_button)
        input_row.addStretch()
        config_layout.addLayout(input_row)
        self.schema_summary = QLabel("加载数据后可定义输入参数和输出目标")
        self.schema_summary.setObjectName("Caption")
        self.schema_summary.setWordWrap(True)
        config_layout.addWidget(self.schema_summary)
        self.save_hint = QLabel("字段名称、输入/输出角色或数据变化后，请保存到当前 DOE。")
        self.save_hint.setObjectName("Caption")
        self.save_hint.setWordWrap(True)
        config_layout.addWidget(self.save_hint)
        config_layout.addWidget(QLabel("待训练模型"))
        model_row = QHBoxLayout()
        self.models: dict[str, QCheckBox] = {}
        for name in ("PRG", "SVR", "RF", "KM", "DNN"):
            checkbox = QCheckBox(name)
            checkbox.setChecked(name in {"PRG", "SVR", "RF", "KM"})
            self.models[name] = checkbox
            model_row.addWidget(checkbox)
        config_layout.addLayout(model_row)
        button_row = QHBoxLayout()
        self.save_button = QPushButton("保存 DOE 配置")
        self.save_button.setToolTip("保存当前数据、字段名称以及输入/输出角色，不会启动训练。")
        self.save_button.clicked.connect(self.save_configuration)
        self.start_button = QPushButton("开始训练")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("Danger")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addStretch()
        config_layout.addLayout(button_row)
        split.addWidget(config, 5)

        status, status_layout = card("训练状态")
        status.setMinimumWidth(310)
        self.status_label = QLabel()
        set_status(self.status_label, "not_started")
        self.stage_label = QLabel("阶段：未开始")
        self.stage_label.setObjectName("Subtitle")
        self.updated_label = QLabel("最近更新：—")
        self.updated_label.setObjectName("Caption")
        self.training_record_hint = QLabel("")
        self.training_record_hint.setObjectName("Caption")
        self.training_record_hint.setWordWrap(True)
        self.progress = QProgressBar()
        status_layout.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addWidget(self.stage_label)
        status_layout.addWidget(self.updated_label)
        status_layout.addWidget(self.training_record_hint)
        status_layout.addWidget(self.progress)
        self.best_model = QLabel("—")
        self.best_model.setObjectName("Metric")
        status_layout.addWidget(QLabel("当前最佳模型"))
        status_layout.addWidget(self.best_model)
        status_layout.addStretch()
        split.addWidget(status, 3)
        layout.addLayout(split)

        tabs = QTabWidget()
        self.preview = DataTable()
        self.preview.enable_wide_columns()
        self.preview.setToolTip("字段较多时可横向滚动；拖动表头分隔线可调整列宽。")
        tabs.addTab(self.preview, "数据预览")
        score_widget = QWidget()
        score_layout = QVBoxLayout(score_widget)
        self.score_chart = QChartView()
        self.score_chart.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.score_chart.setMinimumHeight(280)
        self.score_chart.setMouseTracking(True)
        self.score_chart.viewport().setMouseTracking(True)
        self.score_chart.viewport().installEventFilter(self)
        score_layout.addWidget(self.score_chart)
        self.score_detail = QLabel("悬停或点击柱子可查看模型名称与完整分数")
        self.score_detail.setObjectName("Caption")
        self.score_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(self.score_detail)
        tabs.addTab(score_widget, "模型评分对比")
        layout.addWidget(tabs, 1)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self._refresh_training_actions()

    def refresh_does(self) -> None:
        if self.doe_refresh_in_flight:
            return
        self.doe_refresh_in_flight = True
        self.refresh_doe_button.setEnabled(False)
        self.refresh_doe_button.setText("正在刷新…")
        self.training_record_hint.setText("正在刷新 DOE 列表…")
        self.run_async(
            lambda: self.client_provider().list_doe(),
            self._does_loaded,
            self._doe_refresh_failed,
        )

    def _doe_refresh_failed(self, message: str) -> None:
        self.doe_refresh_in_flight = False
        self.refresh_doe_button.setEnabled(True)
        self.refresh_doe_button.setText("刷新 DOE")
        self.training_record_hint.setText(f"DOE 列表加载失败：{message}；请点击“刷新 DOE”重试。")

    def _doe_selected(self) -> None:
        doe_id = self.current_doe_id()
        if doe_id and doe_id == self.selected_doe_id and self.status_request_in_flight:
            return
        if doe_id != self.selected_doe_id:
            self.selected_doe_id = doe_id
            self._clear_dataset_context()
        self.selection_revision += 1
        self.status_request_id += 1
        self.status_request_in_flight = False
        self.configuration_request_id += 1
        self.configuration_saving = False
        self.pending_configuration_save = False
        self.status_loading = bool(doe_id)
        self.training_state = "not_started"
        if self.status_loading:
            self.training_record_hint.setText("正在读取当前 DOE 的保存配置与训练状态…")
        else:
            self.training_state = "not_started"
            self.training_record_hint.setText("")
        self._refresh_training_actions()
        self.poll()

    def create_doe(self) -> None:
        if self.doe_create_in_flight:
            return
        name, accepted = QInputDialog.getText(self, "新建 DOE", "任务名称")
        if accepted and name.strip():
            self.doe_create_in_flight = True
            self.create_doe_button.setEnabled(False)
            self.run_async(
                lambda: self.client_provider().add_doe({"name": name.strip()}),
                lambda data: self._created_doe(data),
                self._doe_create_failed,
            )

    def _created_doe(self, data: dict[str, Any]) -> None:
        self.doe_create_in_flight = False
        self.create_doe_button.setEnabled(True)
        self.pending_doe_id = str(data.get("id", "")).strip()
        self.refresh_does()

    def _doe_create_failed(self, message: str) -> None:
        self.doe_create_in_flight = False
        self.create_doe_button.setEnabled(True)
        self.show_error(message)

    def _does_loaded(self, data: dict[str, Any]) -> None:
        self.doe_refresh_in_flight = False
        self.refresh_doe_button.setEnabled(True)
        self.refresh_doe_button.setText("刷新 DOE")
        current = self.pending_doe_id or self.current_doe_id()
        self.pending_doe_id = ""
        self.doe_id.clear()
        for item in data.get("items", []):
            self.doe_id.addItem(item.get("name") or item["id"], item["id"])
        if current:
            index = self.doe_id.findData(current)
            if index >= 0:
                self.doe_id.setCurrentIndex(index)
            else:
                self.doe_id.setCurrentIndex(-1)
        else:
            self.doe_id.setCurrentIndex(-1)
        self.training_record_hint.setText("")
        self._doe_selected()

    def _clear_dataset_context(self) -> None:
        self.headers = []
        self.rows = []
        self.field_roles = []
        self.configuration_dirty = False
        self.pending_configuration_save = False
        self.dataset_source_name = ""
        self.dataset_request_key = ""
        self.loaded_dataset_key = ""
        self.preview.set_data([], [])
        self.dataset.edit.blockSignals(True)
        self.dataset.edit.clear()
        self.dataset.edit.blockSignals(False)
        self.data_quality.setText("尚未加载训练数据")
        self.schema_summary.setText("加载数据后可定义输入参数和输出目标")
        self.save_hint.setText("字段名称、输入/输出角色或数据变化后，请保存到当前 DOE。")
        self.save_button.setText("保存 DOE 配置")

    def load_dataset(self, path: str) -> None:
        if not Path(path).is_file():
            return
        try:
            raw_headers, raw_rows = read_tabular_file(path)
            cleaned = prepare_numeric_table(raw_headers, raw_rows)
            # Invalidate any restore request for the previously saved dataset so
            # it cannot overwrite the file the user has just selected.
            self.selection_revision += 1
            self.dataset_request_key = ""
            self.loaded_dataset_key = ""
            self.headers, self.rows = cleaned.headers, cleaned.rows
            self.dataset_source_name = Path(path).name
            self.preview.set_data(self.headers, self.rows[:200])
            self.input_count.setMaximum(max(1, len(self.headers) - 1))
            self._reset_default_roles()
            messages = [f"可用数据：{len(self.rows)} 行 × {len(self.headers)} 列"]
            if cleaned.dropped_columns:
                messages.append("已移除无有效数值列：" + "、".join(cleaned.dropped_columns))
            if cleaned.dropped_row_count:
                messages.append(f"已排除含缺失或非有限值的 {cleaned.dropped_row_count} 行")
            self.data_quality.setText("；".join(messages))
            self._mark_configuration_dirty()
            self._refresh_training_actions()
            if self.current_doe_id():
                self._submit_configuration(automatic=True)
            else:
                self.save_hint.setText("文件已载入本地；请选择或新建 DOE 后保存配置。")
        except Exception as exc:
            self.show_error(str(exc))

    def load_demo_dataset(self) -> None:
        doe_id = self.current_doe_id()
        if not doe_id:
            self.show_error("请先选择或新建 DOE")
            return
        self.demo_button.setEnabled(False)
        self.demo_button.setText("正在生成…")

        def action():
            headers = ["temperature", "speed", "grain", "load"]
            generated = self.client_provider().generate_training_dataset({
                "id": doe_id,
                "input_names": headers[:2],
                "target_names": headers[2:],
                "param_ranges": {
                    "temperature": [900, 1100],
                    "speed": [10, 50],
                },
                "n_samples": 80,
                "seed": 42,
            })
            loaded = self.client_provider().get_data(
                doe_id, str(generated["resource_id"]), headers
            )
            return headers, rows_by_fields(loaded.get("values", {}), headers)

        self.run_async(action, self._demo_loaded, self._demo_failed)

    def _demo_loaded(self, result: tuple[list[str], list[list[Any]]]) -> None:
        self.headers, self.rows = result
        self.dataset_source_name = "演示训练数据.tsv"
        self.preview.set_data(self.headers, self.rows[:200])
        self.input_count.setMaximum(max(1, len(self.headers) - 1))
        self.input_count.setValue(2)
        self._reset_default_roles()
        self.dataset.setText(f"DOE 演示数据 · {len(self.rows)} 行")
        self.data_quality.setText(f"数据检查通过：{len(self.rows)} 行 × {len(self.headers)} 列")
        self._mark_configuration_dirty()
        self.demo_button.setEnabled(True)
        self.demo_button.setText("重新生成演示数据")
        self._refresh_training_actions()

    def _demo_failed(self, message: str) -> None:
        self.demo_button.setEnabled(True)
        self.demo_button.setText("生成并载入演示数据")
        self.show_error(message)

    def _reset_default_roles(self) -> None:
        if not self.headers:
            return
        count = min(self.input_count.value(), max(1, len(self.headers) - 1))
        self.field_roles = ["input" if index < count else "output" for index in range(len(self.headers))]
        self._update_schema_summary()
        self._mark_configuration_dirty()

    def edit_fields(self) -> None:
        if not self.headers:
            self.show_error("请先选择训练数据文件或载入演示数据")
            return
        if len(self.field_roles) != len(self.headers):
            self._reset_default_roles()
        dialog = FieldEditorDialog(self.headers, self.field_roles, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.headers, self.field_roles = dialog.values()
        input_count = self.field_roles.count("input")
        self.input_count.blockSignals(True)
        self.input_count.setValue(input_count)
        self.input_count.blockSignals(False)
        self.preview.set_data(self.headers, self.rows[:200])
        self._update_schema_summary()
        self._mark_configuration_dirty()

    def _update_schema_summary(self) -> None:
        inputs = [name for name, role in zip(self.headers, self.field_roles, strict=True) if role == "input"]
        outputs = [name for name, role in zip(self.headers, self.field_roles, strict=True) if role == "output"]
        self.schema_summary.setText(
            f"输入参数（{len(inputs)}）：{', '.join(inputs) or '未定义'}  ·  "
            f"输出目标（{len(outputs)}）：{', '.join(outputs) or '未定义'}"
        )
        self._refresh_training_actions()

    def _mark_configuration_dirty(self) -> None:
        if not self.headers:
            return
        self.configuration_dirty = True
        if self.configuration_saving:
            self.pending_configuration_save = True
        self.save_button.setText("保存 DOE 配置")
        self.save_hint.setText("当前数据或字段定义尚未保存到 DOE。")
        self._refresh_training_actions()

    def _refresh_training_actions(self) -> None:
        active = self.training_state in {"queued", "running", "stopping"}
        ready = bool(self.current_doe_id() and self.rows and self.headers and self.field_roles)
        self.save_button.setEnabled(not self.configuration_saving)
        self.start_button.setEnabled(ready and not active and not self.status_loading)
        self.stop_button.setEnabled(self.training_state in {"queued", "running"})

    def _current_schema(self) -> dict[str, Any]:
        input_indices = [index for index, role in enumerate(self.field_roles) if role == "input"]
        output_indices = [index for index, role in enumerate(self.field_roles) if role == "output"]
        inputs = []
        for index in input_indices:
            values = [float(row[index]) for row in self.rows]
            inputs.append({"name": self.headers[index], "lower": min(values), "upper": max(values)})
        return {
            "input_indices": input_indices,
            "output_indices": output_indices,
            "inputs": inputs,
            "outputs": [self.headers[index] for index in output_indices],
        }

    def current_doe_id(self) -> str:
        return str(self.doe_id.currentData() or self.doe_id.currentText()).strip()

    def _training_data_source(self, schema: dict[str, Any]) -> dict[str, Any]:
        input_indices = schema["input_indices"]
        output_indices = schema["output_indices"]
        source = {
            "input_data": {
                "labels": [self.headers[index] for index in input_indices],
                "samples": [[row[index] for index in input_indices] for row in self.rows],
            },
            "output_data": {
                "labels": [self.headers[index] for index in output_indices],
                "samples": [[row[index] for index in output_indices] for row in self.rows],
            },
        }
        if self.dataset_source_name:
            source["source_name"] = self.dataset_source_name
        return source

    def save_configuration(self) -> None:
        self._submit_configuration(automatic=False)

    def _submit_configuration(self, *, automatic: bool) -> None:
        if self.configuration_saving:
            return
        if self.training_state in {"queued", "running", "stopping"}:
            message = "当前任务正在训练中，请停止或等待训练结束后再保存 DOE 配置"
            if automatic:
                self.save_hint.setText(f"文件已载入本地，但{message}。")
            else:
                self.show_error(message)
            return
        doe_id = self.current_doe_id()
        if not doe_id or not self.rows or not self.headers:
            message = "请先选择 DOE 并载入有效训练数据"
            if automatic:
                self.save_hint.setText(message)
            else:
                self.show_error(message)
            return
        schema = self._current_schema()
        if not schema["input_indices"] or not schema["output_indices"]:
            message = "至少需要定义一个输入参数和一个输出目标"
            if automatic:
                self.save_hint.setText(message)
            else:
                self.show_error(message)
            return
        payload = {"id": doe_id, "data_source": self._training_data_source(schema)}
        revision = self.selection_revision
        self.configuration_request_id += 1
        request_id = self.configuration_request_id
        self.configuration_saving = True
        self.pending_configuration_save = False
        self.save_button.setText("正在上传…" if automatic else "正在保存…")
        self.save_hint.setText(
            "正在上传训练数据并绑定到当前 DOE…"
            if automatic
            else "正在把训练数据和字段定义保存到当前 DOE…"
        )
        self._refresh_training_actions()
        self.run_async(
            lambda: self.client_provider().save_training_dataset(payload),
            lambda data: self._configuration_saved(
                doe_id, revision, request_id, data, automatic
            ),
            lambda message: self._configuration_save_failed(
                doe_id, revision, request_id, message
            ),
        )

    def _configuration_saved(
        self,
        doe_id: str,
        revision: int,
        request_id: int,
        data: dict[str, Any],
        automatic: bool = False,
    ) -> None:
        if request_id != self.configuration_request_id:
            return
        self.configuration_saving = False
        if doe_id != self.current_doe_id():
            self._refresh_training_actions()
            return
        if revision != self.selection_revision or self.pending_configuration_save:
            self.pending_configuration_save = False
            self.configuration_dirty = True
            self.save_button.setText("正在更新…")
            self.save_hint.setText("数据或字段在上传期间发生变化，正在保存最新配置…")
            self._refresh_training_actions()
            self._submit_configuration(automatic=True)
            return
        self.configuration_dirty = False
        resource_id = str(data.get("resource_id", "")).strip()
        if resource_id:
            self.loaded_dataset_key = f"{doe_id}:{resource_id}"
        self.save_button.setText("已上传" if automatic else "已保存")
        self.save_hint.setText(
            f"{'已上传并绑定' if automatic else '已保存'}到当前 DOE："
            f"{data.get('sample_count', len(self.rows))} 行数据，"
            f"{len(data.get('input_names', []))} 个输入，"
            f"{len(data.get('target_names', []))} 个输出。"
        )
        self._refresh_training_actions()

    def _configuration_save_failed(
        self, doe_id: str, revision: int, request_id: int, message: str
    ) -> None:
        if request_id != self.configuration_request_id:
            return
        self.configuration_saving = False
        self.pending_configuration_save = False
        if doe_id != self.current_doe_id():
            self._refresh_training_actions()
            return
        self.save_button.setText("保存 DOE 配置")
        self.save_hint.setText(f"DOE 配置保存失败：{message}")
        self._refresh_training_actions()

    def start_training(self) -> None:
        if not self.rows or not self.headers:
            self.show_error("请先选择有效的训练数据文件")
            return
        if len(self.field_roles) != len(self.headers):
            self._reset_default_roles()
        schema = self._current_schema()
        input_indices = schema["input_indices"]
        output_indices = schema["output_indices"]
        selected = [name for name, checkbox in self.models.items() if checkbox.isChecked()]
        if not selected:
            self.show_error("至少选择一种代理模型")
            return
        doe_id = self.current_doe_id()
        if not doe_id:
            self.show_error("请选择 DOE")
            return
        input_names = [self.headers[index] for index in input_indices]
        target_names = [self.headers[index] for index in output_indices]
        if not input_names or not target_names:
            self.show_error("至少需要定义一个输入参数和一个输出目标")
            return
        if self.has_completed_training:
            answer = QMessageBox.question(
                self,
                "确认重新训练",
                "当前 DOE 已有完成的训练记录和可用模型，通常无需重复训练。\n\n"
                "只有训练数据、字段定义或模型参数变化时才建议重新训练。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        payload = {
            "id": doe_id,
            "data_source": self._training_data_source(schema),
            "models": [{"name": name, "params": {}} for name in selected],
            "evaluation": {"enabled": True, "method": "k_fold", "n_splits": self.folds.value(), "random_state": 42},
        }
        self.start_button.setEnabled(False)
        self.start_button.setText("训练中…")
        self.has_completed_training = False
        self.training_state = "queued"
        set_status(self.status_label, "queued")
        self._refresh_training_actions()
        self.schema_ready.emit(schema)
        revision = self.selection_revision
        self.run_async(
            lambda: self.client_provider().start_training(payload),
            lambda data: self._training_started(doe_id, revision, data),
            lambda message: self._training_failed(doe_id, revision, message),
        )

    def _training_started(
        self, doe_id: str, revision: int, _data: dict[str, Any]
    ) -> None:
        if doe_id != self.current_doe_id() or revision != self.selection_revision:
            return
        self.timer.start()
        self.configuration_dirty = False
        self.save_button.setText("已保存")
        self.poll()

    def _training_failed(self, doe_id: str, revision: int, message: str) -> None:
        if doe_id != self.current_doe_id() or revision != self.selection_revision:
            return
        self.training_state = "failed"
        self.start_button.setText("开始训练")
        set_status(self.status_label, "failed")
        self._refresh_training_actions()
        self.show_error(message)

    def stop_training(self) -> None:
        doe_id = self.current_doe_id()
        if doe_id:
            self.training_state = "stopping"
            self._refresh_training_actions()
            self.run_async(lambda: self.client_provider().stop_training(doe_id), lambda _data: self.poll())

    def poll(self) -> None:
        doe_id = self.current_doe_id()
        if not doe_id:
            self.status_loading = False
            self._refresh_training_actions()
            return
        if self.status_request_in_flight:
            return
        self.status_request_in_flight = True
        self.status_request_id += 1
        request_id = self.status_request_id
        revision = self.selection_revision
        self.run_async(
            lambda: self.client_provider().training_progress(doe_id),
            lambda data: self._apply_progress(doe_id, revision, data, request_id),
            lambda message: self._progress_failed(doe_id, revision, message, request_id),
        )
        QTimer.singleShot(
            20000,
            lambda: self._progress_timed_out(doe_id, revision, request_id),
        )

    def _apply_progress(
        self,
        doe_id: str,
        revision: int,
        data: dict[str, Any],
        request_id: int | None = None,
    ) -> None:
        if request_id is not None and request_id != self.status_request_id:
            return
        self.status_request_in_flight = False
        if doe_id == self.current_doe_id() and revision == self.selection_revision:
            self.status_loading = False
            self.update_progress(data)
        elif self.current_doe_id():
            self.poll()

    def _progress_failed(
        self,
        doe_id: str,
        revision: int,
        message: str,
        request_id: int | None = None,
    ) -> None:
        if request_id is not None and request_id != self.status_request_id:
            return
        self.status_request_in_flight = False
        if doe_id != self.current_doe_id() or revision != self.selection_revision:
            if self.current_doe_id():
                self.poll()
            return
        self.status_loading = False
        self.training_state = "failed"
        self.training_record_hint.setText(f"训练状态读取失败：{message}")
        self._refresh_training_actions()

    def _progress_timed_out(
        self, doe_id: str, revision: int, request_id: int
    ) -> None:
        if (
            request_id != self.status_request_id
            or not self.status_request_in_flight
            or doe_id != self.current_doe_id()
            or revision != self.selection_revision
        ):
            return
        self.status_request_in_flight = False
        self.status_loading = False
        self.training_record_hint.setText(
            "读取 DOE 配置超时，请重新选择任务或点击“刷新 DOE”重试。"
        )
        self._refresh_training_actions()

    def update_progress(self, data: dict[str, Any]) -> None:
        self.status_loading = False
        state = str(data.get("status", "not_started"))
        self.training_state = state
        set_status(self.status_label, state)
        raw_updated_at = data.get("updated_at")
        self.stage_label.setText(f"阶段：{stage_text(data.get('stage'))}")
        self.updated_label.setText(f"最近更新：{format_timestamp(raw_updated_at)}")
        self.updated_label.setToolTip(str(raw_updated_at or ""))
        self.progress.setValue(int(data.get("progress", 0)))
        models = data.get("models", [])
        self.has_completed_training = state == "finished" and bool(models)
        self.start_button.setText("重新训练" if self.has_completed_training else "开始训练")
        if self.has_completed_training:
            self.training_record_hint.setText(
                "已检测到当前 DOE 的训练记录和可用模型，无需重复训练；"
                "可直接进入优化，数据或配置变化后再重新训练。"
            )
        elif data.get("dataset"):
            self.training_record_hint.setText("当前 DOE 已保存训练数据与字段定义，尚未开始训练。")
        else:
            self.training_record_hint.setText("")
        self._restore_saved_dataset(data)
        if not self.headers and data.get("input_names") and data.get("target_names"):
            self.schema_ready.emit({
                "inputs": list(data.get("input_bounds") or [
                    {"name": name, "lower": 0.0, "upper": 1.0}
                    for name in data["input_names"]
                ]),
                "outputs": list(data["target_names"]),
            })
        scored = [item for item in models if item.get("score") is not None]
        if scored:
            best = max(scored, key=lambda item: item["score"])
            self.best_model.setText(f"{best.get('model_family')}  {best['score']:.3f}")
        self.update_score_chart(models)
        if state in {"finished", "failed", "stopped", "not_started"}:
            self.timer.stop()
        self._refresh_training_actions()

    def _restore_saved_dataset(self, data: dict[str, Any]) -> None:
        dataset = dict(data.get("dataset") or {})
        doe_id = self.current_doe_id()
        resource_id = str(dataset.get("resource_id", "")).strip()
        columns = list(dataset.get("columns") or [])
        if not doe_id or not resource_id or not columns or self.configuration_dirty:
            return
        key = f"{doe_id}:{resource_id}"
        if key in {self.loaded_dataset_key, self.dataset_request_key}:
            return
        self.dataset_request_key = key
        revision = self.selection_revision
        self.data_quality.setText("正在恢复 DOE 中已保存的训练数据…")
        self.run_async(
            lambda: self.client_provider().get_data(doe_id, resource_id, columns),
            lambda loaded: self._saved_dataset_loaded(
                doe_id, revision, key, dataset, data, loaded
            ),
            lambda message: self._saved_dataset_failed(doe_id, revision, key, message),
        )
        QTimer.singleShot(
            20000,
            lambda: self._saved_dataset_timed_out(doe_id, revision, key),
        )

    def _saved_dataset_loaded(
        self,
        doe_id: str,
        revision: int,
        key: str,
        dataset: dict[str, Any],
        progress: dict[str, Any],
        loaded: dict[str, Any],
    ) -> None:
        if self.dataset_request_key == key:
            self.dataset_request_key = ""
        if doe_id != self.current_doe_id() or revision != self.selection_revision:
            return
        columns = list(dataset.get("columns") or [])
        rows = rows_by_fields(loaded.get("values", {}), columns)
        input_names = set(progress.get("input_names") or [])
        target_names = set(progress.get("target_names") or [])
        self.headers = columns
        self.rows = rows
        self.field_roles = [
            "input" if name in input_names else "output"
            for name in columns
        ]
        self.input_count.setMaximum(max(1, len(columns) - 1))
        self.input_count.blockSignals(True)
        self.input_count.setValue(max(1, len(input_names)))
        self.input_count.blockSignals(False)
        self.preview.set_data(columns, rows[:200])
        self.dataset_source_name = str(dataset.get("source_name") or "DOE 已保存训练数据")
        label = f"{self.dataset_source_name}（DOE 副本，{len(rows)} 行）"
        self.dataset.edit.blockSignals(True)
        self.dataset.edit.setText(label)
        self.dataset.edit.blockSignals(False)
        self.data_quality.setText(f"已从 DOE 恢复：{len(rows)} 行 × {len(columns)} 列")
        self.configuration_dirty = False
        self.loaded_dataset_key = key
        self.save_button.setText("已保存")
        self.save_hint.setText(
            f"已恢复当前 DOE 的保存配置：{len(input_names)} 个输入，"
            f"{len(target_names)} 个输出。"
        )
        self._update_schema_summary()
        self.schema_ready.emit(self._current_schema())

    def _saved_dataset_failed(
        self, doe_id: str, revision: int, key: str, message: str
    ) -> None:
        if self.dataset_request_key == key:
            self.dataset_request_key = ""
        if doe_id != self.current_doe_id() or revision != self.selection_revision:
            return
        self.data_quality.setText(f"DOE 已保存数据恢复失败：{message}")
        self._refresh_training_actions()

    def _saved_dataset_timed_out(
        self, doe_id: str, revision: int, key: str
    ) -> None:
        if (
            self.dataset_request_key != key
            or doe_id != self.current_doe_id()
            or revision != self.selection_revision
        ):
            return
        self.dataset_request_key = ""
        self.data_quality.setText("DOE 数据表恢复超时，请重新选择任务或刷新 DOE 后重试。")
        self._refresh_training_actions()

    def update_score_chart(self, models: list[dict[str, Any]]) -> None:
        chart = QChart()
        chart.setBackgroundBrush(QColor("#111d31"))
        chart.setPlotAreaBackgroundBrush(QColor("#0d1829"))
        chart.setPlotAreaBackgroundVisible(True)
        chart.setTitle("交叉验证平均评分（越高越好）")
        chart.setTitleBrush(QColor("#dce7f5"))
        series = QBarSeries()
        values = QBarSet("综合评分")
        values.setColor(QColor("#4dc9d3"))
        values.setLabelColor(QColor("#f4f8ff"))
        categories = []
        self.score_entries = []
        self.selected_score_index = None
        self.hovered_score_index = None
        self.score_detail.setText("悬停或点击柱子可查看模型名称与完整分数")
        for model in models:
            if model.get("score") is not None:
                name = str(model.get("model_family", model.get("model_id", "model")))
                score = float(model["score"])
                categories.append(name)
                self.score_entries.append((name, score))
                values.append(score)
        series.append(values)
        self.score_series = series
        series.setLabelsVisible(True)
        series.setLabelsFormat("@value")
        series.setLabelsPrecision(3)
        series.setLabelsPosition(QAbstractBarSeries.LabelsPosition.LabelsInsideEnd)
        values.clicked.connect(self._score_bar_clicked)
        values.hovered.connect(self._score_bar_hovered)
        chart.addSeries(series)
        axis_x = QBarCategoryAxis()
        axis_x.append(categories or ["暂无评分"])
        axis_x.setLabelsColor(QColor("#a9bad0"))
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        axis_y = QValueAxis()
        axis_y.setRange(min(0.0, min((float(model["score"]) for model in models if model.get("score") is not None), default=0.0)), 1.0)
        axis_y.setLabelsColor(QColor("#a9bad0"))
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        chart.legend().setLabelColor(QColor("#a9bad0"))
        self.score_chart.setChart(chart)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.score_chart.viewport():
            if event.type() == QEvent.Type.MouseMove:
                self._score_chart_mouse_moved(event.position())
            elif event.type() == QEvent.Type.Leave:
                self._score_bar_hovered(False, -1)
        return super().eventFilter(watched, event)

    def _score_chart_mouse_moved(self, position: Any) -> None:
        chart = self.score_chart.chart()
        series = self.score_series
        if not self.score_entries or series is None or not chart.plotArea().contains(position):
            if self.hovered_score_index is not None:
                self._score_bar_hovered(False, -1)
            return
        value = chart.mapToValue(position, series)
        index = int(value.x())
        score = self.score_entries[index][1] if 0 <= index < len(self.score_entries) else 0.0
        half_width = max(0.05, float(series.barWidth()) / 2.0)
        inside_x = (
            0 <= index < len(self.score_entries)
            and abs(value.x() - (index + 0.5)) <= half_width
        )
        inside_y = min(0.0, score) <= value.y() <= max(0.0, score)
        hovered_index = index if inside_x and inside_y else None
        if hovered_index == self.hovered_score_index:
            return
        if hovered_index is None:
            self._score_bar_hovered(False, -1)
        else:
            self._score_bar_hovered(True, hovered_index)

    def _score_bar_hovered(self, hovered: bool, index: int) -> None:
        if hovered and 0 <= index < len(self.score_entries):
            self.hovered_score_index = index
            name, score = self.score_entries[index]
            message = f"{name}：{score:.6f}"
            self.score_detail.setText(f"当前模型  {message}")
            QToolTip.showText(QCursor.pos(), message, self.score_chart)
        else:
            self.hovered_score_index = None
            QToolTip.hideText()
            if self.selected_score_index is None:
                self.score_detail.setText("悬停或点击柱子可查看模型名称与完整分数")
            else:
                name, score = self.score_entries[self.selected_score_index]
                self.score_detail.setText(f"已选择  {name}：{score:.6f}")

    def _score_bar_clicked(self, index: int) -> None:
        if 0 <= index < len(self.score_entries):
            name, score = self.score_entries[index]
            self.selected_score_index = index
            message = f"{name}：{score:.6f}"
            self.score_detail.setText(f"已选择  {message}")
            QToolTip.showText(QCursor.pos(), message, self.score_chart)


class OptimizationPage(QWidget, AsyncMixin):
    result_loaded = Signal(list, list)

    def __init__(self, pool: QThreadPool, client_provider: Callable[[], ApiClient]):
        super().__init__()
        self.pool = pool
        self.client_provider = client_provider
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.poll)
        self.has_completed_optimization = False
        self.training_bounds: dict[str, tuple[float, float]] = {}
        self.configuration_valid = False
        self.schema_ready_for_optimization = False
        self.optimization_state = "not_started"
        self.selection_revision = 0
        self.doe_refresh_in_flight = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        layout.addLayout(page_header("优化中心", "配置单目标、多目标 NSGA-II 或强化学习 PPO，并跟踪执行状态。"))

        top = QHBoxLayout()
        config, config_layout = card("优化配置")
        form = QGridLayout()
        self.doe_id = QComboBox()
        self.doe_id.setObjectName("TaskSelector")
        self.doe_id.setEditable(False)
        self.doe_id.setPlaceholderText("请选择已有 DOE 任务")
        self.doe_id.activated.connect(lambda _index: self.load_selected_doe())
        self.mode = QComboBox()
        self.mode.addItem("多目标优化", "multi")
        self.mode.addItem("单目标优化", "single")
        self.mode.addItem("强化学习", "reinforcement_learning")
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.algorithm = QComboBox()
        self.algorithm.addItem("NSGA-II 遗传算法", "nsga2")
        self.refresh_doe_button = QPushButton("刷新 DOE")
        self.refresh_doe_button.clicked.connect(self.refresh_does)
        form.addWidget(QLabel("优化任务 / DOE"), 0, 0)
        form.addWidget(self.doe_id, 0, 1)
        form.addWidget(self.refresh_doe_button, 0, 2)
        form.addWidget(QLabel("优化模式"), 1, 0)
        form.addWidget(self.mode, 1, 1)
        form.addWidget(QLabel("优化算法"), 2, 0)
        form.addWidget(self.algorithm, 2, 1)
        config_layout.addLayout(form)
        self.parameter_stack = QStackedWidget()
        ga_parameters = QWidget()
        ga_form = QGridLayout(ga_parameters)
        ga_form.setContentsMargins(0, 0, 0, 0)
        self.ga_population = QSpinBox()
        self.ga_population.setRange(4, 10000)
        self.ga_population.setValue(100)
        self.ga_offspring = QSpinBox()
        self.ga_offspring.setRange(2, 10000)
        self.ga_offspring.setValue(50)
        self.ga_generations = QSpinBox()
        self.ga_generations.setRange(1, 1000000)
        self.ga_generations.setValue(200)
        ga_form.addWidget(QLabel("种群规模"), 0, 0)
        ga_form.addWidget(self.ga_population, 0, 1)
        ga_form.addWidget(QLabel("子代数量"), 1, 0)
        ga_form.addWidget(self.ga_offspring, 1, 1)
        ga_form.addWidget(QLabel("迭代代数"), 2, 0)
        ga_form.addWidget(self.ga_generations, 2, 1)
        self.parameter_stack.addWidget(ga_parameters)
        ppo_parameters = QWidget()
        ppo_form = QGridLayout(ppo_parameters)
        ppo_form.setContentsMargins(0, 0, 0, 0)
        self.ppo_timesteps = QSpinBox()
        self.ppo_timesteps.setRange(1, 1_000_000_000)
        self.ppo_timesteps.setValue(20000)
        self.ppo_episode_steps = QSpinBox()
        self.ppo_episode_steps.setRange(1, 100000)
        self.ppo_episode_steps.setValue(100)
        self.ppo_evaluation_episodes = QSpinBox()
        self.ppo_evaluation_episodes.setRange(1, 10000)
        self.ppo_evaluation_episodes.setValue(10)
        self.ppo_learning_rate = QDoubleSpinBox()
        self.ppo_learning_rate.setDecimals(6)
        self.ppo_learning_rate.setRange(0.000001, 1.0)
        self.ppo_learning_rate.setValue(0.001)
        self.ppo_constraint_penalty = QDoubleSpinBox()
        self.ppo_constraint_penalty.setDecimals(4)
        self.ppo_constraint_penalty.setRange(0.001, 1_000_000.0)
        self.ppo_constraint_penalty.setValue(5.0)
        for row, (label, widget) in enumerate((
            ("训练时间步", self.ppo_timesteps),
            ("每回合步数", self.ppo_episode_steps),
            ("评估回合数", self.ppo_evaluation_episodes),
            ("学习率", self.ppo_learning_rate),
            ("约束惩罚系数", self.ppo_constraint_penalty),
        )):
            ppo_form.addWidget(QLabel(label), row, 0)
            ppo_form.addWidget(widget, row, 1)
        self.parameter_stack.addWidget(ppo_parameters)
        config_layout.addWidget(self.parameter_stack)
        # Compatibility aliases for integrations that used the former GA widgets.
        self.population = self.ga_population
        self.generations = self.ga_generations
        self.schema_hint = QLabel("请选择已有 DOE；目标函数和设计变量将从该任务的训练记录加载。")
        self.schema_hint.setObjectName("Caption")
        self.schema_hint.setWordWrap(True)
        config_layout.addWidget(self.schema_hint)
        top.addWidget(config, 1)

        status, status_layout = card("任务状态")
        self.status_label = QLabel()
        set_status(self.status_label, "not_started")
        self.stage_label = QLabel("阶段：未开始")
        self.stage_label.setObjectName("Subtitle")
        self.updated_label = QLabel("最近更新：—")
        self.updated_label.setObjectName("Caption")
        self.progress = QProgressBar()
        status_layout.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addWidget(self.stage_label)
        status_layout.addWidget(self.updated_label)
        status_layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("启动优化")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.start_optimization)
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("Danger")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch()
        status_layout.addLayout(buttons)
        status_layout.addStretch()
        top.addWidget(status, 1)
        layout.addLayout(top)

        tables = QHBoxLayout()
        objective_card, objective_layout = card("目标函数")
        self.objectives = DataTable()
        self.objectives.enable_wide_columns()
        self.objectives.setItemDelegateForColumn(
            1, ChoiceDelegate([("最小化", "min"), ("最大化", "max")], self.objectives)
        )
        self.objectives.setItemDelegateForColumn(
            2, NumericDelegate(0.0, 1.0, self.objectives)
        )
        self.objectives.set_data(["名称", "方向", "权重"], [])
        self.objectives.itemChanged.connect(self.validate_configuration)
        objective_layout.addWidget(self.objectives)
        objective_buttons = QHBoxLayout()
        add_objective = QPushButton("＋ 添加目标")
        add_objective.clicked.connect(lambda: self._add_empty_row(self.objectives))
        delete_objective = QPushButton("删除选中行")
        delete_objective.setObjectName("Danger")
        delete_objective.clicked.connect(lambda: self._delete_selected_rows(self.objectives))
        objective_buttons.addWidget(add_objective)
        objective_buttons.addWidget(delete_objective)
        objective_buttons.addStretch()
        objective_layout.addLayout(objective_buttons)
        tables.addWidget(objective_card, 1)
        variable_card, variable_layout = card("设计变量与边界")
        self.variables = DataTable()
        self.variables.enable_wide_columns()
        self.variables.setItemDelegateForColumn(
            1, NumericDelegate(-1e100, 1e100, self.variables)
        )
        self.variables.setItemDelegateForColumn(
            2, NumericDelegate(-1e100, 1e100, self.variables)
        )
        self.variables.set_data(["名称", "下限", "上限"], [])
        self.variables.itemChanged.connect(self.validate_configuration)
        variable_layout.addWidget(self.variables)
        variable_buttons = QHBoxLayout()
        add_variable = QPushButton("＋ 添加变量")
        add_variable.clicked.connect(lambda: self._add_empty_row(self.variables))
        delete_variable = QPushButton("删除选中行")
        delete_variable.setObjectName("Danger")
        delete_variable.clicked.connect(lambda: self._delete_selected_rows(self.variables))
        variable_buttons.addWidget(add_variable)
        variable_buttons.addWidget(delete_variable)
        variable_buttons.addStretch()
        variable_layout.addLayout(variable_buttons)
        tables.addWidget(variable_card, 1)
        layout.addLayout(tables, 1)
        self.validation_hint = QLabel("选择 DOE 后将检查优化配置。")
        self.validation_hint.setObjectName("Caption")
        self.validation_hint.setWordWrap(True)
        layout.addWidget(self.validation_hint)
        self._refresh_optimization_actions()

    def refresh_does(self) -> None:
        if self.doe_refresh_in_flight:
            return
        self.doe_refresh_in_flight = True
        self.refresh_doe_button.setEnabled(False)
        self.refresh_doe_button.setText("正在刷新…")
        self.schema_hint.setText("正在刷新 DOE 列表…")
        self.run_async(
            lambda: self.client_provider().list_doe(),
            self._does_loaded,
            self._doe_refresh_failed,
        )

    def _doe_refresh_failed(self, message: str) -> None:
        self.doe_refresh_in_flight = False
        self.refresh_doe_button.setEnabled(True)
        self.refresh_doe_button.setText("刷新 DOE")
        self.schema_hint.setText(f"DOE 列表加载失败：{message}；请点击“刷新 DOE”重试。")

    def set_schema(self, schema: dict[str, Any]) -> None:
        inputs = list(schema.get("inputs") or [])
        outputs = list(schema.get("outputs") or [])
        self.schema_ready_for_optimization = bool(inputs and outputs)
        self.training_bounds = {
            str(item["name"]): (float(item.get("lower", 0.0)), float(item.get("upper", 1.0)))
            for item in inputs
        }
        weight = 1.0 / len(outputs) if outputs and self.mode.currentData() != "multi" else 1.0
        self.objectives.blockSignals(True)
        self.variables.blockSignals(True)
        self.variables.set_data(
            ["名称", "下限", "上限"],
            [
                [
                    item["name"],
                    f"{float(item.get('lower', 0.0)):.8g}",
                    f"{float(item.get('upper', 1.0)):.8g}",
                ]
                for item in inputs
            ],
        )
        self.objectives.set_data(
            ["名称", "方向", "权重"],
            [[name, "min", f"{weight:.8g}"] for name in outputs],
        )
        self.objectives.blockSignals(False)
        self.variables.blockSignals(False)
        self.validate_configuration()

    def _does_loaded(self, data: dict[str, Any]) -> None:
        self.doe_refresh_in_flight = False
        self.refresh_doe_button.setEnabled(True)
        self.refresh_doe_button.setText("刷新 DOE")
        current = str(self.doe_id.currentData() or self.doe_id.currentText()).strip()
        self.doe_id.clear()
        for item in data.get("items", []):
            name = item.get("name") or item["id"]
            self.doe_id.addItem(str(name), item["id"])
        if current:
            index = self.doe_id.findData(current)
            if index >= 0:
                self.doe_id.setCurrentIndex(index)
            else:
                self.doe_id.setCurrentIndex(-1)
        else:
            self.doe_id.setCurrentIndex(-1)
        if self.doe_id.currentIndex() >= 0:
            self.load_selected_doe()
        else:
            self.set_schema({})
            self.optimization_state = "not_started"
            self._refresh_optimization_actions()

    def load_selected_doe(self) -> None:
        self.selection_revision += 1
        revision = self.selection_revision
        doe_id = str(self.doe_id.currentData() or "").strip()
        if not doe_id:
            self.set_schema({})
            self.optimization_state = "not_started"
            self.schema_hint.setText("请选择已有 DOE；目标函数和设计变量将从该任务的训练记录加载。")
            self._refresh_optimization_actions()
            return
        self.set_schema({})
        self.optimization_state = "loading"
        self._refresh_optimization_actions()
        self.schema_hint.setText(f"正在读取 {doe_id} 的代理模型训练字段和输入边界…")
        self.poll()
        self.run_async(
            lambda: self.client_provider().training_progress(doe_id),
            lambda data: self._training_schema_loaded(doe_id, revision, data),
            lambda message: self._training_schema_failed(doe_id, revision, message),
        )

    def _training_schema_loaded(
        self, doe_id: str, revision: int, data: dict[str, Any]
    ) -> None:
        if revision != self.selection_revision or doe_id != self.current_doe_id():
            return
        inputs = list(data.get("input_bounds") or [])
        input_names = list(data.get("input_names") or [])
        targets = list(data.get("target_names") or [])
        bounds_by_name = {item.get("name"): item for item in inputs}
        schema_inputs = [
            {
                "name": name,
                "lower": bounds_by_name.get(name, {}).get("lower", 0.0),
                "upper": bounds_by_name.get(name, {}).get("upper", 1.0),
            }
            for name in input_names
        ]
        self.set_schema({"inputs": schema_inputs, "outputs": targets})
        if not input_names or not targets:
            self.schema_hint.setText("当前 DOE 没有可用于优化的代理模型训练字段，请先完成训练。")
            self._refresh_optimization_actions()
            return
        constant_names = [
            item["name"] for item in schema_inputs
            if float(item["lower"]) >= float(item["upper"])
        ]
        detail = f"已加载 {len(input_names)} 个设计变量和 {len(targets)} 个目标函数。"
        if constant_names:
            detail += " 以下变量在训练数据中为常量，请手动设置有效上下界：" + "、".join(constant_names)
        self.schema_hint.setText(detail)
        self._refresh_optimization_actions()

    def _training_schema_failed(self, doe_id: str, revision: int, message: str) -> None:
        if revision != self.selection_revision or doe_id != self.current_doe_id():
            return
        self.set_schema({})
        self.schema_hint.setText(f"训练字段读取失败：{message}")
        self._refresh_optimization_actions()

    def current_doe_id(self) -> str:
        return str(self.doe_id.currentData() or "").strip()

    def mode_changed(self) -> None:
        rl = self.mode.currentData() == "reinforcement_learning"
        self.algorithm.clear()
        self.algorithm.addItem("PPO 强化学习", "ppo") if rl else self.algorithm.addItem("NSGA-II 遗传算法", "nsga2")
        self.parameter_stack.setCurrentIndex(1 if rl else 0)
        self._normalize_default_weights()
        self.validate_configuration()

    def _normalize_default_weights(self) -> None:
        if self.mode.currentData() == "multi" or not self.objectives.rowCount():
            return
        values = []
        for row in range(self.objectives.rowCount()):
            try:
                values.append(float(self.objectives.item(row, 2).text()))
            except (AttributeError, ValueError):
                return
        if not values or abs(sum(values) - len(values)) > 1e-8:
            return
        weight = 1.0 / len(values)
        self.objectives.blockSignals(True)
        for row in range(self.objectives.rowCount()):
            self.objectives.item(row, 2).setText(f"{weight:.8g}")
        self.objectives.blockSignals(False)

    def validate_configuration(self, _item=None) -> bool:
        errors: list[str] = []
        warnings: list[str] = []
        error_rows: set[tuple[DataTable, int]] = set()
        objective_rows = self._cell_rows(self.objectives)
        variable_rows = self._cell_rows(self.variables)
        objective_names = [row[0] for row in objective_rows]
        variable_names = [row[0] for row in variable_rows]
        if not objective_rows:
            errors.append("至少需要一个目标函数")
        if len(objective_names) != len(set(objective_names)):
            errors.append("目标函数名称不能重复")
        weights = []
        for row_index, row in enumerate(objective_rows):
            try:
                if row[1] not in {"min", "max"}:
                    raise ValueError
                weight = float(row[2])
                if not 0 <= weight <= 1:
                    raise ValueError
                weights.append(weight)
            except (ValueError, IndexError):
                error_rows.add((self.objectives, row_index))
                errors.append(f"目标第 {row_index + 1} 行的方向或权重无效")
        if self.mode.currentData() != "multi" and weights and abs(sum(weights) - 1.0) > 1e-8:
            errors.append("单目标/强化学习模式的目标权重总和必须为 1")
        if not variable_rows:
            errors.append("至少需要一个设计变量")
        if len(variable_names) != len(set(variable_names)):
            errors.append("设计变量名称不能重复")
        for row_index, row in enumerate(variable_rows):
            try:
                lower, upper = float(row[1]), float(row[2])
                if lower >= upper:
                    raise ValueError
            except (ValueError, IndexError):
                error_rows.add((self.variables, row_index))
                errors.append(f"变量第 {row_index + 1} 行必须满足下限小于上限")
                continue
            trained = self.training_bounds.get(row[0])
            if trained and (lower < trained[0] or upper > trained[1]):
                warnings.append(
                    f"{row[0]} 超出训练范围 [{trained[0]:.8g}, {trained[1]:.8g}]"
                )
        for table in (self.objectives, self.variables):
            table.blockSignals(True)
            for row in range(table.rowCount()):
                color = QColor("#4d262d") if (table, row) in error_rows else QColor("transparent")
                for column in range(table.columnCount()):
                    item = table.item(row, column)
                    if item:
                        item.setBackground(color)
            table.blockSignals(False)
        self.configuration_valid = not errors
        if errors:
            self.validation_hint.setText("配置错误：" + "；".join(dict.fromkeys(errors)))
            self.validation_hint.setStyleSheet("color:#ffaaa8")
        elif warnings:
            self.validation_hint.setText("范围警告：" + "；".join(warnings))
            self.validation_hint.setStyleSheet("color:#ffd47c")
        else:
            message = "配置检查通过"
            if self.mode.currentData() == "multi":
                message += "；多目标模式的权重仅记录，不参与 NSGA-II 的 Pareto 排序"
            self.validation_hint.setText(message)
            self.validation_hint.setStyleSheet("color:#75e1bd")
        self._refresh_optimization_actions()
        return self.configuration_valid

    def _refresh_optimization_actions(self) -> None:
        active = self.optimization_state in {"loading", "queued", "running", "stopping"}
        ready = bool(
            self.current_doe_id()
            and self.schema_ready_for_optimization
            and self.configuration_valid
        )
        if self.optimization_state in {"queued", "running"}:
            self.start_button.setText("正在优化…")
        elif self.optimization_state == "stopping":
            self.start_button.setText("正在停止…")
        elif self.optimization_state == "loading":
            self.start_button.setText("正在加载…")
        elif self.has_completed_optimization:
            self.start_button.setText("重新优化")
        else:
            self.start_button.setText("启动优化")
        self.start_button.setEnabled(ready and not active)
        self.stop_button.setEnabled(self.optimization_state in {"queued", "running"})

    def _cell_rows(self, table: DataTable) -> list[list[str]]:
        rows = [[cell.strip() for cell in row] for row in table.rows()]
        return [row for row in rows if row and row[0]]

    @staticmethod
    def _add_empty_row(table: DataTable) -> None:
        row = table.rowCount()
        table.insertRow(row)
        for column in range(table.columnCount()):
            table.setItem(row, column, QTableWidgetItem(""))
        table.setCurrentCell(row, 0)
        table.editItem(table.item(row, 0))

    def _delete_selected_rows(self, table: DataTable) -> None:
        selected = sorted(
            {index.row() for index in table.selectionModel().selectedRows()}, reverse=True
        )
        if not selected:
            QMessageBox.information(self, "未选择行", "请先在表格中选择要删除的整行。")
            return
        for row in selected:
            table.removeRow(row)

    def start_optimization(self) -> None:
        doe_id = str(self.doe_id.currentData() or self.doe_id.currentText()).strip()
        if not doe_id:
            self.show_error("请选择 DOE")
            return
        if not self.validate_configuration():
            self.show_error(self.validation_hint.text())
            return
        if self.has_completed_optimization:
            answer = QMessageBox.question(
                self,
                "确认重新优化",
                "当前 DOE 已有完成的优化结果，可直接在结果分析中查看。\n\n"
                "确定要使用当前配置重新优化吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            objectives = [
                {"name": row[0], "direction": row[1] or "min", "weight": float(row[2] or 1)}
                for row in self._cell_rows(self.objectives)
            ]
            variables = [
                {"name": row[0], "lower": float(row[1]), "upper": float(row[2])}
                for row in self._cell_rows(self.variables)
            ]
        except (ValueError, IndexError):
            self.show_error("请检查目标权重和设计变量上下界")
            return
        if not objectives or not variables:
            self.show_error("当前 DOE 没有可用的目标函数或设计变量，请先完成代理模型训练")
            return
        mode = str(self.mode.currentData())
        if mode == "reinforcement_learning":
            params = {
                "total_timesteps": self.ppo_timesteps.value(),
                "episode_steps": self.ppo_episode_steps.value(),
                "evaluation_episodes": self.ppo_evaluation_episodes.value(),
                "learning_rate": self.ppo_learning_rate.value(),
                "constraint_penalty": self.ppo_constraint_penalty.value(),
                "seed": 42,
            }
        else:
            params = {
                "pop_size": self.ga_population.value(),
                "n_offsprings": self.ga_offspring.value(),
                "eliminate_duplicates": True,
                "n_gen": self.ga_generations.value(),
                "seed": 42,
            }
        payload = {
            "id": doe_id,
            "mode": mode,
            "objectives": objectives,
            "constraints": [],
            "decision_variables": variables,
            "algorithm": {"name": self.algorithm.currentData(), "params": params},
        }
        self.start_button.setEnabled(False)
        self.has_completed_optimization = False
        self.optimization_state = "queued"
        set_status(self.status_label, "queued")
        self._refresh_optimization_actions()
        revision = self.selection_revision
        self.run_async(
            lambda: self.client_provider().start_optimization(payload),
            lambda data: self._started(doe_id, revision, data),
            lambda message: self._failed(doe_id, revision, message),
        )

    def _started(self, doe_id: str, revision: int, _data: dict[str, Any]) -> None:
        if doe_id != self.current_doe_id() or revision != self.selection_revision:
            return
        self.timer.start()
        self.poll()

    def _failed(self, doe_id: str, revision: int, message: str) -> None:
        if doe_id != self.current_doe_id() or revision != self.selection_revision:
            return
        self.optimization_state = "failed"
        set_status(self.status_label, "failed")
        self._refresh_optimization_actions()
        self.show_error(message)

    def stop_optimization(self) -> None:
        doe_id = str(self.doe_id.currentData() or self.doe_id.currentText()).strip()
        if doe_id:
            self.optimization_state = "stopping"
            self._refresh_optimization_actions()
            self.run_async(lambda: self.client_provider().stop_optimization(doe_id), lambda _data: self.poll())

    def poll(self) -> None:
        doe_id = self.current_doe_id()
        if doe_id:
            revision = self.selection_revision
            self.run_async(
                lambda: self.client_provider().optimization_progress(doe_id),
                lambda data: self._apply_progress(doe_id, revision, data),
                lambda message: self._progress_failed(doe_id, revision, message),
            )

    def _apply_progress(self, doe_id: str, revision: int, data: dict[str, Any]) -> None:
        if doe_id == self.current_doe_id() and revision == self.selection_revision:
            self.update_progress(data)

    def _progress_failed(self, doe_id: str, revision: int, message: str) -> None:
        if doe_id != self.current_doe_id() or revision != self.selection_revision:
            return
        self.optimization_state = "failed"
        self.stage_label.setText(f"优化状态读取失败：{message}")
        self._refresh_optimization_actions()

    def update_progress(self, data: dict[str, Any]) -> None:
        state = str(data.get("status", "not_started"))
        self.optimization_state = state
        self.has_completed_optimization = state == "finished" and bool(data.get("result"))
        set_status(self.status_label, state)
        raw_updated_at = data.get("updated_at")
        self.stage_label.setText(f"阶段：{stage_text(data.get('stage'))}")
        self.updated_label.setText(f"最近更新：{format_timestamp(raw_updated_at)}")
        self.updated_label.setToolTip(str(raw_updated_at or ""))
        self.progress.setValue(int(data.get("progress", 0)))
        if state in {"finished", "failed", "stopped", "not_started"}:
            self.timer.stop()
        self._refresh_optimization_actions()


class ResultsPage(QWidget, AsyncMixin):
    def __init__(
        self,
        pool: QThreadPool | None = None,
        client_provider: Callable[[], ApiClient] | None = None,
    ):
        super().__init__()
        self.pool = pool or QThreadPool.globalInstance()
        self.client_provider = client_provider
        self.headers: list[str] = []
        self.rows: list[list[Any]] = []
        self.tasks_refresh_in_flight = False
        self.chart_windows: list[QDialog] = []
        self.history_windows: list[QDialog] = []
        self.history_headers = [
            "运行版本", "状态", "模式", "算法", "解数量", "耗时(s)", "更新时间"
        ]
        self.history_rows: list[list[Any]] = []
        self.history_run_ids: list[str] = []
        self.selected_run_id = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        header = QHBoxLayout()
        header.addLayout(page_header("结果分析", "任意选择输入或输出字段，创建二维或三维图表并与原始数据对照。"), 1)
        open_button = QPushButton("打开本地数据")
        open_button.clicked.connect(self.open_file)
        export_button = QPushButton("导出当前表格")
        export_button.clicked.connect(self.export_table)
        header.addWidget(open_button)
        header.addWidget(export_button)
        layout.addLayout(header)

        task_card, task_layout = card("从已完成任务加载")
        task_row = QHBoxLayout()
        self.result_task = QComboBox()
        self.result_task.setMinimumWidth(280)
        self.result_task.setPlaceholderText("刷新后选择已完成的优化任务")
        self.result_task.currentIndexChanged.connect(self._result_task_changed)
        self.refresh_tasks_button = QPushButton("刷新任务")
        self.refresh_tasks_button.clicked.connect(self.refresh_tasks)
        self.load_task_button = QPushButton("加载任务结果")
        self.load_task_button.setObjectName("Primary")
        self.load_task_button.clicked.connect(self.load_selected_task)
        self.load_task_button.setEnabled(False)
        self.open_history_button = QPushButton("查看运行历史")
        self.open_history_button.clicked.connect(self.open_history_window)
        self.open_history_button.setEnabled(False)
        task_row.addWidget(QLabel("优化任务"))
        task_row.addWidget(self.result_task, 1)
        task_row.addWidget(self.refresh_tasks_button)
        task_row.addWidget(self.open_history_button)
        task_row.addWidget(self.load_task_button)
        task_layout.addLayout(task_row)
        self.task_hint = QLabel("直接从后端优化目录读取资源，无需手动查找本地结果文件。")
        self.task_hint.setObjectName("Caption")
        self.task_hint.setWordWrap(True)
        task_layout.addWidget(self.task_hint)

        layout.addWidget(task_card)

        controls, controls_layout = card()
        type_row = QHBoxLayout()
        self.style = QComboBox()
        self.style.addItem("二维散点图", "scatter")
        self.style.addItem("二维折线图", "line")
        self.style.addItem("二维柱状图", "bar")
        self.style.addItem("三维散点图", "scatter3d")
        self.style.currentIndexChanged.connect(self._chart_type_changed)
        self.chart_hint = QLabel("所有数值列均可作为坐标轴，不区分输入工艺参数与输出目标。")
        self.chart_hint.setObjectName("Caption")
        type_row.addWidget(QLabel("图表类型"))
        type_row.addWidget(self.style)
        type_row.addSpacing(12)
        type_row.addWidget(self.chart_hint, 1)
        controls_layout.addLayout(type_row)

        axis_row = QHBoxLayout()
        self.x_field = QComboBox()
        self.y_field = QComboBox()
        self.z_label = QLabel("Z 轴")
        self.z_field = QComboBox()
        for field in (self.x_field, self.y_field, self.z_field):
            field.setMinimumWidth(130)
        update = QPushButton("更新图表")
        update.setObjectName("Primary")
        update.clicked.connect(self.update_chart)
        axis_row.addWidget(QLabel("X 轴"))
        axis_row.addWidget(self.x_field, 1)
        axis_row.addWidget(QLabel("Y 轴"))
        axis_row.addWidget(self.y_field, 1)
        axis_row.addWidget(self.z_label)
        axis_row.addWidget(self.z_field, 1)
        axis_row.addWidget(update)
        controls_layout.addLayout(axis_row)
        self.plot_status = QLabel("图表会按数据量自动抽样，原始表格始终保留全部数据。")
        self.plot_status.setObjectName("Caption")
        controls_layout.addWidget(self.plot_status)
        layout.addWidget(controls)
        split = QHBoxLayout()
        table_card, table_layout = card("原始表格")
        self.table = VirtualDataTable()
        self.table.setToolTip("字段较多时可横向滚动；拖动表头分隔线可调整列宽。")
        table_layout.addWidget(self.table)
        split.addWidget(table_card, 3)
        chart_card, chart_layout = card("可视化")
        zoom_row = QHBoxLayout()
        zoom_hint = QLabel("双击可单独查看；滚轮或框选可缩放二维图，三维图可拖动旋转。")
        zoom_hint.setObjectName("Caption")
        zoom_in = QPushButton("放大")
        zoom_out = QPushButton("缩小")
        zoom_reset = QPushButton("重置视图")
        self.open_chart_button = QPushButton("单独查看图表")
        self.open_chart_button.setObjectName("Primary")
        zoom_in.clicked.connect(lambda: self.zoom_chart(1.25))
        zoom_out.clicked.connect(lambda: self.zoom_chart(0.8))
        zoom_reset.clicked.connect(self.reset_chart_zoom)
        self.open_chart_button.clicked.connect(self.open_chart_window)
        zoom_row.addWidget(zoom_hint, 1)
        zoom_row.addWidget(zoom_in)
        zoom_row.addWidget(zoom_out)
        zoom_row.addWidget(zoom_reset)
        zoom_row.addWidget(self.open_chart_button)
        chart_layout.addLayout(zoom_row)
        self.chart_stack = QStackedWidget()
        self.chart = ZoomableChartView()
        self.chart.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart.setMinimumWidth(450)
        self.chart.double_clicked.connect(self.open_chart_window)
        self.chart_stack.addWidget(self.chart)
        self.scatter3d: Scatter3DCanvas | None = None
        self.scatter3d_container: QWidget | None = None
        self.three_d_placeholder = QLabel("选择三个数值字段并点击“更新图表”加载三维视图")
        self.three_d_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.three_d_placeholder.setObjectName("Subtitle")
        self.chart_stack.addWidget(self.three_d_placeholder)
        chart_layout.addWidget(self.chart_stack)
        split.addWidget(chart_card, 2)
        layout.addLayout(split, 1)
        self._chart_type_changed()
        self._empty_chart()

    def refresh_tasks(self) -> None:
        if self.client_provider is None:
            self.task_hint.setText("当前结果页未配置后端连接。")
            return
        if self.tasks_refresh_in_flight:
            return
        self.tasks_refresh_in_flight = True
        self.refresh_tasks_button.setEnabled(False)
        self.refresh_tasks_button.setText("正在刷新…")
        self.task_hint.setText("正在读取已完成的优化任务…")
        self.run_async(
            lambda: self.client_provider().list_doe(),
            self._tasks_loaded,
            self._task_failed,
        )

    def _tasks_loaded(self, data: dict[str, Any]) -> None:
        self.tasks_refresh_in_flight = False
        self.refresh_tasks_button.setEnabled(True)
        self.refresh_tasks_button.setText("刷新任务")
        current = self.result_task.currentData()
        completed = [
            item for item in data.get("items", [])
            if item.get("has_optimization_result") or item.get("stage") == "optimization_finished"
        ]
        self.result_task.blockSignals(True)
        self.result_task.clear()
        for item in completed:
            name = item.get("name") or item["id"]
            self.result_task.addItem(str(name), item["id"])
        if current:
            index = self.result_task.findData(current)
            if index >= 0:
                self.result_task.setCurrentIndex(index)
        if not current:
            self.result_task.setCurrentIndex(-1)
        self.result_task.blockSignals(False)
        self._result_task_changed(self.result_task.currentIndex())
        self.task_hint.setText(
            f"已找到 {len(completed)} 个完成的优化任务，选择后可直接加载表格和图表。"
            if completed else "当前后端没有已完成且带结果资源的优化任务。"
        )

    def _result_task_changed(self, index: int) -> None:
        self.load_task_button.setEnabled(index >= 0)
        if index >= 0:
            self.load_selected_task()
            return
        self.history_rows = []
        self.history_run_ids = []
        self.selected_run_id = ""
        self.open_history_button.setEnabled(False)

    def open_history_window(self) -> None:
        if not self.history_rows:
            QMessageBox.information(self, "没有运行历史", "请先选择并加载一个优化任务。")
            return
        dialog = QDialog(self, Qt.WindowType.Window)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("优化运行历史")
        dialog.resize(1050, 600)
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(18, 18, 18, 18)
        dialog_layout.setSpacing(12)
        title = QLabel(f"{self.result_task.currentText()} · 运行历史")
        title.setObjectName("Section")
        hint = QLabel("双击表头分隔线可适配内容；拖动分隔线可调整列宽。")
        hint.setObjectName("Caption")
        dialog_layout.addWidget(title)
        dialog_layout.addWidget(hint)
        table = DataTable()
        table.enable_wide_columns()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        table.set_data(self.history_headers, self.history_rows)
        dialog_layout.addWidget(table, 1)
        buttons = QDialogButtonBox()
        load_button = buttons.addButton(
            "加载选中版本", QDialogButtonBox.ButtonRole.ActionRole
        )
        close_button = buttons.addButton(QDialogButtonBox.StandardButton.Close)
        load_button.setObjectName("Primary")
        load_button.setEnabled(False)
        table.itemSelectionChanged.connect(
            lambda: self._history_selection_changed(table, load_button)
        )
        load_button.clicked.connect(
            lambda: self._load_history_from_dialog(table.currentRow(), dialog)
        )
        table.cellDoubleClicked.connect(
            lambda row, _column: self._load_history_from_dialog(row, dialog)
        )
        close_button.clicked.connect(dialog.close)
        buttons.rejected.connect(dialog.close)
        dialog_layout.addWidget(buttons)
        if self.selected_run_id in self.history_run_ids:
            table.selectRow(self.history_run_ids.index(self.selected_run_id))
        self.history_windows.append(dialog)
        dialog.destroyed.connect(
            lambda _object=None, window=dialog: self._forget_history_window(window)
        )
        dialog.show()

    def _history_selection_changed(
        self, table: DataTable, load_button: QPushButton
    ) -> None:
        row = table.currentRow()
        run_id = self.history_run_ids[row] if 0 <= row < len(self.history_run_ids) else ""
        result = (self.run_records.get(run_id) or {}).get("result") or {}
        load_button.setEnabled(self._result_is_readable(result))

    def _load_history_from_dialog(self, row: int, dialog: QDialog) -> None:
        if not 0 <= row < len(self.history_run_ids):
            return
        run_id = self.history_run_ids[row]
        result = (self.run_records.get(run_id) or {}).get("result") or {}
        if not self._result_is_readable(result):
            QMessageBox.information(self, "版本不可用", "该运行版本没有可读取的完成结果。")
            return
        dialog.close()
        self.load_selected_run(run_id)

    def _forget_history_window(self, window: QDialog) -> None:
        if window in self.history_windows:
            self.history_windows.remove(window)

    def _task_failed(self, message: str) -> None:
        self.tasks_refresh_in_flight = False
        self.refresh_tasks_button.setEnabled(True)
        self.refresh_tasks_button.setText("刷新任务")
        self.load_task_button.setEnabled(bool(self.result_task.count()))
        self.task_hint.setText(f"任务列表加载失败：{message}")

    def load_selected_task(self) -> None:
        doe_id = str(self.result_task.currentData() or "").strip()
        if not doe_id or self.client_provider is None:
            QMessageBox.information(self, "未选择任务", "请先刷新并选择一个已完成的优化任务。")
            return
        self.load_task_button.setEnabled(False)
        self.load_task_button.setText("加载中…")
        self.history_rows = []
        self.history_run_ids = []
        self.selected_run_id = ""
        self.open_history_button.setEnabled(False)
        self.task_hint.setText(f"正在加载任务 {doe_id} 的优化结果…")

        self.run_async(
            lambda: (doe_id, self.client_provider().optimization_progress(doe_id)),
            self._task_progress_loaded,
            lambda message: self._task_result_failed(message)
            if doe_id == str(self.result_task.currentData() or "").strip()
            else None,
        )

    def _task_progress_loaded(self, loaded: tuple[str, dict[str, Any]]) -> None:
        doe_id, progress = loaded
        if doe_id != str(self.result_task.currentData() or "").strip():
            return
        self._populate_run_history(progress)
        result = progress.get("result") or {}
        run_id = str(progress.get("current_run_id") or "")
        if self._result_is_readable(result) and not run_id:
            resource_id = result.get("resource_id")
            for run in reversed(progress.get("history") or []):
                if (run.get("result") or {}).get("resource_id") == resource_id:
                    run_id = str(run.get("run_id") or "")
                    break
            run_id = run_id or "legacy-current"
        if progress.get("status") != "finished" or not self._result_is_readable(result):
            result = {}
            run_id = ""
            for run in reversed(progress.get("history") or []):
                candidate = run.get("result") or {}
                if run.get("status") == "finished" and self._result_is_readable(candidate):
                    result = candidate
                    run_id = str(run.get("run_id") or "")
                    break
        if not result:
            self.load_task_button.setEnabled(True)
            self.load_task_button.setText("重新加载任务结果")
            self.task_hint.setText("已加载运行历史，但当前没有可读取的已完成结果。")
            return
        self.selected_run_id = run_id
        fields = list(result.get("columns") or [])
        resource_id = str(result["resource_id"])
        self.task_hint.setText(f"已加载运行历史，正在读取版本 {run_id} 的结果数据…")
        self.run_async(
            lambda: self.client_provider().get_data(doe_id, resource_id, fields),
            lambda data: self._task_result_loaded(
                doe_id,
                run_id,
                fields,
                rows_by_fields(data.get("values", {}), fields),
            ),
            lambda message: self._task_result_failed(message)
            if (
                doe_id == str(self.result_task.currentData() or "").strip()
                and run_id == self.selected_run_id
            )
            else None,
        )

    @staticmethod
    def _result_is_readable(result: dict[str, Any]) -> bool:
        return bool(result.get("columns") and result.get("resource_id"))

    def _task_result_loaded(
        self, doe_id: str, run_id: str, headers: list[str], rows: list[list[Any]]
    ) -> None:
        if (
            doe_id != str(self.result_task.currentData() or "").strip()
            or run_id != self.selected_run_id
        ):
            return
        self.set_data(headers, rows)
        self.load_task_button.setEnabled(True)
        self.load_task_button.setText("重新加载任务结果")
        self.task_hint.setText(
            f"已加载任务 {doe_id} 的版本 {run_id}：{len(rows)} 行 × {len(headers)} 列。"
        )

    def _populate_run_history(self, progress: dict[str, Any]) -> None:
        history = list(progress.get("history") or [])
        if not history and progress.get("result"):
            history = [{
                "run_id": progress.get("current_run_id") or "legacy-current",
                "status": progress.get("status"),
                "stage": progress.get("stage"),
                "request": progress.get("request"),
                "result": progress.get("result"),
                "updated_at": progress.get("updated_at"),
            }]
        self.run_records = {
            str(run.get("run_id")): run for run in history if run.get("run_id")
        }
        summary_rows = []
        run_ids = []
        for run in reversed(history):
            run_id = str(run.get("run_id", "—"))
            run_ids.append(run_id)
            request = run.get("request") or {}
            result = run.get("result") or {}
            info = result.get("task_info") or {}
            algorithm = "PPO" if request.get("optimizer") == "rl" else "NSGA-II"
            summary_rows.append([
                run_id,
                status_view(run.get("status")).text,
                request.get("requested_mode", request.get("mode", "—")),
                algorithm,
                (result.get("constraint_check") or {}).get("solution_count", "—"),
                info.get("run_time_sec", "—"),
                format_timestamp(run.get("updated_at")),
            ])
        self.history_rows = summary_rows
        self.history_run_ids = run_ids
        self.open_history_button.setEnabled(bool(summary_rows))

    def load_selected_run(self, run_id: str) -> None:
        run = self.run_records.get(run_id) or {}
        result = run.get("result") or {}
        doe_id = str(self.result_task.currentData() or "")
        fields = list(result.get("columns") or [])
        resource_id = result.get("resource_id")
        if not doe_id or not fields or not resource_id or self.client_provider is None:
            QMessageBox.information(self, "版本不可用", "所选运行版本没有可读取的结果资源。")
            return
        self.selected_run_id = run_id
        self.run_async(
            lambda: self.client_provider().get_data(doe_id, str(resource_id), fields),
            lambda loaded: self._history_result_loaded(
                doe_id, run_id, fields, rows_by_fields(loaded.get("values", {}), fields)
            ),
            lambda message: self._task_result_failed(message)
            if (
                doe_id == str(self.result_task.currentData() or "").strip()
                and run_id == self.selected_run_id
            )
            else None,
        )

    def _history_result_loaded(
        self, doe_id: str, run_id: str, fields: list[str], rows: list[list[Any]]
    ) -> None:
        if (
            doe_id != str(self.result_task.currentData() or "").strip()
            or run_id != self.selected_run_id
        ):
            return
        self.set_data(fields, rows)
        self.task_hint.setText(f"已加载历史运行 {run_id}：{len(rows)} 行 × {len(fields)} 列。")

    def _task_result_failed(self, message: str) -> None:
        self.load_task_button.setEnabled(True)
        self.load_task_button.setText("加载任务结果")
        self.task_hint.setText(f"结果加载失败：{message}")
        self.show_error(message)

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开结果数据", "", "表格数据 (*.txt *.tsv *.csv);;所有文件 (*)")
        if path:
            try:
                headers, rows = read_tabular_file(path)
                self.set_data(headers, rows)
            except Exception as exc:
                QMessageBox.critical(self, "无法打开数据", str(exc))

    def set_data(self, headers: list[str], rows: list[list[Any]]) -> None:
        self.headers, self.rows = list(headers), list(rows)
        self.table.set_data(self.headers, self.rows)
        self.x_field.clear()
        self.y_field.clear()
        self.z_field.clear()
        plottable = numeric_field_names(self.headers, self.rows)
        excluded = [name for name in self.headers if name not in plottable]
        self.x_field.addItems(plottable)
        self.y_field.addItems(plottable)
        self.z_field.addItems(plottable)
        if excluded:
            self.chart_hint.setText(
                "坐标轴可选择任意数值列；已排除无有效数值列：" + "、".join(excluded)
            )
        else:
            self.chart_hint.setText("所有数值列均可作为坐标轴，不区分输入工艺参数与输出目标。")
        if len(plottable) > 1:
            self.y_field.setCurrentIndex(1)
        if len(plottable) > 2:
            self.z_field.setCurrentIndex(2)
        self.update_chart()

    def export_table(self) -> None:
        if not self.headers:
            QMessageBox.information(self, "没有数据", "请先打开或加载结果数据。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出当前表格", "optimization_result.tsv", "TSV (*.tsv);;CSV (*.csv)")
        if not path:
            return
        delimiter = "," if path.lower().endswith(".csv") else "\t"
        lines = [delimiter.join(self.headers)] + [delimiter.join(map(str, row)) for row in self.rows]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _numeric_points(self) -> tuple[list[float], list[float], int]:
        x_index = self.x_field.currentIndex()
        points = []
        for row_index, row in enumerate(self.rows):
            try:
                x_column = self.headers.index(self.x_field.currentText())
                y_column = self.headers.index(self.y_field.currentText())
                x = finite_float(row[x_column]) if x_index >= 0 else float(row_index + 1)
                y = finite_float(row[y_column])
                if x is not None and y is not None:
                    points.append((x, y))
            except (ValueError, IndexError):
                continue
        total = len(points)
        sampled = evenly_sample(points, MAX_2D_PLOT_POINTS)
        return [point[0] for point in sampled], [point[1] for point in sampled], total

    def _numeric_points_3d(self) -> tuple[list[tuple[float, float, float]], int]:
        try:
            indices = tuple(
                self.headers.index(field.currentText())
                for field in (self.x_field, self.y_field, self.z_field)
            )
        except ValueError:
            return [], 0
        points = []
        for row in self.rows:
            try:
                values = tuple(finite_float(row[index]) for index in indices)
                if all(value is not None for value in values):
                    points.append(tuple(float(value) for value in values if value is not None))
            except IndexError:
                continue
        return evenly_sample(points, MAX_3D_PLOT_POINTS), len(points)

    def _chart_type_changed(self) -> None:
        is_3d = self.style.currentData() == "scatter3d"
        self.z_label.setVisible(is_3d)
        self.z_field.setVisible(is_3d)
        if is_3d:
            target = self.scatter3d_container or self.three_d_placeholder
            self.chart_stack.setCurrentWidget(target)
        else:
            self.chart_stack.setCurrentWidget(self.chart)

    def zoom_chart(self, factor: float) -> None:
        if self.style.currentData() != "scatter3d":
            self.chart.chart().zoom(factor)
            return
        if self.scatter3d is None:
            return
        self.scatter3d.zoom_by(factor)

    def reset_chart_zoom(self) -> None:
        if self.style.currentData() != "scatter3d":
            self.chart.chart().zoomReset()
            return
        if self.scatter3d is not None:
            self.scatter3d.reset_view()

    def update_chart(self) -> None:
        if not self.headers or not self.rows:
            self._empty_chart()
            return
        if self.style.currentData() == "scatter3d":
            self._update_3d_chart()
            return
        built = self._build_2d_chart()
        if built is None:
            self._empty_chart("所选字段没有可绘制的数值")
            return
        chart, total, shown, warning = built
        self._set_plot_status(total, shown, warning)
        self.chart.setChart(chart)
        self.chart_stack.setCurrentIndex(0)

    def _build_2d_chart(self) -> tuple[QChart, int, int, str] | None:
        xs, ys, total = self._numeric_points()
        if not ys:
            return None
        chart = self._base_chart(f"{self.y_field.currentText()} / {self.x_field.currentText()}")
        style = self.style.currentData()
        if style == "bar":
            shown = min(80, len(ys))
            series = QBarSeries()
            values = QBarSet(self.y_field.currentText())
            values.setColor(QColor("#4bc8d2"))
            values.append(ys[:80])
            series.append(values)
            chart.addSeries(series)
            axis_x = QBarCategoryAxis()
            axis_x.append([f"{value:.6g}" for value in xs[:80]])
            axis_x.setTitleText(self.x_field.currentText())
            axis_x.setLabelsAngle(-45)
            chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            series.attachAxis(axis_x)
            axis_y = QValueAxis()
            axis_y.setTitleText(self.y_field.currentText())
            axis_y.setLabelFormat("%.6g")
            axis_y.setTickCount(6)
            axis_y.setRange(
                *self._padded_axis_range(ys[:shown], include_zero=True)
            )
            chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            series.attachAxis(axis_y)
        else:
            shown = len(ys)
            series = QScatterSeries() if style == "scatter" else QLineSeries()
            series.setName(self.y_field.currentText())
            series.setColor(QColor("#54d3dc"))
            if isinstance(series, QScatterSeries):
                series.setMarkerSize(8)
                series.setBorderColor(QColor("#b8f5f7"))
            points = list(zip(xs, ys, strict=True))
            if isinstance(series, QLineSeries):
                points.sort(key=lambda point: point[0])
            for x, y in points:
                series.append(x, y)
            chart.addSeries(series)
            axis_x = QValueAxis()
            axis_x.setTitleText(self.x_field.currentText())
            axis_x.setLabelFormat("%.6g")
            axis_x.setTickCount(6)
            axis_x.setRange(*self._padded_axis_range(xs))
            chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
            series.attachAxis(axis_x)
            axis_y = QValueAxis()
            axis_y.setTitleText(self.y_field.currentText())
            axis_y.setLabelFormat("%.6g")
            axis_y.setTickCount(6)
            axis_y.setRange(*self._padded_axis_range(ys))
            chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
            series.attachAxis(axis_y)
        for axis in chart.axes():
            axis.setLabelsColor(QColor("#9fb1c7"))
            axis.setTitleBrush(QColor("#9fb1c7"))
            axis.setGridLineColor(QColor("#23364e"))
        chart.legend().setLabelColor(QColor("#a9bad0"))
        if style == "scatter":
            chart.legend().hide()
        warning = ""
        x_span = max(xs) - min(xs)
        x_scale = max(abs(min(xs)), abs(max(xs)), 1.0)
        if x_span / x_scale < 1e-5:
            warning = "X 轴数据变化很小；折线可能接近竖直，建议使用散点图或更换 X 轴字段。"
        return chart, total, shown, warning

    @staticmethod
    def _padded_axis_range(
        values: list[float], *, include_zero: bool = False
    ) -> tuple[float, float]:
        low, high = min(values), max(values)
        if include_zero:
            low, high = min(0.0, low), max(0.0, high)
        span = high - low
        if span == 0:
            padding = max(abs(low) * 0.01, 1.0)
        else:
            padding = max(span * 0.08, max(abs(low), abs(high), 1.0) * 1e-9)
        return low - padding, high + padding

    def open_chart_window(self) -> None:
        if not self.headers or not self.rows:
            QMessageBox.information(self, "没有图表", "请先加载优化结果数据。")
            return
        dialog = QDialog(self, Qt.WindowType.Window)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("优化结果图表")
        dialog.resize(1180, 760)
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(16, 16, 16, 16)
        dialog_layout.setSpacing(10)
        title = QLabel(
            f"{self.style.currentText()} · {self.y_field.currentText()} / "
            f"{self.x_field.currentText()}"
        )
        title.setObjectName("CardTitle")
        dialog_layout.addWidget(title)

        if self.style.currentData() == "scatter3d":
            points, total = self._numeric_points_3d()
            if not points:
                dialog.deleteLater()
                QMessageBox.information(
                    self, "无法绘图", "所选 X、Y、Z 字段没有共同的数值数据。"
                )
                return
            scatter3d = Scatter3DCanvas(dialog)
            self._populate_3d_chart(scatter3d, points)
            scatter3d.setMinimumSize(800, 520)
            dialog_layout.addWidget(scatter3d, 1)
            detail = QLabel(f"显示 {len(points)} / {total} 个有效数据点；拖动旋转，滚轮缩放。")
        else:
            built = self._build_2d_chart()
            if built is None:
                dialog.deleteLater()
                QMessageBox.information(self, "无法绘图", "所选字段没有可绘制的数值。")
                return
            chart, total, shown, warning = built
            view = ZoomableChartView()
            view.setRenderHint(QPainter.RenderHint.Antialiasing)
            view.setChart(chart)
            dialog_layout.addWidget(view, 1)
            toolbar = QHBoxLayout()
            toolbar.addStretch(1)
            zoom_in = QPushButton("放大")
            zoom_out = QPushButton("缩小")
            zoom_reset = QPushButton("重置视图")
            zoom_in.clicked.connect(lambda: view.chart().zoom(1.25))
            zoom_out.clicked.connect(lambda: view.chart().zoom(0.8))
            zoom_reset.clicked.connect(view.chart().zoomReset)
            toolbar.addWidget(zoom_in)
            toolbar.addWidget(zoom_out)
            toolbar.addWidget(zoom_reset)
            dialog_layout.addLayout(toolbar)
            detail_text = f"显示 {shown} / {total} 个有效数据点；滚轮或框选可缩放。"
            detail = QLabel(f"{detail_text} {warning}".strip())
        detail.setObjectName("Caption")
        detail.setWordWrap(True)
        dialog_layout.addWidget(detail)
        self.chart_windows.append(dialog)
        dialog.destroyed.connect(
            lambda _object=None, window=dialog: self._forget_chart_window(window)
        )
        dialog.show()

    def _forget_chart_window(self, window: QDialog) -> None:
        if window in self.chart_windows:
            self.chart_windows.remove(window)

    def _update_3d_chart(self) -> None:
        points, total = self._numeric_points_3d()
        if not points:
            QMessageBox.information(self, "无法绘图", "所选 X、Y、Z 字段没有共同的数值数据。")
            return
        scatter3d = self._ensure_3d_chart()
        self._populate_3d_chart(scatter3d, points)
        self._set_plot_status(total, len(points))
        self.chart_stack.setCurrentWidget(self.scatter3d_container)

    def _populate_3d_chart(
        self, scatter3d: Scatter3DCanvas, points: list[tuple[float, float, float]]
    ) -> None:
        scatter3d.set_points(
            points,
            (
                self.x_field.currentText(),
                self.y_field.currentText(),
                self.z_field.currentText(),
            ),
        )

    def _set_plot_status(self, total: int, shown: int, warning: str = "") -> None:
        if shown < total:
            text = (
                f"共有 {total} 个有效数据点；为保持交互流畅，当前均匀抽样显示 {shown} 个。"
            )
        else:
            text = f"当前显示全部 {shown} 个有效数据点。"
        self.plot_status.setText(f"{text} {warning}".strip())

    def _ensure_3d_chart(self) -> Scatter3DCanvas:
        if self.scatter3d is None:
            self.scatter3d = Scatter3DCanvas(self.chart_stack)
            self.scatter3d_container = self.scatter3d
            self.scatter3d_container.setMinimumSize(450, 360)
            self.chart_stack.addWidget(self.scatter3d_container)
        return self.scatter3d

    def _base_chart(self, title: str) -> QChart:
        chart = QChart()
        chart.setTitle(title)
        chart.setTitleBrush(QColor("#dce7f5"))
        chart.setBackgroundBrush(QColor("#111d31"))
        chart.setPlotAreaBackgroundBrush(QColor("#0d1829"))
        chart.setPlotAreaBackgroundVisible(True)
        return chart

    def _empty_chart(self, text: str = "加载结果后即可绘图") -> None:
        chart = self._base_chart(text)
        chart.legend().hide()
        self.chart.setChart(chart)
        self.plot_status.setText(text)


class MainWindow(QMainWindow):
    connection_checked = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOBO · 锻造工艺优化工作台")
        self.resize(1440, 900)
        self.setMinimumSize(1120, 720)
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(max(4, QThreadPool.globalInstance().maxThreadCount()))
        self.network = QNetworkAccessManager(self)
        self.connection_reply: QNetworkReply | None = None
        self.connection_timeout_ms = 3000

        root = QWidget()
        root.setObjectName("Root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._top_bar())
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        content.addWidget(self._sidebar())
        self.stack = QStackedWidget()
        self.pages = [
            DashboardPage(self.pool),
            AutomationPage(self.pool, multi=False),
            AutomationPage(self.pool, multi=True),
            ModelPage(self.pool, self.api_client),
            OptimizationPage(self.pool, self.api_client),
            ResultsPage(self.pool, self.api_client),
        ]
        for page in self.pages:
            self.stack.addWidget(page)
        self.pages[0].navigate.connect(self.navigate)
        self.pages[1].result_ready.connect(self.open_result_file)
        self.pages[2].result_ready.connect(self.open_result_file)
        self.pages[4].result_loaded.connect(self.open_result_data)
        content.addWidget(self.stack, 1)
        outer.addLayout(content, 1)
        self.setCentralWidget(root)
        status = QStatusBar()
        status.showMessage("就绪")
        self.setStatusBar(status)
        self.nav_buttons[0].setChecked(True)
        QTimer.singleShot(250, self.check_connection)

    def _top_bar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("TopBar")
        frame.setFixedHeight(64)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 0, 20, 0)
        brand = QLabel("MOBO")
        brand.setStyleSheet("font-size:20px;font-weight:800;color:#edf9ff;letter-spacing:2px")
        product = QLabel("锻造工艺优化工作台")
        product.setObjectName("Subtitle")
        self.api_url = QLineEdit(os.environ.get("MOBO_API_URL", "http://127.0.0.1:5000"))
        self.api_url.setFixedWidth(235)
        self.api_url.setPlaceholderText("后端服务地址")
        self.connect_button = QPushButton("检测连接")
        self.connect_button.clicked.connect(self.check_connection)
        self.connection = QLabel()
        set_status(self.connection, "not_started")
        self.connection.setText("●  未检测")
        layout.addWidget(brand)
        layout.addSpacing(10)
        layout.addWidget(product)
        layout.addStretch()
        layout.addWidget(QLabel("API"))
        layout.addWidget(self.api_url)
        layout.addWidget(self.connect_button)
        layout.addWidget(self.connection)
        return frame

    def _sidebar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Sidebar")
        frame.setFixedWidth(215)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 18, 12, 18)
        layout.setSpacing(5)
        groups = [
            ("工作台", [
                (0, "工作台概览"),
                (1, "单工步计算"),
                (2, "多工步批处理"),
            ]),
            ("优化模块", [
                (3, "代理模型"),
                (4, "优化中心"),
                (5, "结果分析"),
            ]),
        ]
        self.nav_buttons = []
        self.nav_group_buttons: list[QToolButton] = []
        self.nav_group_contents: list[QWidget] = []
        for group_index, (group_name, entries) in enumerate(groups):
            if group_index:
                layout.addSpacing(8)
            group_button = QToolButton()
            group_button.setObjectName("NavGroup")
            group_button.setText(group_name)
            group_button.setCheckable(True)
            group_button.setChecked(True)
            group_button.setArrowType(Qt.ArrowType.DownArrow)
            group_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            group_content = QWidget()
            group_layout = QVBoxLayout(group_content)
            group_layout.setContentsMargins(8, 0, 0, 0)
            group_layout.setSpacing(4)
            for page_index, name in entries:
                button = QPushButton(name)
                button.setObjectName("Nav")
                button.setCheckable(True)
                button.clicked.connect(partial(self.navigate, page_index))
                group_layout.addWidget(button)
                self.nav_buttons.append(button)
            group_button.toggled.connect(
                partial(self._toggle_nav_group, group_button, group_content)
            )
            layout.addWidget(group_button)
            layout.addWidget(group_content)
            self.nav_group_buttons.append(group_button)
            self.nav_group_contents.append(group_content)
        layout.addStretch()
        version = QLabel("MOBO Desktop\n后端协议 v1")
        version.setObjectName("Caption")
        layout.addWidget(version)
        return frame

    @staticmethod
    def _toggle_nav_group(
        button: QToolButton, content: QWidget, expanded: bool
    ) -> None:
        content.setVisible(expanded)
        button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def api_client(self) -> ApiClient:
        return ApiClient(self.api_url.text().strip() or "http://127.0.0.1:5000")

    def navigate(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == index)
        if index == 3:
            self.pages[3].refresh_does()
        elif index == 4:
            self.pages[4].refresh_does()
        elif index == 5:
            self.pages[5].refresh_tasks()

    def check_connection(self) -> None:
        if self.connection_reply is not None and self.connection_reply.isRunning():
            return
        base_url = self.api_url.text().strip().rstrip("/")
        url = QUrl(f"{base_url}/health")
        if not base_url or not url.isValid() or url.scheme() not in {"http", "https"}:
            self._connection_result(False)
            self.statusBar().showMessage("请输入有效的 HTTP 或 HTTPS 后端地址", 5000)
            return
        set_status(self.connection, "queued")
        self.connection.setText("●  检测中")
        self.connect_button.setEnabled(False)
        request = QNetworkRequest(url)
        request.setTransferTimeout(self.connection_timeout_ms)
        request.setRawHeader(b"Accept", b"application/json")
        reply = self.network.get(request)
        self.connection_reply = reply
        reply.finished.connect(partial(self._connection_finished, reply))

    def _connection_finished(self, reply: QNetworkReply) -> None:
        if reply is not self.connection_reply:
            reply.deleteLater()
            return
        connected = reply.error() == QNetworkReply.NetworkError.NoError
        if connected:
            try:
                payload = json.loads(bytes(reply.readAll()).decode("utf-8"))
                connected = isinstance(payload, dict) and payload.get("code") == 0
            except (UnicodeDecodeError, ValueError):
                connected = False
        self.connection_reply = None
        reply.deleteLater()
        self._connection_result(connected)

    def _connection_result(self, connected: bool) -> None:
        self.connect_button.setEnabled(True)
        set_status(self.connection, "finished" if connected else "failed")
        self.connection.setText("●  已连接" if connected else "●  未连接")
        self.statusBar().showMessage("后端服务连接正常" if connected else "后端服务不可用；DEFORM 本地功能仍可使用", 5000)
        self.connection_checked.emit(connected)

    def open_result_file(self, path: str) -> None:
        if Path(path).is_file():
            headers, rows = read_tabular_file(path)
            self.open_result_data(headers, rows)

    def open_result_data(self, headers: list[str], rows: list[list[Any]]) -> None:
        self.pages[5].set_data(headers, rows)
        self.navigate(5)


def configure_application(app: QApplication) -> None:
    """Apply the shared palette, typeface and stylesheet to a QApplication."""
    app.setApplicationName("MOBO Desktop")
    app.setOrganizationName("MOBO")
    app.setStyle("Fusion")
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0b1220"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#dce7f5"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0d1829"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#dce7f5"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#26889a"))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLE)
    app.setFont(QFont("Microsoft YaHei UI", 10))


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    configure_application(app)
    window = MainWindow()
    window.show()
    return app.exec()


__all__ = ["MainWindow", "configure_application", "main"]
