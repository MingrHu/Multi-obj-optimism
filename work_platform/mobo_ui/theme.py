"""Global display themes shared by widgets and scientific charts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

from PySide6.QtCharts import QAbstractBarSeries, QChart, QXYSeries
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


DisplayMode = Literal["engineering", "publication"]


@dataclass(frozen=True)
class Theme:
    mode: DisplayMode
    window: str
    surface: str
    plot: str
    text: str
    muted: str
    accent: str
    accent_soft: str
    grid: str
    point_edge: str
    chart_colors: tuple[str, ...]


THEMES: dict[DisplayMode, Theme] = {
    "engineering": Theme(
        mode="engineering",
        window="#1e252d",
        surface="#29323d",
        plot="#222a34",
        text="#e4eaf2",
        muted="#a5b2c1",
        accent="#65afe8",
        accent_soft="#c0e2ff",
        grid="#394553",
        point_edge="#c0e2ff",
        chart_colors=("#65afe8", "#efb366", "#8fc57a", "#b69ad9", "#df7d81"),
    ),
    "publication": Theme(
        mode="publication",
        window="#edf1f5",
        surface="#ffffff",
        plot="#ffffff",
        text="#243447",
        muted="#627487",
        accent="#005b96",
        accent_soft="#d8e9f7",
        grid="#e4eaf0",
        point_edge="#ffffff",
        chart_colors=("#005b96", "#b01f24", "#238b75", "#d17c00", "#6f4c9b"),
    ),
}


# One control system for both palettes: density, borders and hierarchy stay identical.
def _workbench_style(*, dark: bool) -> str:
    colors = (
        dict(window='#1e252d', surface='#29323d', input='#222a34', raised='#303b48',
             border='#414e5d', subtle='#35414e', text='#e4eaf2', muted='#a5b2c1',
             accent='#65afe8', primary='#287ab7', hover='#364b60', selected='#304d68',
             sidebar='#222a34', disabled='#758393', danger='#f19b9b', grid='#35414e')
        if dark else
        dict(window='#edf1f5', surface='#ffffff', input='#ffffff', raised='#f4f7fa',
             border='#c5d0dc', subtle='#e2e8ef', text='#243447', muted='#627487',
             accent='#005b96', primary='#136da9', hover='#eaf3fb', selected='#e1eef9',
             sidebar='#f7f9fc', disabled='#93a0ae', danger='#b54348', grid='#edf1f5')
    )
    template = """
* { font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 13px; }
QWidget { color: @text@; }
QMainWindow, QWidget#Root, QDialog { background: @window@; }
QFrame { border: none; }
QFrame#TopBar { background: @surface@; border-bottom: 1px solid @border@; }
QFrame#Sidebar { background: @sidebar@; border-right: 1px solid @subtle@; }
QFrame#Card { background: @surface@; border: 1px solid @border@; border-radius: 5px; }
QFrame#StatCard { background: @surface@; border: 1px solid @subtle@; border-radius: 5px; }
QFrame#CardHeader { background: transparent; border-bottom: 1px solid @subtle@; }
QLabel { background: transparent; border: none; }
QLabel#Brand { font-size: 20px; font-weight: 700; color: @accent@; letter-spacing: 1px; }
QLabel#Title { font-size: 21px; font-weight: 600; color: @text@; }
QLabel#Subtitle { color: @muted@; font-size: 12px; }
QLabel#Section, QLabel#CardTitle { font-size: 14px; font-weight: 600; color: @text@; }
QLabel#StepBadge { color: @accent@; background: @selected@; border-radius: 3px;
  font-size: 12px; font-weight: 600; min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px; }
QLabel#Metric { font-size: 26px; font-weight: 600; color: @accent@; }
QLabel#StepNumber { color: @accent@; font-size: 17px; font-weight: 600; }
QLabel#Caption { color: @muted@; font-size: 11px; }
QPushButton, QToolButton { background: @surface@; border: 1px solid @border@;
  border-radius: 3px; color: @text@; padding: 5px 11px; min-height: 21px; }
QPushButton:hover, QToolButton:hover { background: @hover@; border-color: @accent@; }
QPushButton:pressed, QToolButton:pressed { background: @selected@; }
QPushButton:focus, QToolButton:focus { border-color: @accent@; }
QPushButton#Primary { background: @primary@; border-color: @primary@; color: #ffffff; font-weight: 600; }
QPushButton#Primary:hover { background: @accent@; border-color: @accent@; }
QPushButton#Danger { color: @danger@; background: transparent; }
QPushButton:disabled, QToolButton:disabled, QPushButton#Primary:disabled, QPushButton#Danger:disabled {
  background: @raised@; border-color: @subtle@; color: @disabled@; }
QPushButton#Nav { border: 0; border-left: 3px solid transparent; background: transparent;
  text-align: left; padding: 10px 12px; color: @muted@; border-radius: 3px; }
QPushButton#Nav:hover { background: @hover@; color: @text@; }
QPushButton#Nav:checked { background: @selected@; color: @accent@; border-left: 3px solid @accent@; font-weight: 600; }
QToolButton#NavGroup { border: none; background: transparent; color: @muted@;
  padding: 10px 5px 6px 5px; font-size: 12px; font-weight: 600; text-align: left; }
QToolButton#NavGroup:hover { color: @accent@; }
QToolButton#Help { border-radius: 10px; padding: 0; min-height: 20px; min-width: 20px;
  max-height: 20px; max-width: 20px; color: @muted@; background: transparent; border: 1px solid @border@; }
QToolButton#Help:hover { color: @accent@; background: @hover@; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
  background: @input@; border: 1px solid @border@; border-radius: 3px; padding: 5px 8px;
  selection-background-color: @selected@; selection-color: @text@; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 21px; }
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: @muted@; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
  border-color: @accent@; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
  color: @disabled@; background: @raised@; border-color: @subtle@; }
QComboBox { padding-right: 28px; }
QComboBox::drop-down { subcontrol-origin: border; subcontrol-position: top right; width: 24px;
  border-left: 1px solid @subtle@; background: @raised@; border-top-right-radius: 3px; border-bottom-right-radius: 3px; }
QComboBox::down-arrow { image: url(@DOWN@); width: 10px; height: 7px; }
QComboBox QAbstractItemView { background: @surface@; border: 1px solid @border@; padding: 3px;
  color: @text@; selection-background-color: @selected@; selection-color: @text@; outline: none; }
QSpinBox, QDoubleSpinBox { padding-right: 25px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
  width: 21px; background: @raised@; border-left: 1px solid @subtle@; border-bottom: 1px solid @subtle@; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right;
  width: 21px; background: @raised@; border-left: 1px solid @subtle@; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: @hover@; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(@UP@); width: 10px; height: 7px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(@DOWN@); width: 10px; height: 7px; }
QCheckBox, QRadioButton { spacing: 7px; padding: 3px 0; }
QCheckBox:disabled, QRadioButton:disabled { color: @disabled@; }
QCheckBox::indicator { width: 14px; height: 14px; background: @input@; border: 1px solid @border@; border-radius: 2px; }
QCheckBox::indicator:hover { border-color: @accent@; }
QCheckBox::indicator:checked { background: @primary@; border-color: @primary@; image: url(@CHECK@); }
QCheckBox::indicator:disabled { background: @raised@; border-color: @subtle@; }
QTableWidget, QTableView { background: @input@; border: 1px solid @border@; border-radius: 3px;
  padding: 0; gridline-color: @grid@; alternate-background-color: @raised@;
  selection-background-color: @selected@; selection-color: @text@; outline: none; }
QTableWidget::item, QTableView::item { padding: 3px 8px; }
QTableWidget::item:selected, QTableView::item:selected { background: @selected@; color: @text@; }
QTableWidget QComboBox { background: @input@; border: none; border-radius: 0; padding: 3px 6px; padding-right: 24px; }
QTableWidget QComboBox::drop-down { background: transparent; border-left: none; }
QHeaderView::section { background: @raised@; color: @muted@; border: none;
  border-right: 1px solid @subtle@; border-bottom: 1px solid @border@; padding: 7px 10px; font-weight: 600; }
QTableCornerButton::section { background: @raised@; border: none; }
QTabWidget::pane { background: @surface@; border: 1px solid @border@; border-radius: 3px; top: -1px; }
QTabBar::tab { background: @raised@; color: @muted@; padding: 8px 16px;
  border: none; border-bottom: 2px solid transparent; margin-right: 2px; }
QTabBar::tab:selected { background: @surface@; color: @accent@; border-bottom: 2px solid @accent@; font-weight: 600; }
QTabBar::tab:hover { background: @hover@; }
QTabBar QToolButton { padding: 2px; min-width: 18px; min-height: 18px; }
QProgressBar { background: @raised@; border: none; border-radius: 3px; height: 7px; text-align: center; color: transparent; }
QProgressBar::chunk { background: @primary@; border-radius: 3px; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle { background: @border@; border-radius: 4px; min-height: 28px; min-width: 28px; }
QScrollBar::handle:hover { background: @muted@; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QSplitter::handle { background: @subtle@; }
QStatusBar { background: @surface@; color: @muted@; border-top: 1px solid @subtle@; font-size: 11px; }
QStatusBar::item { border: none; }
QToolTip { background: @surface@; color: @text@; border: 1px solid @border@; padding: 6px; }
QMenu { background: @surface@; border: 1px solid @border@; padding: 4px; }
QMenu::item { padding: 6px 24px; }
QMenu::item:selected { background: @selected@; }
"""
    for name, color in colors.items():
        template = template.replace(f"@{name}@", color)
    template = template.replace("@UP@", "@SPIN_UP_LIGHT@" if dark else "@SPIN_UP@")
    return template.replace("@DOWN@", "@SPIN_DOWN_LIGHT@" if dark else "@SPIN_DOWN@")


ENGINEERING_STYLE = _workbench_style(dark=True)
PUBLICATION_STYLE = _workbench_style(dark=False)


def normalize_mode(value: object) -> DisplayMode:
    return "publication" if value == "publication" else "engineering"


def current_mode() -> DisplayMode:
    app = QApplication.instance()
    return normalize_mode(app.property("displayMode") if app is not None else None)


def current_theme() -> Theme:
    return THEMES[current_mode()]


def apply_application_theme(app: QApplication, mode: DisplayMode) -> Theme:
    mode = normalize_mode(mode)
    theme = THEMES[mode]
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(theme.window))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(theme.plot))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme.window))
    palette.setColor(QPalette.ColorRole.Text, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(theme.surface))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(theme.accent))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    for role in (QPalette.ColorRole.Light, QPalette.ColorRole.Midlight,
                 QPalette.ColorRole.Mid, QPalette.ColorRole.Dark):
        palette.setColor(role, QColor(theme.grid))
    app.setProperty("displayMode", mode)
    app.setPalette(palette)
    if mode == "publication":
        stylesheet = PUBLICATION_STYLE
        stylesheet = stylesheet.replace("@SPIN_UP@", ui_asset_path("spin_up.svg"))
        stylesheet = stylesheet.replace("@SPIN_DOWN@", ui_asset_path("spin_down.svg"))
    else:
        stylesheet = ENGINEERING_STYLE
        stylesheet = stylesheet.replace(
            "@SPIN_UP_LIGHT@", ui_asset_path("spin_up_light.svg")
        )
        stylesheet = stylesheet.replace(
            "@SPIN_DOWN_LIGHT@", ui_asset_path("spin_down_light.svg")
        )
    stylesheet = stylesheet.replace("@CHECK@", ui_asset_path("check.svg"))
    app.setStyleSheet(stylesheet)
    return theme


def ui_asset_path(name: str) -> str:
    return str(files(__package__).joinpath("assets", name)).replace("\\", "/")


def style_chart(chart: QChart, *, hide_legend: bool = False) -> Theme:
    theme = current_theme()
    chart.setBackgroundBrush(QColor(theme.surface))
    chart.setBackgroundVisible(False)
    chart.setPlotAreaBackgroundBrush(QColor(theme.plot))
    chart.setPlotAreaBackgroundVisible(True)
    chart.setTitleBrush(QColor(theme.text))
    for axis in chart.axes():
        axis.setLabelsColor(QColor(theme.muted))
        axis.setTitleBrush(QColor(theme.muted))
        axis.setGridLineColor(QColor(theme.grid))
    chart.legend().setLabelColor(QColor(theme.muted))
    for index, series in enumerate(chart.series()):
        color = QColor(theme.chart_colors[index % len(theme.chart_colors)])
        if isinstance(series, QAbstractBarSeries):
            for bar_set in series.barSets():
                bar_set.setColor(color)
                bar_set.setBorderColor(QColor(theme.point_edge))
                bar_set.setLabelColor(QColor(theme.text))
        elif isinstance(series, QXYSeries):
            series.setColor(color)
            if hasattr(series, "setBorderColor"):
                series.setBorderColor(QColor(theme.point_edge))
    if hide_legend:
        chart.legend().hide()
    return theme


def status_colors(tone: str) -> tuple[str, str]:
    if current_mode() == "publication":
        colors = {
            "neutral": ("#e6ebef", "#455764"),
            "info": ("#dcecf5", "#005b96"),
            "warning": ("#fff1cf", "#805700"),
            "success": ("#dcefe8", "#1d6a50"),
            "danger": ("#f6dfe1", "#a51d25"),
        }
    else:
        colors = {
            "neutral": ("#24344b", "#adbed2"),
            "info": ("#173d55", "#73d9e1"),
            "warning": ("#503e1c", "#ffd47c"),
            "success": ("#153e35", "#75e1bd"),
            "danger": ("#4d262d", "#ffaaa8"),
        }
    return colors[tone]


__all__ = [
    "DisplayMode",
    "Theme",
    "THEMES",
    "apply_application_theme",
    "current_mode",
    "current_theme",
    "normalize_mode",
    "status_colors",
    "style_chart",
    "ui_asset_path",
]
