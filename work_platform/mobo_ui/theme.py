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
        window="#08111f",
        surface="#132139",
        plot="#0d1829",
        text="#dce7f5",
        muted="#9fb1c7",
        accent="#4dc9d3",
        accent_soft="#b8f5f7",
        grid="#23364e",
        point_edge="#b8f5f7",
        chart_colors=("#4dc9d3", "#efb366", "#8fc57a", "#b69ad9", "#df7d81"),
    ),
    "publication": Theme(
        mode="publication",
        window="#e4e6e8",
        surface="#f2f3f4",
        plot="#ffffff",
        text="#172433",
        muted="#5c6b78",
        accent="#005b96",
        accent_soft="#a9c8dc",
        grid="#d7dde3",
        point_edge="#ffffff",
        chart_colors=("#005b96", "#b01f24", "#238b75", "#d17c00", "#6f4c9b"),
    ),
}


ENGINEERING_STYLE = """
* { font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 14px; }
QMainWindow, QWidget#Root { background: #08111f; color: #dce7f5; }
QDialog { background: #08111f; color: #dce7f5; }
QFrame#TopBar { background: #101a2d; border-bottom: 1px solid #24344d; }
QFrame#Sidebar { background: #0d1727; border-right: 1px solid #24344d; }
QFrame#Card { background: #132139; border: 1px solid #2b405d; border-radius: 6px; }
QFrame#StatCard { background: #162843; border: 1px solid #31506f; border-radius: 6px; }
QLabel#Brand { font-size: 20px; font-weight: 800; color: #edf9ff; letter-spacing: 2px; }
QLabel#Title { font-size: 26px; font-weight: 700; color: #f4f8ff; }
QLabel#Subtitle { color: #8fa3bd; font-size: 13px; }
QLabel#Section, QLabel#CardTitle { font-size: 17px; font-weight: 600; color: #eff6ff; }
QLabel#Metric { font-size: 25px; font-weight: 700; color: #69d6df; }
QLabel#StepNumber { color: #5ed1da; font-size: 19px; font-weight: 700; }
QLabel#Caption { color: #93a6bf; font-size: 12px; }
QLabel#StatusPill { padding: 5px 10px; border-radius: 9px; font-weight: 600; }
QPushButton, QToolButton { background: #172740; border: 1px solid #304864; border-radius: 5px;
  color: #e7f0fb; padding: 7px 13px; min-height: 20px; }
QPushButton:hover, QToolButton:hover { background: #1d3352; border-color: #42b9c7; }
QPushButton:pressed { background: #132239; }
QPushButton:disabled { color: #7f91a7; background: #111a29; border-color: #27384d; }
QPushButton#Primary { background: #1b8190; border-color: #35a9b6; color: white; font-weight: 600; }
QPushButton#Primary:hover { background: #2397a5; }
QPushButton#Danger { color: #ffb6b3; border-color: #7f4448; }
QPushButton#Nav { border: 0; background: transparent; text-align: left; padding: 11px 15px;
  color: #a9bad0; border-radius: 5px; }
QPushButton#Nav:hover { background: #14243b; color: #f6fbff; }
QPushButton#Nav:checked { background: #17364d; color: #71dbe2; border-left: 3px solid #5bd0d8; }
QToolButton#NavGroup { border: 0; background: transparent; color: #edf5ff; padding: 9px 8px;
  font-weight: 700; text-align: left; }
QToolButton#NavGroup:hover { background: #14243b; color: #71dbe2; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget, QTableView { background: #0d1829; color: #deebf8;
  border: 1px solid #2c4059; border-radius: 4px; padding: 6px; selection-background-color: #256d82; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus { border-color: #43bdc8; }
QComboBox { padding-right: 25px; }
QComboBox::drop-down { subcontrol-origin: border; subcontrol-position: top right; width: 23px;
  background: #172740; border-left: 1px solid #304864; }
QComboBox::down-arrow { image: url(@SPIN_DOWN_LIGHT@); width: 10px; height: 7px; }
QSpinBox, QDoubleSpinBox { padding-right: 24px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
  width: 21px; background: #172740; border-left: 1px solid #304864; border-bottom: 1px solid #304864; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right;
  width: 21px; background: #172740; border-left: 1px solid #304864; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #23506a; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(@SPIN_UP_LIGHT@); width: 10px; height: 7px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(@SPIN_DOWN_LIGHT@); width: 10px; height: 7px; }
QComboBox QAbstractItemView { background: #202b3a; color: #d8e0e9; selection-background-color: #586575; selection-color: #ffffff; }
QComboBox#TaskSelector { background: #1b2635; color: #d8e0e9; border-color: #465466; }
QHeaderView::section { background: #17263b; color: #9fb2c9; border: 0; border-right: 1px solid #263a53;
  border-bottom: 1px solid #263a53; padding: 8px; font-weight: 600; }
QTableWidget, QTableView { gridline-color: #21334a; alternate-background-color: #101d30; }
QProgressBar { background: #0d1828; border: 1px solid #2a4059; border-radius: 5px; height: 10px;
  text-align: center; color: transparent; }
QProgressBar::chunk { background: #4ac7d1; border-radius: 4px; }
QTabWidget::pane { border: 1px solid #263a52; border-radius: 5px; top: -1px; }
QTabBar::tab { background: #111d30; color: #91a5bd; padding: 9px 18px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #62d4dc; border-bottom-color: #62d4dc; }
QScrollBar:vertical { background: #0d1725; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #2a425d; border-radius: 5px; min-height: 30px; }
QStatusBar { background: #0e1727; color: #91a3b8; }
"""


PUBLICATION_STYLE = """
* { font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 14px; }
QMainWindow, QWidget#Root, QDialog { background: #e4e6e8; color: #172433; }
QFrame#TopBar { background: #f0f1f2; border-bottom: 2px solid #005b96; }
QFrame#Sidebar { background: #e9ebed; border-right: 1px solid #aeb5bb; }
QFrame#Card { background: #f2f3f4; border: 1px solid #aeb5bb; border-radius: 6px; }
QFrame#StatCard { background: #eceeef; border: 1px solid #b5bcc2; border-radius: 6px; }
QLabel#Brand { font-size: 20px; font-weight: 800; color: #005b96; letter-spacing: 2px; }
QLabel#Title { font-size: 26px; font-weight: 700; color: #122435; }
QLabel#Subtitle { color: #5c6b78; font-size: 13px; }
QLabel#Section, QLabel#CardTitle { font-size: 17px; font-weight: 600; color: #172433; }
QLabel#Metric { font-size: 25px; font-weight: 700; color: #005b96; }
QLabel#StepNumber { color: #b01f24; font-size: 19px; font-weight: 700; }
QLabel#Caption { color: #61707d; font-size: 12px; }
QLabel#StatusPill { padding: 5px 10px; border-radius: 9px; font-weight: 600; }
QPushButton, QToolButton { background: #e1e3e5; border: 1px solid #858f97; border-radius: 5px;
  color: #203040; padding: 7px 13px; min-height: 20px; }
QPushButton:hover, QToolButton:hover { background: #d5e5ef; border-color: #005b96; }
QPushButton:pressed { background: #c9d8e1; border-color: #005b96; }
QPushButton:disabled { color: #91989e; background: #d8dadd; border-color: #b9bec2; }
QPushButton#Primary { background: #005b96; border-color: #005b96; color: white; font-weight: 600; }
QPushButton#Primary:hover { background: #004876; }
QPushButton#Danger { color: #a51d25; border-color: #bf777c; }
QPushButton#Nav { border: 0; background: transparent; text-align: left; padding: 11px 15px;
  color: #465766; border-radius: 6px; }
QPushButton#Nav:hover { background: #dce5eb; color: #005b96; }
QPushButton#Nav:checked { background: #dce8ef; color: #004b7a; border-left: 3px solid #005b96; font-weight: 600; }
QToolButton#NavGroup { border: 0; background: transparent; color: #172433; padding: 9px 8px;
  font-weight: 700; text-align: left; }
QToolButton#NavGroup:hover { background: #dce5eb; color: #005b96; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget, QTableView { background: #fbfbfb; color: #172433;
  border: 1px solid #7f8992; border-radius: 4px; padding: 6px; selection-background-color: #bad6e7; selection-color: #172433; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus { border: 1px solid #0078d4; }
QComboBox { padding-right: 25px; }
QComboBox::drop-down { subcontrol-origin: border; subcontrol-position: top right; width: 23px;
  background: #dfe2e4; border-left: 1px solid #8b949b; }
QComboBox::down-arrow { image: url(@SPIN_DOWN@); width: 10px; height: 7px; }
QSpinBox, QDoubleSpinBox { padding-right: 24px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
  width: 21px; background: #dfe2e4; border-left: 1px solid #7f8992; border-bottom: 1px solid #a7adb2; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right;
  width: 21px; background: #dfe2e4; border-left: 1px solid #7f8992; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #c8dce8; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(@SPIN_UP@); width: 10px; height: 7px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(@SPIN_DOWN@); width: 10px; height: 7px; }
QComboBox QAbstractItemView { background: #ffffff; color: #172433; selection-background-color: #d8e8f2; selection-color: #172433; }
QComboBox#TaskSelector { background: #e5e7e9; color: #172433; border-color: #858f97; }
QHeaderView::section { background: #d9dde0; color: #243746; border: 0; border-right: 1px solid #adb5bc;
  border-bottom: 1px solid #9fa8af; padding: 8px; font-weight: 600; }
QTableWidget, QTableView { gridline-color: #c1c7cc; alternate-background-color: #eef0f1; }
QProgressBar { background: #d5d8da; border: 1px solid #969fa6; border-radius: 4px; height: 10px;
  text-align: center; color: transparent; }
QProgressBar::chunk { background: #005b96; border-radius: 3px; }
QTabWidget::pane { border: 1px solid #aeb5bb; border-radius: 5px; top: -1px; background: #f2f3f4; }
QTabBar::tab { background: #d8dbde; color: #43515d; padding: 8px 16px; border: 1px solid #aeb5bb; }
QTabBar::tab:selected { color: #005b96; border-top: 2px solid #005b96; background: #f2f3f4; }
QScrollBar:vertical { background: #d9dcde; width: 13px; margin: 0; border-left: 1px solid #bcc2c7; }
QScrollBar::handle:vertical { background: #9ea7ae; border: 2px solid #d9dcde; border-radius: 4px; min-height: 30px; }
QStatusBar { background: #e9ebed; color: #536574; border-top: 1px solid #aeb5bb; }
"""


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
