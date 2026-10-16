"""Small reusable interaction helpers for the desktop UI."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QMessageBox, QTabBar, QTabWidget, QToolButton, QWidget


class HelpButton(QToolButton):
    """Compact question-mark button that opens contextual help."""

    def __init__(
        self, title: str, message: str, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.help_title = title
        self.help_message = message
        self.setObjectName("Help")
        self.setText("?")
        self.setFixedSize(24, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"查看“{title}”说明")
        self.setAccessibleName(f"{title}说明")
        self.clicked.connect(self._show_help)

    def _show_help(self) -> None:
        QMessageBox.information(self.window(), self.help_title, self.help_message)


class ScrollableTabBar(QTabBar):
    """Tab bar with arrows plus wheel navigation for long tab collections."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.count() <= 1:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().x() or event.angleDelta().y()
        if delta:
            step = -1 if delta > 0 else 1
            self.setCurrentIndex((self.currentIndex() + step) % self.count())
            event.accept()
            return
        super().wheelEvent(event)


def configure_scrollable_tabs(tabs: QTabWidget) -> QTabWidget:
    """Keep tabs reachable when labels exceed the available width."""
    bar = ScrollableTabBar(tabs)
    bar.setUsesScrollButtons(True)
    bar.setExpanding(False)
    bar.setElideMode(Qt.TextElideMode.ElideRight)
    bar.setAccessibleName("可滚动标签页")
    tabs.setTabBar(bar)
    tabs.setDocumentMode(True)
    return tabs


__all__ = ["HelpButton", "ScrollableTabBar", "configure_scrollable_tabs"]
