"""
CONVIO - Reusable UI Components
================================

Reusable Qt widgets for result presentation.  No business logic here —
components receive pre-formatted strings from ``results_formatter``.

Components
----------
MetricsPanel        QGroupBox with a 2-column key/value grid.
SortableTable       QTableWidget with alternating rows + sort on header click.
TopologyResultWidget  QWidget that stacks a PlotWidget above a MetricsPanel.
"""

from typing import Any, Dict, List, Optional, Tuple

import pyqtgraph as pg

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QFrame, QGridLayout, QGroupBox, QHeaderView,
    QLabel, QScrollArea, QSizePolicy, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)


# ── Shared style constants ─────────────────────────────────────────────────────
_HEADER_BG  = QColor("#37474F")   # dark blue-grey
_HEADER_FG  = QColor("#FFFFFF")
_ALT_ROW    = QColor("#F5F7F8")   # very light grey
_BEST_ROW   = QColor("#E8F5E9")   # light green
_KEY_FONT   = QFont("Arial", 9)
_KEY_FONT.setBold(True)
_VAL_FONT   = QFont("Arial", 9)


class MetricsPanel(QGroupBox):
    """
    A labelled panel that displays key → value pairs in a 2-column grid.

    Usage::

        panel = MetricsPanel("Bus Topology Metrics")
        panel.update_metrics({"Total Length": "1234.56 mm", ...})
    """

    def __init__(self, title: str, parent: QWidget = None):
        super().__init__(title, parent)
        self._grid = QGridLayout(self)
        self._grid.setSpacing(4)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._labels: List[Tuple[QLabel, QLabel]] = []

    def update_metrics(self, metrics: Dict[str, str]) -> None:
        """Replace all rows with new key/value pairs."""
        # Clear existing widgets
        for key_lbl, val_lbl in self._labels:
            key_lbl.deleteLater()
            val_lbl.deleteLater()
        self._labels.clear()

        for row, (key, val) in enumerate(metrics.items()):
            key_lbl = QLabel(key)
            key_lbl.setFont(_KEY_FONT)
            key_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            val_lbl = QLabel(val)
            val_lbl.setFont(_VAL_FONT)
            val_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            val_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

            self._grid.addWidget(key_lbl, row, 0)
            self._grid.addWidget(val_lbl, row, 1)
            self._labels.append((key_lbl, val_lbl))

    def clear(self) -> None:
        self.update_metrics({})


class SortableTable(QTableWidget):
    """
    A read-only, sortable QTableWidget with alternating row colours and a
    styled header.

    Usage::

        t = SortableTable(["Linkage", "Wire (mm)", "CAN (mm)", "Total (mm)", "Saving"])
        t.populate([("complete", "1200.0", "300.0", "1500.0", "+12.3%"), ...],
                   best_row=0)   # optional: highlight row index
    """

    def __init__(self, headers: List[str], parent: QWidget = None):
        super().__init__(parent)
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setStyleSheet(
            f"QHeaderView::section {{"
            f"background-color: {_HEADER_BG.name()};"
            f"color: {_HEADER_FG.name()};"
            f"padding: 4px; border: none; font-weight: bold; }}"
        )
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.verticalHeader().setVisible(False)
        self.setStyleSheet("alternate-background-color: #F5F7F8;")

    def populate(
        self,
        rows: List[Tuple],
        best_row: Optional[int] = None,
        center_cols: Optional[List[int]] = None,
    ) -> None:
        """
        Fill the table with rows.

        Args:
            rows:        Each tuple maps 1-to-1 to the column headers.
            best_row:    If given, paint that row with _BEST_ROW background.
            center_cols: Column indices to centre-align (default: all but first).
        """
        self.setSortingEnabled(False)
        self.setRowCount(len(rows))

        if center_cols is None:
            center_cols = list(range(1, self.columnCount()))

        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                if c in center_cols:
                    item.setTextAlignment(Qt.AlignCenter)
                if r == best_row:
                    item.setBackground(_BEST_ROW)
                self.setItem(r, c, item)

        self.setSortingEnabled(True)
        self.resizeRowsToContents()

    def clear_data(self) -> None:
        self.setRowCount(0)


class MetricsTable(QTableWidget):
    """
    Two-column read-only table (Metric | Value) for displaying key/value
    pairs in a tabular layout instead of a free-form grid.

    Section headers (rows where ``value`` is empty) are rendered with a
    coloured background spanning both columns.
    """

    _HEADER_ROW_BG = QColor("#37474F")
    _HEADER_ROW_FG = QColor("#FFFFFF")

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Metric", "Value"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setStyleSheet(
            f"QHeaderView::section {{"
            f"background-color: {_HEADER_BG.name()};"
            f"color: {_HEADER_FG.name()};"
            f"padding: 4px; border: none; font-weight: bold; }}"
        )
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("alternate-background-color: #F5F7F8;")
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)

    def update_metrics(self, metrics: Dict[str, str]) -> None:
        """Replace all rows. Empty-value rows render as section headers."""
        self.setRowCount(len(metrics))
        bold = QFont(); bold.setBold(True)

        for r, (key, val) in enumerate(metrics.items()):
            # Section header (no value)
            if val == "" or val is None:
                header_item = QTableWidgetItem(key.strip(" —"))
                header_item.setBackground(self._HEADER_ROW_BG)
                header_item.setForeground(self._HEADER_ROW_FG)
                header_item.setFont(bold)
                header_item.setTextAlignment(Qt.AlignCenter)
                self.setItem(r, 0, header_item)
                # Span both columns
                self.setSpan(r, 0, 1, 2)
                continue

            key_item = QTableWidgetItem(key)
            val_item = QTableWidgetItem(val)
            # Sub-rows (indented in the formatter) get muted styling but
            # everything is centered for a consistent tabular look.
            if key.startswith("    "):
                key_item.setForeground(QColor("#666"))
            elif not key.startswith("  "):
                key_item.setFont(bold)
            key_item.setTextAlignment(Qt.AlignCenter)
            val_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(r, 0, key_item)
            self.setItem(r, 1, val_item)

        self.resizeRowsToContents()

    def clear_data(self) -> None:
        self.setRowCount(0)


class TopologyResultWidget(QWidget):
    """
    Composite widget: plot view on the **left**, metrics table on the **right**.

    Horizontal split (60/40) — the visualisation gets the wider portion, the
    metrics table sits next to it without needing the user to drag a divider.

    Outer code accesses ``widget.plot_view`` for visualisation and calls
    ``widget.metrics_panel.update_metrics(dict)`` to refresh the numbers.
    The attribute name ``metrics_panel`` is preserved for backward compat;
    the widget behind it is now a ``MetricsTable``.
    """

    def __init__(self, plot_title: str, metrics_title: str, parent: QWidget = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal, self)

        # Left: plot
        self.plot_view = pg.PlotWidget()
        splitter.addWidget(self.plot_view)

        # Right: titled metrics table inside a group box
        right = QGroupBox(metrics_title)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)
        self.metrics_panel = MetricsTable()
        right_layout.addWidget(self.metrics_panel)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 6)   # plot
        splitter.setStretchFactor(1, 4)   # metrics
        splitter.setSizes([900, 500])     # initial sizes

        outer.addWidget(splitter)
