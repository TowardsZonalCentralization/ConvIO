"""
CONVIO - Dialog Windows
=======================

Modal dialogs shown on top of the main window.
"""

from typing import Any, Dict

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHeaderView, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)


class CostSettingsDialog(QDialog):
    """
    Modal dialog for editing cost and weight parameters at runtime.

    Reads from and writes back to a cost_cfg dict with the keys:
        currency             (str)
        wire_price_per_m     (float)
        CAN_bus_price_per_m  (float)
        wire_weight_per_m_kg (float)

    The dialog itself does not persist changes to disk — the caller is
    responsible for re-running any analysis that should reflect the new values.
    """

    def __init__(self, cost_cfg: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cost & Weight Settings")
        self.setMinimumWidth(360)
        self._initial_cfg = dict(cost_cfg)

        layout = QVBoxLayout(self)

        # ─── Cost section ─────────────────────────────────────────────
        cost_group = QGroupBox("Cost (per metre)")
        cost_form  = QFormLayout(cost_group)

        self.currency_edit = QLineEdit(str(cost_cfg.get("currency", "EURO")))
        self.currency_edit.setMaxLength(8)
        cost_form.addRow("Currency:", self.currency_edit)

        self.wire_price = self._make_spin(cost_cfg.get("wire_price_per_m", 0.0),
                                           suffix=" / m", decimals=3, maximum=1000.0)
        cost_form.addRow("Wire Price:", self.wire_price)

        self.can_price = self._make_spin(cost_cfg.get("CAN_bus_price_per_m", 0.0),
                                          suffix=" / m", decimals=3, maximum=1000.0)
        cost_form.addRow("CAN/Network Bus Price:", self.can_price)
        layout.addWidget(cost_group)

        # ─── Weight section ───────────────────────────────────────────
        weight_group = QGroupBox("Weight")
        weight_form  = QFormLayout(weight_group)
        self.wire_weight = self._make_spin(cost_cfg.get("wire_weight_per_m_kg", 0.0),
                                            suffix=" kg / m", decimals=4, maximum=10.0)
        weight_form.addRow("Wire Weight:", self.wire_weight)
        layout.addWidget(weight_group)

        # ─── Note ─────────────────────────────────────────────────────
        note = QLabel(
            "<i>Changes are applied immediately to all displayed metrics. "
            "Re-run Step 5 to refresh topology results.</i>"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555; font-size: 10px;")
        layout.addWidget(note)

        # ─── OK / Cancel ──────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _make_spin(value: float, suffix: str = "", decimals: int = 3,
                    maximum: float = 100.0) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setMinimum(0.0)
        spin.setMaximum(maximum)
        spin.setSingleStep(10 ** -min(decimals, 2))
        spin.setSuffix(suffix)
        spin.setValue(float(value))
        return spin

    def get_settings(self) -> Dict[str, Any]:
        """Return the edited values as a cost_cfg-shaped dict."""
        return {
            "currency":             self.currency_edit.text().strip() or "EURO",
            "wire_price_per_m":     float(self.wire_price.value()),
            "CAN_bus_price_per_m":  float(self.can_price.value()),
            "wire_weight_per_m_kg": float(self.wire_weight.value()),
        }


class ComparisonWindow(QDialog):
    """
    A dialog window to display the comparison of clustering linkage methods.

    Used after a "Run Full Analysis" run: shows the wiring length for each
    linkage method and highlights the best one.
    """

    _LINKAGE_METHODS = ["average", "complete", "single"]
    _COLUMNS = ["Linkage Method", "I/O Wiring (mm)", "CAN Bus (mm)", "Total Length (mm)"]
    _BEST_ROW_BG = QColor("#d4edda")

    def __init__(self, data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Linkage Method Comparison")
        self.setMinimumSize(600, 200)

        layout = QVBoxLayout(self)

        title = QLabel("<b>Comprehensive Analysis Results</b>")
        title.setFont(QFont("Arial", 14))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setRowCount(len(self._LINKAGE_METHODS))
        self.table.setColumnCount(len(self._COLUMNS))
        self.table.setHorizontalHeaderLabels(self._COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        self._populate_table(data)

        best_method = data.get("best_method", "N/A")
        recommendation = QLabel(
            f"<b>Recommendation:</b> The '<b>{best_method}</b>' linkage method produced "
            f"the shortest overall wiring harness length and has been selected for "
            f"detailed visualization."
        )
        recommendation.setWordWrap(True)
        layout.addWidget(recommendation)

    # ------------------------------------------------------------------
    def _populate_table(self, data: Dict[str, Any]):
        results = data.get("results", {})
        best_method = data.get("best_method")

        for row, method in enumerate(self._LINKAGE_METHODS):
            res = results.get(method, {})

            wire_len = res.get("total_wire_length", 0.0)
            can_len = res.get("can_bus", {}).get("total_length", 0.0)
            total_len = res.get("overall_wiring_harness_length", 0.0)

            items = [
                QTableWidgetItem(method),
                QTableWidgetItem(f"{wire_len:.2f}"),
                QTableWidgetItem(f"{can_len:.2f}"),
                QTableWidgetItem(f"{total_len:.2f}"),
            ]

            # Bold + green-tinted background for the row corresponding to the
            # best method found in the sweep.
            if method == best_method:
                bold_font = QFont()
                bold_font.setBold(True)
                for item in items:
                    item.setFont(bold_font)
                    item.setBackground(self._BEST_ROW_BG)

            for col, item in enumerate(items):
                self.table.setItem(row, col, item)
