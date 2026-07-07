"""
CONVIO - Reusable GUI Widgets
=============================

Small, reusable QWidget subclasses used by the main window.
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QVBoxLayout, QWidget


class MatplotlibWidget(QWidget):
    """A thin wrapper around a Matplotlib :class:`FigureCanvas` for Qt embedding."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
