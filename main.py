"""
CONVIO - Main Entry Point
=========================

Thin entry point. Creates the :class:`QApplication`, loads configuration,
builds the main window, and starts the Qt event loop.

Implementation is split across:
    - ``config_manager.py``  : :class:`ConfigManager` (no Qt deps).
    - ``gui/workers.py``     : background :class:`OptimizationWorker` thread.
    - ``gui/dialogs.py``     : :class:`ComparisonWindow`.
    - ``gui/widgets.py``     : :class:`MatplotlibWidget`.
    - ``gui/visualization.py`` : :class:`VisualizationMixin` (drawing + export).
    - ``gui/main_window.py`` : :class:`WiringHarnessOptimizer` (the main window).
    - ``modules/``           : analysis, reporting, graph utilities.
"""

import sys

# QApplication must be instantiated before ANY QWidget subclass is imported.
# Keeping this here (rather than inside ``main()``) preserves the original
# behaviour: some Qt platform plugins allocate globals at import time.
from PyQt5.QtWidgets import QApplication, QMessageBox

app = QApplication(sys.argv)

from config_manager import ConfigManager  # noqa: E402
from gui.main_window import WiringHarnessOptimizer  # noqa: E402


def main():
    """Application entry point."""
    try:
        config_manager = ConfigManager()
        window = WiringHarnessOptimizer(config_manager)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "Startup Error", f"Failed to start application:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
