from typing import Tuple

import pyqtgraph as pg

LEGEND_PRIORITY_PREFIXES = (
    "Chassis",
    "Chassis junction points",
    "High Performance Computer nodes",
    "Input/Output nodes",
    "Direct HPC Wiring",
    "Cluster",
    "Bus Path",
    "Redundant",
    "Star",
    "Ring",
)

def reorder(view: pg.PlotWidget) -> None:
    """
    Rebuild the PyQtGraph legend in a deterministic order without changing
    draw order / visual stacking.
    """
    plot_item = view.getPlotItem()
    legend = getattr(plot_item, "legend", None)

    if legend is None or not getattr(legend, "items", None):
        return

    entries = []
    for index, entry in enumerate(list(legend.items)):
        sample, label = entry
        item = getattr(sample, "item", None)
        name = _label_text(label)

        if item is None or not name:
            continue

        entries.append((index, name, item))

    if not entries:
        return

    legend.clear()

    for _index, name, item in sorted(entries, key=_sort_key):
        legend.addItem(item, name)


def _sort_key(entry) -> Tuple[int, int]:
    original_index, name, _item = entry

    for priority, prefix in enumerate(LEGEND_PRIORITY_PREFIXES):
        if prefix in name:
            return priority, original_index

    return len(LEGEND_PRIORITY_PREFIXES), original_index


def _label_text(label) -> str:
    """Extract text from a PyQtGraph legend LabelItem."""
    text = getattr(label, "text", None)

    if callable(text):
        try:
            return str(text())
        except TypeError:
            pass

    if text is not None:
        return str(text)

    item = getattr(label, "item", None)
    if item is not None and hasattr(item, "toPlainText"):
        return str(item.toPlainText())

    return ""