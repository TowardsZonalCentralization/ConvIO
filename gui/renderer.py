"""
CONVIO - Graph Renderer
=======================

Stateless, pure-function drawing layer.  Every visualizer in the application
calls this module so node/edge styling is enforced from one place.

Design rules
------------
- No ``self`` / no class — pass what you need as arguments.
- ``config_dict`` is the raw ``ConfigManager.config`` dict.
- Callers are responsible for ``view.clear()`` and tab switching.

Visual standards
----------------
  Chassis edges  : black, width=1, alpha=0.2  (background context)
  Chassis nodes  : styled per config, alpha=0.3
  I/O nodes      : cluster color, size=7, circle, thin black border
  Aggregators    : cluster color, size=14, square, 1.5px black border + label
  Wiring paths   : cluster color, width=2.5, DashLine
  Bus path       : #2962FF (blue), solid, width=3
  Redundant path : #D32F2F (red),  dashed, width=2
  Star paths     : #1565C0 (dark blue), solid, width=2.5
  Ring paths     : #E65100 (orange), dashed, width=2
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt


# ── Palette constants ──────────────────────────────────────────────────────────
BUS_PEN       = pg.mkPen("#2962FF", width=2)
REDUNDANT_PEN = pg.mkPen("#D32F2F", width=2, style=Qt.DashLine)
STAR_PEN      = pg.mkPen("#1565C0", width=2.5)
RING_PEN      = pg.mkPen("#E65100", width=2, style=Qt.DashLine)
WIRING_WIDTH  = 2.5
IO_SIZE       = 7
AGG_SIZE      = 14


# ── View setup ─────────────────────────────────────────────────────────────────

def setup_view(view: pg.PlotWidget, title: str) -> None:
    """Apply the standard pyqtgraph settings to any plot view."""
    view.setBackground("w")
    view.setLabel("left", "Y (mm)")
    view.setLabel("bottom", "X (mm)")
    view.setTitle(title)
    view.showGrid(x=True, y=True)
    view.setAspectLocked(True, ratio=1.0)
    view.addLegend()


def reset_view(view: pg.PlotWidget) -> None:
    """Clear content and legend."""
    view.clear()
    view.addLegend().clear()


# ── Chassis background ─────────────────────────────────────────────────────────

def draw_base_graph(
    view: pg.PlotWidget,
    graph,
    pos: Dict[str, Tuple[float, float]],
    config_dict: Dict[str, Any],
    edge_alpha: float = 0.5,
    node_alpha: float = 0.3,
    include_io: bool = False,
) -> None:
    """
    Draw the chassis frame as a faded backdrop.

    Args:
        include_io: If True, I/O nodes are drawn too.  Default is False —
                    callers draw I/O nodes in a separate, coloured pass.
    """
    if not include_io:
        chassis_nodes = [n for n, d in graph.nodes(data=True) if not d.get("is_io")]
        subg = graph.subgraph(chassis_nodes)
    else:
        subg = graph

    _draw_edges(view, subg, pos, alpha=1, name="Chassis")
    _draw_nodes(view, subg, pos, config_dict, alpha=node_alpha)


def draw_io_nodes(
    view: pg.PlotWidget,
    graph,
    pos: Dict[str, Tuple[float, float]],
    config_dict: Dict[str, Any],
    alpha: float = 1.0,
) -> None:
    """
    Draw only I/O nodes on top of a pre-drawn chassis (e.g. baseline wiring view).

    Uses the ``io`` style from config so appearance matches every other tab.
    """
    io_only = graph.subgraph([n for n, d in graph.nodes(data=True) if d.get("is_io")])
    _draw_nodes(view, io_only, pos, config_dict, alpha=alpha)


# ── Cluster layer ──────────────────────────────────────────────────────────────

def draw_cluster_layer(
    view: pg.PlotWidget,
    clusters: Dict[str, Any],
    pos: Dict[str, Tuple[float, float]],
    colors: List,
    draw_wiring: bool = True,
) -> None:
    """
    Draw I/O nodes, wiring paths, and aggregator markers for each cluster.

    Args:
        draw_wiring: If False, only I/O nodes and aggregators are drawn
                     (used for the Step 3 raw-assignment view).
    """
    legend_added: set = set()

    for cluster_id, cluster_data in clusters.items():
        is_can_bus = cluster_id == "can_bus"
        cluster_num = _cluster_number(cluster_id)

        # Skip non-cluster entries except can_bus.
        if cluster_num is None and not is_can_bus:
            continue

        color = _cluster_color(cluster_id, colors)

        # I/O nodes (coloured scatter)
        io_pts = [pos[n] for n in cluster_data.get("io_nodes", []) if n in pos]
        if io_pts:
            arr = np.array(io_pts)
            view.addItem(pg.ScatterPlotItem(
                arr[:, 0], arr[:, 1],
                size=IO_SIZE, symbol="o",
                brush=pg.mkBrush(color),
                pen=pg.mkPen("k", width=0.5),
            ))

        if draw_wiring and not is_can_bus:
            _draw_wiring_paths(view, cluster_data, pos, color, cluster_num, legend_added)
            _draw_aggregator(view, cluster_id, cluster_data, color)

        if draw_wiring and is_can_bus:
            _draw_wiring_paths(view, cluster_data, pos, color, None, legend_added)


def draw_cluster_io_only(
    view: pg.PlotWidget,
    clusters: Dict[str, Any],
    pos: Dict[str, Tuple[float, float]],
    colors: List,
) -> None:
    """Step 3 – colour I/O nodes by cluster; no paths, no centroids."""
    for cluster_id, cluster_data in clusters.items():
        cluster_num = _cluster_number(cluster_id)
        if cluster_num is None:
            continue

        color = _cluster_color(cluster_id, colors)
        io_nodes = cluster_data.get("io_nodes", [])
        pts = [pos[n] for n in io_nodes if n in pos]

        if not pts:
            continue

        arr = np.array(pts)
        view.addItem(pg.ScatterPlotItem(
            arr[:, 0], arr[:, 1],
            size=10, symbol="o",
            brush=pg.mkBrush(color),
            pen=pg.mkPen("k", width=0.5),
            name=f"Cluster {cluster_num}  ({len(io_nodes)} I/O)",
        ))


# ── Network path helpers ───────────────────────────────────────────────────────

def draw_path(
    view: pg.PlotWidget,
    path_nodes: List[str],
    pos: Dict[str, Tuple[float, float]],
    pen,
    legend_label: Optional[str] = None,
) -> None:
    """Draw a sequence of nodes as a connected line."""
    xs = [pos[n][0] for n in path_nodes if n in pos]
    ys = [pos[n][1] for n in path_nodes if n in pos]
    if len(xs) > 1:
        view.addItem(pg.PlotCurveItem(xs, ys, pen=pen, name=legend_label))


def draw_bus_topology(
    view: pg.PlotWidget,
    clustering_results: Dict[str, Any],
    pos: Dict[str, Tuple[float, float]],
) -> None:
    """Draw the forward greedy-NNS CAN bus path."""
    can_bus  = clustering_results.get("can_bus", {})
    can_path = can_bus.get("path", [])
    if len(can_path) > 1:
        draw_path(view, can_path, pos, BUS_PEN, legend_label="Bus Path (CAN FD)")


# Note: the redundant return path is no longer computed here.
# main_window.WiringHarnessOptimizer._compute_redundant_return computes an
# edge-disjoint return path on the chassis (excluding all forward bus edges)
# plus the EXT→chassis connector segment. The mixin reads the cached path
# and calls draw_path() directly with REDUNDANT_PEN.


# Note: star and ring paths are no longer computed here.
# main_window.WiringHarnessOptimizer._compute_star_topology /
# _compute_ring_topology compute them on a bus_graph (chassis + EXT
# split edges) so lengths automatically include EXT→chassis connector
# segments and Dijkstra picks the optimal entry side. The mixin reads the
# cached paths and calls draw_path() directly with STAR_PEN / RING_PEN.


# ── View limits ────────────────────────────────────────────────────────────────

def set_view_limits(view: pg.PlotWidget, pos: Dict[str, Tuple[float, float]]) -> None:
    """Fit the view to data with 10% padding, preserving square aspect."""
    if not pos:
        view.setXRange(-1, 1, padding=0.05)
        view.setYRange(-1, 1, padding=0.05)
        return
    xs        = np.array([p[0] for p in pos.values()])
    ys        = np.array([p[1] for p in pos.values()])
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
    padding   = 0.1 * max_range
    xc, yc    = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    view.setXRange(xc - max_range / 2 - padding, xc + max_range / 2 + padding)
    view.setYRange(yc - max_range / 2 - padding, yc + max_range / 2 + padding)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _draw_edges(view, graph, pos, alpha: float, name: str = None) -> None:
    xs, ys = [], []
    for u, v in graph.edges():
        if u in pos and v in pos:
            xs.extend([pos[u][0], pos[v][0], np.nan])
            ys.extend([pos[u][1], pos[v][1], np.nan])
    if xs:
        view.addItem(pg.PlotCurveItem(
            xs, ys,
            pen=pg.mkPen((0, 0, 0, int(255 * alpha)), width=1.2),
            connect="finite", name=name,
        ))


def _draw_nodes(view, graph, pos, config_dict: Dict[str, Any], alpha: float = 1.0) -> None:
    """Group nodes by style (one ScatterPlotItem per type) for performance."""
    node_defs  = config_dict.get("node_definitions", {})
    node_types = node_defs.get("node_types", {})
    by_style: Dict[str, Dict] = {}

    for node, data in graph.nodes(data=True):
        if node not in pos:
            continue
        type_name = "default"
        if data.get("is_io"):
            type_name = "io"
        else:
            for tname, tinfo in node_types.items():
                if any(node.startswith(p) for p in tinfo.get("prefixes", [])):
                    type_name = tname
                    break

        style = _resolve_style(type_name, node_defs, node_types, alpha)
        bucket = by_style.setdefault(style["label"], {"style": style, "points": []})
        bucket["points"].append(pos[node])

    for _key, group in by_style.items():
        pts   = np.array(group["points"])
        style = group["style"]
        view.addItem(pg.ScatterPlotItem(
            pts[:, 0], pts[:, 1],
            size=style["size"], symbol=style["symbol"],
            brush=style["brush"], pen=style["pen"],
            name=style["label"],
        ))


def _resolve_style(type_name: str, node_defs: Dict, node_types: Dict, alpha: float) -> Dict:
    info     = node_types.get(type_name, node_defs.get("default_node", {}))
    hex_col  = info.get("color", "#9B9B9B").lstrip("#")
    r, g, b  = (int(hex_col[i:i+2], 16) for i in (0, 2, 4))
    a        = int(255 * alpha)
    return {
        "label":  info.get("description", "Unknown"),
        "brush":  pg.mkBrush(r, g, b, a),
        "pen":    pg.mkPen(int(r*0.6), int(g*0.6), int(b*0.6), a),
        "size":   info.get("size", 6),
        "symbol": info.get("symbol", "o"),
    }


def _draw_wiring_paths(view, cluster_data, pos, color, cluster_num: Optional[int], legend_added: set) -> None:
    centroid = cluster_data.get("centroid")
    wiring_paths = cluster_data.get("wiring_paths", {})

    cluster_label = None
    if cluster_num is not None:
        io_nodes = cluster_data.get("io_nodes", [])
        cluster_label = f"Cluster {cluster_num} Wiring ({len(io_nodes)} I/O)"

    for _io, path_data in wiring_paths.items():
        path = path_data.get("path", [])
        if not (centroid and "pos" in centroid):
            continue
        vis_pts = [centroid["pos"]] + [pos[n] for n in path if n in pos]
        if len(vis_pts) <= 1:
            continue
        xs = [p[0] for p in vis_pts]
        ys = [p[1] for p in vis_pts]

        name = None
        if cluster_label and cluster_label not in legend_added:
            name = cluster_label
            legend_added.add(cluster_label)

        view.addItem(pg.PlotCurveItem(
            xs, ys,
            pen=pg.mkPen(color, width=WIRING_WIDTH, style=Qt.DashLine),
            name=name,
        ))


def _draw_aggregator(view, cluster_id: str, cluster_data: Dict, color) -> None:
    centroid = cluster_data.get("centroid")
    if not (centroid and "pos" in centroid):
        return
    cx, cy = centroid["pos"]
    view.addItem(pg.ScatterPlotItem(
        [cx], [cy],
        size=AGG_SIZE, symbol="s",
        brush=pg.mkBrush(color),
        pen=pg.mkPen("k", width=1.5),
    ))
    label = pg.TextItem(
        f"Agg {cluster_id.split('_')[-1]}",
        color=(30, 30, 30),
        anchor=(0.5, 1.3),
    )
    label.setPos(cx, cy)
    view.addItem(label)

def _cluster_number(cluster_id: str) -> Optional[int]:
    """Return 1-based cluster number from IDs like 'cluster_1'."""
    if not isinstance(cluster_id, str) or not cluster_id.startswith("cluster_"):
        return None

    try:
        return int(cluster_id.rsplit("_", 1)[-1])
    except ValueError:
        return None


def _cluster_color(cluster_id: str, colors):
    cluster_num = _cluster_number(cluster_id)

    if not colors:
        return pg.mkColor("#9B9B9B")

    # Non-cluster fallback, e.g. can_bus.
    if cluster_num is None:
        if isinstance(colors, dict):
            return colors.get(1, next(iter(colors.values())))
        return colors[0]

    # Dict case: cluster_1 -> colors[1]
    if isinstance(colors, dict):
        if cluster_num in colors:
            return colors[cluster_num]

        # Wrap safely if cluster count exceeds palette length.
        keys = sorted(colors.keys())
        return colors[keys[(cluster_num - 1) % len(keys)]]

    # List case: cluster_1 -> colors[0]
    return colors[(cluster_num - 1) % len(colors)]