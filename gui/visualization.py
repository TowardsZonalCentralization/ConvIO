"""
CONVIO - Visualization Mixin
============================

All drawing logic delegates to ``gui.renderer`` so every tab uses the same
node/edge palette and positioning rules.

The mixin assumes the host provides:
    self.config, self.current_graph, self.elbow_data, self.hpc_results,
    self.cluster_view (pg.PlotWidget),
    self.elbow_widget (MatplotlibWidget),
    self.tab_widget (QTabWidget),
    self.log(message),
    self._setup_pg_view(view, title)   — delegates to renderer.setup_view
"""

from typing import Any, Dict, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QBuffer, QIODevice

from gui import renderer
from gui import legend


class VisualizationMixin:

    # ==================================================================
    # Step 1 – Full network graph
    # ==================================================================
    def _visualize_graph(self, graph):
        renderer.reset_view(self.graph_view)
        pos = self._get_node_positions(graph)
        if not pos:
            self.log("No positions to plot")
            return
        renderer.draw_base_graph(self.graph_view, graph, pos, self.config.config, node_alpha=1.0, include_io=True)
        renderer.set_view_limits(self.graph_view, pos)
        # Tab switch handled by on_graph_loaded

    # ==================================================================
    # Step 2 – Elbow analysis (Matplotlib)
    # ==================================================================
    def _visualize_elbow_analysis(self):
        if not self.elbow_data:
            return
        fig = self.elbow_widget.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(self.elbow_data.get("k_values", []),
                self.elbow_data.get("wcss", []),
                "bo-", markersize=8, linewidth=2)
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Within-Cluster Sum of Squares")
        ax.set_title("Elbow Method for Optimal k")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.elbow_widget.canvas.draw()
        # Tab switch handled by on_elbow_completed

    # ==================================================================
    # Step 3 – Initial clustering (raw assignments, no paths)
    # ==================================================================
    def _visualize_initial_clustering(self, view: pg.PlotWidget, initial_result: Dict[str, Any]):
        if not initial_result or not self.current_graph:
            self.log("No initial clustering result available.")
            return
        renderer.reset_view(view)
        pos = self._get_node_positions(self.current_graph)
        renderer.draw_base_graph(view, self.current_graph, pos, self.config.config, node_alpha=1.0)
        renderer.draw_cluster_io_only(view, initial_result.get("clusters", {}), pos,
                                       self._get_cluster_colors())
        renderer.set_view_limits(view, pos)
        # Tab switch handled by on_initial_clustering_completed

    # ==================================================================
    # Step 4 / Bus Topology – full clustering (paths + centroids)
    # ==================================================================
    def _visualize_clustering_results(self, view: pg.PlotWidget, clustering_results: Dict[str, Any]):
        """
        Shared renderer for Step 4 optimisation view, Bus Topology tab,
        and off-screen PDF report export.
        """
        if not clustering_results or not self.current_graph:
            self.log("Clustering results or graph not available.")
            return
        renderer.reset_view(view)
        pos = self._get_node_positions(self.current_graph)
        renderer.draw_cluster_layer(view, clustering_results.get("clusters", {}), pos,
                                     self._get_cluster_colors())
        renderer.draw_bus_topology(view, clustering_results, pos)
        renderer.draw_base_graph(view, self.current_graph, pos, self.config.config, node_alpha=1.0)
        legend.reorder(view)
        renderer.set_view_limits(view, pos)
        # Tab switch handled by the caller (on_optimization_completed or run_communication_network)

    # ==================================================================
    # Baseline – direct HPC wiring
    # ==================================================================
    def _visualize_hpc_wiring(self):
        if not self.hpc_results or not self.current_graph:
            return
        renderer.reset_view(self.hpc_view)
        pos = self._get_node_positions(self.current_graph)

        paths = self.hpc_results.get("paths", {})
        legend_added = False
        for _io, path_data in paths.items():
            path = path_data.get("path", [])
            if len(path) > 1:
                renderer.draw_path(self.hpc_view, path, pos,
                                   pen=renderer.BUS_PEN,
                                   legend_label="Direct HPC Wiring" if not legend_added else None)
                legend_added = True

        renderer.draw_io_nodes(self.hpc_view, self.current_graph, pos, self.config.config, alpha=1.0)
        # Faded chassis backdrop + full-colour I/O nodes on top.
        renderer.draw_base_graph(self.hpc_view, self.current_graph, pos, self.config.config, node_alpha=1.0)
        legend.reorder(self.hpc_view)
        renderer.set_view_limits(self.hpc_view, pos)
        # Tab switch handled by on_hpc_completed (so the Baseline *container*
        # tab is selected, not the inner plot widget).

    # ==================================================================
    # Step 5b – Redundant Bus
    # ==================================================================
    def _visualize_redundant_bus(self, view: pg.PlotWidget, clustering_results: Dict[str, Any]):
        if not clustering_results or not self.current_graph:
            return
        renderer.reset_view(view)
        pos = self._get_node_positions(self.current_graph)
        renderer.draw_bus_topology(view, clustering_results, pos)

        # Redundant return path is precomputed in
        # main_window.run_communication_network and cached on the host as
        # ``redundant_return``. Drawn for strict disjoint paths and shared-route
        # fallback paths.
        return_info = getattr(self, "redundant_return", None)
        if return_info and return_info.get("status") in {"ok", "shared_route_fallback"} and return_info.get("path"):
            legend_label = (
                "Redundant Return (edge-disjoint)"
                if return_info.get("status") == "ok"
                else "Redundant Return (shared route)"
            )
            renderer.draw_path(view, return_info["path"], pos,
                               renderer.REDUNDANT_PEN,
                               legend_label=legend_label)

        renderer.draw_cluster_layer(view, clustering_results.get("clusters", {}), pos,
                                     self._get_cluster_colors())
        renderer.draw_base_graph(view, self.current_graph, pos, self.config.config, node_alpha=1.0)
        legend.reorder(view)
        renderer.set_view_limits(view, pos)

    # ==================================================================
    # Step 5c – Star and Ring
    # ==================================================================
    def _visualize_star_ring(self, view: pg.PlotWidget, clustering_results: Dict[str, Any]):
        if not clustering_results or not self.current_graph:
            return
        renderer.reset_view(view)
        pos = self._get_node_positions(self.current_graph)
        renderer.draw_cluster_layer(view, clustering_results.get("clusters", {}), pos,
                                     self._get_cluster_colors(), draw_wiring=True)
        renderer.draw_base_graph(view, self.current_graph, pos, self.config.config, node_alpha=1.0)

        # Star and ring paths are precomputed in main_window.run_communication_network
        # so the metric and the visual always match.
        star_topo = getattr(self, "star_topology", None)
        if star_topo:
            first = True
            for _ext_id, path in star_topo.get("paths", {}).items():
                renderer.draw_path(view, path, pos, renderer.STAR_PEN,
                                   legend_label="Star Path" if first else None)
                first = False

        ring_topo = getattr(self, "ring_topology", None)
        if ring_topo:
            first = True
            for path in ring_topo.get("paths", []):
                renderer.draw_path(view, path, pos, renderer.RING_PEN,
                                   legend_label="Ring Path" if first else None)
                first = False

        legend.reorder(view)
        renderer.set_view_limits(view, pos)

    # ==================================================================
    # Position + color helpers
    # ==================================================================
    def _get_node_positions(self, graph) -> Dict[str, Tuple[float, float]]:
        pos = {}
        for node, data in graph.nodes(data=True):
            node_pos = data.get("pos")
            if node_pos and len(node_pos) >= 2:
                try:
                    pos[node] = (float(node_pos[0]), float(node_pos[1]))
                except (ValueError, TypeError):
                    continue
        return pos

    def _get_cluster_colors(self):
        palette = self.config.get("gui", {}).get("color_palette")
        if palette:
            colors = []
            for h in palette:
                h = h.lstrip("#")
                colors.append(pg.mkColor(tuple(int(h[i:i+2], 16) for i in (0, 2, 4))))
            return colors
        return [pg.mkColor(c) for c in [
            (141,211,199),(255,255,179),(190,186,218),(251,128,114),
            (128,177,211),(253,180,98),(179,222,105),(252,205,229),
        ]]

    # ==================================================================
    # Off-screen PNG export for PDF report generator
    # ==================================================================
    def generate_clustering_plot_for_report(self, clustering_results: Dict[str, Any], title: str) -> bytes:
        temp = pg.PlotWidget()
        self._setup_pg_view(temp, title)
        self._visualize_clustering_results(temp, clustering_results)
        return self._export_plot_to_image_bytes(temp)

    def _export_plot_to_image_bytes(self, plot_widget: pg.PlotWidget) -> bytes:
        pixmap = plot_widget.grab()
        buf    = QBuffer()
        buf.open(QIODevice.ReadWrite)
        pixmap.save(buf, "PNG")
        buf.seek(0)
        return buf.readAll().data()
