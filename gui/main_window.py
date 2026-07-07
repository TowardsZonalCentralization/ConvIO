"""
CONVIO - Main Window
====================

WiringHarnessOptimizer – the QMainWindow that hosts the full UI.

Workflow (left control panel):
  Step 1  Build Network Graph    (k-d Tree + chassis graph)
  Step 2  k-Means Elbow          (WCSS curve, optimal k)
  —       Baseline Wiring        (direct HPC, comparison reference)
  Step 3  Agglomerative Clust.   (raw I/O assignments, no optimisation)
  Step 4  Iterative Optimisation (centroid refinement + iteration table)
  Step 5  Communication Network  (Bus / Redundant Bus / Star and Ring)
          Results                (summary + per-cluster + linkage comparison)

Scaling fix: step groups live in a QScrollArea; topology tabs are
TopologyResultWidget (plot + scrollable metrics panel).
"""

import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

from config_manager import ConfigManager
from modules.graph_utils import get_graph_statistics, validate_graph
from modules.report_generator import ReportGenerator
from gui import renderer
from gui import results_formatter as rf
from gui.dialogs import ComparisonWindow, CostSettingsDialog
from gui.ui_components import MetricsPanel, SortableTable, TopologyResultWidget
from gui.visualization import VisualizationMixin
from gui.widgets import MatplotlibWidget
from gui.workers import OptimizationWorker


class WiringHarnessOptimizer(VisualizationMixin, QMainWindow):

    # Backward compat wrappers for report_generator
    @staticmethod
    def _get_graph_statistics(graph):
        return get_graph_statistics(graph)

    @staticmethod
    def _validate_graph(graph):
        return validate_graph(graph)

    # ==================================================================
    # Construction
    # ==================================================================
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config = config_manager
        self.logger = logging.getLogger(__name__)

        # Analysis state
        self.current_graph:             Optional[Any]            = None
        self.graph_loader:              Optional[Any]            = None
        self.elbow_data:                Optional[Dict[str, Any]] = None
        self.initial_clustering_result: Optional[Dict[str, Any]] = None
        self.clustering_results:        Optional[Dict[str, Any]] = None
        self.hpc_results:               Optional[Dict[str, Any]] = None
        self.all_linkage_results:       Optional[Dict[str, Any]] = None  # full_analysis sweep
        self.topology_lengths:          Dict[str, Dict[str, Any]] = {}    # set by run_communication_network
        self.chassis_file_path:         Optional[str]            = None
        self.io_file_path:              Optional[str]            = None

        self._apply_config()
        self._init_ui()
        self._init_menu()
        self._setup_reproducibility()

    def _apply_config(self):
        gui_cfg     = self.config.get("gui", {})
        window_size = gui_cfg.get("window_size", [1500, 900])
        self.setWindowTitle("CONVIO – Automotive Wiring Harness Optimizer")
        self.setGeometry(100, 100, int(window_size[0]), int(window_size[1]))
        cost = self.config.get("cost", {})
        self.cost_cfg = {
            "currency":             cost.get("currency", "EURO"),
            "wire_price_per_m":     float(cost.get("wire_price_per_m", 0.0)),
            "CAN_bus_price_per_m":  float(cost.get("CAN_bus_price_per_m", 0.0)),
            "wire_weight_per_m_kg": float(cost.get("wire_weight_per_m_kg", 0.0)),
        }

    def _setup_reproducibility(self):
        repro = self.config.get("reproducibility", {})
        if bool(repro.get("set_global_seeds", False)):
            seed = int(repro.get("numpy_seed", 42))
            np.random.seed(seed)
            random.seed(seed)

    # ==================================================================
    # UI construction
    # ==================================================================
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_control_panel())
        splitter.addWidget(self._create_visualization_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    def _init_menu(self):
        menubar   = self.menuBar()
        file_menu = menubar.addMenu("File")
        for label, slot in [
            ("Open Chassis JSON…",     self.load_chassis_file),
            ("Open I/O CSV…",          self.load_io_file),
            ("Load Default Files",     self.load_default_files),
        ]:
            act = QAction(label, self); act.triggered.connect(slot)
            file_menu.addAction(act)
        file_menu.addSeparator()
        for label, slot in [
            ("Export Results (JSON)…", self.export_results_json),
            ("Export Report (PDF)…",   self.export_report_pdf),
        ]:
            act = QAction(label, self); act.triggered.connect(slot)
            file_menu.addAction(act)
        file_menu.addSeparator()
        act = QAction("Exit", self); act.triggered.connect(self.close)
        file_menu.addAction(act)
        tools_menu = menubar.addMenu("Tools")
        act_costs  = QAction("Cost && Weight Settings…", self)
        act_costs.triggered.connect(self.show_cost_settings)
        tools_menu.addAction(act_costs)
        help_menu = menubar.addMenu("Help")
        act = QAction("About", self); act.triggered.connect(self.show_about_dialog)
        help_menu.addAction(act)

    # ------------------------------------------------------------------
    # Control panel
    # ------------------------------------------------------------------
    def _create_control_panel(self) -> QWidget:
        outer = QWidget()
        outer.setMaximumWidth(380)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.setSpacing(4)

        title = QLabel("CONVIO Control Panel")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        outer_layout.addWidget(title)

        self.file_status_label = QLabel("No files loaded  •  Use File menu")
        self.file_status_label.setWordWrap(True)
        self.file_status_label.setStyleSheet("color: #555; font-size: 11px;")
        outer_layout.addWidget(self.file_status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        steps = QWidget()
        steps_layout = QVBoxLayout(steps)
        steps_layout.setSpacing(6)
        for grp in [
            self._build_step1_group(), self._build_step2_group(),
            self._build_baseline_group(), self._build_step3_group(),
            self._build_step4_group(), self._build_step5_group(),
            self._build_comparison_group(),
        ]:
            steps_layout.addWidget(grp)
        steps_layout.addStretch()
        scroll.setWidget(steps)
        outer_layout.addWidget(scroll, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        outer_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        outer_layout.addWidget(self.status_label)
        outer_layout.addWidget(self._build_log_group())
        return outer

    # --- Step group builders ------------------------------------------
    def _build_step1_group(self) -> QGroupBox:
        g = QGroupBox("Step 1  ·  Network Graph  (k-d Tree)")
        lay = QVBoxLayout(g)
        self.btn_process = QPushButton("Build Network Graph")
        self.btn_process.clicked.connect(self.process_graph)
        self.btn_process.setEnabled(False)
        lay.addWidget(self.btn_process)
        self.graph_stats_label = QLabel("")
        self.graph_stats_label.setWordWrap(True)
        lay.addWidget(self.graph_stats_label)
        return g

    def _build_step2_group(self) -> QGroupBox:
        g = QGroupBox("Step 2  ·  k-Means Elbow  (Euclidean WCSS)")
        lay = QVBoxLayout(g)
        self.elbow_btn = QPushButton("Run Elbow Analysis")
        self.elbow_btn.clicked.connect(self.run_elbow_analysis)
        self.elbow_btn.setEnabled(False)
        lay.addWidget(self.elbow_btn)
        self.optimal_clusters_label = QLabel("Optimal clusters: —")
        lay.addWidget(self.optimal_clusters_label)
        return g

    def _build_baseline_group(self) -> QGroupBox:
        g = QGroupBox("Baseline Wiring  (direct HPC, for comparison)")
        lay = QVBoxLayout(g)
        self.hpc_btn = QPushButton("Calculate Baseline Wiring")
        self.hpc_btn.clicked.connect(self.run_hpc_analysis)
        self.hpc_btn.setEnabled(False)
        lay.addWidget(self.hpc_btn)
        self.hpc_total_label  = QLabel("Total Length: —")
        self.hpc_cost_label   = QLabel("Cost: —")
        self.hpc_weight_label = QLabel("Weight: —")
        for w in (self.hpc_total_label, self.hpc_cost_label, self.hpc_weight_label):
            lay.addWidget(w)
        return g

    def _build_step3_group(self) -> QGroupBox:
        g = QGroupBox("Step 3  ·  Agglomerative Clustering  (graph-based)")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("Clusters:"), 0, 0)
        max_k = int(self.config.get("clustering.max_clusters_supported", 100))
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(1, max_k)
        self.n_clusters_spin.setValue(3)
        lay.addWidget(self.n_clusters_spin, 0, 1)

        lay.addWidget(QLabel("Linkage:"), 1, 0)
        self.linkage_combo = QComboBox()
        self.linkage_combo.addItems(["average", "complete", "single"])
        lay.addWidget(self.linkage_combo, 1, 1)

        self.clustering_btn = QPushButton("Run Clustering")
        self.clustering_btn.clicked.connect(self.run_initial_clustering)
        self.clustering_btn.setEnabled(False)
        lay.addWidget(self.clustering_btn, 2, 0, 1, 2)

        self.compare_linkage_btn = QPushButton("Compare All Linkage Methods")
        self.compare_linkage_btn.clicked.connect(self.run_linkage_comparison)
        self.compare_linkage_btn.setEnabled(False)
        self.compare_linkage_btn.setToolTip("Runs average / complete / single and populates the Linkage Comparison table in Results.")
        lay.addWidget(self.compare_linkage_btn, 3, 0, 1, 2)
        return g

    def _build_step4_group(self) -> QGroupBox:
        g = QGroupBox("Step 4  ·  Iterative Centroid Optimisation")
        lay = QGridLayout(g)
        lay.addWidget(QLabel("Iterations:"), 0, 0)
        self.n_iterations_spin = QSpinBox()
        self.n_iterations_spin.setRange(1, 20)
        self.n_iterations_spin.setValue(5)
        lay.addWidget(self.n_iterations_spin, 0, 1)
        self.optimization_btn = QPushButton("Run Iterative Optimisation")
        self.optimization_btn.clicked.connect(self.run_iterative_optimization)
        self.optimization_btn.setEnabled(False)
        lay.addWidget(self.optimization_btn, 1, 0, 1, 2)
        self.optim_total_label  = QLabel("I/O Wire Length: —")
        self.optim_cost_label   = QLabel("Cost: —")
        self.optim_weight_label = QLabel("Weight: —")
        for row, w in enumerate([self.optim_total_label, self.optim_cost_label, self.optim_weight_label], 2):
            lay.addWidget(w, row, 0, 1, 2)
        return g

    def _build_step5_group(self) -> QGroupBox:
        g = QGroupBox("Step 5  ·  Communication Network  (Greedy NNS)")
        lay = QVBoxLayout(g)
        self.comm_network_btn = QPushButton("Run Communication Network")
        self.comm_network_btn.clicked.connect(self.run_communication_network)
        self.comm_network_btn.setEnabled(False)
        lay.addWidget(self.comm_network_btn)
        self.comm_total_label  = QLabel("Total with CAN FD: —")
        self.comm_cost_label   = QLabel("Cost: —")
        self.comm_weight_label = QLabel("Weight: —")
        for w in (self.comm_total_label, self.comm_cost_label, self.comm_weight_label):
            lay.addWidget(w)
        return g

    def _build_comparison_group(self) -> QGroupBox:
        g = QGroupBox("Bus Topology vs Baseline")
        lay = QVBoxLayout(g)
        self.comparison_length_saving_label = QLabel("Length Saving: —")
        self.comparison_cost_saving_label   = QLabel("Cost Saving: —")
        self.comparison_weight_saving_label = QLabel("Weight Saving: —")
        for w in (self.comparison_length_saving_label, self.comparison_cost_saving_label,
                  self.comparison_weight_saving_label):
            lay.addWidget(w)
        hint = QLabel("<i>See Results tab for all-topology comparison.</i>")
        hint.setStyleSheet("color: #666; font-size: 10px;")
        lay.addWidget(hint)
        return g

    def _build_log_group(self) -> QGroupBox:
        g = QGroupBox("CONVIO Log")
        lay = QVBoxLayout(g)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(160)
        lay.addWidget(self.log_text)
        return g

    # ------------------------------------------------------------------
    # Visualization panel
    # ------------------------------------------------------------------
    def _create_visualization_panel(self) -> QWidget:
        panel  = QWidget()
        layout = QVBoxLayout(panel)
        self.tab_widget = QTabWidget()

        # ① Network Graph
        self.graph_view = pg.PlotWidget()
        renderer.setup_view(self.graph_view, "Network Graph")
        self.tab_widget.addTab(self.graph_view, "① Network Graph")

        # ② Elbow Analysis
        self.elbow_widget = MatplotlibWidget()
        self.tab_widget.addTab(self.elbow_widget, "② Elbow Analysis")

        # Baseline Wiring
        self.hpc_topo = TopologyResultWidget("Baseline Wiring", "Baseline Metrics")
        self.hpc_view = self.hpc_topo.plot_view
        renderer.setup_view(self.hpc_view, "Baseline: Direct HPC Wiring")
        self.tab_widget.addTab(self.hpc_topo, "Baseline Wiring")

        # ③ Initial Clustering
        self.initial_cluster_view = pg.PlotWidget()
        renderer.setup_view(self.initial_cluster_view, "Initial Clustering")
        self.tab_widget.addTab(self.initial_cluster_view, "③ Initial Clustering")

        # ④ Centroid Optimisation  (plot left, iteration table right)
        self.optim_tab = QWidget()
        optim_layout   = QHBoxLayout(self.optim_tab)
        optim_layout.setContentsMargins(0, 0, 0, 0)
        optim_splitter = QSplitter(Qt.Horizontal)

        self.optim_view = pg.PlotWidget()
        renderer.setup_view(self.optim_view, "Centroid Optimisation")
        optim_splitter.addWidget(self.optim_view)

        iter_container = QWidget()
        iter_layout = QVBoxLayout(iter_container)
        iter_layout.setContentsMargins(4, 4, 4, 4)
        iter_layout.addWidget(QLabel("<b>Iteration Progress</b>"))
        self.iteration_table = SortableTable(
            ["Iteration", "I/O Reassignments", "Total Wire Length (mm)", "Δ vs Previous (mm)"]
        )
        iter_layout.addWidget(self.iteration_table)
        optim_splitter.addWidget(iter_container)

        optim_splitter.setStretchFactor(0, 6)
        optim_splitter.setStretchFactor(1, 4)
        optim_splitter.setSizes([900, 500])
        optim_layout.addWidget(optim_splitter)
        self.tab_widget.addTab(self.optim_tab, "④ Centroid Optimisation")

        # ⑤ Bus Topology  (TopologyResultWidget — also legacy cluster_view)
        self.bus_topo   = TopologyResultWidget("Bus Topology", "Bus Topology Metrics")
        self.cluster_view = self.bus_topo.plot_view   # report_generator compat
        renderer.setup_view(self.cluster_view, "⑤ Bus Topology (Greedy NNS)")
        self.tab_widget.addTab(self.bus_topo, "⑤ Bus Topology")

        # ⑤ Redundant Bus
        self.redundant_bus_topo = TopologyResultWidget("Redundant Bus", "Redundant Bus Metrics")
        self.redundant_bus_view = self.redundant_bus_topo.plot_view
        renderer.setup_view(self.redundant_bus_view, "⑤ Redundant Bus Topology")
        self.tab_widget.addTab(self.redundant_bus_topo, "⑤ Redundant Bus")

        # ⑤ Star and Ring
        self.star_ring_topo = TopologyResultWidget("Star and Ring", "Star and Ring Metrics")
        self.star_ring_view = self.star_ring_topo.plot_view
        renderer.setup_view(self.star_ring_view, "⑤ Star and Ring Topology")
        self.tab_widget.addTab(self.star_ring_topo, "⑤ Star and Ring")

        # Results tab
        self.results_tab = self._build_results_tab()
        self.tab_widget.addTab(self.results_tab, "Results")

        layout.addWidget(self.tab_widget)
        return panel

    def _build_results_tab(self) -> QWidget:
        tab    = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner  = QWidget()
        lay    = QVBoxLayout(inner)
        lay.setSpacing(10)

        # Topology comparison summary
        lay.addWidget(self._section_header("Architecture Comparison Summary"))
        lay.addWidget(QLabel(
            "All lengths in mm. 'Saving vs Baseline' compares to direct HPC wiring; "
            "'Saving vs Bus' compares to the simple Bus Topology."
        ))
        cur = self.cost_cfg["currency"]
        self.results_summary_table = SortableTable([
            "Architecture", "Wiring Harness", "Network Cable", "Total",
            f"Cost ({cur})", "Saving vs Baseline", "Saving vs Bus",
        ])
        lay.addWidget(self.results_summary_table)

        # Per-cluster breakdown
        lay.addWidget(self._section_header("Per-Cluster Breakdown"))
        self.results_cluster_table = SortableTable(
            ["Cluster", "I/O Count", "Wire Length (mm)", "Estimated Cost"]
        )
        lay.addWidget(self.results_cluster_table)

        # Linkage comparison
        lay.addWidget(self._section_header("Linkage Method Comparison"))
        lay.addWidget(QLabel(
            "Run 'Compare All Linkage Methods' in Step 3 to populate this table.",
        ))
        self.linkage_table = SortableTable(
            ["Linkage", "Wire (mm)", "CAN (mm)", "Total (mm)",
             f"Cost ({cur})", "Saving vs Baseline"]
        )
        lay.addWidget(self.linkage_table)
        lay.addStretch()

        scroll.setWidget(inner)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab

    @staticmethod
    def _section_header(text: str) -> QLabel:
        lbl = QLabel(f"<b>{text}</b>")
        lbl.setStyleSheet(
            "background: #37474F; color: white; padding: 4px 8px; border-radius: 3px;"
        )
        return lbl

    def _setup_pg_view(self, view: pg.PlotWidget, title: str):
        """Called by VisualizationMixin.generate_clustering_plot_for_report."""
        renderer.setup_view(view, title)

    # ==================================================================
    # Analysis control flow
    # ==================================================================
    def process_graph(self):
        if not (self.chassis_file_path and self.io_file_path):
            QMessageBox.warning(self, "Missing files", "Open both a Chassis JSON and an I/O CSV first (File menu).")
            return
        self._start_worker("load_graph", chassis_file=self.chassis_file_path,
                           io_file=self.io_file_path, on_finish=self.on_graph_loaded, show_progress=True)

    def run_elbow_analysis(self):
        if not self.current_graph:
            QMessageBox.warning(self, "No graph", "Build the network graph first.")
            return
        self._start_worker("elbow_analysis", graph=self.current_graph, on_finish=self.on_elbow_completed)

    def run_hpc_analysis(self):
        if not self.current_graph:
            QMessageBox.warning(self, "No graph", "Build the network graph first.")
            return
        self._start_worker("hpc_wiring", graph=self.current_graph, on_finish=self.on_hpc_completed)

    def run_initial_clustering(self):
        if not self.current_graph:
            QMessageBox.warning(self, "No graph", "Build the network graph first.")
            return
        self._start_worker(
            "clustering_initial", graph=self.current_graph,
            n_clusters=self.n_clusters_spin.value(),
            linkage_method=self.linkage_combo.currentText(),
            on_finish=self.on_initial_clustering_completed,
        )

    def run_linkage_comparison(self):
        """Full sweep of all linkage methods; results go to the Results tab."""
        if not self.current_graph:
            QMessageBox.warning(self, "No graph", "Build the network graph first.")
            return
        self._start_worker(
            "full_analysis", graph=self.current_graph,
            n_clusters=self.n_clusters_spin.value(),
            max_iterations=self.n_iterations_spin.value(),
            on_finish=self.on_linkage_comparison_completed,
            show_progress=True,
        )

    def run_iterative_optimization(self):
        if not self.current_graph:
            QMessageBox.warning(self, "No graph", "Build the network graph first.")
            return
        self._start_worker(
            "clustering_iterative", graph=self.current_graph,
            n_clusters=self.n_clusters_spin.value(),
            linkage_method=self.linkage_combo.currentText(),
            max_iterations=self.n_iterations_spin.value(),
            on_finish=self.on_optimization_completed,
        )

    def run_communication_network(self):
        """Step 5 – render all three topology tabs and their metrics panels."""
        if not self.clustering_results:
            QMessageBox.warning(self, "No data", "Run Step 4 first.")
            return

        bus_len = self.clustering_results.get("can_bus", {}).get("total_length", 0.0)

        # Edge-disjoint return — cached for the renderer to reuse.
        can_path = self.clustering_results.get("can_bus", {}).get("path", [])

        self.redundant_return = {
            "path": list(can_path),
            "chassis_length": float(bus_len),
            "connector_length": 0.0,
            "total_length": float(bus_len),
            "status": "ok",
        }

        redundant_len = bus_len

        # Star and Ring — also precomputed and cached so visualisation
        # uses the exact same paths the metrics report.
        self.star_topology = self._compute_star_topology()
        self.ring_topology = self._compute_ring_topology()
        star_len = self.star_topology["total_length"]
        ring_len = self.ring_topology["total_length"]

        # Cache segment dicts so the Results tab can build the architecture
        # comparison table.  Keys must match the labels passed to
        # update_metrics() below so _build_topology_metrics can find them.
        self.topology_lengths = {
            "Bus Topology":   {"network_segments": {"CAN Bus": bus_len}},
            "Redundant Bus":  {"network_segments": {"Forward Bus": bus_len, "Return Path": redundant_len}},
            "Star and Ring":  {"network_segments": {"Star Paths": star_len, "Ring": ring_len}},
        }

        # ⑤ Bus Topology
        self._visualize_clustering_results(self.cluster_view, self.clustering_results)
        self.bus_topo.metrics_panel.update_metrics(self._build_topology_metrics("Bus Topology"))

        # ⑤ Redundant Bus
        self._visualize_redundant_bus(self.redundant_bus_view, self.clustering_results)
        self.redundant_bus_topo.metrics_panel.update_metrics(self._build_topology_metrics("Redundant Bus"))

        # ⑤ Star and Ring
        self._visualize_star_ring(self.star_ring_view, self.clustering_results)
        self.star_ring_topo.metrics_panel.update_metrics(self._build_topology_metrics("Star and Ring"))

        self._update_comm_labels()
        if self.hpc_results:
            self._compare_results()
        self._update_results_tables()
        self.tab_widget.setCurrentWidget(self.bus_topo)
        self.log(
            f"Communication network rendered. "
            f"Bus: {bus_len:,.0f} mm  |  "
            f"Redundant return: {redundant_len:,.0f} mm ({self.redundant_return['status']})  |  "
            f"Star: {star_len:,.0f} mm  Ring: {ring_len:,.0f} mm"
        )

    def _build_topology_metrics(self, topology_label: str) -> dict:
        """
        Build the metrics dict for a topology tab, including any annotations
        specific to that topology (e.g. Redundant Bus gets a status row).

        Used by both ``run_communication_network`` (initial render) and
        ``show_cost_settings`` (refresh after cost edit) so the annotations
        survive a settings change.
        """
        info     = self.topology_lengths.get(topology_label, {})
        segments = info.get("network_segments", {})
        metrics  = rf.architecture_metrics(
            topology_label, self.clustering_results, segments, self.cost_cfg,
        )
        # Redundancy status row for the Redundant Bus tab
        if topology_label == "Redundant Bus" and getattr(self, "redundant_return", None):
            status = self.redundant_return.get("status", "unavailable")
            status_text = {
                "ok":               "✓ Edge-disjoint",
                "shared_route_fallback": "⚠ Shared route fallback",
                "no_disjoint_path": "✗ No return route available",
                "unavailable":      "—",
            }.get(status, "—")
            metrics["  Redundancy"] = status_text
        return metrics

    # ==================================================================
    # Topology length helpers (for metrics only, not re-drawing)
    # ==================================================================
    def _compute_redundant_return(self) -> dict:
        """
        Compute the redundant return path from the last bus aggregator back
        to HPC, including the EXT→chassis connector segment.

        Strategy
        --------
        1. Identify the last EXT aggregator on the forward bus path and the
           chassis node it was reached through (``forward_entry_chassis``).
        2. For an edge-centroid aggregator, deliberately exit via the
           **other** chassis endpoint (the unused split edge) so even the
           EXT→chassis hop is not shared with the forward bus.
        3. Build a chassis subgraph minus every chassis edge used by the
           forward bus path, then run Dijkstra from the chosen return start
           back to HPC.
        4. If no edge-disjoint path exists, fall back to the normal chassis
           shortest path so the redundant cable may share the same route.
        5. Total return length = chassis Dijkstra length + connector length.

        If no chassis path exists even with shared routes allowed, status is
        set to ``no_disjoint_path``, length = 0, and the renderer will draw
        nothing.

        Returns:
            dict with keys
              path             — chassis-only return path nodes (or [])
              chassis_length   — Dijkstra length on chassis (mm)
              connector_length — t·edge_len for edge-centroid, 0 otherwise
              total_length     — chassis_length + connector_length
              status           — 'ok' | 'shared_route_fallback' |
                                 'no_disjoint_path' | 'unavailable'
        """
        import networkx as nx

        none_result = {
            "path": [], "chassis_length": 0.0, "connector_length": 0.0,
            "total_length": 0.0, "status": "unavailable",
        }
        if not (self.clustering_results and self.current_graph):
            return none_result

        can_path = self.clustering_results.get("can_bus", {}).get("path", [])
        if len(can_path) < 2:
            return none_result

        chassis_nodes = set(n for n, d in self.current_graph.nodes(data=True) if not d.get("is_io"))
        chassis = self.current_graph.subgraph(chassis_nodes).copy()
        hpc = can_path[0]
        if hpc not in chassis_nodes:
            return none_result

        # ── 1. Last EXT and the chassis node forward bus reached it through
        last_ext = can_path[-1]
        forward_entry_chassis = None
        for n in reversed(can_path):
            if n in chassis_nodes:
                forward_entry_chassis = n
                break
        if forward_entry_chassis is None:
            return none_result

        # ── 2. Resolve return start chassis node + connector length
        suffix   = last_ext.split("_")[-1]
        centroid = (self.clustering_results.get("clusters", {})
                    .get(f"cluster_{suffix}", {})
                    .get("centroid"))

        return_start    = forward_entry_chassis  # default for node-centroid
        connector_length = 0.0
        if centroid:
            ctype = centroid.get("type")
            if ctype == "node":
                # Aggregator sits at a chassis node — single connection,
                # no connector length, no choice.
                return_start    = centroid["node"]
                connector_length = 0.0
            elif ctype == "edge":
                u, v, t = centroid["u"], centroid["v"], centroid.get("t", 0.5)
                edge_len = (self.current_graph.edges[u, v].get("weight", 1.0)
                            if self.current_graph.has_edge(u, v) else 0.0)
                # Use the OTHER chassis endpoint than forward bus did.
                # Connector for u side has weight t·edge_len, for v side (1-t)·edge_len.
                if forward_entry_chassis == u:
                    return_start, connector_length = v, (1 - t) * edge_len
                else:
                    return_start, connector_length = u, t * edge_len

        # ── 3. Chassis subgraph minus all forward chassis edges
        forward_chassis_edges: set = set()
        for i in range(len(can_path) - 1):
            a, b = can_path[i], can_path[i + 1]
            if a in chassis_nodes and b in chassis_nodes:
                forward_chassis_edges.add(frozenset([a, b]))

        disjoint = chassis.copy()
        for edge in forward_chassis_edges:
            ab = list(edge)
            if len(ab) == 2 and disjoint.has_edge(ab[0], ab[1]):
                disjoint.remove_edge(ab[0], ab[1])

        def build_result(graph, status: str) -> dict | None:
            try:
                return_path    = nx.dijkstra_path(graph, return_start, hpc, weight="weight")
                chassis_length = nx.dijkstra_path_length(graph, return_start, hpc, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

            return {
                "path":             return_path,
                "chassis_length":   float(chassis_length),
                "connector_length": float(connector_length),
                "total_length":     float(chassis_length + connector_length),
                "status":           status,
            }

        # ── 4. Prefer Dijkstra from return_start to HPC on the disjoint graph
        disjoint_result = build_result(disjoint, "ok")
        if disjoint_result:
            return disjoint_result

        # ── 5. Fallback: use a separate cable that may share chassis route
        shared_result = build_result(chassis, "shared_route_fallback")
        if shared_result:
            return shared_result

        return {**none_result, "connector_length": connector_length,
                "status": "no_disjoint_path"}

    def _build_bus_graph(self):
        """
        Build chassis subgraph augmented with EXT_ aggregator virtual nodes
        and their connector edges, mirroring how
        ``ClusteringDijkstra._format_and_export_output`` constructs ``G_out``.

        For each cluster centroid:
          - node-centroid: 0-weight edge from EXT to the centroid node
          - edge-centroid: t·edge_len edge to u, (1-t)·edge_len edge to v

        Dijkstra on this graph automatically picks the optimal entry side
        and includes the EXT→chassis connector — identical behaviour to
        the forward bus path computation.
        """
        import networkx as nx
        chassis_nodes = [n for n, d in self.current_graph.nodes(data=True) if not d.get("is_io")]
        bus = self.current_graph.subgraph(chassis_nodes).copy()
        if not self.clustering_results:
            return bus

        for cid, cdata in self.clustering_results.get("clusters", {}).items():
            if cid == "can_bus":
                continue
            centroid = cdata.get("centroid")
            if not centroid:
                continue
            ext_id = f"EXT_{cid.split('_')[-1]}"
            bus.add_node(ext_id, pos=centroid.get("pos"))

            if centroid.get("type") == "node":
                target = centroid.get("node")
                if target in bus:
                    bus.add_edge(ext_id, target, weight=0.0)
            elif centroid.get("type") == "edge":
                u, v, t = centroid["u"], centroid["v"], centroid.get("t", 0.5)
                if not self.current_graph.has_edge(u, v):
                    continue
                edge_len = self.current_graph.edges[u, v].get("weight", 1.0)
                if u in bus:
                    bus.add_edge(ext_id, u, weight=t * edge_len)
                if v in bus:
                    bus.add_edge(ext_id, v, weight=(1 - t) * edge_len)
        return bus

    def _compute_star_topology(self) -> dict:
        """
        Star: HPC connects directly to each aggregator via shortest cable.

        Uses ``_build_bus_graph`` so length includes EXT→chassis connectors
        and the entry side is chosen by Dijkstra (not by ``t`` alone).

        Returns:
            ``{"paths": {ext_id: [chassis_path]}, "total_length": float}``
        """
        import networkx as nx
        result = {"paths": {}, "total_length": 0.0}
        if not (self.clustering_results and self.current_graph):
            return result
        bus = self._build_bus_graph()
        hpc = self.config.get("node_configuration.hpc_node_name", "H1")
        if hpc not in bus:
            return result

        total = 0.0
        for cid, cdata in self.clustering_results.get("clusters", {}).items():
            if cid == "can_bus" or not cdata.get("centroid"):
                continue
            ext_id = f"EXT_{cid.split('_')[-1]}"
            if ext_id not in bus:
                continue
            try:
                length    = nx.dijkstra_path_length(bus, hpc, ext_id, weight="weight")
                full_path = nx.dijkstra_path(bus, hpc, ext_id, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            # Strip EXT_ nodes for visualisation (they don't have positions
            # in current_graph; the aggregator marker is drawn separately).
            chassis_path = [n for n in full_path if not n.startswith("EXT_")]
            result["paths"][ext_id] = chassis_path
            total += length
        result["total_length"] = total
        return result

    def _compute_ring_topology(self) -> dict:
        """
        Ring: aggregators connected in a closed loop (cluster-id insertion order).

        Uses ``_build_bus_graph`` so each segment includes both EXT→chassis
        connectors and uses optimal entry sides.

        Note: the ring direction is the iteration order of the clusters dict
        — not TSP-optimal. A different ordering could yield a shorter ring.

        Returns:
            ``{"paths": [[chassis_path], ...], "total_length": float}``
        """
        import networkx as nx
        result = {"paths": [], "total_length": 0.0}
        if not (self.clustering_results and self.current_graph):
            return result
        bus = self._build_bus_graph()

        ext_ids = []
        for cid, cdata in self.clustering_results.get("clusters", {}).items():
            if cid == "can_bus" or not cdata.get("centroid"):
                continue
            ext_id = f"EXT_{cid.split('_')[-1]}"
            if ext_id in bus:
                ext_ids.append(ext_id)
        if len(ext_ids) < 2:
            return result

        total = 0.0
        paths = []
        for i in range(len(ext_ids)):
            src = ext_ids[i]
            dst = ext_ids[(i + 1) % len(ext_ids)]
            if src == dst:
                continue
            try:
                length    = nx.dijkstra_path_length(bus, src, dst, weight="weight")
                full_path = nx.dijkstra_path(bus, src, dst, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            chassis_path = [n for n in full_path if not n.startswith("EXT_")]
            if chassis_path:
                paths.append(chassis_path)
            total += length
        result["paths"] = paths
        result["total_length"] = total
        return result

    # ==================================================================
    # Worker factory
    # ==================================================================
    def _start_worker(self, task_type: str, on_finish, show_progress: bool = False, **kwargs):
        self._lock_controls()
        if show_progress:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
        self.worker = OptimizationWorker(task_type, self.config, **kwargs)
        if show_progress:
            self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.finished.connect(on_finish)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    # ==================================================================
    # Worker signal handlers
    # ==================================================================
    def on_graph_loaded(self, results):
        self.current_graph = results["graph"]
        self.graph_loader  = results["loader"]
        self._unlock_controls()
        stats      = results.get("statistics", {})
        validation = results.get("validation", {})
        self.graph_stats_label.setText(
            f"{stats.get('chassis_nodes',0)} chassis  •  {stats.get('io_nodes',0)} I/O"
        )
        for w in validation.get("warnings", []):
            self.log(f"  Warning: {w}")
        self.log(f"Graph loaded: {stats.get('chassis_nodes',0)} chassis, {stats.get('io_nodes',0)} I/O")
        self._visualize_graph(self.current_graph)
        self.tab_widget.setCurrentWidget(self.graph_view)
        self.elbow_btn.setEnabled(True)
        self.hpc_btn.setEnabled(True)
        self.clustering_btn.setEnabled(True)
        self.compare_linkage_btn.setEnabled(True)

    def on_elbow_completed(self, results):
        self.elbow_data = results["elbow_data"]
        optimal_k       = results["optimal_k"]
        self._unlock_controls()
        self.optimal_clusters_label.setText(f"Optimal clusters: {optimal_k}")
        self.log(f"Elbow done. Optimal k = {optimal_k}")
        max_k = int(self.config.get("clustering.max_clusters_supported", 100))
        self.n_clusters_spin.setValue(min(optimal_k, max_k))
        self._visualize_elbow_analysis()
        self.tab_widget.setCurrentWidget(self.elbow_widget)

    def on_hpc_completed(self, results):
        self.hpc_results = results["hpc_wiring_results"]
        self._unlock_controls()
        if not self.hpc_results:
            self.log("  Baseline wiring failed.")
            return
        total  = self.hpc_results.get("total_length", 0.0)
        p_m    = self.cost_cfg["wire_price_per_m"]
        w_m    = self.cost_cfg["wire_weight_per_m_kg"]
        cur    = self.cost_cfg["currency"]
        self.hpc_total_label.setText(f"Total Length: {total:,.2f} mm")
        self.hpc_cost_label.setText(f"Cost: {(total/1000)*p_m:,.2f} {cur}")
        self.hpc_weight_label.setText(f"Weight: {(total/1000)*w_m:,.3f} kg")
        self._visualize_hpc_wiring()
        self.hpc_topo.metrics_panel.update_metrics(rf.baseline_metrics(self.hpc_results, self.cost_cfg))
        self.tab_widget.setCurrentWidget(self.hpc_topo)
        if self.clustering_results:
            self._compare_results()

    def on_initial_clustering_completed(self, results):
        self.initial_clustering_result = results
        self._unlock_controls()
        n = len(results.get("clusters", {}))
        self.log(f"Initial clustering: {n} clusters (no optimisation)")
        self._visualize_initial_clustering(self.initial_cluster_view, results)
        self.tab_widget.setCurrentWidget(self.initial_cluster_view)
        self.optimization_btn.setEnabled(True)

    def on_optimization_completed(self, results):
        self.clustering_results = results
        self._invalidate_communication_network_results()
        self._unlock_controls()
        if self.clustering_results:
            self._update_optim_labels()
            self.iteration_table.populate(self._format_iteration_rows(results))
            self._visualize_clustering_results(self.optim_view, self.clustering_results)
            self.tab_widget.setCurrentWidget(self.optim_tab)
        if self.hpc_results:
            self._compare_results()
        self.comm_network_btn.setEnabled(True)
        self.log("Iterative optimisation complete.")

    def on_linkage_comparison_completed(self, results):
        self.all_linkage_results = results.get("results", {})
        best = results.get("best_method")
        self._unlock_controls()
        self.log(f"Linkage comparison done. Best: {best}")
        rows = rf.linkage_comparison_rows(self.all_linkage_results, self.hpc_results, self.cost_cfg)
        best_row = next((i for i, r in enumerate(rows) if r[0] == best), None)
        self.linkage_table.populate(rows, best_row=best_row)
        if best and best in self.all_linkage_results:
            self.linkage_combo.setCurrentText(best)
            # Apply best result but don't let it switch to the optim tab;
            # we want Results tab to stay visible after the dialog closes.
            self._apply_clustering_result(self.all_linkage_results[best])
        self.tab_widget.setCurrentWidget(self.results_tab)
        ComparisonWindow(results, self).exec_()

    def _apply_clustering_result(self, results: dict):
        """Store and reflect a clustering result without switching tabs."""
        self.clustering_results = results
        self._invalidate_communication_network_results()
        if self.clustering_results:
            self._update_optim_labels()
            self.iteration_table.populate(self._format_iteration_rows(results))
            self._visualize_clustering_results(self.optim_view, self.clustering_results)
        if self.hpc_results:
            self._compare_results()
        self.comm_network_btn.setEnabled(True)

    def _invalidate_communication_network_results(self):
        """Clear topology results derived from an older clustering run."""
        self.topology_lengths = {}
        self.redundant_return = None
        self.star_topology = None
        self.ring_topology = None
        if hasattr(self, "results_summary_table"):
            self.results_summary_table.clear_data()

    def on_error(self, error_message: str):
        self._unlock_controls()
        self.log(f"  ERROR: {error_message}")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_message}")

    # ==================================================================
    # Label / table updaters
    # ==================================================================
    def _update_optim_labels(self):
        p_m   = self.cost_cfg["wire_price_per_m"]
        w_m   = self.cost_cfg["wire_weight_per_m_kg"]
        cur   = self.cost_cfg["currency"]
        total = self.clustering_results.get("total_wire_length", 0.0)
        self.optim_total_label.setText(f"I/O Wire Length: {total:,.2f} mm")
        self.optim_cost_label.setText(f"Cost: {(total/1000)*p_m:,.2f} {cur}")
        self.optim_weight_label.setText(f"Weight: {(total/1000)*w_m:,.3f} kg")

    def _format_iteration_rows(self, results: dict) -> list:
        """
        Convert iteration_results from the worker into table rows.

        Renders the delta column with a sign prefix (+/-). The first
        iteration is compared against the initial agglomerative baseline.
        """
        rows = []
        for r in results.get("iteration_results", []):
            delta = r.get("delta_vs_previous", 0.0)
            iteration = r["iteration"]
            delta_str = f"{delta:+,.2f}"
            rows.append((
                iteration,
                r["assignments_changed"],
                f'{r["total_wire_length"]:,.2f}',
                delta_str,
            ))
        return rows

    def _update_comm_labels(self):
        wire  = self.clustering_results.get("total_wire_length", 0.0)
        can   = self.clustering_results.get("can_bus", {}).get("total_length", 0.0)
        total = wire + can
        p_m   = self.cost_cfg["wire_price_per_m"]
        cp_m  = self.cost_cfg["CAN_bus_price_per_m"]
        w_m   = self.cost_cfg["wire_weight_per_m_kg"]
        cur   = self.cost_cfg["currency"]
        cost  = (wire/1000)*p_m + (can/1000)*cp_m
        self.comm_total_label.setText(f"Total with CAN FD: {total:,.2f} mm")
        self.comm_cost_label.setText(f"Cost: {cost:,.2f} {cur}")
        self.comm_weight_label.setText(f"Weight: {(total/1000)*w_m:,.3f} kg")

    def _compare_results(self):
        if not (self.clustering_results and self.hpc_results):
            return
        hpc_l = self.hpc_results.get("total_length", 0.0)
        opt_l = self.clustering_results.get("overall_wiring_harness_length", 0.0)
        if hpc_l <= 0 or opt_l <= 0:
            return
        p_m   = self.cost_cfg["wire_price_per_m"]
        cp_m  = self.cost_cfg["CAN_bus_price_per_m"]
        w_m   = self.cost_cfg["wire_weight_per_m_kg"]
        hpc_c = (hpc_l/1000)*p_m;  hpc_w = (hpc_l/1000)*w_m
        wire  = self.clustering_results.get("total_wire_length", 0)
        can   = self.clustering_results.get("can_bus", {}).get("total_length", 0)
        opt_c = (wire/1000)*p_m + (can/1000)*cp_m
        opt_w = ((wire+can)/1000)*w_m
        def pct(a, b): return f"{((a-b)/a*100):+.1f}%" if a > 0 else "—"
        self.comparison_length_saving_label.setText(f"Length Saving: {pct(hpc_l, opt_l)}")
        self.comparison_cost_saving_label.setText(f"Cost Saving:   {pct(hpc_c, opt_c)}")
        self.comparison_weight_saving_label.setText(f"Weight Saving: {pct(hpc_w, opt_w)}")
        self.log(f"  Comparison – Length: {pct(hpc_l,opt_l)}  Cost: {pct(hpc_c,opt_c)}")

    def _update_results_tables(self):
        """Refresh the Architecture Comparison and Per-Cluster tables."""
        # Architecture comparison — populated only once Step 5 has been run
        if self.topology_lengths:
            rows = rf.topology_comparison_rows(
                self.clustering_results, self.topology_lengths,
                self.hpc_results, self.cost_cfg,
            )
            # Find best (lowest total) row to highlight
            best_row = None
            best_total = float("inf")
            for i, r in enumerate(rows):
                if r[0] == "Baseline (Direct HPC)":
                    continue
                try:
                    total = float(r[3])
                    if total < best_total:
                        best_total, best_row = total, i
                except (ValueError, IndexError):
                    pass
            self.results_summary_table.populate(rows, best_row=best_row,
                                                 center_cols=[1, 2, 3, 4, 5, 6])

        # Per-cluster breakdown
        cluster_rows = rf.cluster_breakdown_rows(self.clustering_results, self.cost_cfg)
        self.results_cluster_table.populate(cluster_rows)
        # Linkage table is populated in on_linkage_comparison_completed

    # ==================================================================
    # Worker lifecycle helpers
    # ==================================================================
    def _lock_controls(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        for btn in (self.btn_process, self.elbow_btn, self.hpc_btn,
                    self.clustering_btn, self.compare_linkage_btn,
                    self.optimization_btn, self.comm_network_btn):
            btn.setEnabled(False)

    def _unlock_controls(self):
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        files = bool(self.chassis_file_path and self.io_file_path)
        graph = self.current_graph is not None
        self.btn_process.setEnabled(files)
        self.elbow_btn.setEnabled(graph)
        self.hpc_btn.setEnabled(graph)
        self.clustering_btn.setEnabled(graph)
        self.compare_linkage_btn.setEnabled(graph)
        self.optimization_btn.setEnabled(graph)
        self.comm_network_btn.setEnabled(self.clustering_results is not None)

    # ==================================================================
    # File operations
    # ==================================================================
    def _update_file_status(self):
        ch = os.path.basename(self.chassis_file_path) if self.chassis_file_path else "—"
        io = os.path.basename(self.io_file_path)      if self.io_file_path      else "—"
        self.file_status_label.setText(f"Chassis: {ch}   I/O: {io}")
        self.btn_process.setEnabled(bool(self.chassis_file_path and self.io_file_path))

    def load_chassis_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load Chassis Graph", self.config.get("paths.data_dir", "./data"), "JSON Files (*.json)")
        if fp:
            self.chassis_file_path = fp; self._update_file_status()
            self.log(f"Chassis: {os.path.basename(fp)}")

    def load_io_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load I/O Coordinates", self.config.get("paths.data_dir", "./data"), "CSV Files (*.csv)")
        if fp:
            self.io_file_path = fp; self._update_file_status()
            self.log(f"I/O: {os.path.basename(fp)}")

    def load_default_files(self):
        paths = self.config.get("paths", {})
        for attr, key, label in [
            ("chassis_file_path", "default_chassis_json", "chassis"),
            ("io_file_path",      "default_io_csv",       "I/O"),
        ]:
            val = paths.get(key)
            if val and os.path.exists(val):
                setattr(self, attr, val)
                self.log(f"Default {label}: {os.path.basename(val)}")
            else:
                self.log(f"Default {label} not found: {val}")
        self._update_file_status()

    def export_results_json(self):
        if not (self.clustering_results or self.hpc_results):
            QMessageBox.warning(self, "No Results", "Run analysis first.")
            return
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export Results",
            os.path.join(self.config.get("paths.export_dir", "./export"), f"results_{ts}.json"),
            "JSON Files (*.json)")
        if not fp:
            return
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump({"timestamp": ts, "chassis_file": self.chassis_file_path,
                           "io_file": self.io_file_path,
                           "clustering_results": self.clustering_results,
                           "hpc_results": self.hpc_results, "elbow_data": self.elbow_data,
                           "configuration": self.config.config}, f, indent=2)
            self.log(f"Exported: {os.path.basename(fp)}")
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{fp}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_report_pdf(self):
        if not self.current_graph:
            QMessageBox.warning(self, "No Data", "Process files first.")
            return
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export Report",
            os.path.join(self.config.get("paths.export_dir", "./export"), f"report_{ts}.pdf"),
            "PDF Files (*.pdf)")
        if not fp:
            return
        try:
            ReportGenerator(self).generate_pdf(fp)
            self.log(f"PDF: {os.path.basename(fp)}")
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{fp}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ==================================================================
    # Misc
    # ==================================================================
    def show_cost_settings(self):
        """Open the cost/weight settings dialog and apply changes live."""
        dlg = CostSettingsDialog(self.cost_cfg, self)
        if dlg.exec_() != dlg.Accepted:
            return
        self.cost_cfg = dlg.get_settings()
        self.log(
            f"Cost settings updated — wire: {self.cost_cfg['wire_price_per_m']:.3f}/m, "
            f"CAN: {self.cost_cfg['CAN_bus_price_per_m']:.3f}/m, "
            f"weight: {self.cost_cfg['wire_weight_per_m_kg']:.4f} kg/m  "
            f"({self.cost_cfg['currency']})"
        )

        # Refresh every display that uses cost_cfg
        if self.hpc_results:
            total = self.hpc_results.get("total_length", 0.0)
            p_m   = self.cost_cfg["wire_price_per_m"]
            w_m   = self.cost_cfg["wire_weight_per_m_kg"]
            cur   = self.cost_cfg["currency"]
            self.hpc_total_label.setText(f"Total Length: {total:,.2f} mm")
            self.hpc_cost_label.setText(f"Cost: {(total/1000)*p_m:,.2f} {cur}")
            self.hpc_weight_label.setText(f"Weight: {(total/1000)*w_m:,.3f} kg")
            self.hpc_topo.metrics_panel.update_metrics(
                rf.baseline_metrics(self.hpc_results, self.cost_cfg)
            )
        if self.clustering_results:
            self._update_optim_labels()
            self._update_comm_labels()
            if self.hpc_results:
                self._compare_results()
            # Re-apply topology metrics if Step 5 was already run
            if self.topology_lengths:
                panels = {
                    "Bus Topology":  self.bus_topo.metrics_panel,
                    "Redundant Bus": self.redundant_bus_topo.metrics_panel,
                    "Star and Ring": self.star_ring_topo.metrics_panel,
                }
                for label, panel in panels.items():
                    if label in self.topology_lengths:
                        panel.update_metrics(self._build_topology_metrics(label))
            self._update_results_tables()
        # Linkage table needs recompute too
        if self.all_linkage_results:
            rows = rf.linkage_comparison_rows(
                self.all_linkage_results, self.hpc_results, self.cost_cfg,
            )
            self.linkage_table.populate(rows)

    def show_about_dialog(self):
        QMessageBox.about(self, "About CONVIO",
            "<h3>CONVIO: Automotive Wiring Harness Optimizer</h3><p>Version 1.0</p>"
            "<p>Graph-based I/O clustering and wiring harness optimisation.</p>")

    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {message}")
        self.logger.info(message)

    def closeEvent(self, event):
        self.log("Application closing...")
        event.accept()
