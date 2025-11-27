"""
CONVIO - Main Application
=========================

This is the main entry point for the CONVIO application.
It provides a graphical user interface (GUI) built with PyQt5 for loading,
processing, and visualizing wiring harness data.

The application is structured as follows:
- ConfigManager: Handles loading and validation of the `config.yaml` file.
- OptimizationWorker: Runs heavy computations (graph loading, clustering, etc.)
  in a separate thread to keep the GUI responsive.
- WiringHarnessOptimizer: The main window class that orchestrates the UI,
  event handling, and visualization.

The workflow is designed to be sequential:
1.  Load a chassis graph (JSON) and I/O coordinates (CSV).
2.  Run an elbow method analysis to determine the optimal number of clusters.
3.  Perform clustering and centroid optimization.
4.  Calculate a baseline direct wiring from the HPC for comparison.
"""

import sys

# Initialization of QApplication BEFORE QWidget is called
from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)

import os
from typing import Optional, Dict, Any, Tuple, List
import json
import yaml
import logging
import numpy as np
import networkx as nx
import pyqtgraph as pg
from datetime import datetime


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTextEdit, QTabWidget, QSpinBox, QProgressBar, QMessageBox,
    QSplitter, QGroupBox, QGridLayout, QAction, QFormLayout, QDoubleSpinBox, QComboBox,
    QScrollArea, QCheckBox, QDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QBuffer, QIODevice
from PyQt5.QtGui import QFont, QPixmap, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# Import custom modules
from modules.report_generator import ReportGenerator
from modules.graph_loader import create_graph_loader_from_config
from modules.elbow_method import ElbowMethodAnalyzer
from modules.clustering_dijkstra import ClusteringDijkstra
from modules.hpc_connector import calculate_direct_hpc_wiring


class ConfigManager:
    """
    Manages application configuration from config.yaml (YAML-first, single source).
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        self.config_path = config_path
        self.config = self._load_config_yaml_only()
        self._setup_logging()
        self._create_directories()

    def _load_config_yaml_only(self) -> Dict[str, Any]:
        """Load configuration strictly from config.yaml."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at: {self.config_path}. Please add config.yaml next to main.py.")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("Top-level configuration must be a mapping (YAML object).")
            
            # Validate required sections/keys
            self._require_keys(data, ["paths", "graph_loader", "logging", "error_handling", "reproducibility"])
            self._require_keys(data["paths"], ["data_dir", "export_dir", "log_dir"])
            self._require_keys(data["graph_loader"], ["min_direct_node_distance_mm", "allow_projection_on_edge", "skip_self_loops"])
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load/validate configuration: {e}") from e

    def _require_keys(self, node: Dict[str, Any], keys: List[str]) -> None:
        missing = [k for k in keys if k not in node]
        if missing:
            raise ValueError(f"Missing required configuration keys: {missing}")

    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format=log_config.get("fmt", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
            datefmt=log_config.get("datefmt", "%H:%M:%S"),
        )
        
        # File logging
        if log_config.get("enable_log_to_file", True):
            log_dir = self.config["paths"]["log_dir"]
            os.makedirs(log_dir, exist_ok=True)
            log_file = self.config["paths"].get("log_file", os.path.join(log_dir, "optimizer.log"))
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(logging.Formatter(
                log_config.get("fmt", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
                datefmt=log_config.get("datefmt", "%H:%M:%S"),
            ))
            logging.getLogger().addHandler(fh)

    def _create_directories(self) -> None:
        """Create necessary directories from YAML."""
        paths = self.config["paths"]
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Adjust paths to be relative to the script's directory
        paths["data_dir"] = os.path.join(script_dir, paths["data_dir"])
        paths["export_dir"] = os.path.join(script_dir, paths["export_dir"])
        paths["log_dir"] = os.path.join(script_dir, paths["log_dir"])

        for key in ["data_dir", "export_dir", "log_dir"]:
            path = paths.get(key)
            if path:
                os.makedirs(path, exist_ok=True)

    def get(self, path: str, default=None):
        """Get configuration value using dot notation."""
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class OptimizationWorker(QThread):
    """Worker thread for running optimization tasks."""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, task_type, config_manager: ConfigManager, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.config = config_manager
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task_type == "load_graph":
                self.load_graph_task()
            elif self.task_type == "elbow_analysis":
                self.elbow_analysis_task()
            elif self.task_type == "clustering":
                self.clustering_task()
            elif self.task_type == "full_analysis":
                self.full_analysis_task()
            elif self.task_type == "hpc_wiring":
                self.hpc_wiring_task()
        except Exception as e:
            if self.config.get("error_handling.log_stack_trace_on_error", True):
                logging.exception(f"Error in {self.task_type}: {e}")
            self.error_occurred.emit(str(e))

    def load_graph_task(self):
        
        def _load_graph_internal():
            self.status_updated.emit("Loading graph data...")
            self.progress_updated.emit(10)
        
            loader = create_graph_loader_from_config(self.config.config)
            self.progress_updated.emit(25)
            
            # Load chassis
            graph = loader.load_chassis_graph(self.kwargs['chassis_file'])
            self.progress_updated.emit(50)
            
            self.status_updated.emit("Loading I/O coordinates...")
            io_points = loader.load_io_coordinates_from_csv(self.kwargs['io_file'])
            network_graph = loader.add_io_nodes_to_graph(graph, io_points)
            self.progress_updated.emit(75)
            
            export_path = loader.export_network_graph_json()
            self.status_updated.emit(f"Exported to: {os.path.basename(export_path)}")
            self.progress_updated.emit(90)
            
            # Stats and validation
            stats = WiringHarnessOptimizer._get_graph_statistics(network_graph)
            validation = WiringHarnessOptimizer._validate_graph(network_graph)
            self.progress_updated.emit(100)
            
            self.status_updated.emit("Graph loading completed!")
            
            return {"graph": network_graph, 
                    "loader": loader,
                    "statistics": stats,
                    "validation": validation,
                    "export_path": export_path
                    }
        try:
            result = _load_graph_internal()
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"Graph loading failed: {e}")
            
    def elbow_analysis_task(self):
        
        def _elbow_analysis_internal():
            self.status_updated.emit("Running elbow method analysis...")
            elbow_config = self.config.get("elbow_method", {})
            analyzer = ElbowMethodAnalyzer(
                k_min=int(elbow_config.get("k_min", 1)),
                k_max=int(elbow_config.get("k_max", 12)),
                random_state=int(elbow_config.get("random_state", 42)),
                n_init=int(elbow_config.get("n_init", 10)),
            )
            optimal_k, elbow_data = analyzer.find_optimal_clusters(self.kwargs['graph'])
            return {"optimal_k": optimal_k, "elbow_data": elbow_data}
    
        try:
            result = _elbow_analysis_internal()
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"Elbow analysis failed: {e}")


    def clustering_task(self):
        
        def _clustering_internal():
            self.status_updated.emit("Performing clustering and shortest path analysis...")
            clusterer = ClusteringDijkstra(config=self.config.config)
            n_clusters = int(self.kwargs['n_clusters'])
            linkage = self.kwargs.get('linkage_method', 'complete')  # Default to 'complete' if not provided
            results = clusterer.cluster_and_optimize(self.kwargs['graph'], n_clusters, linkage_method=linkage)
            return results
    
        try:
            result = _clustering_internal()
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"Clustering failed: {e}")

    def full_analysis_task(self):
        def _full_analysis_internal():
            self.status_updated.emit("Starting comprehensive analysis...")
            clusterer = ClusteringDijkstra(config=self.config.config)
            n_clusters = int(self.kwargs['n_clusters'])
            graph = self.kwargs['graph']
            
            linkage_methods = ['average', 'complete', 'single']
            results = {}
            best_length = float('inf')
            best_method = None

            for i, method in enumerate(linkage_methods):
                self.status_updated.emit(f"Running analysis with '{method}' linkage...")
                self.progress_updated.emit(int((i / len(linkage_methods)) * 100))
                
                run_result = clusterer.cluster_and_optimize(graph, n_clusters, linkage_method=method)
                results[method] = run_result
                
                total_length = run_result.get("overall_wiring_harness_length", float('inf'))
                if total_length < best_length:
                    best_length = total_length
                    best_method = method
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Comprehensive analysis complete!")
            
            return {"results": results, "best_method": best_method}

        try:
            result = _full_analysis_internal()
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"Full analysis failed: {e}")


    def hpc_wiring_task(self):
        
        def _hpc_wiring_internal():
            self.status_updated.emit("Calculating Overall wiring...")
            results = calculate_direct_hpc_wiring(self.kwargs['graph'], self.config.config)
            return {"hpc_wiring_results": results}
    
        try:
            result = _hpc_wiring_internal()
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"Overall wiring calculation failed: {e}")



class ComparisonWindow(QDialog):
    """A dialog window to display the comparison of clustering methods."""
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Linkage Method Comparison")
        self.setMinimumSize(600, 200)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("<b>Comprehensive Analysis Results</b>")
        title.setFont(QFont("Arial", 14))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Table
        self.table = QTableWidget()
        self.table.setRowCount(3)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Linkage Method", "I/O Wiring (mm)", "CAN Bus (mm)", "Total Length (mm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        # Populate data
        self._populate_table(data)
        
        # Recommendation text
        best_method = data.get('best_method', 'N/A')
        rec_text = f"<b>Recommendation:</b> The '<b>{best_method}</b>' linkage method produced the shortest overall wiring harness length and has been selected for detailed visualization."
        recommendation = QLabel(rec_text)
        recommendation.setWordWrap(True)
        layout.addWidget(recommendation)

    def _populate_table(self, data):
        results = data.get("results", {})
        best_method = data.get("best_method")
        
        methods = ["average", "complete", "single"]
        for row, method in enumerate(methods):
            res = results.get(method, {})
            
            # Method name
            item_method = QTableWidgetItem(method)
            
            # I/O Wiring
            wire_len = res.get("total_wire_length", 0.0)
            item_wire = QTableWidgetItem(f"{wire_len:.2f}")
            
            # Communication Network Bus
            can_len = res.get("can_bus", {}).get("total_length", 0.0)
            item_can = QTableWidgetItem(f"{can_len:.2f}")
            
            # Total Length
            total_len = res.get("overall_wiring_harness_length", 0.0)
            item_total = QTableWidgetItem(f"{total_len:.2f}")
            
            # Highlight the best row
            if method == best_method:
                font = QFont()
                font.setBold(True)
                item_method.setFont(font)
                item_wire.setFont(font)
                item_can.setFont(font)
                item_total.setFont(font)
                
            self.table.setItem(row, 0, item_method)
            self.table.setItem(row, 1, item_wire)
            self.table.setItem(row, 2, item_can)
            self.table.setItem(row, 3, item_total)

            if method == best_method:
                # Set background color for all cells in the best row
                for col in range(4):
                    self.table.item(row, col).setBackground(QColor('#d4edda'))


class MatplotlibWidget(QWidget):
    """Matplotlib widget for plots."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class WiringHarnessOptimizer(QMainWindow):
    """Main application window with integrated processing and visualization."""
    
    @staticmethod
    def _get_graph_statistics(graph):
        """Get basic graph statistics."""
        io_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_io", False)]
        chassis_nodes = [n for n, d in graph.nodes(data=True) if not d.get("is_io", False)]
        
        return {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "io_nodes": len(io_nodes),
            "chassis_nodes": len(chassis_nodes),
            "node_types": {"io": len(io_nodes), "chassis": len(chassis_nodes)}
        }

    @staticmethod
    def _validate_graph(graph):
        """Basic graph validation."""
        warnings = []
        if not nx.is_connected(graph):
            warnings.append("Graph is not connected")
        
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            warnings.append(f"Found {len(isolated_nodes)} isolated nodes")
            
        return {"warnings": warnings, "is_valid": len(warnings) == 0}

    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        self.current_graph = None
        self.graph_loader = None
        self.elbow_data: Optional[Dict[str, Any]] = None
        self.clustering_results: Optional[Dict[str, Any]] = None
        self.comparison_results: Optional[Dict[str, Any]] = None
        self.hpc_results: Optional[Dict[str, Any]] = None
        self.chassis_file_path: Optional[str] = None
        self.io_file_path: Optional[str] = None
        
        self._apply_config()
        self._init_ui()
        self._init_menu()
        self._setup_reproducibility()

    def _apply_config(self):
        """Apply configuration settings to the GUI."""
        gui_cfg = self.config.get("gui", {})
        window_size = gui_cfg.get("window_size", [1500, 900])
        self.setWindowTitle("CONVIO - Automotive Wiring Harness Optimizer")
        self.setGeometry(100, 100, int(window_size[0]), int(window_size[1]))
        self.show_grid = bool(gui_cfg.get("show_grid", True))
        
        cost_cfg = self.config.get("cost", {})
        self.cost_cfg = {
            "currency": cost_cfg.get("currency", "EURO"),
            "wire_price_per_m": float(cost_cfg.get("wire_price_per_m", 0.0)),
            "CAN_bus_price_per_m": float(cost_cfg.get("CAN_bus_price_per_m", 0.0)),
            "wire_weight_per_m_kg": float(cost_cfg.get("wire_weight_per_m_kg", 0.0)),
        }

    def _setup_reproducibility(self):
        """Setup reproducibility settings."""
        repro = self.config.get("reproducibility", {})
        if bool(repro.get("set_global_seeds", False)):
            import random
            np.random.seed(int(repro.get("numpy_seed", 42)))
            random.seed(int(repro.get("numpy_seed", 42)))

    def _init_ui(self):
        """Initialize the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left = self._create_control_panel()
        right = self._create_visualization_panel()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _init_menu(self):
        """Initialize the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        
        act_open_chassis = QAction("Open Chassis JSON...", self)
        act_open_chassis.triggered.connect(self.load_chassis_file)
        file_menu.addAction(act_open_chassis)

        act_open_io = QAction("Open I/O CSV...", self)
        act_open_io.triggered.connect(self.load_io_file)
        file_menu.addAction(act_open_io)

        file_menu.addSeparator()
        
        act_export_json = QAction("Export Results (JSON)...", self)
        act_export_json.triggered.connect(self.export_results_json)
        file_menu.addAction(act_export_json)

        act_export_pdf = QAction("Export Report (PDF)...", self)
        act_export_pdf.triggered.connect(self.export_report_pdf)
        file_menu.addAction(act_export_pdf)

        file_menu.addSeparator()
        
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        #To be discussed


        # Help menu
        help_menu = menubar.addMenu("Help")
        
        act_about = QAction("About", self)
        act_about.triggered.connect(self.show_about_dialog)
        help_menu.addAction(act_about)

    def _create_control_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("CONVIO Control Panel")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        # Step 1: Load Data
        files_group = QGroupBox("Step 1: Load Data Files")
        fg_layout = QVBoxLayout(files_group)

        self.chassis_file_label = QLabel("No chassis file selected")
        btn_chassis = QPushButton("Load Chassis Graph (JSON)")
        btn_chassis.clicked.connect(self.load_chassis_file)

        self.io_file_label = QLabel("No I/O file selected")
        btn_io = QPushButton("Load I/O Coordinates (CSV)")
        btn_io.clicked.connect(self.load_io_file)

        btn_load_defaults = QPushButton("Load Default Files")
        btn_load_defaults.clicked.connect(self.load_default_files)

        self.btn_process = QPushButton("Process Graph")
        self.btn_process.clicked.connect(self.process_graph)
        self.btn_process.setEnabled(False)

        fg_layout.addWidget(self.chassis_file_label)
        fg_layout.addWidget(btn_chassis)
        fg_layout.addWidget(self.io_file_label)
        fg_layout.addWidget(btn_io)
        fg_layout.addWidget(btn_load_defaults)
        fg_layout.addWidget(self.btn_process)

        layout.addWidget(files_group)

        # Step 2: Elbow Analysis
        elbow_group = QGroupBox("Step 2: Find Optimal Clusters")
        el_layout = QVBoxLayout(elbow_group)

        self.elbow_btn = QPushButton("Run Elbow Method Analysis")
        self.elbow_btn.clicked.connect(self.run_elbow_analysis)
        self.elbow_btn.setEnabled(False)

        self.optimal_clusters_label = QLabel("Optimal clusters: Not calculated")

        el_layout.addWidget(self.elbow_btn)
        el_layout.addWidget(self.optimal_clusters_label)

        layout.addWidget(elbow_group)
        
        # Step 3: Overall wiring Analysis
        hpc_group = QGroupBox("Step 3: Overall wiring Analysis")
        hpc_layout = QVBoxLayout(hpc_group)

        self.hpc_btn = QPushButton("Calculate Overall Wiring")
        self.hpc_btn.clicked.connect(self.run_hpc_analysis)
        self.hpc_btn.setEnabled(False)

        self.hpc_total_label = QLabel("Overall Total Length: Not calculated")
        self.hpc_cost_label = QLabel("Cost: Not calculated")
        self.hpc_weight_label = QLabel("Weight: Not calculated")

        hpc_layout.addWidget(self.hpc_btn)
        hpc_layout.addWidget(self.hpc_total_label)
        hpc_layout.addWidget(self.hpc_cost_label)
        hpc_layout.addWidget(self.hpc_weight_label)

        layout.addWidget(hpc_group)
        
        # Step 4: Clustering
        cl_group = QGroupBox("Step 4: Clustering & Optimization")
        cl_layout = QGridLayout(cl_group)

        cl_layout.addWidget(QLabel("Number of Clusters:"), 0, 0)
        
        max_clusters = int(self.config.get("clustering.max_clusters_supported", 100))
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(1, max_clusters)
        self.n_clusters_spin.setValue(3)
        cl_layout.addWidget(self.n_clusters_spin, 0, 1)

        cl_layout.addWidget(QLabel("Linkage Method:"), 1, 0)
        self.linkage_combo = QComboBox()
        self.linkage_combo.addItems(["average", "complete", "single"])
        cl_layout.addWidget(self.linkage_combo, 1, 1)

        self.clustering_btn = QPushButton("Run Clustering & Optimization")
        self.clustering_btn.clicked.connect(self.run_clustering)
        self.clustering_btn.setEnabled(False)
        cl_layout.addWidget(self.clustering_btn, 2, 0, 1, 2)

        self.full_analysis_btn = QPushButton("Run Full Analysis & Compare")
        self.full_analysis_btn.clicked.connect(self.run_full_analysis)
        self.full_analysis_btn.setEnabled(False)
        cl_layout.addWidget(self.full_analysis_btn, 3, 0, 1, 2)

        self.clustering_total_label = QLabel("Clustering Total Length: Not calculated")
        cl_layout.addWidget(self.clustering_total_label, 4, 0, 1, 2)

        self.clustering_cost_label = QLabel("Cost: Not calculated")
        cl_layout.addWidget(self.clustering_cost_label, 5, 0, 1, 2)
        
        self.clustering_weight_label = QLabel("Weight: Not calculated")
        cl_layout.addWidget(self.clustering_weight_label, 6, 0, 1, 2)

        self.clustering_with_can_label = QLabel("Total with CAN FD: Not calculated")
        cl_layout.addWidget(self.clustering_with_can_label, 7, 0, 1, 2)

        self.clustering_with_can_cost_label = QLabel("Cost: Not calculated")
        cl_layout.addWidget(self.clustering_with_can_cost_label, 8, 0, 1, 2)

        self.clustering_with_can_weight_label = QLabel("Weight: Not calculated")
        cl_layout.addWidget(self.clustering_with_can_weight_label, 9, 0, 1, 2)

        layout.addWidget(cl_group)

        # Comparison Results
        comp_group = QGroupBox("Comparison Summary")
        comp_layout = QVBoxLayout(comp_group)

        self.comparison_length_saving_label = QLabel("Length Saving: Not calculated")
        self.comparison_cost_saving_label = QLabel("Cost Saving: Not calculated")
        self.comparison_weight_saving_label = QLabel("Weight Saving: Not calculated")

        comp_layout.addWidget(self.comparison_length_saving_label)
        comp_layout.addWidget(self.comparison_cost_saving_label)
        comp_layout.addWidget(self.comparison_weight_saving_label)
        layout.addWidget(comp_group)

        


        # Progress and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to start")
        layout.addWidget(self.status_label)

        # Logging Display
        log_group = QGroupBox("CONVIO Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200) 
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        layout.addStretch()
        return panel

    def _create_visualization_panel(self) -> QWidget:
        """Create the right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.tab_widget = QTabWidget()

        # Graph View
        self.graph_view = pg.PlotWidget()
        self._setup_pg_view(self.graph_view, 'Network Graph')
        self.tab_widget.addTab(self.graph_view, "Network Graph")

        self.elbow_widget = MatplotlibWidget()
        self.tab_widget.addTab(self.elbow_widget, "Elbow Analysis")
        
        # Overall Wiring
        self.hpc_view = pg.PlotWidget()
        self._setup_pg_view(self.hpc_view, 'Overall Wiring')
        self.tab_widget.addTab(self.hpc_view, "Overall Wiring")        
        
        # Clustering Results
        self.cluster_view = pg.PlotWidget()
        self._setup_pg_view(self.cluster_view, 'EEA with I/O aggregators')
        self.tab_widget.addTab(self.cluster_view, "EEA with I/O aggregators")

        

        layout.addWidget(self.tab_widget)
        return panel

    def _setup_pg_view(self, view: pg.PlotWidget, title: str):
        """Setup PyQtGraph view with common settings."""
        view.setBackground('w')
        view.setLabel('left', 'Y (mm)')
        view.setLabel('bottom', 'X (mm)')
        view.setTitle(title)
        view.showGrid(x=True, y=True)
        view.setAspectLocked(True, ratio=1.0)
        view.addLegend()

    # ==== ROBUST PROCESSING FUNCTIONS ====



    def process_graph(self):
        """Process and load the graph data."""
        if not (self.chassis_file_path and self.io_file_path):
            QMessageBox.warning(self, "Missing files", "Please select both Chassis JSON and I/O CSV.")
            return

        self._start_worker_task()
        self.worker = OptimizationWorker(
            "load_graph", self.config,
            chassis_file=self.chassis_file_path,
            io_file=self.io_file_path
        )
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_graph_loaded)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def run_elbow_analysis(self):
        """Run elbow method analysis."""
        if not self.current_graph:
            QMessageBox.warning(self, "No graph", "Load and process files first.")
            return

        self._start_worker_task()
        self.worker = OptimizationWorker("elbow_analysis", self.config, graph=self.current_graph)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_elbow_completed)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def run_clustering(self):
        """Run clustering and optimization."""
        if not self.current_graph:
            QMessageBox.warning(self, "No graph", "Load and process files first.")
            return

        n_clusters = int(self.n_clusters_spin.value())
        linkage_method = self.linkage_combo.currentText()
        self._start_worker_task()
        self.worker = OptimizationWorker("clustering", self.config, graph=self.current_graph, n_clusters=n_clusters, linkage_method=linkage_method)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_clustering_completed)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def run_full_analysis(self):
        """Run comprehensive analysis comparing all linkage methods."""
        if not self.current_graph:
            QMessageBox.warning(self, "No graph", "Load and process files first.")
            return

        n_clusters = int(self.n_clusters_spin.value())
        self._start_worker_task()
        self.worker = OptimizationWorker("full_analysis", self.config, graph=self.current_graph, n_clusters=n_clusters)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_full_analysis_completed)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def run_hpc_analysis(self):
        """Run Overall wiring analysis."""
        if not self.current_graph:
            QMessageBox.warning(self, "No graph", "Load and process files first.")
            return

        self._start_worker_task()
        self.worker = OptimizationWorker("hpc_wiring", self.config, graph=self.current_graph)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_hpc_completed)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    # ==== EVENT HANDLERS ====

    def on_graph_loaded(self, results):
        """Handle graph loading completion."""
        self.current_graph = results["graph"]
        self.graph_loader = results["loader"]
        self._finish_worker_task()

        stats = results.get("statistics", {})
        validation = results.get("validation", {})
        export_path = results.get("export_path", None)

        num_chassis = stats.get("chassis_nodes", 0)
        num_io = stats.get("io_nodes", 0)

        self.log(f"Graph loaded: {num_chassis} chassis nodes, {num_io} I/O points")
        
        for w in validation.get("warnings", []):
            self.log(f" Warning: {w}")

        # Update visualization
        self._visualize_graph(self.current_graph)

        # Enable next steps
        self.elbow_btn.setEnabled(True)
        self.hpc_btn.setEnabled(True)
        self.full_analysis_btn.setEnabled(True)
      



    def on_elbow_completed(self, results):
        """Handle elbow analysis completion."""
        optimal_k = results["optimal_k"]
        self.elbow_data = results["elbow_data"]
        self._finish_worker_task()

        self.optimal_clusters_label.setText(f"Optimal clusters: {optimal_k}")
        self.log(f"Elbow analysis completed. Optimal clusters: {optimal_k}")
        
        max_clusters = int(self.config.get("clustering.max_clusters_supported", 100))
        clamped_k = min(optimal_k, max_clusters)
        
        if clamped_k != optimal_k:
            self.log(f" Optimal k ({optimal_k}) exceeds max supported ({max_clusters}), using {clamped_k}")
        
        self.n_clusters_spin.setValue(clamped_k)
        self.clustering_btn.setEnabled(True)

        # Visualize elbow curve
        self._visualize_elbow_analysis()
        #self.log(f" Elbow analysis completed. Optimal clusters: {optimal_k}")



    def on_full_analysis_completed(self, results):
        """Handle completion of the full analysis and show comparison window."""
        self.comparison_results = results
        best_method = results.get('best_method')
        
        if best_method and best_method in results.get('results', {}):
            # Set the main clustering_results to the best one found
            best_result_data = results['results'][best_method]
            self.on_clustering_completed(best_result_data) # This will update UI and visuals
        else:
            self.log("Full analysis completed, but no best method was determined.")
            self._finish_worker_task()

        # Show the comparison dialog
        dialog = ComparisonWindow(results, self)
        dialog.exec_()

    def on_clustering_completed(self, results):
        """Handle clustering completion."""
        self.clustering_results = results
        self._finish_worker_task()
        
        if self.clustering_results:
            price_per_m = self.cost_cfg.get("wire_price_per_m", 0.0)
            currency = self.cost_cfg.get("currency", "")

            total_length = self.clustering_results.get("total_wire_length", 0.0)
            self.clustering_total_label.setText(f"Clustering Wiring Length: {total_length:.2f} mm")
            
            price_per_m = self.cost_cfg.get("wire_price_per_m", 0.0)
            currency = self.cost_cfg.get("currency", "")
            wire_weight_per_m_kg = self.cost_cfg.get("wire_weight_per_m_kg", 0.0)

            cost_without_can = (total_length / 1000.0) * price_per_m
            weight_without_can = (total_length / 1000.0) * wire_weight_per_m_kg
            
            self.clustering_cost_label.setText(f"Cost: {cost_without_can:.2f} {currency}")
            self.clustering_weight_label.setText(f"Weight: {weight_without_can:.3f} kg")

            total_with_can = self.clustering_results.get("overall_wiring_harness_length", 0.0)
            self.clustering_with_can_label.setText(f"Total with CAN FD: {total_with_can:.2f} mm")
            
            can_price_per_m = self.cost_cfg.get("CAN_bus_price_per_m", price_per_m)
            can_bus_length = total_with_can - total_length
            can_price_per_m = self.cost_cfg.get("CAN_bus_price_per_m", price_per_m)
            wire_weight_per_m_kg = self.cost_cfg.get("wire_weight_per_m_kg", 0.0)
            can_bus_length = total_with_can - total_length
            
            cost_with_can = ((total_length / 1000.0) * price_per_m) + ((can_bus_length / 1000.0) * can_price_per_m)
            weight_with_can = ((total_length / 1000.0) * wire_weight_per_m_kg) + ((can_bus_length / 1000.0) * wire_weight_per_m_kg) # Assuming CAN bus has same weight per meter for now
            
            self.clustering_with_can_cost_label.setText(f"Cost: {cost_with_can:.2f} {currency}")
            self.clustering_with_can_weight_label.setText(f"Weight: {weight_with_can:.3f} kg")

        # Visualize clustering results and update comparison
        self._visualize_clustering_results(self.cluster_view, self.clustering_results)
        if self.hpc_results:
            self._compare_results()



    def on_hpc_completed(self, results):
        """Handle HPC analysis completion."""
        self.hpc_results = results["hpc_wiring_results"]
        self._finish_worker_task()

        if self.hpc_results:
            total_length = self.hpc_results.get("total_length", 0.0)
            self.hpc_total_label.setText(f"Overall Wiring Length: {total_length:.2f} mm")

            price_per_m = self.cost_cfg.get("wire_price_per_m", 0.0)
            currency = self.cost_cfg.get("currency", "")
            wire_weight_per_m_kg = self.cost_cfg.get("wire_weight_per_m_kg", 0.0)
            
            cost = (total_length / 1000.0) * price_per_m
            weight = (total_length / 1000.0) * wire_weight_per_m_kg # Convert mm to m for weight calculation
            
            self.hpc_cost_label.setText(f"Cost: {cost:.2f} {currency}")
            self.hpc_weight_label.setText(f"Weight: {weight:.3f} kg")
            
            # Visualize HPC wiring
            self._visualize_hpc_wiring()

            # Compare results if clustering is done
            if self.clustering_results:
                self._compare_results()
        else:
            self.log(" Overall wiring analysis failed.")



    def on_error(self, error_message: str):
        """Handle worker errors."""
        self._finish_worker_task()
        self.log(f" ERROR: {error_message}")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_message}")
        
    

    # ==== VISUALIZATION FUNCTIONS ====

    def _visualize_graph(self, graph):
        """Visualize the network graph."""
        self.graph_view.clear()
        self.graph_view.addLegend().clear()
        pos = self._get_node_positions(graph)
        
        if not pos:
            self.log("No positions to plot")
            return

        # Draw edges
        self._draw_graph_edges(self.graph_view, graph, pos, name="Chassis Edge")
        
        # Draw nodes
        self._draw_graph_nodes(self.graph_view, graph, pos)
        
        # Set view limits
        self._set_view_limits(self.graph_view, pos)
        
        # Switch to graph tab
        self.tab_widget.setCurrentWidget(self.graph_view)

    def _visualize_elbow_analysis(self):
        """Visualize elbow analysis results."""
        if not self.elbow_data:
            return

        self.elbow_widget.figure.clear()
        ax = self.elbow_widget.figure.add_subplot(111)
        
        k_values = self.elbow_data.get("k_values", [])
        wcss = self.elbow_data.get("wcss", [])
        optimal_k = self.elbow_data.get("elbow_k", 1)
        
        ax.plot(k_values, wcss, 'bo-', markersize=8, linewidth=2)
        ax.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Within-Cluster Sum of Squares')
        ax.set_title('Elbow Method for Optimal k')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.elbow_widget.figure.tight_layout()
        self.elbow_widget.canvas.draw()
        
        # Switch to elbow tab
        self.tab_widget.setCurrentWidget(self.elbow_widget)

    def _visualize_clustering_results(self, view: pg.PlotWidget, clustering_results: Dict[str, Any]):
        """
        Visualize clustering results on a given PlotWidget.
        This is decoupled from the main UI to allow for off-screen rendering for reports.
        """
        if not clustering_results or not self.current_graph:
            self.log("Clustering results not available or missing graph object for visualization.")
            return

        view.clear()
        view.addLegend().clear()
        
        pos = self._get_node_positions(self.current_graph)
        
        # 1. Draw base chassis graph (nodes and edges)
        chassis_nodes = [n for n, d in self.current_graph.nodes(data=True) if not d.get("is_io")]
        chassis_graph = self.current_graph.subgraph(chassis_nodes)
        self._draw_graph_edges(view, chassis_graph, pos, alpha=0.2, name="Chassis Edge")
        self._draw_graph_nodes(view, chassis_graph, pos)

        # 2. Draw I/O nodes, paths, and centroids by cluster
        clusters = clustering_results.get("clusters", {})
        colors = self._get_cluster_colors()
        legend_items_added = set()

        for i, (cluster_id, cluster_data) in enumerate(clusters.items()):
            color = colors[i % len(colors)]
            
            # Draw I/O nodes for this cluster
            io_nodes_in_cluster = cluster_data.get("io_nodes", [])
            io_points = [pos[node] for node in io_nodes_in_cluster if node in pos]
            if io_points:
                points = np.array(io_points)
                style = self._get_node_style("", type_name_override="io")
                view.addItem(
                    pg.ScatterPlotItem(
                        points[:, 0], points[:, 1],
                        size=style.get("size", 7),
                        symbol=style.get("symbol", 'o'),
                        brush=pg.mkBrush(color),
                        pen=pg.mkPen('k', width=0.5)
                    )
                )

            # Draw wiring paths for this cluster
            centroid_info = cluster_data.get("centroid")
            wiring_paths = cluster_data.get("wiring_paths", {})
            
            is_can_bus = cluster_id == "can_bus"
            path_name = "Communication Network Bus Path" if is_can_bus else f"Cluster {i} Wiring"
            
            for io_node, path_data in wiring_paths.items():
                path = path_data.get("path", [])
                if centroid_info and "pos" in centroid_info:
                    vis_path_points = [centroid_info["pos"]] + [pos[node] for node in path if node in pos]
                    if len(vis_path_points) > 1:
                        xs = [p[0] for p in vis_path_points]
                        ys = [p[1] for p in vis_path_points]
                        
                        current_path_name = None
                        if path_name not in legend_items_added:
                            current_path_name = path_name
                            legend_items_added.add(path_name)
                        
                        view.addItem(
                            pg.PlotCurveItem(xs, ys, pen=pg.mkPen(color, width=2.5, style=Qt.DashLine), name=current_path_name)
                        )

            # Draw I/O aggregator (centroid) node and its label
            if centroid_info and "pos" in centroid_info and not is_can_bus:
                cx, cy = centroid_info["pos"]
                view.addItem(
                    pg.ScatterPlotItem(
                        [cx], [cy],
                        size=12,
                        symbol='s',
                        brush=pg.mkBrush(color),
                        pen=pg.mkPen('k', width=1.5)
                    )
                )
                label = pg.TextItem(f"I/O Aggregator {cluster_id.split('_')[-1]}", color=(0,0,0), anchor=(0.5, -1.0))
                label.setPos(cx, cy)
                view.addItem(label)

        self._set_view_limits(view, pos)
        if view == self.cluster_view:
            self.tab_widget.setCurrentWidget(self.cluster_view)

    def _visualize_hpc_wiring(self):
        """Visualize HPC wiring results."""
        if not self.hpc_results or not self.current_graph:
            return

        self.hpc_view.clear()
        self.hpc_view.addLegend().clear()
        pos = self._get_node_positions(self.current_graph)
        
        # Draw base graph lightly
        self._draw_graph_edges(self.hpc_view, self.current_graph, pos, alpha=0.2, name="Chassis Edge")
        self._draw_graph_nodes(self.hpc_view, self.current_graph, pos, alpha=0.3)
        
        # Draw HPC wiring paths
        hpc_node = self.hpc_results.get("hpc_node", "H1")
        paths = self.hpc_results.get("paths", {})
        legend_item_added = False
        
        for io_node, path_data in paths.items():
            path = path_data.get("path", [])
            if len(path) > 1:
                xs, ys = [], []
                for node in path:
                    if node in pos:
                        xs.append(pos[node][0])
                        ys.append(pos[node][1])
                
                if len(xs) > 1:
                    name = None
                    if not legend_item_added:
                        name = "Direct HPC Wiring"
                        legend_item_added = True
                    
                    self.hpc_view.addItem(
                        pg.PlotCurveItem(
                            xs, ys, pen=pg.mkPen('r', width=3, style=Qt.DashLine), name=name
                        )
                    )

        self._set_view_limits(self.hpc_view, pos)
        
        # Switch to HPC tab
        self.tab_widget.setCurrentWidget(self.hpc_view)

    # ==== UTILITY FUNCTIONS ====

    def generate_clustering_plot_for_report(self, clustering_results: Dict[str, Any], title: str) -> bytes:
        """
        Generates a clustering plot image in memory for a given result set.
        This is used by the report generator to create visuals for all linkage methods
        without needing to render them in the main UI.
        """
        # Create a temporary, off-screen plot widget
        temp_view = pg.PlotWidget()
        self._setup_pg_view(temp_view, title)
        
        # Use the refactored visualization method to draw on the temporary widget
        self._visualize_clustering_results(temp_view, clustering_results)
        
        # Export the contents of the temporary widget to image bytes
        return self._export_plot_to_image_bytes(temp_view)

    def _export_plot_to_image_bytes(self, plot_widget: pg.PlotWidget) -> bytes:
        """
        Exports a pyqtgraph PlotWidget to an in-memory PNG and returns the bytes.
        
        This allows non-GUI modules like the report generator to access plot
        visualizations without directly handling GUI objects.

        Args:
            plot_widget: The pyqtgraph widget to export.

        Returns:
            The PNG image data as a byte string.
        """
        pixmap = plot_widget.grab()
        
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        # Save the pixmap to the buffer in PNG format. PNG is chosen for its
        # lossless compression, which is ideal for line art like graphs.
        pixmap.save(buffer, "PNG")
        buffer.seek(0)
        
        return buffer.readAll().data()

    def _get_node_positions(self, graph) -> Dict[str, Tuple[float, float]]:
        """Extract node positions from graph."""
        pos = {}
        for node, data in graph.nodes(data=True):
            node_pos = data.get('pos')
            if node_pos and len(node_pos) >= 2:
                try:
                    pos[node] = (float(node_pos[0]), float(node_pos[1]))
                except (ValueError, TypeError):
                    continue
        return pos

    def _draw_graph_edges(self, view, graph, pos, alpha=1.0, name=None):
        """Draw graph edges."""
        xs, ys = [], []
        for u, v in graph.edges():
            if u in pos and v in pos:
                xu, yu = pos[u]
                xv, yv = pos[v]
                xs.extend([xu, xv, np.nan])
                ys.extend([yu, yv, np.nan])
        
        if xs:
            pen_color = (0, 0, 0, int(255 * alpha))
            view.addItem(
                pg.PlotCurveItem(
                    xs, ys, pen=pg.mkPen(pen_color, width=1), connect='finite', name=name
                )
            )

    def _get_node_style(self, node_name: str, type_name_override: str = None) -> Dict[str, Any]:
        """Get node style from config based on prefix."""
        node_defs = self.config.get("node_definitions", {})
        node_types = node_defs.get("node_types", {})
        
        # Determine the type name to use
        type_to_find = type_name_override
        if not type_to_find:
            for name, info in node_types.items():
                for prefix in info.get("prefixes", []):
                    if node_name.startswith(prefix):
                        type_to_find = name
                        break
                if type_to_find:
                    break
        
        type_info = node_types.get(type_to_find, node_defs.get("default_node", {}))

        # Convert hex color to pg.Color
        hex_color = type_info.get("color", "#9B9B9B").lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return {
            "name": type_info.get("description", "Unknown Node"),
            "brush": pg.mkBrush(color=rgb),
            "pen": pg.mkPen(color=tuple(c * 0.6 for c in rgb)),
            "size": type_info.get("size", 6),
            "symbol": type_info.get("symbol", "o")
        }
        
    def _draw_graph_nodes(self, view, graph, pos, alpha=1.0):
        """Draw graph nodes based on config-driven styles."""
        
        # Group nodes by style for efficient plotting
        nodes_by_style = {}
        node_types_in_graph = set()

        for node, data in graph.nodes(data=True):
            if node not in pos:
                continue
            
            # Determine node type for styling and legend
            node_type_name = "default"
            node_defs = self.config.get("node_definitions", {}).get("node_types", {})

            if data.get("is_io"):
                node_type_name = "io"
            else:
                for type_name, type_info in node_defs.items():
                    for prefix in type_info.get("prefixes", []):
                        if node.startswith(prefix):
                            node_type_name = type_name
                            break
                    if node_type_name != "default":
                        break
            
            node_types_in_graph.add(node_type_name)
            
            style = self._get_node_style(node, type_name_override=node_type_name)
            style_key = style["name"]
            
            if style_key not in nodes_by_style:
                nodes_by_style[style_key] = {"style": style, "points": []}
            
            nodes_by_style[style_key]["points"].append(pos[node])

        # Plot each group of nodes and add to legend
        for style_key, group in nodes_by_style.items():
            points = np.array(group["points"])
            style = group["style"]
            
            view.addItem(
                pg.ScatterPlotItem(
                    points[:, 0], points[:, 1],
                    size=style["size"],
                    symbol=style["symbol"],
                    brush=style["brush"],
                    pen=style["pen"],
                    name=style["name"]
                )
            )

    def _set_view_limits(self, view, pos):
        """Set appropriate view limits for the plot."""
        if not pos:
            view.setXRange(-1, 1, padding=0.05)
            view.setYRange(-1, 1, padding=0.05)
            return

        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])
        
        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()
        max_range = max(x_range, y_range, 1.0)
        
        x_center = (xs.min() + xs.max()) / 2
        y_center = (ys.min() + ys.max()) / 2
        
        padding = 0.1 * max_range
        
        view.setXRange(x_center - max_range/2 - padding, x_center + max_range/2 + padding)
        view.setYRange(y_center - max_range/2 - padding, y_center + max_range/2 + padding)

    def _get_cluster_colors(self):
        """Get cluster colors from configuration."""
        gui_config = self.config.get("gui", {})
        if "color_palette" in gui_config:
            colors = []
            for hex_color in gui_config["color_palette"]:
                hex_color = hex_color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                colors.append(pg.mkColor(rgb))
            return colors
        
        # Default colors
        return [pg.mkColor(c) for c in [
            (141, 211, 199), (255, 255, 179), (190, 186, 218), (251, 128, 114),
            (128, 177, 211), (253, 180, 98), (179, 222, 105), (252, 205, 229)
        ]]

    def _start_worker_task(self):
        """Start a worker task - show progress and disable controls."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_process.setEnabled(False)
        self.elbow_btn.setEnabled(False)
        self.clustering_btn.setEnabled(False)
        self.full_analysis_btn.setEnabled(False)
        self.hpc_btn.setEnabled(False)
    

    def _finish_worker_task(self):
        """Finish a worker task - hide progress and re-enable controls."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        
        # Re-enable appropriate controls based on current state
        if self.chassis_file_path and self.io_file_path:
            self.btn_process.setEnabled(True)
           
        
        if self.current_graph:
            self.elbow_btn.setEnabled(True)
            self.clustering_btn.setEnabled(True)
            self.full_analysis_btn.setEnabled(True)
            self.hpc_btn.setEnabled(True)

    def _compare_results(self):
        """Compare clustering (with and without CAN) vs HPC results."""
        if not (self.clustering_results and self.hpc_results):
            return

        cluster_length_without_can = self.clustering_results.get("total_wire_length", 0.0)
        cluster_length_with_can = self.clustering_results.get("overall_wiring_harness_length", 0.0)
        hpc_length = self.hpc_results.get("total_length", 0.0)

        if hpc_length > 0:
            self.log(f" Comparison: EEA with I/O Aggregator vs. Overall Wiring")
            self.log(f"   Overall Wiring: {hpc_length:.2f} mm")

            hpc_cost = (hpc_length / 1000.0) * self.cost_cfg.get("wire_price_per_m", 0.0)
            hpc_weight = (hpc_length / 1000.0) * self.cost_cfg.get("wire_weight_per_m_kg", 0.0)

            if cluster_length_with_can > 0:
                cluster_cost = ((self.clustering_results.get("total_wire_length", 0) / 1000.0) * self.cost_cfg.get("wire_price_per_m", 0.0)) + \
                               ((self.clustering_results.get("can_bus", {}).get("total_length", 0) / 1000.0) * self.cost_cfg.get("CAN_bus_price_per_m", 0.0))
                cluster_weight = ((self.clustering_results.get("total_wire_length", 0) / 1000.0) * self.cost_cfg.get("wire_weight_per_m_kg", 0.0)) + \
                                 ((self.clustering_results.get("can_bus", {}).get("total_length", 0) / 1000.0) * self.cost_cfg.get("wire_weight_per_m_kg", 0.0))

                length_saving_percent = ((hpc_length - cluster_length_with_can) / hpc_length) * 100
                cost_saving_percent = ((hpc_cost - cluster_cost) / hpc_cost) * 100 if hpc_cost > 0 else 0
                weight_saving_percent = ((hpc_weight - cluster_weight) / hpc_weight) * 100 if hpc_weight > 0 else 0

                self.comparison_length_saving_label.setText(f"Length Saving: {length_saving_percent:.1f}%")
                self.comparison_cost_saving_label.setText(f"Cost Saving: {cost_saving_percent:.1f}%")
                self.comparison_weight_saving_label.setText(f"Weight Saving: {weight_saving_percent:.1f}%")
                
                self.log(f"   Zonal EEA (with BUS): {cluster_length_with_can:.2f} mm -> Length Improvement: {length_saving_percent:.1f}%")
                self.log(f"   Zonal EEA (with BUS): Cost Improvement: {cost_saving_percent:.1f}%")
                self.log(f"   Zonal EEA (with BUS): Weight Improvement: {weight_saving_percent:.1f}%")
            else:
                self.comparison_length_saving_label.setText("Length Saving: N/A")
                self.comparison_cost_saving_label.setText("Cost Saving: N/A")
                self.comparison_weight_saving_label.setText("Weight Saving: N/A")

    # ==== FILE OPERATIONS ====

    def load_chassis_file(self):
        """Load chassis file."""
        data_dir = self.config.get("paths.data_dir", "./data")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Chassis Graph", data_dir, "JSON Files (*.json)"
        )
        if file_path:
            self.chassis_file_path = file_path
            self.chassis_file_label.setText(f"Chassis: {os.path.basename(file_path)}")
            self._check_files_loaded()
            self.log(f" Loaded chassis: {os.path.basename(file_path)}")

    def load_io_file(self):
        """Load I/O file."""
        data_dir = self.config.get("paths.data_dir", "./data")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load I/O Coordinates", data_dir, "CSV Files (*.csv)"
        )
        if file_path:
            self.io_file_path = file_path
            self.io_file_label.setText(f"I/O: {os.path.basename(file_path)}")
            self._check_files_loaded()
            self.log(f" Loaded I/O: {os.path.basename(file_path)}")

    def load_default_files(self):
        """Load default files from configuration."""
        paths = self.config.get("paths", {})
        default_chassis = paths.get("default_chassis_json")
        default_io = paths.get("default_io_csv")

        if default_chassis and os.path.exists(default_chassis):
            self.chassis_file_path = default_chassis
            self.chassis_file_label.setText(f"Chassis: {os.path.basename(default_chassis)}")
            self.log(f" Loaded default chassis: {os.path.basename(default_chassis)}")
        else:
            self.log(f" Default chassis file not found: {default_chassis}")

        if default_io and os.path.exists(default_io):
            self.io_file_path = default_io
            self.io_file_label.setText(f"I/O: {os.path.basename(default_io)}")
            self.log(f" Loaded default I/O: {os.path.basename(default_io)}")
        else:
            self.log(f" Default I/O file not found: {default_io}")

        self._check_files_loaded()

    def _check_files_loaded(self):
        """Check if both files are loaded and enable processing."""
        both_loaded = bool(self.chassis_file_path and self.io_file_path)
        self.btn_process.setEnabled(both_loaded)
        

    def export_results_json(self):
        """Export results to JSON."""
        if not (self.clustering_results or self.hpc_results):
            QMessageBox.warning(self, "No Results", "No results to export. Run analysis first.")
            return

        export_dir = self.config.get("paths.export_dir", "./export")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(export_dir, f"optimization_results_{timestamp}.json")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", default_path, "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                results = {
                    "timestamp": timestamp,
                    "chassis_file": self.chassis_file_path,
                    "io_file": self.io_file_path,
                    "clustering_results": self.clustering_results,
                    "hpc_results": self.hpc_results,
                    "elbow_data": self.elbow_data,
                    "configuration": self.config.config,
                }
                
                with open(file_path, 'w', encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                
                self.log(f" Results exported: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
            except Exception as e:
                error_msg = f"Failed to export results: {e}"
                self.log(f" {error_msg}")
                QMessageBox.critical(self, "Export Error", error_msg)

    def export_report_pdf(self):
        """Export comprehensive PDF report via ReportGenerator module."""
        if not self.current_graph:
            QMessageBox.warning(self, "No Data", "No data available to generate a report. Please process files first.")
            return

        export_dir = self.config.get("paths.export_dir", "./export")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(export_dir, f"wiring_report_{timestamp}.pdf")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", default_path, "PDF Files (*.pdf)"
        )
        
        if not file_path:
            return

        try:
            report_generator = ReportGenerator(self)
            report_generator.generate_pdf(file_path)
            
            self.log(f" PDF report exported: {os.path.basename(file_path)}")
            QMessageBox.information(self, "Export Complete", f"Report exported to:\n{file_path}")
            
        except Exception as e:
            error_msg = f"Failed to export PDF: {e}"
            self.log(f" {error_msg}")
            QMessageBox.critical(self, "Export Error", error_msg)

    def show_about_dialog(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About CONVIO",
            "<h3>CONVIO: Automotive Wiring Harness Optimizer</h3>"
            "<p>Version 1.0</p>"
            "<p>CONVIO is a specialized engineering tool designed to streamline and "
            "optimize the complex process of automotive wiring harness design. "
            "By leveraging advanced computational methods, CONVIO provides a robust "
            "platform for engineers to design, analyze, and refine wiring layouts "
            "with greater efficiency and precision.</p>"
            "<b>Core Capabilities:</b>"
            "<ul>"
            "<li><b>Graph-Based System Modeling:</b> Accurately models the vehicle chassis and I/O points.</li>"
            "<li><b>Automated Cluster Analysis:</b> Uses the elbow method to scientifically determine the optimal number of I/O clusters.</li>"
            "<li><b>Shortest-Path Optimization:</b> Implements Dijkstra's algorithm for efficient wiring path calculation.</li>"
            "<li><b>Comparative Analysis:</b> Benchmarks optimized zonal architectures against traditional direct-to-HPC wiring.</li>"
            "<li><b>Insightful Reporting:</b> Generates comprehensive reports and visualizations to support design decisions.</li>"
            "</ul>"
            "<p>Built with flexibility in mind, CONVIO's entire workflow is controlled "
            "through a clear and concise YAML configuration, ensuring adaptability and reproducibility.</p>"
        )

    def log(self, message: str):
        """Log message to the log text display and logger."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        self.logger.info(message)

    def closeEvent(self, event):
        """Handle application close event."""
        self.log("Application closing...")
        event.accept()


def main():
    """Main application entry point."""
    # app = QApplication(sys.argv)
    
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
