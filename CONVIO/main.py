""" Automotive Wiring Harness Optimizer - Main Application """

import sys
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
    QScrollArea, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Import custom modules
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
        
           
            # YAML-driven loader
            loader = create_graph_loader_from_config(self.config.config)
            self.progress_updated.emit(25)
            
            # Load chassis
            graph = loader.load_chassis_graph(self.kwargs['chassis_file'])
            self.progress_updated.emit(50)
            
            # Load IO and add IO nodes
            self.status_updated.emit("Loading I/O coordinates...")
            io_points = loader.load_io_coordinates_from_csv(self.kwargs['io_file'])
            enhanced_graph = loader.add_io_nodes_to_graph(graph, io_points)
            self.progress_updated.emit(75)
            
            # Export enhanced graph
            export_path = loader.export_enhanced_graph_json()
            self.status_updated.emit(f"Exported to: {os.path.basename(export_path)}")
            self.progress_updated.emit(90)
            
            # Stats and validation
            stats = self._get_graph_statistics(enhanced_graph)
            validation = self._validate_graph(enhanced_graph)
            self.progress_updated.emit(100)
            
            self.status_updated.emit("Graph loading completed!")
            
            return {"graph": enhanced_graph, 
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
            results = clusterer.cluster_and_optimize(self.kwargs['graph'], n_clusters)
            return results
    
        try:
            result = _clustering_internal()
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"Clustering failed: {e}")


    def hpc_wiring_task(self):
        
        def _hpc_wiring_internal():
            self.status_updated.emit("Calculating HPC wiring...")
            results = calculate_direct_hpc_wiring(self.kwargs['graph'], self.config.config)
            return {"hpc_wiring_results": results}
    
        try:
            result = _hpc_wiring_internal()
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"HPC wiring calculation failed: {e}")

    def _get_graph_statistics(self, graph):
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

    def _validate_graph(self, graph):
        """Basic graph validation."""
        warnings = []
        if not nx.is_connected(graph):
            warnings.append("Graph is not connected")
        
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            warnings.append(f"Found {len(isolated_nodes)} isolated nodes")
            
        return {"warnings": warnings, "is_valid": len(warnings) == 0}


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
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # State variables
      
        self.current_graph = None
        self.graph_loader = None
        self.elbow_data: Optional[Dict[str, Any]] = None
        self.clustering_results: Optional[Dict[str, Any]] = None
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
        self.setWindowTitle("Automotive Wiring Harness Optimizer")
        self.setGeometry(100, 100, int(window_size[0]), int(window_size[1]))
        self.show_grid = bool(gui_cfg.get("show_grid", True))
        
        cost_cfg = self.config.get("cost", {})
        self.cost_cfg = {
            "currency": cost_cfg.get("currency", "USD"),
            "wire_price_per_m": float(cost_cfg.get("wire_price_per_m", 0.0)),
            "connector_price_each": float(cost_cfg.get("connector_price_each", 0.0)),
            "labor_price_per_m": float(cost_cfg.get("labor_price_per_m", 0.0)),
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
        
        act_process_all = QAction("Process All Steps", self)
        act_process_all.triggered.connect(self.process_all_steps)
        tools_menu.addAction(act_process_all)

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
        title = QLabel("Wiring Harness Optimizer")
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

        # Step 3: Clustering
        cl_group = QGroupBox("Step 3: Clustering & Optimization")
        cl_layout = QGridLayout(cl_group)

        cl_layout.addWidget(QLabel("Number of Clusters:"), 0, 0)
        
        max_clusters = int(self.config.get("clustering.max_clusters_supported", 100))
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(1, max_clusters)
        self.n_clusters_spin.setValue(3)
        cl_layout.addWidget(self.n_clusters_spin, 0, 1)

        self.clustering_btn = QPushButton("Run Clustering & Optimization")
        self.clustering_btn.clicked.connect(self.run_clustering)
        self.clustering_btn.setEnabled(False)
        cl_layout.addWidget(self.clustering_btn, 1, 0, 1, 2)

        layout.addWidget(cl_group)

        # Step 4: HPC Analysis
        hpc_group = QGroupBox("Step 4: HPC Baseline Analysis")
        hpc_layout = QVBoxLayout(hpc_group)

        self.hpc_btn = QPushButton("Calculate HPC Wiring")
        self.hpc_btn.clicked.connect(self.run_hpc_analysis)
        self.hpc_btn.setEnabled(False)

        self.hpc_total_label = QLabel("HPC Total Length: Not calculated")

        hpc_layout.addWidget(self.hpc_btn)
        hpc_layout.addWidget(self.hpc_total_label)

        layout.addWidget(hpc_group)

        # Process All Button
        self.btn_process_all = QPushButton("🚀 Process All Steps")
        self.btn_process_all.clicked.connect(self.process_all_steps)
        self.btn_process_all.setEnabled(False)
        self.btn_process_all.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }")
        layout.addWidget(self.btn_process_all)

        # Progress and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to start")
        layout.addWidget(self.status_label)

        # Results/Log
        results_group = QGroupBox("Results Summary")
        rl = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        rl.addWidget(self.results_text)

        layout.addWidget(results_group)

        layout.addStretch()
        return panel

    def _create_visualization_panel(self) -> QWidget:
        """Create the right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.tab_widget = QTabWidget()

        # Graph View
        self.graph_view = pg.PlotWidget()
        self._setup_pg_view(self.graph_view, 'Enhanced Graph')
        self.tab_widget.addTab(self.graph_view, "Enhanced Graph")

        # Elbow Analysis
        self.elbow_widget = MatplotlibWidget()
        self.tab_widget.addTab(self.elbow_widget, "Elbow Analysis")

        # Clustering Results
        self.cluster_view = pg.PlotWidget()
        self._setup_pg_view(self.cluster_view, 'Clustering Results')
        self.tab_widget.addTab(self.cluster_view, "Clustering Results")

        # HPC Wiring
        self.hpc_view = pg.PlotWidget()
        self._setup_pg_view(self.hpc_view, 'HPC Wiring')
        self.tab_widget.addTab(self.hpc_view, "HPC Wiring")

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

    # ==== ROBUST PROCESSING FUNCTIONS ====

    def process_all_steps(self):
        """Robust function to process all steps sequentially."""
        if not (self.chassis_file_path and self.io_file_path):
            QMessageBox.warning(self, "Missing Files", "Please select both chassis and I/O files first.")
            return

        self.log("🚀 Starting complete processing pipeline...")
        
        # Step 1: Process Graph
        self.process_graph()

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
        self._start_worker_task()
        self.worker = OptimizationWorker("clustering", self.config, graph=self.current_graph, n_clusters=n_clusters)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_clustering_completed)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def run_hpc_analysis(self):
        """Run HPC wiring analysis."""
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

        self.log(f"✅ Graph loaded: {num_chassis} chassis nodes, {num_io} I/O points")
        
        if export_path:
            self.log(f"📁 Enhanced graph exported: {os.path.basename(export_path)}")

        for w in validation.get("warnings", []):
            self.log(f"⚠️ Warning: {w}")

        # Update visualization
        self._visualize_graph(self.current_graph)

        # Enable next steps
        self.elbow_btn.setEnabled(True)
        self.hpc_btn.setEnabled(True)
        self.btn_process_all.setEnabled(True)

        # Auto-continue if processing all
        if hasattr(self, '_processing_all') and self._processing_all:
            self.run_elbow_analysis()

    def on_elbow_completed(self, results):
        """Handle elbow analysis completion."""
        optimal_k = results["optimal_k"]
        self.elbow_data = results["elbow_data"]
        self._finish_worker_task()

        self.optimal_clusters_label.setText(f"Optimal clusters: {optimal_k}")
        max_clusters = int(self.config.get("clustering.max_clusters_supported", 100))
        clamped_k = min(optimal_k, max_clusters)
        
        if clamped_k != optimal_k:
            self.log(f"⚠️ Optimal k ({optimal_k}) exceeds max supported ({max_clusters}), using {clamped_k}")
        
        self.n_clusters_spin.setValue(clamped_k)
        self.clustering_btn.setEnabled(True)

        # Visualize elbow curve
        self._visualize_elbow_analysis()
        self.log(f"📊 Elbow analysis completed. Optimal clusters: {optimal_k}")

        # Auto-continue if processing all
        if hasattr(self, '_processing_all') and self._processing_all:
            self.run_clustering()

    def on_clustering_completed(self, results):
        """Handle clustering completion."""
        self.clustering_results = results
        self._finish_worker_task()

        total_length = results.get("total_wire_length", 0.0)
        self.log(f"🎯 Clustering completed. Total wire length: {total_length:.2f} mm")

        # Visualize clustering results
        self._visualize_clustering_results()

        # Auto-continue if processing all
        if hasattr(self, '_processing_all') and self._processing_all:
            self.run_hpc_analysis()

    def on_hpc_completed(self, results):
        """Handle HPC analysis completion."""
        self.hpc_results = results["hpc_wiring_results"]
        self._finish_worker_task()

        if self.hpc_results:
            total_length = self.hpc_results.get("total_length", 0.0)
            self.hpc_total_label.setText(f"HPC Total Length: {total_length:.2f} mm")
            self.log(f"🔌 HPC analysis completed. Total length: {total_length:.2f} mm")
            
            # Visualize HPC wiring
            self._visualize_hpc_wiring()

            # Compare results if clustering is done
            if self.clustering_results:
                self._compare_results()
        else:
            self.log("❌ HPC analysis failed.")

        # Finish processing all
        if hasattr(self, '_processing_all'):
            self._processing_all = False
            self.log("🎉 All processing steps completed!")

    def on_error(self, error_message: str):
        """Handle worker errors."""
        self._finish_worker_task()
        self.log(f"❌ ERROR: {error_message}")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_message}")
        
        if hasattr(self, '_processing_all'):
            self._processing_all = False

    # ==== VISUALIZATION FUNCTIONS ====

    def _visualize_graph(self, graph):
        """Visualize the enhanced graph."""
        self.graph_view.clear()
        pos = self._get_node_positions(graph)
        
        if not pos:
            self.log("No positions to plot")
            return

        # Draw edges
        self._draw_graph_edges(self.graph_view, graph, pos)
        
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

    def _visualize_clustering_results(self):
        """Visualize clustering results with config-driven styling, paths, and labels."""
        if not self.clustering_results or not self.current_graph:
            self.log("Clustering results not available or missing graph object for visualization.")
            return

        self.cluster_view.clear()
        
        pos = self._get_node_positions(self.current_graph)
        
        # 1. Draw base chassis graph (nodes and edges)
        chassis_nodes = [n for n, d in self.current_graph.nodes(data=True) if not d.get("is_io")]
        chassis_graph = self.current_graph.subgraph(chassis_nodes)
        self._draw_graph_edges(self.cluster_view, chassis_graph, pos, alpha=0.2)
        self._draw_graph_nodes(self.cluster_view, chassis_graph, pos)

        # 2. Draw I/O nodes, paths, and centroids by cluster
        clusters = self.clustering_results.get("clusters", {})
        colors = self._get_cluster_colors()

        for i, (cluster_id, cluster_data) in enumerate(clusters.items()):
            color = colors[i % len(colors)]
            
            # Draw I/O nodes for this cluster
            io_nodes_in_cluster = cluster_data.get("io_nodes", [])
            io_points = [pos[node] for node in io_nodes_in_cluster if node in pos]
            if io_points:
                points = np.array(io_points)
                style = self._get_node_style("IO_") 
                self.cluster_view.addItem(
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
            for io_node, path_data in wiring_paths.items():
                path = path_data.get("path", [])
                if centroid_info and "pos" in centroid_info:
                    vis_path_points = [centroid_info["pos"]] + [pos[node] for node in path if node in pos]
                    if len(vis_path_points) > 1:
                        xs = [p[0] for p in vis_path_points]
                        ys = [p[1] for p in vis_path_points]
                        self.cluster_view.addItem(
                            pg.PlotCurveItem(xs, ys, pen=pg.mkPen(color, width=2.5, style=Qt.DashLine))
                        )

            # Draw I/O extender (centroid) node and its label
            if centroid_info and "pos" in centroid_info:
                cx, cy = centroid_info["pos"]
                self.cluster_view.addItem(
                    pg.ScatterPlotItem(
                        [cx], [cy],
                        size=12,
                        symbol='s',
                        brush=pg.mkBrush(color),
                        pen=pg.mkPen('k', width=1.5)
                    )
                )
                label = pg.TextItem(f"I/O Extender {cluster_id.split('_')[-1]}", color=(0,0,0), anchor=(0.5, -1.0))
                label.setPos(cx, cy)
                self.cluster_view.addItem(label)

        self._set_view_limits(self.cluster_view, pos)
        self.tab_widget.setCurrentWidget(self.cluster_view)

    def _visualize_hpc_wiring(self):
        """Visualize HPC wiring results."""
        if not self.hpc_results or not self.current_graph:
            return

        self.hpc_view.clear()
        pos = self._get_node_positions(self.current_graph)
        
        # Draw base graph lightly
        self._draw_graph_edges(self.hpc_view, self.current_graph, pos, alpha=0.2)
        self._draw_graph_nodes(self.hpc_view, self.current_graph, pos, alpha=0.3)
        
        # Draw HPC wiring paths
        hpc_node = self.hpc_results.get("hpc_node", "H1")
        paths = self.hpc_results.get("paths", {})
        
        for io_node, path_data in paths.items():
            path = path_data.get("path", [])
            if len(path) > 1:
                xs, ys = [], []
                for node in path:
                    if node in pos:
                        xs.append(pos[node][0])
                        ys.append(pos[node][1])
                
                if len(xs) > 1:
                    self.hpc_view.addItem(
                        pg.PlotCurveItem(
                            xs, ys, pen=pg.mkPen('r', width=3, style=Qt.DashLine)
                        )
                    )

        self._set_view_limits(self.hpc_view, pos)
        
        # Switch to HPC tab
        self.tab_widget.setCurrentWidget(self.hpc_view)

    # ==== UTILITY FUNCTIONS ====

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

    def _draw_graph_edges(self, view, graph, pos, alpha=1.0):
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
                    xs, ys, pen=pg.mkPen(pen_color, width=1), connect='finite'
                )
            )

    def _get_node_style(self, node_name: str) -> Dict[str, Any]:
        """Get node style from config based on prefix."""
        node_defs = self.config.get("node_definitions", {})
        node_types = node_defs.get("node_types", {})
        
        for type_name, type_info in node_types.items():
            for prefix in type_info.get("prefixes", []):
                if node_name.startswith(prefix):
                    # Convert hex color to pg.Color
                    hex_color = type_info.get("color", "#9B9B9B").lstrip('#')
                    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    return {
                        "brush": pg.mkBrush(color=rgb),
                        "pen": pg.mkPen(color=tuple(c*0.6 for c in rgb)),
                        "size": type_info.get("size", 6),
                        "symbol": type_info.get("symbol", "o")
                    }
        
        # Fallback to default style
        default_style = node_defs.get("default_node", {})
        hex_color = default_style.get("color", "#9B9B9B").lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return {
            "brush": pg.mkBrush(color=rgb),
            "pen": pg.mkPen(color=tuple(c*0.6 for c in rgb)),
            "size": default_style.get("size", 6),
            "symbol": default_style.get("symbol", "o")
        }

    def _draw_graph_nodes(self, view, graph, pos, alpha=1.0):
        """Draw graph nodes based on config-driven styles."""
        
        # Group nodes by style for efficient plotting
        nodes_by_style = {}
        for node, data in graph.nodes(data=True):
            if node not in pos:
                continue
            
            style = self._get_node_style(node)
            style_key = (style["symbol"], style["size"], style["brush"].color().name())
            
            if style_key not in nodes_by_style:
                nodes_by_style[style_key] = {"style": style, "points": []}
            
            nodes_by_style[style_key]["points"].append(pos[node])

        # Plot each group of nodes
        for style_key, group in nodes_by_style.items():
            points = np.array(group["points"])
            style = group["style"]
            
            view.addItem(
                pg.ScatterPlotItem(
                    points[:, 0], points[:, 1],
                    size=style["size"],
                    symbol=style["symbol"],
                    brush=style["brush"],
                    pen=style["pen"]
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
        self.hpc_btn.setEnabled(False)
        self.btn_process_all.setEnabled(False)

    def _finish_worker_task(self):
        """Finish a worker task - hide progress and re-enable controls."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        
        # Re-enable appropriate controls based on current state
        if self.chassis_file_path and self.io_file_path:
            self.btn_process.setEnabled(True)
            self.btn_process_all.setEnabled(True)
        
        if self.current_graph:
            self.elbow_btn.setEnabled(True)
            self.clustering_btn.setEnabled(True)
            self.hpc_btn.setEnabled(True)

    def _compare_results(self):
        """Compare clustering vs HPC results."""
        if not (self.clustering_results and self.hpc_results):
            return
        
        cluster_length = self.clustering_results.get("total_wire_length", 0.0)
        hpc_length = self.hpc_results.get("total_length", 0.0)
        
        if hpc_length > 0:
            improvement = ((hpc_length - cluster_length) / hpc_length) * 100
            self.log(f"📊 Comparison: Clustering vs HPC")
            self.log(f"   Clustering: {cluster_length:.2f} mm")
            self.log(f"   HPC Direct: {hpc_length:.2f} mm")
            self.log(f"   Improvement: {improvement:.1f}%")

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
            self.log(f"📁 Loaded chassis: {os.path.basename(file_path)}")

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
            self.log(f"📁 Loaded I/O: {os.path.basename(file_path)}")

    def load_default_files(self):
        """Load default files from configuration."""
        paths = self.config.get("paths", {})
        default_chassis = paths.get("default_chassis_json")
        default_io = paths.get("default_io_csv")

        if default_chassis and os.path.exists(default_chassis):
            self.chassis_file_path = default_chassis
            self.chassis_file_label.setText(f"Chassis: {os.path.basename(default_chassis)}")
            self.log(f"📁 Loaded default chassis: {os.path.basename(default_chassis)}")
        else:
            self.log(f"⚠️ Default chassis file not found: {default_chassis}")

        if default_io and os.path.exists(default_io):
            self.io_file_path = default_io
            self.io_file_label.setText(f"I/O: {os.path.basename(default_io)}")
            self.log(f"📁 Loaded default I/O: {os.path.basename(default_io)}")
        else:
            self.log(f"⚠️ Default I/O file not found: {default_io}")

        self._check_files_loaded()

    def _check_files_loaded(self):
        """Check if both files are loaded and enable processing."""
        both_loaded = bool(self.chassis_file_path and self.io_file_path)
        self.btn_process.setEnabled(both_loaded)
        if both_loaded:
            self.btn_process_all.setEnabled(True)

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
                
                self.log(f"💾 Results exported: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
            except Exception as e:
                error_msg = f"Failed to export results: {e}"
                self.log(f"❌ {error_msg}")
                QMessageBox.critical(self, "Export Error", error_msg)

    def export_report_pdf(self):
        """Export comprehensive PDF report."""
        export_dir = self.config.get("paths.export_dir", "./export")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(export_dir, f"wiring_report_{timestamp}.pdf")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", default_path, "PDF Files (*.pdf)"
        )
        
        if not file_path:
            return

        try:
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Title
            story.append(Paragraph("Automotive Wiring Harness Optimization Report", styles["Title"]))
            story.append(Spacer(1, 24))
            story.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
            story.append(PageBreak())

            # Add sections based on available results
            if self.current_graph:
                self._add_graph_section(story, styles)
            
            if self.elbow_data:
                self._add_elbow_section(story, styles)
            
            if self.clustering_results:
                self._add_clustering_section(story, styles)
            
            if self.hpc_results:
                self._add_hpc_section(story, styles)

            doc.build(story)
            self.log(f"📄 PDF report exported: {os.path.basename(file_path)}")
            QMessageBox.information(self, "Export Complete", f"Report exported to:\n{file_path}")
            
        except Exception as e:
            error_msg = f"Failed to export PDF: {e}"
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "Export Error", error_msg)

    def _add_graph_section(self, story, styles):
        """Add graph statistics section to PDF."""
        story.append(Paragraph("Graph Statistics", styles["Heading1"]))
        
        stats_data = [
            ["Total Nodes", self.current_graph.number_of_nodes()],
            ["Total Edges", self.current_graph.number_of_edges()],
            ["I/O Nodes", len([n for n, d in self.current_graph.nodes(data=True) if d.get("is_io")])],
            ["Chassis Nodes", len([n for n, d in self.current_graph.nodes(data=True) if not d.get("is_io")])],
        ]
        
        table = Table(stats_data, colWidths=[200, 100])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(table)
        story.append(PageBreak())

    def _add_elbow_section(self, story, styles):
        """Add elbow analysis section to PDF."""
        story.append(Paragraph("Elbow Analysis", styles["Heading1"]))
        story.append(Paragraph(f"Optimal clusters found: {self.elbow_data.get('elbow_k', 'N/A')}", styles["Normal"]))
        story.append(PageBreak())

    def _add_clustering_section(self, story, styles):
        """Add clustering section to PDF."""
        story.append(Paragraph("Clustering Results", styles["Heading1"]))
        
        total_length = self.clustering_results.get("total_wire_length", 0)
        story.append(Paragraph(f"Total wire length: {total_length:.2f} mm", styles["Normal"]))
        
        clusters = self.clustering_results.get("clusters", {})
        story.append(Paragraph(f"Number of clusters: {len(clusters)}", styles["Normal"]))
        story.append(PageBreak())

    def _add_hpc_section(self, story, styles):
        """Add HPC section to PDF."""
        story.append(Paragraph("HPC Wiring Analysis", styles["Heading1"]))
        
        total_length = self.hpc_results.get("total_length", 0)
        story.append(Paragraph(f"Total HPC wire length: {total_length:.2f} mm", styles["Normal"]))
        story.append(PageBreak())

    def show_about_dialog(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About",
            "Automotive Wiring Harness Optimizer\n\n"
            "A comprehensive tool for optimizing automotive wiring harness layouts.\n\n"
            "Features:\n"
            "• Graph-based chassis modeling\n"
            "• I/O clustering and optimization\n"
            "• HPC wiring analysis\n"
            "• Elbow method for optimal cluster detection\n"
            "• Comprehensive visualization and reporting\n\n"
            "Fully configurable via YAML configuration."
        )

    def log(self, message: str):
        """Log message to results text and logger."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.results_text.append(formatted_message)
        self.logger.info(message)

    def closeEvent(self, event):
        """Handle application close event."""
        self.log("Application closing...")
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
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
