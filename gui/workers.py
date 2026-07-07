"""
CONVIO - Background Workers
===========================

``OptimizationWorker`` runs heavy analysis tasks off the GUI thread.

Task types
----------
load_graph            Step 1 - build network graph
elbow_analysis        Step 2 - k-Means elbow method
hpc_wiring            Baseline - direct HPC wiring
clustering_initial    Step 3 - agglomerative clustering (raw assignments only)
clustering_iterative  Step 4 - iterative centroid optimisation with per-iteration stats
clustering            Legacy single-run (used by full_analysis sweep)
full_analysis         All linkage methods in one sweep
"""

import os
import logging

from PyQt5.QtCore import QThread, pyqtSignal

from config_manager import ConfigManager
from modules.graph_loader import create_graph_loader_from_config
from modules.elbow_method import ElbowMethodAnalyzer
from modules.clustering_dijkstra import ClusteringDijkstra
from modules.hpc_connector import calculate_direct_hpc_wiring
from modules.graph_utils import get_graph_statistics, validate_graph


class OptimizationWorker(QThread):
    """Worker thread for long-running optimization tasks."""

    progress_updated = pyqtSignal(int)
    status_updated   = pyqtSignal(str)
    finished         = pyqtSignal(dict)
    error_occurred   = pyqtSignal(str)

    def __init__(self, task_type: str, config_manager: ConfigManager, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.config    = config_manager
        self.kwargs    = kwargs

    def run(self):
        dispatch = {
            "load_graph":           self.load_graph_task,
            "elbow_analysis":       self.elbow_analysis_task,
            "hpc_wiring":           self.hpc_wiring_task,
            "clustering_initial":   self.clustering_initial_task,
            "clustering_iterative": self.clustering_iterative_task,
            "clustering":           self.clustering_task,
            "full_analysis":        self.full_analysis_task,
        }
        handler = dispatch.get(self.task_type)
        if handler is None:
            self.error_occurred.emit(f"Unknown task type: {self.task_type}")
            return
        try:
            handler()
        except Exception as e:
            if self.config.get("error_handling.log_stack_trace_on_error", True):
                logging.exception(f"Error in {self.task_type}: {e}")
            self.error_occurred.emit(str(e))

    def load_graph_task(self):
        try:
            self.status_updated.emit("Loading graph data...")
            self.progress_updated.emit(10)
            loader = create_graph_loader_from_config(self.config.config)
            self.progress_updated.emit(25)
            graph = loader.load_chassis_graph(self.kwargs["chassis_file"])
            self.progress_updated.emit(50)
            self.status_updated.emit("Loading I/O coordinates...")
            io_points     = loader.load_io_coordinates_from_csv(self.kwargs["io_file"])
            network_graph = loader.add_io_nodes_to_graph(graph, io_points)
            self.progress_updated.emit(75)
            export_path = loader.export_network_graph_json()
            self.status_updated.emit(f"Exported: {os.path.basename(export_path)}")
            self.progress_updated.emit(90)
            stats      = get_graph_statistics(network_graph)
            validation = validate_graph(network_graph)
            self.progress_updated.emit(100)
            self.status_updated.emit("Graph loading completed!")
            self.finished.emit({
                "graph": network_graph, "loader": loader,
                "statistics": stats, "validation": validation, "export_path": export_path,
            })
        except Exception as e:
            self.error_occurred.emit(f"Graph loading failed: {e}")

    def elbow_analysis_task(self):
        try:
            self.status_updated.emit("Running elbow method analysis...")
            ecfg = self.config.get("elbow_method", {})
            analyzer = ElbowMethodAnalyzer(
                k_min=int(ecfg.get("k_min", 1)), k_max=int(ecfg.get("k_max", 12)),
                random_state=int(ecfg.get("random_state", 42)), n_init=int(ecfg.get("n_init", 10)),
            )
            optimal_k, elbow_data = analyzer.find_optimal_clusters(self.kwargs["graph"])
            self.finished.emit({"optimal_k": optimal_k, "elbow_data": elbow_data})
        except Exception as e:
            self.error_occurred.emit(f"Elbow analysis failed: {e}")

    def hpc_wiring_task(self):
        try:
            self.status_updated.emit("Calculating overall wiring...")
            results = calculate_direct_hpc_wiring(self.kwargs["graph"], self.config.config)
            self.finished.emit({"hpc_wiring_results": results})
        except Exception as e:
            self.error_occurred.emit(f"Overall wiring failed: {e}")

    def clustering_initial_task(self):
        """Step 3 - raw agglomerative assignments, no centroid optimisation."""
        try:
            self.status_updated.emit("Running agglomerative clustering...")
            clusterer = ClusteringDijkstra(config=self.config.config)
            result = clusterer.cluster_initial(
                self.kwargs["graph"],
                int(self.kwargs["n_clusters"]),
                self.kwargs.get("linkage_method", "complete"),
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"Initial clustering failed: {e}")

    def clustering_iterative_task(self):
        """Step 4 - iterative centroid optimisation with per-iteration tracking."""
        try:
            self.status_updated.emit("Running iterative centroid optimisation...")
            clusterer = ClusteringDijkstra(config=self.config.config)
            result = clusterer.cluster_and_optimize_iterative(
                self.kwargs["graph"],
                int(self.kwargs["n_clusters"]),
                self.kwargs.get("linkage_method", "complete"),
                int(self.kwargs.get("max_iterations", 5)),
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"Iterative optimisation failed: {e}")

    def clustering_task(self):
        try:
            self.status_updated.emit("Performing clustering...")
            clusterer = ClusteringDijkstra(config=self.config.config)
            results = clusterer.cluster_and_optimize(
                self.kwargs["graph"], int(self.kwargs["n_clusters"]),
                linkage_method=self.kwargs.get("linkage_method", "complete"),
            )
            self.finished.emit(results)
        except Exception as e:
            self.error_occurred.emit(f"Clustering failed: {e}")

    def full_analysis_task(self):
        try:
            self.status_updated.emit("Starting comprehensive analysis...")
            clusterer = ClusteringDijkstra(config=self.config.config)
            n_clusters, graph = int(self.kwargs["n_clusters"]), self.kwargs["graph"]
            linkage_methods = ["average", "complete", "single"]
            results, best_length, best_method = {}, float("inf"), None
            max_iterations = int(self.kwargs.get("max_iterations", 5))
            for i, method in enumerate(linkage_methods):
                self.status_updated.emit(f"Running '{method}' linkage...")
                self.progress_updated.emit(int((i / len(linkage_methods)) * 100))
                # Use the same iterative centroid optimisation pipeline as
                # the main Step 4 run so the comparison table and final result
                # measure the same thing.
                run_result = clusterer.cluster_and_optimize_iterative(
                    graph, n_clusters, linkage_method=method,
                    max_iterations=max_iterations,
                )
                results[method] = run_result
                total = run_result.get("overall_wiring_harness_length", float("inf"))
                if total < best_length:
                    best_length, best_method = total, method
            self.progress_updated.emit(100)
            self.status_updated.emit("Comprehensive analysis complete!")
            self.finished.emit({"results": results, "best_method": best_method})
        except Exception as e:
            self.error_occurred.emit(f"Full analysis failed: {e}")
