"""
Clustering & finding shortest path with Dijkstra
========================================================

This module provides the `ClusteringDijkstra` class, which is responsible for
grouping I/O nodes into clusters and finding the optimal placement for I/O
aggregators (centroids) within a chassis graph. It also includes functionality
to calculate a communication network bus path connecting these aggregators to a High-Performance
Computer (HPC).

The process involves these main steps:
1.  Clustering: I/O nodes are clustered based on their graph distance
    from each other.
2.  Centroid Optimization: For each cluster, the optimal location for an
    I/O aggregator is found by evaluating candidate points on chassis nodes and
    edges, The threshold value can be set in configuration file.
3.  Communication Network Calculation: A shortest path is calculated to connect the HPC
    to all newly placed I/O aggregators in a bus topology.
4.  Output Generation: The results, including cluster data, wiring paths,
    and the communication network path, are compiled and exported to a JSON file.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import json
from datetime import datetime
import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import logging
import cProfile
import pstats
import io
from functools import wraps


_active_profiler = None


def profile_function(func):
    """
    A decorator to profile a function's execution time and memory usage.

    Profiling is enabled by setting the environment variable ENABLE_PROFILING=true.
    Results are saved to the './profiling/functions' directory.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _active_profiler
        if os.getenv('ENABLE_PROFILING', 'false').lower() != 'true':
            return func(*args, **kwargs)
        if _active_profiler is not None:
            #print(f"  Skipping profiling for {func.__name__} (profiler already active)")
            return func(*args, **kwargs)
        profiler = cProfile.Profile()
        _active_profiler = profiler
        start_time = datetime.now()
        try:
            profiler.enable()
            return func(*args, **kwargs)
        finally:
            profiler.disable()
            _active_profiler = None
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            timestamp = end_time.strftime('%Y%m%d_%H%M%S')
            profile_dir = './profiling/functions'
            os.makedirs(profile_dir, exist_ok=True)
            base_name = f'{func.__module__}.{func.__name__}_{timestamp}'
            prof_path = os.path.join(profile_dir, base_name + '.prof')
            profiler.dump_stats(prof_path)
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.strip_dirs().sort_stats('cumtime').print_stats(20)
            txt_path = os.path.join(profile_dir, base_name + '.txt')
            with open(txt_path, 'w') as f:
                f.write(s.getvalue())
            if elapsed > 0.1:
                print(f' Profiled {func.__name__} took {elapsed:.2f}s -> {txt_path}')
    return wrapper


class ClusteringDijkstra:
    """
    Orchestrates the clustering of I/O nodes and optimization of aggregator placements.
    """
    @profile_function
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ClusteringDijkstra instance.

        Args:
            config: A dictionary containing the application's configuration,
                    typically loaded from a YAML file.
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configure paths for exporting results.
        paths_cfg = self.config.get("paths", {})
        self.export_dir = paths_cfg.get("export_dir", "./export")
        os.makedirs(self.export_dir, exist_ok=True)

        # Load clustering-specific parameters.
        self.clustering_config = self.config.get("clustering", {})
        self.edge_sample_step = float(self.clustering_config.get("edge_sample_step", 0.25))
        self.include_node_candidates = bool(self.clustering_config.get("include_node_candidates", True))

        # Load node definitions to dynamically identify HPC and aggregator nodes.
        node_cfg = self.config.get("node_definitions", {})
        self.node_types = node_cfg.get("node_types", {})

        # Determine the HPC node name from configuration for flexible hardware setups.
        hpc_type = next((t for t, d in self.node_types.items() if d.get("is_hpc")), "hpc")
        self.hpc_prefixes = self.node_types.get(hpc_type, {}).get("prefixes", ["HPC_", "H_"])
        self.configured_hpc_name = self.config.get("node_configuration", {}).get("hpc_node_name")

        # Get prefixes for I/O aggregators to support custom naming schemes.
        aggregator_type = next((t for t, d in self.node_types.items() if d.get("is_aggregator")), "aggregator")
        self.aggregator_prefixes = self.node_types.get(aggregator_type, {}).get("prefixes", ["EXT_"])

    def _find_hpc_node(self, graph: nx.Graph) -> Optional[str]:
        """
        Determines the HPC node name by checking configuration and then scanning the graph.

        This robust method ensures that the HPC node can be found even if not
        explicitly named in the config, adapting to the actual graph data.

        Args:
            graph: The graph to search for the HPC node.

        Returns:
            The name of the HPC node if found, otherwise None.
        """
        # Priority 1: Use the explicitly configured HPC node name if it exists in the graph.
        if self.configured_hpc_name and self.configured_hpc_name in graph:
            return self.configured_hpc_name

        # Priority 2: If not configured or not found, scan the graph for nodes matching HPC prefixes.
        hpc_candidates = [n for n in graph.nodes() if any(n.startswith(p) for p in self.hpc_prefixes)]

        if len(hpc_candidates) == 1:
            self.logger.info(f"HPC not configured, but found unique candidate '{hpc_candidates[0]}'. Using it.")
            return hpc_candidates[0]
        
        if len(hpc_candidates) > 1:
            self.logger.warning(f"Found multiple possible HPCs: {hpc_candidates}. Using the first one: '{hpc_candidates[0]}'. Please specify 'hpc_node_name' in config for clarity.")
            return hpc_candidates[0]

        # If no candidates are found by any method, log an error.
        self.logger.error("HPC could not be found in the graph. Please check graph data or 'hpc_node_name' in config.")
        return None
    
    @profile_function
    def cluster_and_optimize(self, graph: nx.Graph, n_clusters: int,
                              linkage_method: str = 'complete',
                              refine: bool = True) -> Dict[str, Any]:
        """
        Executes the full clustering and optimization workflow.

        Args:
            graph: The input graph containing chassis and I/O nodes.
            n_clusters: The desired number of clusters to create.
            linkage_method: The linkage criterion to use ('complete', 'average', 'single').
            refine: If True (default), run the k-means-like reassignment loop after
                    initial agglomerative clustering. Set to False to compare pure
                    linkage-method output — otherwise all three linkages converge to
                    the same partition regardless of seeding.

        Returns:
            A dictionary containing the complete results, including cluster
            definitions, wiring paths, total wire length, and CAN bus data.
        """
        # Step 1: Identify all I/O nodes in the graph.
        io_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_io", False)]
        if not io_nodes:
            self.logger.warning("No I/O nodes found in the graph. Aborting.")
            return {"clusters": {}, "total_wire_length": 0.0, "output_path": None}

        # Step 2: Prepare the graph and data for clustering.
        k = min(n_clusters, len(io_nodes))  # Ensure user don't ask for more clusters than there are nodes.
        chassis = self._get_chassis_subgraph(graph)
        pos = {n: chassis.nodes[n]["pos"] for n in chassis.nodes()}
        io_attachment_map = self._infer_attachment_nodes(graph, io_nodes)

        # Step 3: Group I/O nodes into clusters.
        self.logger.info(f"Starting clustering with k={k} and linkage='{linkage_method}'...")
        labels = self._cluster_by_graph_distance(chassis, io_nodes, io_attachment_map, k, linkage_method)
        clusters = {f"cluster_{i + 1}": {"io_nodes": []} for i in range(k)}
        for i, node in enumerate(io_nodes):
            cluster_id = f"cluster_{labels[i] + 1}"
            clusters[cluster_id]["io_nodes"].append(node)

        # Step 4: (optional) Refine clusters by re-assigning I/Os to closest centroids.
        all_pairs_lengths = dict(nx.all_pairs_dijkstra_path_length(chassis, weight="weight"))
        if refine:
            self.logger.info("Refining cluster assignments based on optimal centroids...")
            clusters = self._refine_cluster_assignments(graph, chassis, clusters, all_pairs_lengths)
        else:
            self.logger.info("Skipping refinement (refine=False) — reporting pure linkage output.")

        # Step 5: For each FINAL cluster, find the optimal aggregator location and calculate wire lengths.
        total_wire_length = 0.0
        for cid, cdata in clusters.items():
            if not cdata["io_nodes"]:  # Skip empty clusters that may result from refinement
                continue
            cluster_length = self._process_single_cluster(graph, chassis, pos, cid, cdata, all_pairs_lengths)
            total_wire_length += cluster_length

        # Step 6: Format and export the final results.
        return self._format_and_export_output(graph, clusters, total_wire_length)

    def cluster_initial(self, graph: nx.Graph, n_clusters: int, linkage_method: str = 'complete') -> Dict[str, Any]:
        """
        GUI Step 3 - Agglomerative clustering only, no centroid optimisation.

        Returns raw cluster memberships so the visualization layer can colour
        I/O nodes by cluster without drawing paths or centroids.
        """
        io_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_io", False)]
        if not io_nodes:
            self.logger.warning("No I/O nodes found. Aborting initial clustering.")
            return {"clusters": {}, "io_nodes": []}

        k = min(n_clusters, len(io_nodes))
        chassis = self._get_chassis_subgraph(graph)
        io_attachment_map = self._infer_attachment_nodes(graph, io_nodes)

        self.logger.info(f"Initial agglomerative clustering: k={k}, linkage='{linkage_method}'")
        labels = self._cluster_by_graph_distance(chassis, io_nodes, io_attachment_map, k, linkage_method)

        clusters: Dict[str, Any] = {f"cluster_{i + 1}": {"io_nodes": []} for i in range(k)}
        for node, label in zip(io_nodes, labels):
            cluster_id = f"cluster_{int(label) + 1}"
            clusters[cluster_id]["io_nodes"].append(node)

        return {"clusters": clusters, "io_nodes": io_nodes}

    def cluster_and_optimize_iterative(
        self, graph: nx.Graph, n_clusters: int,
        linkage_method: str = 'complete', max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        GUI Step 4 - Full pipeline with per-iteration convergence tracking.

        The returned dict includes an ``iteration_results`` list used to
        populate the Step 4 iteration table in the GUI:
          [{"iteration": int, "assignments_changed": int, "total_wire_length": float}, ...]
        """
        io_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_io", False)]
        if not io_nodes:
            return {"clusters": {}, "total_wire_length": 0.0, "iteration_results": [], "output_path": None}

        k = min(n_clusters, len(io_nodes))
        chassis = self._get_chassis_subgraph(graph)
        pos = {n: chassis.nodes[n]["pos"] for n in chassis.nodes()}
        io_attachment_map = self._infer_attachment_nodes(graph, io_nodes)

        # Initial clustering
        labels = self._cluster_by_graph_distance(chassis, io_nodes, io_attachment_map, k, linkage_method)
        clusters: Dict[str, Any] = {f"cluster_{i + 1}": {"io_nodes": []} for i in range(k)}
        for node, label in zip(io_nodes, labels):
            cluster_id = f"cluster_{int(label) + 1}"
            clusters[cluster_id]["io_nodes"].append(node)

        all_pairs_lengths = dict(nx.all_pairs_dijkstra_path_length(chassis, weight="weight"))

        # Baseline cost — agglomerative clustering output before any centroid
        # optimisation. Used so the iter-1 delta is meaningful (it shows how
        # much the first reassignment improves on the raw linkage assignment).
        baseline_centroids = {}
        baseline_total = 0.0
        for cid, cdata in clusters.items():
            if not cdata["io_nodes"]:
                continue
            c = self._find_optimal_centroid(graph, chassis, cdata["io_nodes"], all_pairs_lengths)
            baseline_centroids[cid] = c
            if c:
                attachment = self._infer_attachment_nodes(graph, cdata["io_nodes"])
                baseline_total += self._evaluate_candidate_cost(
                    graph, chassis, c, cdata["io_nodes"], attachment, all_pairs_lengths,
                )

        # ── Iterative centroid refinement ─────────────────────────────────
        # At each iteration:
        #   1. Compute optimal centroids for the current assignments
        #   2. Reassign each I/O to the nearest centroid
        #   3. Record the cost of the *new* assignments under the *new*
        #      centroids (i.e. the post-reassignment state)
        #
        # This makes the reported wire length and assignments_changed columns
        # tell a consistent story: when assignments stabilise, the cost stops
        # changing too. Reporting cost mid-iteration (with old assignments and
        # new centroids) leads to the confusing "0 changes but length differs"
        # case the user reported.
        iteration_results = []
        prev_total = baseline_total  # iter-1 delta is relative to baseline
        for iteration in range(max_iterations):
            self.logger.info(f"Centroid optimisation iteration {iteration + 1}/{max_iterations}...")

            # Step 1: optimal centroid for each cluster's current members
            centroids: Dict[str, Any] = {}
            for cid, cdata in clusters.items():
                if not cdata["io_nodes"]:
                    continue
                centroids[cid] = self._find_optimal_centroid(
                    graph, chassis, cdata["io_nodes"], all_pairs_lengths,
                )

            # Step 2: reassign each I/O to the nearest centroid
            assignments_changed = 0
            all_io = [n for cdata in clusters.values() for n in cdata["io_nodes"]]
            for io_node in all_io:
                current_cid = next(
                    (cid for cid, cdata in clusters.items() if io_node in cdata["io_nodes"]), None,
                )
                min_dist, best_cid = float("inf"), None
                attachment_node = self._infer_attachment_nodes(graph, [io_node])[io_node]
                for cid, centroid in centroids.items():
                    if not centroid:
                        continue
                    _, length = self._get_path_and_length_from_centroid(chassis, centroid, attachment_node)
                    total_dist = length + graph[attachment_node][io_node].get("weight", 0.0)
                    if total_dist < min_dist:
                        min_dist, best_cid = total_dist, cid
                if best_cid and best_cid != current_cid:
                    clusters[current_cid]["io_nodes"].remove(io_node)
                    clusters[best_cid]["io_nodes"].append(io_node)
                    assignments_changed += 1

            # Step 3: cost of the new state (post-reassignment, current centroids)
            total_wire_this_iter = 0.0
            for cid, cdata in clusters.items():
                if not cdata["io_nodes"] or not centroids.get(cid):
                    continue
                attachment = self._infer_attachment_nodes(graph, cdata["io_nodes"])
                total_wire_this_iter += self._evaluate_candidate_cost(
                    graph, chassis, centroids[cid], cdata["io_nodes"],
                    attachment, all_pairs_lengths,
                )

            delta = total_wire_this_iter - prev_total
            iteration_results.append({
                "iteration":           iteration + 1,
                "assignments_changed": assignments_changed,
                "total_wire_length":   round(total_wire_this_iter, 2),
                "delta_vs_previous":   round(delta, 2),
            })
            prev_total = total_wire_this_iter

            if assignments_changed == 0:
                self.logger.info(f"Assignments stabilised at iteration {iteration + 1}.")
                break

        # Final pass: compute wiring paths and export
        total_wire_length = 0.0
        for cid, cdata in clusters.items():
            if not cdata["io_nodes"]:
                continue
            total_wire_length += self._process_single_cluster(
                graph, chassis, pos, cid, cdata, all_pairs_lengths,
            )

        result = self._format_and_export_output(graph, clusters, total_wire_length)
        result["iteration_results"] = iteration_results
        result["baseline_wire_length"] = round(baseline_total, 2)
        return result

    # --- Helper Methods for Graph Preparation ---

    def _get_chassis_subgraph(self, graph: nx.Graph) -> nx.Graph:
        """Extracts the chassis subgraph (nodes that are not I/O)."""
        return graph.subgraph([n for n, d in graph.nodes(data=True) if not d.get("is_io", False)]).copy()

    def _infer_attachment_nodes(self, graph: nx.Graph, io_nodes: List[str]) -> Dict[str, str]:
        """Infers the chassis node to which each I/O node is attached."""
        return {io: list(graph.neighbors(io))[0] for io in io_nodes if list(graph.neighbors(io))}

    @profile_function
    def _cluster_by_graph_distance(self, chassis: nx.Graph, io_nodes: List[str],
                                   io_attachment_map: Dict[str, str], k: int, linkage: str) -> np.ndarray:
        """
        Clusters I/O nodes based on their shortest path distances within the chassis graph.
        """
        # Create a distance matrix based on the shortest path length between each pair of I/O attachment nodes.
        attachment_nodes = [io_attachment_map[io] for io in io_nodes]
        all_lengths = dict(nx.all_pairs_dijkstra_path_length(chassis, weight="weight"))
        dist_matrix = np.array([[all_lengths.get(u, {}).get(v, float('inf')) for v in attachment_nodes] for u in attachment_nodes])

        # If the graph is disconnected, some distances will be infinity. Replace them with a large value.
        if np.isinf(dist_matrix).any():
            max_dist = np.max(dist_matrix[np.isfinite(dist_matrix)]) if np.isfinite(dist_matrix).any() else 1.0
            dist_matrix[np.isinf(dist_matrix)] = max_dist * 10

        # Use Agglomerative Clustering with the precomputed distance matrix.
        clusterer = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage=linkage)
        return clusterer.fit_predict(dist_matrix)

    @profile_function
    def _find_optimal_centroid(self, graph: nx.Graph, chassis: nx.Graph,
                               io_nodes_in_cluster: List[str],
                               all_pairs_lengths: Dict) -> Optional[Dict[str, Any]]:
        """Finds the single best candidate location for an aggregator for a given set of I/O nodes."""
        if not io_nodes_in_cluster:
            return None

        pos = {n: chassis.nodes[n]["pos"] for n in chassis.nodes()}
        attachment_nodes_map = self._infer_attachment_nodes(graph, io_nodes_in_cluster)
        candidates = self._generate_candidates(chassis, pos)

        best_cost = float('inf')
        best_candidate = None
        for cand in candidates:
            cost = self._evaluate_candidate_cost(graph, chassis, cand, io_nodes_in_cluster, attachment_nodes_map, all_pairs_lengths)
            if cost < best_cost:
                best_cost, best_candidate = cost, cand
        
        return best_candidate

    @profile_function
    def _refine_cluster_assignments(self, graph: nx.Graph, chassis: nx.Graph,
                                    initial_clusters: Dict[str, Any],
                                    all_pairs_lengths: Dict) -> Dict[str, Any]:
        """
        Iteratively refines cluster assignments by moving I/O nodes to the zone
        with the nearest centroid.
        """
        clusters = initial_clusters
        max_iterations = 5  # Prevents infinite loops, usually converges in 2-3 steps.
        
        for i in range(max_iterations):
            self.logger.info(f"Refinement iteration {i + 1}...")
            
            # 1. Find the optimal centroid for each current cluster.
            centroids = {}
            for cid, cdata in clusters.items():
                if cdata["io_nodes"]:
                    centroids[cid] = self._find_optimal_centroid(graph, chassis, cdata["io_nodes"], all_pairs_lengths)

            # 2. For each I/O, find its closest centroid and re-assign if necessary.
            assignments_changed = False
            all_io_nodes = [node for cdata in clusters.values() for node in cdata["io_nodes"]]
            
            for io_node in all_io_nodes:
                current_cid = next((cid for cid, cdata in clusters.items() if io_node in cdata["io_nodes"]), None)
                
                min_dist = float('inf')
                best_cid = None
                attachment_node = self._infer_attachment_nodes(graph, [io_node])[io_node]

                for cid, centroid in centroids.items():
                    if not centroid: continue
                    _, length = self._get_path_and_length_from_centroid(chassis, centroid, attachment_node)
                    io_edge_length = graph[attachment_node][io_node].get("weight", 0.0)
                    total_dist = length + io_edge_length
                    if total_dist < min_dist:
                        min_dist, best_cid = total_dist, cid
                
                if best_cid and best_cid != current_cid:
                    clusters[current_cid]["io_nodes"].remove(io_node)
                    clusters[best_cid]["io_nodes"].append(io_node)
                    assignments_changed = True
            
            if not assignments_changed:
                self.logger.info("Cluster assignments have stabilized. Finishing refinement.")
                break
        
        return clusters

    @profile_function
    def _process_single_cluster(self, graph: nx.Graph, chassis: nx.Graph,
                              pos: Dict[str, Tuple[float, float]],
                              cid: str, cdata: Dict[str, Any],
                              all_pairs_lengths: Dict[Any, Any]) -> float:
        """
        Finds the optimal centroid for a single cluster and calculates wiring paths.
        """
        io_nodes = cdata["io_nodes"]
        if not io_nodes: return 0.0

        # Find the candidate that results in the shortest total wire length for the cluster.
        best_candidate = self._find_optimal_centroid(graph, chassis, io_nodes, all_pairs_lengths)
        
        # Calculate the final cost using the best candidate
        attachment_nodes = self._infer_attachment_nodes(graph, io_nodes)
        best_cost = self._evaluate_candidate_cost(graph, chassis, best_candidate, io_nodes, attachment_nodes, all_pairs_lengths)

        # Store the best found centroid and its associated cost.
        cdata["centroid"] = best_candidate
        cdata["cluster_wire_length"] = best_cost

        # Calculate the individual wiring paths from the optimal centroid to each I/O node.
        wiring_paths = {}
        if best_candidate:
            for io_node in io_nodes:
                attach_node = attachment_nodes[io_node]
                path, length = self._get_path_and_length_from_centroid(chassis, best_candidate, attach_node)
                
                # Add the distance from the attachment node to the I/O node.
                io_edge_length = graph[attach_node][io_node].get("weight", 0.0)
                length += io_edge_length
                
                full_path = path + [io_node]  # Append the I/O node itself to complete the path.
                wiring_paths[io_node] = {"path": full_path, "length": length}
        cdata["wiring_paths"] = wiring_paths
        
        return best_cost

    @profile_function
    def _generate_candidates(self, chassis: nx.Graph, pos: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
        """
        Generates candidate locations for centroids, including chassis nodes and points along edges.
        """
        candidates = []
        # Option 1: Consider every node in the chassis as a potential centroid location.
        if self.include_node_candidates:
            for n in chassis.nodes():
                candidates.append({"type": "node", "node": n, "pos": pos[n]})
        
        # Option 2: Consider points along each edge of the chassis as potential locations.
        t_values = np.arange(0, 1 + self.edge_sample_step, self.edge_sample_step)
        for u, v in chassis.edges():
            p_u, p_v = np.array(pos[u]), np.array(pos[v])
            for t in t_values:
                if 1e-6 < t < 1 - 1e-6:  # Exclude the exact endpoints to avoid duplicates with node candidates.
                    # Interpolate to find the position of the candidate point.
                    p = (1 - t) * p_u + t * p_v
                    candidates.append({"type": "edge", "u": u, "v": v, "t": t, "pos": tuple(p)})
        return candidates

    @profile_function
    def _evaluate_candidate_cost(self, graph: nx.Graph, chassis: nx.Graph, candidate: Dict[str, Any],
                                 io_nodes_in_cluster: List[str],
                                 attachment_nodes_map: Dict[str, str],
                                 all_pairs_lengths: Dict) -> float:
        """
        Calculates the total wiring cost from a single candidate centroid to all nodes in a cluster.
        """
        total_cost = 0.0
        for io_node in io_nodes_in_cluster:
            attach_node = attachment_nodes_map[io_node]
            cost_to_attach = 0.0
            # Case 1: The candidate is located directly on a chassis node.
            if candidate["type"] == "node":
                cost_to_attach = all_pairs_lengths.get(candidate["node"], {}).get(attach_node, float('inf'))
            # Case 2: The candidate is located on an edge between two nodes.
            elif candidate["type"] == "edge":
                u, v, t = candidate["u"], candidate["v"], candidate["t"]
                edge_len = chassis[u][v].get("weight", 1.0)
                # The path to the attachment node can go via either end of the edge.
                cost_u = all_pairs_lengths.get(u, {}).get(attach_node, float('inf')) + t * edge_len
                cost_v = all_pairs_lengths.get(v, {}).get(attach_node, float('inf')) + (1 - t) * edge_len
                cost_to_attach = min(cost_u, cost_v)
            
            # Add distance from attachment node to I/O node
            cost_from_attach_to_io = graph[attach_node][io_node].get("weight", 0.0)
            total_cost += cost_to_attach + cost_from_attach_to_io
            
        return total_cost

    @profile_function
    def _get_path_and_length_from_centroid(self, chassis: nx.Graph, centroid: Dict[str, Any],
                                           target_node: str) -> Tuple[List[str], float]:
        """
        Calculates the shortest path and length from a centroid candidate to a target node.
        """
        if centroid["type"] == "node":
            source_node = centroid["node"]
            length = nx.dijkstra_path_length(chassis, source=source_node, target=target_node, weight="weight")
            path = nx.dijkstra_path(chassis, source=source_node, target=target_node, weight="weight")
            return path, length
        elif centroid["type"] == "edge":
            u, v, t = centroid["u"], centroid["v"], centroid["t"]
            edge_len = chassis[u][v].get("weight", 1.0)
            len_u = nx.dijkstra_path_length(chassis, source=u, target=target_node, weight="weight")
            len_v = nx.dijkstra_path_length(chassis, source=v, target=target_node, weight="weight")
            if len_u + t * edge_len < len_v + (1 - t) * edge_len:
                path = nx.dijkstra_path(chassis, source=u, target=target_node, weight="weight")
                return path, len_u + t * edge_len
            else:
                path = nx.dijkstra_path(chassis, source=v, target=target_node, weight="weight")
                return path, len_v + (1 - t) * edge_len
        return [], float('inf')

    @profile_function
    def _calculate_can_bus_path(self, graph: nx.Graph, hpc_node: str, aggregator_nodes: List[str]) -> Dict[str, Any]:
        """
        Calculates the shortest CAN bus path connecting the HPC to all I/O aggregators.
        This uses a greedy approach (nearest neighbor), finding the next closest
        aggregator from the last point in the path.
        """
        if not hpc_node or hpc_node not in graph:
            self.logger.warning(f"HPC  '{hpc_node}' not found for CAN bus calculation.")
            return {"path": [], "total_length": 0.0}
        if not aggregator_nodes:
            return {"path": [], "total_length": 0.0}

        # Define which nodes are part of the bus network (chassis, aggregators, and HPC).
        bus_nodes = [n for n, d in graph.nodes(data=True) if not d.get("is_io")]
        bus_graph = graph.subgraph(bus_nodes).copy()

        # Iteratively find the nearest aggregator and add it to the bus.
        remaining_aggregators = set(e for e in aggregator_nodes if e in bus_graph)
        can_bus_path = [hpc_node]
        total_can_length = 0.0
        current_node = hpc_node
        while remaining_aggregators:
            shortest_dist = float('inf')
            best_path = []
            next_aggregator = None

            # Find the closest aggregator from the current end of the bus.
            for aggregator in remaining_aggregators:
                try:
                    length, path = nx.single_source_dijkstra(bus_graph, source=current_node, target=aggregator, weight="weight")
                    if length < shortest_dist:
                        shortest_dist, best_path, next_aggregator = length, path, aggregator
                except nx.NetworkXNoPath:
                    continue
            
            # If a path is found, add it to our main CAN bus path.
            if next_aggregator:
                can_bus_path.extend(best_path[1:])  # Exclude the first node to avoid duplicates.
                total_can_length += shortest_dist
                current_node = next_aggregator
                remaining_aggregators.remove(next_aggregator)
            else:
                # This case occurs if some aggregators are on disconnected parts of the graph.
                self.logger.error("Could not find a path to all I/O aggregators for CAN bus.")
                break
        
        return {"path": can_bus_path, "total_length": total_can_length}

    def _format_and_export_output(self, original_graph: nx.Graph, clusters: Dict[str, Any], total_wire_length: float) -> Dict[str, Any]:
        """
        Formats the final results, calculates the CAN bus path, and exports to JSON.
        """
        G_out = original_graph.copy()
        
        # Find the HPC node before proceeding.
        hpc_node_name = self._find_hpc_node(G_out)
        if not hpc_node_name:
            # Error is logged within the find method. Return an error state.
            return {"clusters": {}, "total_wire_length": 0.0, "output_path": None, "error": "HPC node not found"}

        # Create and connect I/O aggregator nodes in the output graph.
        for cluster_id, cluster_data in clusters.items():
            centroid = cluster_data.get("centroid")
            if not centroid: continue
            
            aggregator_prefix = self.aggregator_prefixes[0] if self.aggregator_prefixes else "EXT_"
            aggregator_id = f"{aggregator_prefix}{cluster_id.split('_')[-1]}"
            G_out.add_node(aggregator_id, pos=centroid["pos"], type="aggregator", is_io=False)

            # Connect the new aggregator node to the chassis graph to make it routable.
            if centroid["type"] == "node":
                G_out.add_edge(aggregator_id, centroid["node"], weight=0)
            elif centroid["type"] == "edge":
                u, v, t = centroid["u"], centroid["v"], centroid["t"]
                edge_len = original_graph.edges[u, v].get("weight", 1.0)
                G_out.add_edge(aggregator_id, u, weight=t * edge_len)
                G_out.add_edge(aggregator_id, v, weight=(1 - t) * edge_len)

            # Add edges representing the optimized wiring from the aggregator to its I/O nodes.
            for io_node, path_data in cluster_data.get("wiring_paths", {}).items():
                G_out.add_edge(aggregator_id, io_node, weight=path_data.get("length", 0.0), edge_type="optimized_wire")
        
        # Now that aggregators are in the graph, calculate the communication network bus path.
        aggregator_nodes = [n for n in G_out.nodes() if any(n.startswith(p) for p in self.aggregator_prefixes)]
        can_bus_results = self._calculate_can_bus_path(G_out, hpc_node_name, aggregator_nodes)

        # For visualization purposes, format the communication network bus path as a pseudo-cluster.
        if can_bus_results and can_bus_results["path"] and hpc_node_name in G_out:
            can_path = can_bus_results["path"]
            if len(can_path) > 1:
                clusters["can_bus"] = {
                    "io_nodes": [],
                    "centroid": {"type": "node", "node": hpc_node_name, "pos": G_out.nodes[hpc_node_name].get("pos")},
                    "wiring_paths": {"can_bus_path": {"path": can_path[1:], "length": can_bus_results["total_length"]}},
                    "cluster_wire_length": can_bus_results["total_length"]
                }

        # Compile all data into the final output dictionary.
        can_bus_length = can_bus_results.get("total_length", 0.0)
        overall_wiring_harness_length = total_wire_length + can_bus_length
        output_data = {
            "nodes": list(G_out.nodes()),
            "coordinates": {n: d.get("pos") for n, d in G_out.nodes(data=True) if "pos" in d},
            "edges": {u: [[v, d.get("weight", 1.0)] for _, v, d in G_out.edges(u, data=True)] for u in G_out.nodes()},
            "clusters": clusters,
            "total_wire_length": total_wire_length,
            "can_bus": can_bus_results,
            "overall_wiring_harness_length": overall_wiring_harness_length,
        }
        
        # Save the output to a timestamped JSON file.
        filename = f"Zonal_EEA{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(self.export_dir, filename)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        
        output_data["output_path"] = output_path
        return output_data
