"""
Clustering & Dijkstra - Integrated Centroid Optimization
========================================================
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



_active_profiler = None  # Global tracker to prevent overlaps

def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _active_profiler
        
        # Check if profiling is enabled
        if os.getenv('ENABLE_PROFILING', 'false').lower() != 'true':
            return func(*args, **kwargs)

        # Skip if another profiler is already active
        if _active_profiler is not None:
            print(f"⚠️  Skipping profiling for {func.__name__} (profiler already active)")
            return func(*args, **kwargs)

        # Start new profiler
        profiler = cProfile.Profile()
        _active_profiler = profiler
        start_time = datetime.now()
        
        try:
            profiler.enable()
            result = func(*args, **kwargs)
            return result
        finally:
            # Always clean up, even if function throws exception
            profiler.disable()
            _active_profiler = None  # Reset global tracker
            
            # Save profiling results
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            timestamp = end_time.strftime('%Y%m%d_%H%M%S')
            
            # Create profiling directory
            profile_dir = './profiling/functions'
            os.makedirs(profile_dir, exist_ok=True)
            
            # Save files
            base_name = f'{func.__module__}.{func.__name__}_{timestamp}'
            prof_path = os.path.join(profile_dir, base_name + '.prof')
            profiler.dump_stats(prof_path)
            
            # Create readable report
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.strip_dirs()
            ps.sort_stats('cumtime')
            ps.print_stats(20)
            
            txt_path = os.path.join(profile_dir, base_name + '.txt')
            with open(txt_path, 'w') as f:
                f.write(s.getvalue())
            
            # Log results
            if elapsed > 0.1:
                print(f'⚡ Profiled {func.__name__} took {elapsed:.2f}s -> {txt_path}')
    
    return wrapper


class ClusteringDijkstra:
    @profile_function
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        paths_cfg = self.config.get("paths", {})
        self.export_dir = paths_cfg.get("export_dir", "./export")
        os.makedirs(self.export_dir, exist_ok=True)
        self.clustering_config = self.config.get("clustering", {})
        self.edge_sample_step = float(self.clustering_config.get("edge_sample_step", 0.25))
        self.include_node_candidates = bool(self.clustering_config.get("include_node_candidates", True))
    
    @profile_function
    def cluster_and_optimize(self, graph: nx.Graph, n_clusters: int) -> Dict[str, Any]:
        io_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_io", False)]
        if not io_nodes:
            return {"clusters": {}, "total_wire_length": 0.0, "output_path": None}

        k = min(n_clusters, len(io_nodes))
        chassis = self._get_chassis_subgraph(graph)
        pos = {n: chassis.nodes[n]["pos"] for n in chassis.nodes()}
        io_attachment_map = self._infer_attachment_nodes(graph, io_nodes)
        
        labels = self._cluster_by_graph_distance(chassis, io_nodes, io_attachment_map, k)
        clusters = {f"cluster_{i}": {"io_nodes": []} for i in range(k)}
        for i, node in enumerate(io_nodes):
            clusters[f"cluster_{labels[i]}"]["io_nodes"].append(node)

        total_wire_length = 0.0
        all_pairs_lengths = dict(nx.all_pairs_dijkstra_path_length(chassis, weight="weight"))

        for cid, cdata in clusters.items():
            cluster_length = self._process_single_cluster(graph, chassis, pos, cid, cdata, all_pairs_lengths)
            total_wire_length += cluster_length

        return self._format_and_export_output(graph, clusters, total_wire_length)


    def _get_chassis_subgraph(self, graph: nx.Graph) -> nx.Graph:
        return graph.subgraph([n for n, d in graph.nodes(data=True) if not d.get("is_io", False)]).copy()

    def _infer_attachment_nodes(self, graph: nx.Graph, io_nodes: List[str]) -> Dict[str, str]:
        return {io: list(graph.neighbors(io))[0] for io in io_nodes if list(graph.neighbors(io))}

    
    @profile_function
    def _cluster_by_graph_distance(self, chassis: nx.Graph, io_nodes: List[str], 
                                   io_attachment_map: Dict[str, str], k: int) -> np.ndarray:
        attachment_nodes = [io_attachment_map[io] for io in io_nodes]
        all_lengths = dict(nx.all_pairs_dijkstra_path_length(chassis, weight="weight"))
        dist_matrix = np.array([[all_lengths.get(u, {}).get(v, float('inf')) for v in attachment_nodes] for u in attachment_nodes])
        
        if np.isinf(dist_matrix).any():
            max_dist = np.max(dist_matrix[np.isfinite(dist_matrix)]) if np.isfinite(dist_matrix).any() else 1.0
            dist_matrix[np.isinf(dist_matrix)] = max_dist * 10

        clusterer = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
        return clusterer.fit_predict(dist_matrix)

    
    @profile_function
    def _process_single_cluster(self, graph: nx.Graph, chassis: nx.Graph,
                              pos: Dict[str, Tuple[float, float]],
                              cid: str, cdata: Dict[str, Any], 
                              all_pairs_lengths: Dict[Any, Any]) -> float:
        io_nodes = cdata["io_nodes"]
        if not io_nodes: return 0.0

        attachment_nodes = self._infer_attachment_nodes(graph, io_nodes)
        attachment_nodes_in_cluster = [attachment_nodes[ion] for ion in io_nodes]
        
        candidates = self._generate_candidates(chassis, pos)
        
        best_cost = float('inf')
        best_candidate = None
        for cand in candidates:
            cost = self._evaluate_candidate_cost(chassis, cand, attachment_nodes_in_cluster, all_pairs_lengths)
            if cost < best_cost:
                best_cost, best_candidate = cost, cand
        
        cdata["centroid"] = best_candidate
        cdata["cluster_wire_length"] = best_cost
        
        wiring_paths = {}
        if best_candidate:
            for io_node in io_nodes:
                attach_node = attachment_nodes[io_node]
                path, length = self._get_path_and_length_from_centroid(chassis, best_candidate, attach_node)
                # Append the I/O node to the path to make a complete path for visualization
                full_path = path + [io_node]
                wiring_paths[io_node] = {"path": full_path, "length": length}
        cdata["wiring_paths"] = wiring_paths
        
        return best_cost

    
    
    @profile_function
    def _generate_candidates(self, chassis, pos):
        candidates = []
        if self.include_node_candidates:
            for n in chassis.nodes():
                candidates.append({"type": "node", "node": n, "pos": pos[n]})
        
        t_values = np.arange(0, 1 + self.edge_sample_step, self.edge_sample_step)
        for u, v in chassis.edges():
            p_u, p_v = np.array(pos[u]), np.array(pos[v])
            for t in t_values:
                if t > 1e-6 and t < 1 - 1e-6:
                    p = (1 - t) * p_u + t * p_v
                    candidates.append({"type": "edge", "u": u, "v": v, "t": t, "pos": tuple(p)})
        return candidates

    
    
    
    @profile_function
    def _evaluate_candidate_cost(self, chassis, candidate, attachment_nodes_in_cluster, all_pairs_lengths):
        total_cost = 0.0
        if candidate["type"] == "node":
            for attach_node in attachment_nodes_in_cluster:
                total_cost += all_pairs_lengths.get(candidate["node"], {}).get(attach_node, float('inf'))
        elif candidate["type"] == "edge":
            u, v, t = candidate["u"], candidate["v"], candidate["t"]
            edge_len = chassis[u][v].get("weight", 1.0)
            for attach_node in attachment_nodes_in_cluster:
                cost = min(all_pairs_lengths.get(u, {}).get(attach_node, float('inf')) + t * edge_len, 
                           all_pairs_lengths.get(v, {}).get(attach_node, float('inf')) + (1 - t) * edge_len)
                total_cost += cost
        return total_cost

    
    
    
    @profile_function
    def _get_path_and_length_from_centroid(self, chassis, centroid, target_node):
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
                return nx.dijkstra_path(chassis, source=u, target=target_node, weight="weight"), len_u + t * edge_len
            else:
                return nx.dijkstra_path(chassis, source=v, target=target_node, weight="weight"), len_v + (1 - t) * edge_len
        return [], float('inf')

    def _format_and_export_output(self, original_graph: nx.Graph, clusters: Dict[str, Any], total_wire_length: float) -> Dict[str, Any]:
        G_out = original_graph.copy()
        for cluster_id, cluster_data in clusters.items():
            centroid = cluster_data.get("centroid")
            if not centroid: continue
            extender_id = f"EXT_{cluster_id.split('_')[-1]}"
            G_out.add_node(extender_id, pos=centroid["pos"], type="extender", is_io=False)
            for io_node, path_data in cluster_data.get("wiring_paths", {}).items():
                G_out.add_edge(extender_id, io_node, weight=path_data.get("length", 0.0), edge_type="optimized_wire")
        
        # Format G_out into the desired dictionary structure
        nodes = list(G_out.nodes())
        coordinates = {n: d.get("pos") for n, d in G_out.nodes(data=True) if "pos" in d}
        
        edges_dict = {}
        for u, v, d in G_out.edges(data=True):
            weight = d.get("weight", 1.0)
            if u not in edges_dict:
                edges_dict[u] = []
            if v not in edges_dict:
                edges_dict[v] = []
            edges_dict[u].append([v, weight])
            edges_dict[v].append([u, weight])

        output_data = {
            "nodes": nodes,
            "coordinates": coordinates,
            "edges": edges_dict,
            "clusters": clusters,
            "total_wire_length": total_wire_length
            
        }
        
        filename = f"clustered_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(self.export_dir, filename)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        self.logger.info(f"Successfully exported clustered graph to {output_path}")
        
        output_data["output_path"] = output_path
        return output_data
