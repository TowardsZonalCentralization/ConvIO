import csv
import json
import math
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx
from scipy.spatial import KDTree
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


Point = Tuple[float, float]


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

@profile_function
def project_point_to_segment(p: Point, a: Point, b: Point) -> Tuple[Point, float]:
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    v_len_sq = vx * vx + vy * vy
    if v_len_sq == 0.0:
        return a, 0.0
    t = (wx * vx + wy * vy) / v_len_sq
    t_clamped = max(0.0, min(1.0, t))
    qx = ax + t_clamped * vx
    qy = ay + t_clamped * vy
    return (qx, qy), t_clamped


def _setup_logger(config: Dict[str, Any]) -> logging.Logger:
    log_cfg = config.get("logging", {})
    paths_cfg = config.get("paths", {})
    log_dir = paths_cfg.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("graph_loader")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(log_dir, "graph_loader_debug.log"), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"))
    logger.addHandler(fh)
    if log_cfg.get("enable_log_to_console", True):
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO))
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(ch)
    return logger


class GraphLoader:
    @profile_function
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.logger = _setup_logger(config_dict)
        if "graph_loader" not in config_dict:
            raise ValueError("Missing 'graph_loader' section in configuration")
        gl = config_dict["graph_loader"]
        self.min_direct_node_distance_mm = float(gl["min_direct_node_distance_mm"])
        self.allow_projection_on_edge = bool(gl["allow_projection_on_edge"])
        self.skip_self_loops = bool(gl["skip_self_loops"])
        paths_cfg = config_dict.get("paths", {})
        self.export_dir = paths_cfg.get("export_dir", "./export")
        os.makedirs(self.export_dir, exist_ok=True)
        self.enhanced_graph: Optional[nx.Graph] = None

    
    @profile_function
    def load_chassis_graph(self, json_file_path: str) -> nx.Graph:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.logger.info(f"DEBUG: Nodes from json file: {data['nodes']}")
        G = nx.Graph()
        for nid in data["nodes"]:
            coord = data["coordinates"].get(nid)
            x, y, z = float(coord[0]), float(coord[1]), float(coord[2]) if len(coord) > 2 else 0.0
            G.add_node(nid, pos=(x, y), x=x, y=y, z=z, type="chassis", is_io=False)
        for src, nbrs in data["edges"].items():
            if src not in G: continue
            for entry in nbrs:
                dst, weight = entry[0], float(entry[1])
                if self.skip_self_loops and src == dst: continue
                if dst not in G: continue
                G.add_edge(src, dst, weight=weight, length=weight, edge_type="chassis")
        self.logger.info("Chassis loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    
    @profile_function
    def load_io_coordinates_from_csv(self, csv_file_path: str) -> List[Point]:
        pts: List[Point] = []
        with open(csv_file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pts.append((float(row["X"]), float(row["Y"])))
        self.logger.info("Loaded %d I/O points", len(pts))
        return pts

    
    @profile_function
    def add_io_nodes_to_graph(self, graph: nx.Graph, points: List[Point]) -> nx.Graph:
        G = graph.copy()
        chassis_nodes = [n for n, d in G.nodes(data=True) if not d.get("is_io", False)]
        chassis_positions = [G.nodes[n]["pos"] for n in chassis_nodes]
        kdtree = KDTree(chassis_positions)
        
        for idx, io_pos in enumerate(points, start=1):
            io_id = f"IO_{idx}"
            G.add_node(io_id, pos=io_pos, x=io_pos[0], y=io_pos[1], z=0.0, type="io", is_io=True)
            dist, nearest_idx = kdtree.query(io_pos)
            nearest_node = chassis_nodes[nearest_idx]
            
            if dist > self.min_direct_node_distance_mm and self.allow_projection_on_edge:
                best_d, best_proj = float('inf'), None
                for u, v in G.edges():
                    if G.nodes[u].get("is_io") or G.nodes[v].get("is_io"): continue
                    q, t = project_point_to_segment(io_pos, G.nodes[u]["pos"], G.nodes[v]["pos"])
                    d = euclidean(io_pos, q)
                    if d < best_d:
                        best_d, best_proj = d, (u, v, q, t)
                
                if best_proj:
                    u, v, q, t = best_proj
                    if t > 1e-6 and t < 1 - 1e-6:
                        proj_node = f"{u}_{v}_proj_{idx}"
                        G.add_node(proj_node, pos=q, x=q[0], y=q[1], z=0.0, type="chassis", is_io=False)
                        w = G[u][v].get('weight', 1.0)
                        G.remove_edge(u, v)
                        G.add_edge(u, proj_node, weight=w*t, edge_type="chassis")
                        G.add_edge(proj_node, v, weight=w*(1-t), edge_type="chassis")
                        nearest_node = proj_node
                        dist = best_d
                    elif t <= 1e-6:
                        nearest_node = u
                        dist = euclidean(io_pos, G.nodes[u]["pos"])
                    else:
                        nearest_node = v
                        dist = euclidean(io_pos, G.nodes[v]["pos"])
            G.add_edge(io_id, nearest_node, weight=dist, edge_type="io_edge")
        self.enhanced_graph = G
        return G

    @profile_function
    def export_enhanced_graph_json(self, filename: Optional[str] = None) -> str:
        if not self.enhanced_graph:
            raise ValueError("No enhanced graph to export")
        if not filename:
            filename = f"enhanced_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path = os.path.join(self.export_dir, filename)
        data = nx.node_link_data(self.enhanced_graph)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        self.logger.info("Exported enhanced graph to %s", out_path)
        return out_path

@profile_function
def create_graph_loader_from_config(config_dict: Dict[str, Any]) -> GraphLoader:
    return GraphLoader(config_dict)
