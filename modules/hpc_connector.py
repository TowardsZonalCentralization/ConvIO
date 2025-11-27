"""
Direct Wiring Connector
===========================

This module provides a baseline calculation for the total wiring length
required to connect every I/O node directly to the High-Performance Computer
(HPC). This serves as a benchmark against which the optimized, clustered
solution can be compared.

The process is straightforward:
1.  Identify the HPC node and all I/O nodes in the graph.
2.  For each I/O node, calculate the shortest path to the HPC using
    Dijkstra's algorithm.
3.  Sum the lengths of all these paths to get the total direct wiring length.
4.  Export the results, including the paths, for visualization.
"""

import networkx as nx
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import cProfile
import pstats
import io
from functools import wraps


_active_profiler = None


def profile_function(func):
    """
    A decorator for profiling function performance.

    To enable, set the environment variable ENABLE_PROFILING=true. Profiling
    data is saved to the './profiling/functions' directory.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _active_profiler
        if os.getenv('ENABLE_PROFILING', 'false').lower() != 'true':
            return func(*args, **kwargs)
        if _active_profiler is not None:
            #print(f" Skipping profiling for {func.__name__} (profiler already active)")
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


@profile_function
def get_io_nodes(graph: nx.Graph) -> List[str]:
    """
    Extracts a list of I/O node identifiers from the graph.

    Args:
        graph: The NetworkX graph to search.

    Returns:
        A list of node names that are marked as I/O nodes.
    """
    return [node for node, data in graph.nodes(data=True) if data.get('is_io', False)]

@profile_function
def export_hpc_wiring_graph(graph: nx.Graph, paths: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Exports a copy of the graph with the direct HPC wiring paths highlighted.

    This is a utility for visualization, allowing the direct wiring solution
    to be displayed in the GUI.

    Args:
        graph: The original Network graph.
        paths: A dictionary of paths from the HPC to each I/O node.
        config: The application configuration dictionary.

    Returns:
        The file path of the exported JSON graph.
    """
    paths_cfg = config.get("paths", {})
    export_dir = paths_cfg.get("export_dir", "./export")
    os.makedirs(export_dir, exist_ok=True)

    g_out = graph.copy()
    # Mark the edges that are part of the direct HPC wiring paths.
    for path_data in paths.values():
        path = path_data.get("path", [])
        if len(path) > 1:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if g_out.has_edge(u, v):
                    g_out[u][v]['edge_type'] = 'hpc_wire'

    filename = f"Overall_wiring{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(export_dir, filename)
    data = nx.node_link_data(g_out)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return output_path


@profile_function
def calculate_direct_hpc_wiring(graph: nx.Graph, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates the total wiring length for a direct, point-to-point connection
    from the HPC to every I/O node.

    This serves as a baseline to measure the effectiveness of the clustering
    and optimization algorithm.

    Args:
        graph: The Network graph containing the HPC and I/O nodes.
        config: The application configuration dictionary.

    Returns:
        A dictionary containing the total wiring length, the individual paths,
        and the path to the exported visualization graph.
    """
    node_cfg = config.get("node_configuration", {})
    hpc_node_name = node_cfg.get("hpc_node_name", "H1")
    
    io_nodes = get_io_nodes(graph)
    
    if hpc_node_name not in graph:
        logging.error(f"HPC node '{hpc_node_name}' not found in the graph.")
        return None
        
    if not io_nodes:
        return {'total_length': 0, 'paths': {}, 'output_path': None}

    total_length = 0
    paths = {}
    
    # For each I/O node, find the shortest path to the HPC and sum the lengths.
    for io_node in io_nodes:
        try:
            length = nx.dijkstra_path_length(graph, source=hpc_node_name, target=io_node, weight='weight')
            path = nx.dijkstra_path(graph, source=hpc_node_name, target=io_node, weight='weight')
            total_length += length
            paths[io_node] = {'path': path, 'length': length}
        except nx.NetworkXNoPath:
            # This handles cases where an I/O node is on a disconnected part of the graph.
            paths[io_node] = {'path': [], 'length': float('inf')}

    output_path = export_hpc_wiring_graph(graph, paths, config)
    
    return {
        'hpc_node': hpc_node_name,
        'total_length': total_length,
        'paths': paths,
        'output_path': output_path
    }
