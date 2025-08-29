import networkx as nx
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any
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

@profile_function
def get_io_nodes(graph: nx.Graph) -> list:
    """Extracts IO nodes from the graph."""
    return [node for node, data in graph.nodes(data=True) if data.get('is_io', False)]

@profile_function
def export_hpc_wiring_graph(graph: nx.Graph, paths: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Exports a new graph showing the direct wiring from HPC to all I/O nodes.
    """
    paths_cfg = config.get("paths", {})
    export_dir = paths_cfg.get("export_dir", "./export")
    os.makedirs(export_dir, exist_ok=True)

    g_out = graph.copy()
    for io_node, path_data in paths.items():
        path = path_data.get("path", [])
        if len(path) > 1:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if g_out.has_edge(u, v):
                    g_out[u][v]['edge_type'] = 'hpc_wire'

    filename = f"hpc_wiring_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(export_dir, filename)
    data = nx.node_link_data(g_out)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Successfully exported HPC wiring graph to {output_path}")
    return output_path


@profile_function
def calculate_direct_hpc_wiring(graph: nx.Graph, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates the total wiring length for connecting all IO nodes directly to the HPC.
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
    
    for io_node in io_nodes:
        try:
            length = nx.dijkstra_path_length(graph, source=hpc_node_name, target=io_node, weight='weight')
            path = nx.dijkstra_path(graph, source=hpc_node_name, target=io_node, weight='weight')
            total_length += length
            paths[io_node] = {'path': path, 'length': length}
        except nx.NetworkXNoPath:
            paths[io_node] = {'path': [], 'length': float('inf')}

    output_path = export_hpc_wiring_graph(graph, paths, config)
    
    return {
        'hpc_node': hpc_node_name,
        'total_length': total_length,
        'paths': paths,
        'output_path': output_path
    }
