"""
Graph Utilities
===============

Lightweight, dependency-free helpers used to inspect and sanity-check a
Network graph (chassis + I/O). Kept separate from ``graph_loader`` so that
both the GUI worker thread and the PDF report generator can import from here
without pulling in any other concerns.
"""

from typing import Any, Dict, List

import networkx as nx


def get_graph_statistics(graph: nx.Graph) -> Dict[str, Any]:
    """
    Return basic counts for nodes and edges, split by I/O vs. chassis.

    Args:
        graph: The combined network graph.

    Returns:
        A dict with counts that can be serialised to JSON.
    """
    io_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_io", False)]
    chassis_nodes = [n for n, d in graph.nodes(data=True) if not d.get("is_io", False)]

    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "io_nodes": len(io_nodes),
        "chassis_nodes": len(chassis_nodes),
        "node_types": {"io": len(io_nodes), "chassis": len(chassis_nodes)},
    }


def validate_graph(graph: nx.Graph) -> Dict[str, Any]:
    """
    Run cheap structural checks on the graph and collect human-readable warnings.

    Args:
        graph: The combined network graph.

    Returns:
        A dict with a ``warnings`` list and an ``is_valid`` boolean.
    """
    warnings: List[str] = []

    if not nx.is_connected(graph):
        warnings.append("Graph is not connected")

    isolated_nodes = list(nx.isolates(graph))
    if isolated_nodes:
        warnings.append(f"Found {len(isolated_nodes)} isolated nodes")

    return {"warnings": warnings, "is_valid": len(warnings) == 0}
