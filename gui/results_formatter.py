"""
CONVIO - Results Formatter
==========================

Pure computation functions — no Qt imports, no side effects.

Every function takes raw result dicts and the cost_cfg dict from the main
window, and returns plain Python data structures ready for display.

``cost_cfg`` shape:
    {"currency": str, "wire_price_per_m": float,
     "CAN_bus_price_per_m": float, "wire_weight_per_m_kg": float}
"""

from typing import Any, Dict, List, Optional, Tuple


# ── Helpers ────────────────────────────────────────────────────────────────────

def _m(length_mm: float) -> float:
    """Convert mm to metres."""
    return length_mm / 1000.0


def _pct(a: float, b: float) -> str:
    """Percentage saving of b relative to a."""
    if a > 0:
        return f"{((a - b) / a * 100):+.1f}%"
    return "—"


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


# ── Single-architecture metrics ────────────────────────────────────────────────

def baseline_metrics(hpc_results: Dict[str, Any], cost_cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Flat metrics dict for the Baseline Wiring panel.

    Returns:
        {"Total Wire Length": "12345.67 mm", "Estimated Cost": "18.52 EURO", ...}
    """
    if not hpc_results:
        return {}
    total  = hpc_results.get("total_length", 0.0)
    p_m    = cost_cfg.get("wire_price_per_m", 0.0)
    w_m    = cost_cfg.get("wire_weight_per_m_kg", 0.0)
    cur    = cost_cfg.get("currency", "")
    n_io   = len(hpc_results.get("paths", {}))
    return {
        "Total Wire Length":   f"{total:,.2f} mm",
        "Connections":         str(n_io),
        "Estimated Cost":      f"{_m(total) * p_m:,.2f} {cur}",
        "Estimated Weight":    f"{_m(total) * w_m:,.3f} kg",
    }


def optimised_metrics(clustering_results: Dict[str, Any], cost_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Flat metrics dict for the Step 4 / Bus Topology panel."""
    if not clustering_results:
        return {}
    wire_len = clustering_results.get("total_wire_length", 0.0)
    can_len  = clustering_results.get("can_bus", {}).get("total_length", 0.0)
    total    = clustering_results.get("overall_wiring_harness_length", 0.0)
    p_m      = cost_cfg.get("wire_price_per_m", 0.0)
    cp_m     = cost_cfg.get("CAN_bus_price_per_m", 0.0)
    w_m      = cost_cfg.get("wire_weight_per_m_kg", 0.0)
    cur      = cost_cfg.get("currency", "")
    n_clust  = len([k for k in clustering_results.get("clusters", {}) if k != "can_bus"])

    wire_cost = _m(wire_len) * p_m
    can_cost  = _m(can_len)  * cp_m
    weight    = _m(total)    * w_m
    return {
        "I/O Wiring Length":    f"{wire_len:,.2f} mm",
        "CAN Bus Length":       f"{can_len:,.2f} mm",
        "Total Harness Length": f"{total:,.2f} mm",
        "I/O Clusters":         str(n_clust),
        "Wiring Cost":          f"{wire_cost:,.2f} {cur}",
        "CAN Bus Cost":         f"{can_cost:,.2f} {cur}",
        "Total Cost":           f"{wire_cost + can_cost:,.2f} {cur}",
        "Estimated Weight":     f"{weight:,.3f} kg",
    }


def architecture_metrics(
    topology_label: str,
    clustering_results: Dict[str, Any],
    network_segments: Dict[str, float],
    cost_cfg: Dict[str, Any],
) -> Dict[str, str]:
    """
    Metrics panel data for any communication-network topology.

    The wiring harness (I/O → aggregator) is common to all topologies, only
    the network cable varies. Passing segments as a dict lets each topology
    describe its own layout:

    - Bus Topology         : ``{"CAN Bus": bus_len}``
    - Redundant Bus        : ``{"Forward Bus": bus_len, "Return Path": extra}``
    - Star and Ring        : ``{"Star Paths": star_len, "Ring": ring_len}``
    """
    if not clustering_results:
        return {}
    wire_len = clustering_results.get("total_wire_length", 0.0)
    net_total = sum(network_segments.values())
    n_clust   = len([k for k in clustering_results.get("clusters", {}) if k != "can_bus"])

    p_m  = cost_cfg.get("wire_price_per_m", 0.0)
    cp_m = cost_cfg.get("CAN_bus_price_per_m", 0.0)
    w_m  = cost_cfg.get("wire_weight_per_m_kg", 0.0)
    cur  = cost_cfg.get("currency", "")

    wire_cost = _m(wire_len)  * p_m
    net_cost  = _m(net_total) * cp_m
    weight    = _m(wire_len + net_total) * w_m

    result: Dict[str, str] = {
        "Topology":               topology_label,
        "— Wiring Harness —":     "",
        "  Length":               f"{wire_len:,.2f} mm",
        "  Connections":          str(n_clust) + " clusters",
        "  Cost":                 f"{wire_cost:,.2f} {cur}",
        "— Network Cable —":      "",
        "  Total Length":         f"{net_total:,.2f} mm",
    }
    for segment_name, segment_len in network_segments.items():
        result[f"    · {segment_name}"] = f"{segment_len:,.2f} mm"
    result["  Cost"]                  = f"{net_cost:,.2f} {cur}"
    result["— Combined —"]            = ""
    result["  Overall Wiring Length"] = f"{wire_len + net_total:,.2f} mm"
    result["  Total Cost"]            = f"{wire_cost + net_cost:,.2f} {cur}"
    result["  Total Weight"]          = f"{weight:,.3f} kg"
    return result


# Kept as a thin alias so older callers still work during transition.
def topology_metrics(topology_label, clustering_results, extra_length_mm, cost_cfg):
    """Deprecated — use architecture_metrics instead."""
    bus_len = clustering_results.get("can_bus", {}).get("total_length", 0.0) if clustering_results else 0.0
    return architecture_metrics(
        topology_label, clustering_results,
        {"Forward Bus": bus_len, topology_label: extra_length_mm}, cost_cfg,
    )


# ── Summary comparison ─────────────────────────────────────────────────────────

def summary_rows(
    hpc_results: Optional[Dict],
    clustering_results: Optional[Dict],
    cost_cfg: Dict[str, Any],
) -> List[Tuple[str, str, str, str]]:
    """
    Rows for the Results tab summary table.

    Returns: [(metric, baseline_val, optimised_val, saving), ...]
    """
    p_m  = cost_cfg.get("wire_price_per_m", 0.0)
    cp_m = cost_cfg.get("CAN_bus_price_per_m", 0.0)
    w_m  = cost_cfg.get("wire_weight_per_m_kg", 0.0)
    cur  = cost_cfg.get("currency", "")

    hpc_len  = hpc_results.get("total_length", 0.0)           if hpc_results    else 0.0
    wire_len = clustering_results.get("total_wire_length", 0.0) if clustering_results else 0.0
    can_len  = clustering_results.get("can_bus", {}).get("total_length", 0.0) if clustering_results else 0.0
    opt_len  = clustering_results.get("overall_wiring_harness_length", 0.0)  if clustering_results else 0.0

    hpc_cost = _m(hpc_len)  * p_m
    opt_cost = _m(wire_len) * p_m + _m(can_len) * cp_m
    hpc_wt   = _m(hpc_len)  * w_m
    opt_wt   = _m(wire_len + can_len) * w_m

    return [
        ("Wiring Harness (mm)",        _fmt(hpc_len),      _fmt(wire_len),       _pct(hpc_len, wire_len)),
        ("CAN Bus / Network (mm)",      "—",                _fmt(can_len),        "—"),
        ("Total Length (mm)",           _fmt(hpc_len),      _fmt(opt_len),        _pct(hpc_len, opt_len)),
        (f"Est. Cost ({cur})",          _fmt(hpc_cost),     _fmt(opt_cost),       _pct(hpc_cost, opt_cost)),
        ("Est. Weight (kg)",            _fmt(hpc_wt, 3),    _fmt(opt_wt, 3),      _pct(hpc_wt, opt_wt)),
    ]


def topology_comparison_rows(
    clustering_results: Dict[str, Any],
    topology_lengths: Dict[str, Dict[str, float]],
    hpc_results: Optional[Dict],
    cost_cfg: Dict[str, Any],
) -> List[Tuple[str, str, str, str, str, str, str]]:
    """
    Rows for the Overall Summary table — one row per architecture.

    Args:
        clustering_results: The Step 4 result; provides the wiring harness length.
        topology_lengths:   ``{topology_name: {"network_segments": {label: mm, ...}}}``.
                            Keys must include "Bus Topology" (used as savings reference).
        hpc_results:        Baseline result; used to compute saving vs. baseline.

    Returns:
        [(topology, harness_mm, network_mm, total_mm, cost, vs_baseline, vs_bus), ...]

    Sorted with Baseline first (if present), then topologies by total ascending.
    """
    p_m  = cost_cfg.get("wire_price_per_m", 0.0)
    cp_m = cost_cfg.get("CAN_bus_price_per_m", 0.0)
    cur  = cost_cfg.get("currency", "")

    rows: List[Tuple] = []

    # Baseline row (no clusters, no network — direct HPC)
    if hpc_results:
        hpc_len = hpc_results.get("total_length", 0.0)
        rows.append((
            "Baseline (Direct HPC)",
            _fmt(hpc_len), "—", _fmt(hpc_len),
            f"{_m(hpc_len) * p_m:.2f} {cur}",
            "—", "—",
        ))

    if not clustering_results or not topology_lengths:
        return rows

    wire_len = clustering_results.get("total_wire_length", 0.0)
    hpc_total = hpc_results.get("total_length", 0.0) if hpc_results else 0.0

    # Bus topology total acts as the "vs Bus Topology" reference
    bus_segments = topology_lengths.get("Bus Topology", {}).get("network_segments", {})
    bus_network_total = sum(bus_segments.values())
    bus_total = wire_len + bus_network_total

    topology_rows: List[Tuple] = []
    for topo_name, info in topology_lengths.items():
        net_total = sum(info.get("network_segments", {}).values())
        total     = wire_len + net_total
        cost      = _m(wire_len) * p_m + _m(net_total) * cp_m
        topology_rows.append((
            topo_name,
            _fmt(wire_len), _fmt(net_total), _fmt(total),
            f"{cost:.2f} {cur}",
            _pct(hpc_total, total) if hpc_total > 0 else "—",
            _pct(bus_total, total) if bus_total > 0 else "—",
        ))

    # Sort topologies by total length ascending
    topology_rows.sort(key=lambda r: float(r[3]))
    rows.extend(topology_rows)
    return rows


def cluster_breakdown_rows(
    clustering_results: Dict[str, Any],
    cost_cfg: Dict[str, Any],
) -> List[Tuple[str, str, str, str]]:
    """
    Rows for the per-cluster breakdown table.

    Returns: [(cluster_id, io_count, wire_length_mm, cost), ...]
    """
    if not clustering_results:
        return []
    p_m  = cost_cfg.get("wire_price_per_m", 0.0)
    cur  = cost_cfg.get("currency", "")
    rows = []
    for cid, cdata in clustering_results.get("clusters", {}).items():
        if cid == "can_bus":
            continue
        n_io   = len(cdata.get("io_nodes", []))
        length = cdata.get("cluster_wire_length", 0.0)
        cost   = _m(length) * p_m
        rows.append((cid, str(n_io), _fmt(length), f"{cost:.2f} {cur}"))
    return rows


def linkage_comparison_rows(
    all_linkage_results: Dict[str, Any],
    hpc_results: Optional[Dict],
    cost_cfg: Dict[str, Any],
) -> List[Tuple[str, str, str, str, str, str]]:
    """
    Rows for the linkage method comparison table.

    Returns:
        [(linkage, wire_mm, can_mm, total_mm, cost, saving_vs_baseline), ...]
    """
    p_m   = cost_cfg.get("wire_price_per_m", 0.0)
    cp_m  = cost_cfg.get("CAN_bus_price_per_m", 0.0)
    cur   = cost_cfg.get("currency", "")
    hpc_l = hpc_results.get("total_length", 0.0) if hpc_results else 0.0

    rows = []
    for method, result in all_linkage_results.items():
        wire = result.get("total_wire_length", 0.0)
        can  = result.get("can_bus", {}).get("total_length", 0.0)
        tot  = result.get("overall_wiring_harness_length", 0.0)
        cost = _m(wire) * p_m + _m(can) * cp_m
        rows.append((
            method,
            _fmt(wire), _fmt(can), _fmt(tot),
            f"{cost:.2f} {cur}",
            _pct(hpc_l, tot),
        ))
    # Sort by total ascending (best first)
    rows.sort(key=lambda r: float(r[3]))
    return rows
