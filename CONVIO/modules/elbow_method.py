"""
Elbow Method Analyzer - File 2
==============================

This module determines the optimal number of clusters (I/O extenders) for the
I/O nodes in the enhanced graph produced by File 1 (graph_loader.py).

Workflow:
- Extract I/O nodes and their 2D coordinates from the graph
- Run KMeans for k in [1..k_max]
- Compute WCSS (within-cluster sum of squares) for each k
- Use an elbow (knee) detection heuristic (max distance to chord) to find optimal k
- Return optimal k and all elbow data for plotting in the GUI

Requirements:
- scikit-learn
- numpy
"""

from typing import Tuple, Dict, Any, List

import logging
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import cProfile
import pstats
import io
import os
from functools import wraps
from datetime import datetime
# Module-level logger
logger = logging.getLogger(__name__)


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



class ElbowMethodAnalyzer:
    """
    Analyze I/O node coordinates using the elbow method to find the optimal
    number of clusters (i.e., number of I/O extenders).
    """
    @profile_function
    def __init__(self, k_min: int = 1, k_max: int = 12, random_state: int = 42, n_init: int = 10):
        """
        Args:
            k_min: minimum number of clusters to evaluate
            k_max: maximum number of clusters to evaluate
            random_state: RNG seed for reproducibility
            n_init: KMeans n_init parameter
        """
        logger.debug("Initializing ElbowMethodAnalyzer(k_min=%s, k_max=%s, random_state=%s, n_init=%s)",
                     k_min, k_max, random_state, n_init)

        if k_min < 1:
            logger.error("Invalid k_min: %s (must be >=1)", k_min)
            raise ValueError("k_min must be >= 1")
        if k_max < k_min:
            logger.error("Invalid k_max: %s (must be >= k_min=%s)", k_max, k_min)
            raise ValueError("k_max must be >= k_min")

        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state
        self.n_init = n_init
    
    
    @profile_function
    def find_optimal_clusters(self, graph: nx.Graph) -> Tuple[int, Dict[str, Any]]:
        """
        Compute WCSS over a range of k and detect the elbow.

        Args:
            graph: enhanced NetworkX graph containing I/O nodes (is_io=True) with 'pos' attributes

        Returns:
            (optimal_k, elbow_data)
            where elbow_data = {
                "k_values": List[int],
                "wcss": List[float],
                "elbow_k": int,
                "io_count": int
            }
        """
        logger.info("Starting elbow analysis")
        # Extract IO points from graph
        try:
            io_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_io", False)]
        except Exception as e:
            logger.exception("Failed to extract nodes from graph: %s", e)
            raise

        logger.debug("Found %d I/O nodes", len(io_nodes))

        if not io_nodes:
            logger.warning("No I/O nodes found; defaulting to k=1")
            elbow_data = {
                "k_values": [1],
                "wcss": [0.0],
                "elbow_k": 1,
                "io_count": 0,
            }
            return 1, elbow_data

        coords: List[List[float]] = []
        for n in io_nodes:
            pos = graph.nodes[n].get("pos")
            if not pos or len(pos) < 2:
                logger.error("IO node '%s' missing valid 'pos' attribute: %s", n, pos)
                raise ValueError(f"IO node '{n}' missing valid 'pos' attribute")
            try:
                coords.append([float(pos[0]), float(pos[1])])
            except Exception as e:
                logger.exception("Failed to parse 'pos' for node '%s': %s", n, e)
                raise

        X = np.array(coords, dtype=float)
        io_count = X.shape[0]
        logger.debug("Prepared coordinate matrix X with shape %s and io_count=%d", X.shape, io_count)

        # If only one I/O node, force k=1
        if io_count == 1:
            logger.info("Only one I/O node; returning k=1")
            elbow_data = {
                "k_values": [1],
                "wcss": [0.0],
                "elbow_k": 1,
                "io_count": 1,
            }
            return 1, elbow_data

        # Ensure k_max does not exceed number of points and is >= k_min
        k_max = min(self.k_max, max(self.k_min, io_count))
        k_values = list(range(self.k_min, k_max + 1))
        logger.debug("Evaluating k_values=%s", k_values)

        # Compute WCSS for each k
        wcss: List[float] = []
        for k in k_values:
            try:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init)
                km.fit(X)
                inertia = float(km.inertia_)
                wcss.append(inertia)
                logger.debug("k=%d, WCSS=%.6f", k, inertia)
            except Exception as e:
                logger.exception("KMeans failed for k=%d: %s", k, e)
                raise

        # Detect elbow using max distance to the line between first and last points
        try:
            elbow_k = self._detect_elbow(k_values, wcss)
        except Exception as e:
            logger.exception("Elbow detection failed: %s", e)
            raise

        elbow_data = {
            "k_values": k_values,
            "wcss": wcss,
            "elbow_k": elbow_k,
            "io_count": int(io_count),
        }
        logger.info("Elbow analysis complete: elbow_k=%d, io_count=%d", elbow_k, io_count)
        return elbow_k, elbow_data

    
    @profile_function
    def _detect_elbow(self, k_values: List[int], wcss: List[float]) -> int:
        """
        Heuristic: Find k that maximizes the perpendicular distance from the straight
        line connecting the first and last points on the (k, wcss) curve.

        Args:
            k_values: list of k evaluated
            wcss: corresponding within-cluster sum of squares for each k

        Returns:
            elbow_k: detected optimal k (falls back to smallest k on degenerate cases)
        """
        logger.debug("Detecting elbow for k_values=%s", k_values)

        if len(k_values) <= 2:
            logger.warning("Insufficient points for robust elbow detection (len=%d); returning %s",
                           len(k_values), k_values[0])
            return k_values[0]

        x = np.array(k_values, dtype=float)
        y = np.array(wcss, dtype=float)

        # Normalize points to a [0, 1] range for stable distance calculation
        if x.max() > x.min():
            x_norm = (x - x.min()) / (x.max() - x.min())
        else:
            x_norm = np.zeros_like(x) # All k values are the same
        
        if y.max() > y.min():
            y_norm = (y - y.min()) / (y.max() - y.min())
        else:
            y_norm = np.zeros_like(y) # All wcss values are the same

        # Line between the first and last points (in normalized space)
        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])

        # Calculate the perpendicular distance of each point to the line
        # The line is p1 + t * (p2 - p1). The distance from a point p0 is |(p1-p0) x (p1-p2)| / |p1-p2|
        d = p2 - p1
        d_mag = np.linalg.norm(d)
        if d_mag == 0:
            # All points are the same, no elbow
            return k_values[0]

        distances = np.abs(np.cross(p2 - p1, np.column_stack((x_norm, y_norm)) - p1)) / d_mag
        
        # The elbow is the point with the maximum distance
        elbow_idx = np.argmax(distances)
        elbow_k = k_values[elbow_idx]
        
        logger.debug("Distances=%s, max_idx=%d, elbow_k=%d", distances, elbow_idx, elbow_k)
        return elbow_k
