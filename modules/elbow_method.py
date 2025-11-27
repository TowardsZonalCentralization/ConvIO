"""
Elbow Method for finding optimum number of clusters using K-Means
=================================================================

This module provides the `ElbowMethodAnalyzer` class, which is used to
determine the optimal number of clusters for a given set of I/O nodes. This is
a crucial step before the main clustering algorithm runs, as it provides a
data-driven suggestion for the number of I/O aggregators needed.

The workflow is as follows:
1.  Extract the 2D coordinates of all I/O nodes from the input graph.
2.  Iteratively run the KMeans clustering algorithm for a range of `k` values
    (e.g., from 1 to 12 clusters).
3.  For each `k`, calculate the Within-Cluster Sum of Squares (WCSS), which
    measures the compactness of the clusters.
4.  Identify the "elbow point" on the plot of WCSS vs. `k`. This point
    represents the best trade-off between the number of clusters and the
    total error.
5.  Return the optimal `k` value and the data needed for visualization.
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

logger = logging.getLogger(__name__)


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


class ElbowMethodAnalyzer:
    """
    Analyzes I/O node coordinates to find the optimal number of clusters.
    """
    @profile_function
    def __init__(self, k_min: int = 1, k_max: int = 12, random_state: int = 42, n_init: int = 10):
        """
        Initializes the analyzer with parameters for the KMeans algorithm.

        Args:
            k_min: The minimum number of clusters to test.
            k_max: The maximum number of clusters to test.
            random_state: A seed for the random number generator for reproducibility.
            n_init: The number of times KMeans will be run with different seeds.
        """
        logger.debug(f"Initializing ElbowMethodAnalyzer with k_min={k_min}, k_max={k_max}")
        if k_min < 1:
            raise ValueError("k_min must be >= 1")
        if k_max < k_min:
            raise ValueError("k_max must be >= k_min")

        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state
        self.n_init = n_init

    @profile_function
    def find_optimal_clusters(self, graph: nx.Graph) -> Tuple[int, Dict[str, Any]]:
        """
        Calculates the optimal number of clusters for the I/O nodes in a graph.

        Args:
            graph: A NetworkX graph containing I/O nodes with position data.

        Returns:
            A tuple containing the optimal number of clusters (k) and a
            dictionary with the data used for the elbow plot.
        """
        logger.info("Starting elbow analysis to find optimal k.")
        
        # Extract 2D coordinates of all I/O nodes from the graph.
        io_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_io", False)]
        if not io_nodes:
            logger.warning("No I/O nodes found; defaulting to k=1.")
            return 1, {"k_values": [1], "wcss": [0.0], "elbow_k": 1, "io_count": 0}

        coords = []
        for n in io_nodes:
            pos = graph.nodes[n].get("pos")
            if not pos or len(pos) < 2:
                raise ValueError(f"I/O node '{n}' is missing a valid 'pos' attribute.")
            coords.append([float(pos[0]), float(pos[1])])
        
        X = np.array(coords, dtype=float)
        io_count = X.shape[0]

        # Handle edge case where there's only one I/O node.
        if io_count <= 1:
            logger.info("Only one I/O node exists; optimal k is 1.")
            return 1, {"k_values": [1], "wcss": [0.0], "elbow_k": 1, "io_count": io_count}

        # Determine the range of k values to test.
        k_max = min(self.k_max, io_count - 1) # k cannot be greater than the number of samples.
        k_values = list(range(self.k_min, k_max + 1))

        # Calculate WCSS for each value of k.
        wcss = []
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init)
            km.fit(X)
            wcss.append(float(km.inertia_))
        
        # Find the elbow point in the WCSS curve.
        elbow_k = self._detect_elbow(k_values, wcss)

        elbow_data = {
            "k_values": k_values,
            "wcss": wcss,
            "elbow_k": elbow_k,
            "io_count": int(io_count),
        }
        logger.info(f"Elbow analysis complete. Optimal k found: {elbow_k}")
        return elbow_k, elbow_data

    @profile_function
    def _detect_elbow(self, k_values: List[int], wcss: List[float]) -> int:
        """
        Finds the elbow point in a curve using the maximum distance from a line method.

        This heuristic identifies the point that is farthest from the straight line
        drawn between the first and last points of the curve, which typically
        corresponds to the "elbow" where the rate of descent slows.

        Args:
            k_values: The list of k values (x-coordinates).
            wcss: The list of WCSS values (y-coordinates).

        Returns:
            The k value corresponding to the detected elbow point.
        """
        if len(k_values) <= 2:
            return k_values[0]

        points = np.array([k_values, wcss]).T
        
        # Normalize the points to a [0, 1] range. This is crucial for the distance
        # calculation to work correctly, as it prevents the result from being
        # skewed by the scale of the axes.
        points_normalized = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

        # The line is defined by the first and last points of the normalized curve.
        line_start = points_normalized[0]
        line_end = points_normalized[-1]
        
        # Calculate the perpendicular distance of each point from the line.
        line_vec = line_end - line_start
        point_vecs = points_normalized - line_start
        distances = np.abs(np.cross(line_vec, point_vecs)) / np.linalg.norm(line_vec)
        
        # The elbow is the point with the maximum distance from the line.
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]
