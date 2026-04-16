# CONVIO: Automotive Wiring Harness Optimizer

## Overview

**CONVIO** (Convergence of I/O) is a specialized engineering tool designed to streamline and optimize the complex process of automotive wiring harness design. By leveraging advanced computational methods, CONVIO provides a robust platform for engineers to design, analyze, and refine wiring layouts with greater efficiency and precision.

The increasing complexity of modern vehicles, driven by advanced driver-assistance systems (ADAS), infotainment, and vehicle connectivity, has led to a proportional increase in the complexity of the automotive wiring harness. Traditional wiring architectures, which connect each component directly to a central computer, are becoming untenable due to weight, cost, and manufacturing challenges. The Zonal Electrical/Electronic (EE) Architecture with I/O aggregator concept paradigm addresses these issues by decentralizing intelligence and I/O management into regional zones.

CONVIO models, analyzes, and optimizes such architectures, comparing them against traditional direct-to-HPC (High-Performance Computer) wiring to demonstrate the benefits of a zonal approach.

## How to Run
Download suitable prebuilt executable from one of the releases [here](https://github.com/TowardsZonalCentralization/ConvIO/releases).

Or build locally:
```bash
pip install -r requirements.txt
python main.py
```


## Core Capabilities

*   **Graph-Based System Modeling:** Accurately models the vehicle chassis and I/O points as a weighted (distance), undirected graph.
*   **Automated Cluster Analysis:** Uses the elbow method to scientifically determine the optimal number of I/O clusters (zones).
*   **Shortest-Path Optimization:** Implements Dijkstra's algorithm for efficient wiring path calculation within the chassis graph.
*   **Zonal Architecture Optimization:** Employs Agglomerative Clustering and an iterative refinement process to group I/O points into optimal zones, minimizing wiring to I/O aggregators.
*   **Baseline Architecture Analysis:** Calculates the total wiring length for a traditional direct-to-HPC architecture for comparative benchmarking.
*   **Comparative Analysis:** Benchmarks optimized zonal architectures against traditional direct-to-HPC wiring, providing quantitative improvements in length and cost.
*   **Insightful Reporting:** Generates comprehensive PDF reports and visualizations to support design decisions.
*   **Configurable Workflow:** The entire workflow is controlled through a clear and concise YAML configuration (`config.yaml`), ensuring adaptability and reproducibility.

## Methodology

The CONVIO application follows a sequential workflow to achieve optimal wiring harness designs:

1.  **System Modeling (`graph_loader.py`)**:
    *   The physical structure of the vehicle is translated into a computational model.
    *   A JSON file describing the vehicle chassis (nodes as structural points, edges as valid wire pathways) forms a weighted, undirected graph.
    *   A separate CSV file with 2D coordinates of I/O devices is processed.
    *   Each I/O point is mapped onto the chassis graph, either by connecting to the nearest existing node or by projecting onto the closest edge, creating a new node.
    *   This results in a unified network graph for subsequent analysis.
    *   The algorithm considers the special case of "Branch-out points" where it checks the distance of the I/O from the nearest node is less than the set threashold if not then it creates a Branch-out point from the wiring harness directly.

2.  **Optimal Cluster Analysis (`elbow_method.py`)**:
    *   To determine the most efficient number of I/O aggregators (clusters) for the Zonal EE Architecture, a quantitative approach is employed.
    *   The `elbow_method` module applies the K-Means clustering algorithm to the 2D spatial data of the I/O nodes for a range of cluster counts (k).
    *   It calculates the Within-Cluster Sum of Squares (WCSS) for each k. The 'elbow point' on the WCSS curve, where the rate of decrease sharply changes, is identified as a data-driven suggestion for the optimal number of clusters. This balances harness simplification against the number of required I/O aggregators.

3.  **Zonal Architecture Optimization (`clustering_dijkstra.py`)**:
    *   With the optimal number of clusters determined, this module executes a sophisticated two-stage optimization.
    *   **First**, it constructs a distance matrix representing the true shortest-path distance within the chassis graph between every pair of I/O nodes. Using this matrix, it performs an initial partitioning of the I/O nodes using Agglomerative Clustering to establish a strong baseline grouping.
    *   **Second**, it enters a refinement phase. For each initial cluster, an optimal I/O aggregator location (centroid) is calculated. The algorithm then iteratively re-assigns each I/O node to the zone with the closest centroid, ensuring that the final groupings are not just based on I/O-to-I/O proximity, but on the more critical I/O-to-aggregator wiring distance. This process repeats until the cluster memberships stabilize, guaranteeing a more logically and efficiently partitioned zonal architecture.

4.  **Baseline Architecture Analysis (`hpc_connector.py`)**:
    *   To provide a benchmark for evaluation, this module calculates the total wiring length of a traditional, non-zonal architecture.
    *   In this baseline scenario, every I/O node is assumed to be wired directly to the High-Performance Computer (HPC).
    *   The module computes the shortest path on the graph from each I/O node to the designated HPC node. The sum of the lengths of these paths represents the total harness length for the baseline architecture, against which the zonal model is compared.

## Algorithmic Framework

The optimization process is underpinned by several key algorithms and data structures, each chosen for a specific purpose:

*   **KD-Tree (k-dimensional tree)**:
    *   **Purpose:** To perform efficient nearest neighbor searches when mapping I/O nodes to the chassis graph.
    *   **Rationale:** Organizes points in a multi-dimensional space for exceptionally fast queries (logarithmic time complexity on average), avoiding slow, brute-force comparisons.

*   **K-Means Clustering**:
    *   **Purpose:** To provide a data-driven suggestion for the optimal number of clusters (k).
    *   **Rationale:** Provides a fast heuristic for analyzing the spatial distribution of nodes. The Within‑Cluster Sum of Squares (WCSS) is computed for a range of k values, with k‑means minimizing the squared Euclidean distances within each cluster. The elbow point, indicating diminishing returns in WCSS reduction, can guide the selection of the number of zones.

*   **Dijkstra's Shortest Path Algorithm**:
    *   **Purpose:** To calculate the shortest path distances between nodes on the chassis graph.
    *   **Rationale:** Wiring must follow the vehicle's structure. Dijkstra's algorithm finds the shortest path by traversing graph edges, respecting constraints, and is essential for calculating true wiring distances for clustering and total path lengths.

*   **Agglomerative Hierarchical Clustering**:
    *   **Purpose:** To perform the final, high-fidelity clustering of I/O nodes into zones.
    *   **Rationale:** Operates on a precomputed distance matrix (true wiring distances from Dijkstra's algorithm), ensuring clusters are based on actual path lengths, more specifically the Manhattan distances, through the chassis, resulting in more practical and accurately optimized zones.

*   **Greedy Nearest Neighbor Heuristic**:
    *   **Purpose:** To calculate a practical and efficient path for the CAN bus connecting all I/O aggregators.
    *   **Rationale:** Provides an excellent and fast approximation for connecting multiple points (a computationally hard problem like the Travelling Salesperson Problem). It constructs the path by iteratively traveling from the current point to the nearest unvisited I/O aggregator, ensuring a short and logical bus topology without excessive computation.

## Installation and Usage

### Prerequisites

*   Python 3.x
*   `pip` (Python package installer)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TowardsZonalCentralization/ConvIO.git
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The application's behavior is controlled by the `config.yaml` file located in the `CONVIO/` directory. This file allows you to specify:

*   **`paths`**: Directories for data, exports, and logs, including default chassis and I/O files.
*   **`graph_loader`**: Parameters for graph creation, such as `min_direct_node_distance_mm`, `allow_projection_on_edge`, and `skip_self_loops`.
*   **`logging`**: Configuration for console and file logging.
*   **`error_handling`**: Settings for error reporting.
*   **`reproducibility`**: Options to set global random seeds for consistent results.
*   **`gui`**: Window size and color palette for visualizations.
*   **`cost`**: Price per meter for wiring and CAN bus, and currency.
*   **`elbow_method`**: Range for `k_min`, `k_max`, `random_state`, and `n_init` for K-Means.
*   **`clustering`**: Maximum supported clusters.

Ensure `config.yaml` is correctly configured before running the application.

### Running the Application

To start the CONVIO GUI:

```bash
python main.py
```

### Workflow Steps in the GUI

1.  **Load Data Files (Step 1)**:
    *   Click "Load Chassis Graph (JSON)" to select your chassis definition file.
    *   Click "Load I/O Coordinates (CSV)" to select your I/O points file.
    *   Alternatively, click "Load Default Files" to load pre-configured files from `config.yaml`.
    *   Once both files are loaded, click "Process Graph" to build the network model. The "Network Graph" tab will display the loaded chassis and I/O points.

2.  **Find Optimal Clusters (Step 2)**:
    *   Click "Run Elbow Method Analysis" to determine the optimal number of clusters for your I/O points.
    *   The "Elbow Analysis" tab will display the WCSS curve, and the "Optimal clusters" label will update with the recommended `k` value. This value will also pre-fill the "Number of Clusters" spin box.

3.  **Overall Wiring Analysis (Step 3)**:
    *   Click "Calculate Overall Wiring" to compute the total wiring length and cost for the baseline direct-to-HPC architecture.
    *   Results will be displayed in the control panel, and the "Overall Wiring" tab will visualize the direct connections.

4.  **Clustering & Optimization (Step 4)**:
    *   Adjust the "Number of Clusters" (pre-filled from elbow method or manually set) and select a "Linkage Method" (`average`, `complete`, `single`).
    *   Click "Run Clustering & Optimization" to perform the zonal optimization for the selected method.
    *   Alternatively, click "Run Full Analysis & Compare" to run the optimization with all linkage methods and get a comparison table, with the best method automatically selected for visualization.
    *   The "EEA with I/O aggregators" tab will display the optimized zonal architecture, showing clusters, I/O aggregators (centroids), and wiring paths.
    *   Detailed length and cost metrics for the zonal architecture will be updated in the control panel.

### Exporting Results

*   **Export Results (JSON)**: From the `File` menu, select `Export Results (JSON)...` to save all analysis data (clustering, HPC, elbow, configuration) to a JSON file.
*   **Export Report (PDF)**: From the `File` menu, select `Export Report (PDF)...` to generate a comprehensive PDF report detailing the methodology, algorithmic framework, results, and comparative analysis, including all relevant figures and tables.

## Project Structure

```
.
├── CONVIO/
│   ├── config.yaml                 # Application configuration
│   ├── main.py                     # Main GUI application entry point
│   ├── requirements.txt            # Python dependencies
│   ├── data/                       # Sample input data (chassis JSON, I/O CSV)
│   ├── export/                     # Directory for exported results and reports
│   ├── logs/                       # Application logs
│   └── modules/
│       ├── clustering_dijkstra.py  # Implements clustering and Dijkstra's algorithm
│       ├── elbow_method.py         # Elbow method for optimal cluster determination
│       ├── graph_loader.py         # Loads chassis graph and I/O coordinates
│       ├── hpc_connector.py        # Calculates baseline direct-to-HPC wiring
│       └── report_generator.py     # Generates comprehensive PDF reports
└── README.md                       # This file
