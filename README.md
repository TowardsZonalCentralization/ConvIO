# Zonal Convergence

## Overview
Zonal Convergence is a Python-based toolkit designed for optimizing the I/O wiring architecture in commercial truck platforms. The project leverages clustering and pathfinding algorithms to achieve efficient wiring topology by:
- Clustering I/O signals based on spatial proximity and functional zones using **K-Means**.
- Computing the shortest paths between I/O nodes, extenders, and HPCs using **Dijkstra's Algorithm**.

This repository facilitates the reduction of wiring complexity and supports the design of a modular, scalable zonal architecture.

---

## Features
- **K-Means Clustering**:
  - Groups I/O points based on spatial location and signal grouping.
  - Determines optimal number and placement of I/O extenders (Zonal ECUs).

- **Dijkstra Pathfinding**:
  - Calculates the shortest path for wiring between I/O devices, extenders, and central units (e.g., HPC).
  - Generates visual and quantitative analysis of wiring length and efficiency.

---

## Directory Structure
```text
ZonalConvergence/
├── data/                   # Input datasets (CSV, JSON, etc.)
├── results/                # Output clusters and path metrics
├── scripts/
│   ├── kmeans_clustering.py
│   ├── dijkstra_pathfinder.py
│   └── utils.py
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── main.py                # Execution entry point

