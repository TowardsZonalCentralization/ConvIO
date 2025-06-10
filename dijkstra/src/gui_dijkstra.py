"""
Dijkstra Algorithm for optimising the wiring harness of a truck, with a GUI for user interaction.
"""

import json
import os
import heapq
import logging
import threading
from typing import Tuple, List, Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 
import networkx as nx

# ----------------------------- Logging Setup -----------------------------

def configure_logging(log_file: str):
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# ----------------------------- Graph Utilities -----------------------------

class Graph:
    """Graph representation including nodes, edges, and coordinates."""
    def __init__(self, data: Dict):
        self.nodes = data.get("nodes", [])
        self.edges = data.get("edges", {})
        self.coordinates = data.get("coordinates", {})
        self.validate()

    def validate(self):
        """Validates graph data integrity."""
        if len(self.nodes) != len(set(self.nodes)):
            raise ValueError("Duplicate nodes found in graph.")

        for node, neighbors in self.edges.items():
            for neighbor, weight in neighbors:
                if neighbor not in self.nodes:
                    raise ValueError(f"Undefined node in edges: {neighbor}")
                if float(weight) < 0:
                    raise ValueError(f"Negative weight detected between {node} and {neighbor}")

    @staticmethod
    def load_from_file(file_path: str) -> 'Graph':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return Graph(data)

# ----------------------------- Dijkstra Algorithm -----------------------------

class DijkstraSolver:
    """Solves shortest path using Dijkstra's algorithm."""
    def __init__(self, graph: Graph):
        self.graph = graph

    def solve(self, start: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        distances = {node: float('inf') for node in self.graph.nodes}
        predecessors = {node: None for node in self.graph.nodes}
        visited = set()

        distances[start] = 0
        pq = [(0, start)]  

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            if current_node in visited:
                continue
            visited.add(current_node)

            for neighbor, weight in self.graph.edges.get(current_node, []):
                distance = current_distance + float(weight)
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

        return distances, predecessors

    @staticmethod
    def reconstruct_path(predecessors: Dict[str, Optional[str]], target: str) -> List[str]:
        """Reconstruct the shortest path to the target node."""
        path = []
        while target:
            path.insert(0, target)
            target = predecessors[target]
        return path

# ----------------------------- Visualization -----------------------------

class GraphVisualizer:
    @staticmethod
    def animate(graph: Graph, path: List[str]):
        """3D animation of the shortest path traversal."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Optimised Wiring Harness Path")
        ax.set_xlabel("X-axis (cm)")
        ax.set_ylabel("Y-axis (cm)")
        ax.set_zlabel("Z-axis (cm)")

        distance_text = ax.text2D(0.05, 0.95, "Total length of wiring harness : 0 cm", transform=ax.transAxes)

        all_coords = list(graph.coordinates.values())
        x_vals, y_vals, z_vals = zip(*all_coords)
        max_range = max(
            max(x_vals) - min(x_vals),
            max(y_vals) - min(y_vals),
            max(z_vals) - min(z_vals)
        ) / 2
        mid_x = (max(x_vals) + min(x_vals)) / 2
        mid_y = (max(y_vals) + min(y_vals)) / 2
        mid_z = (max(z_vals) + min(z_vals)) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        def draw_base():
            for node, (x, y, z) in graph.coordinates.items():
                ax.scatter(x, y, z, color='blue', s=40)
                ax.text(x, y, z + 1, node, fontsize=9)
            for node, edges in graph.edges.items():
                for neighbor, _ in edges:
                    x_vals = [graph.coordinates[node][0], graph.coordinates[neighbor][0]]
                    y_vals = [graph.coordinates[node][1], graph.coordinates[neighbor][1]]
                    z_vals = [graph.coordinates[node][2], graph.coordinates[neighbor][2]]
                    ax.plot(x_vals, y_vals, z_vals, color='gray', linewidth=0.8)

        draw_base()

        line, = ax.plot([], [], [], color='red', linewidth=3)
        cumulative_distance = [0.0]

        def euclidean(p1, p2):
            return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

        def update(frame):
            xs, ys, zs = zip(*[graph.coordinates[n] for n in path[:frame + 1]])
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

            if frame > 0:
                last = graph.coordinates[path[frame - 1]]
                current = graph.coordinates[path[frame]]
                step = euclidean(last, current)
                cumulative_distance[0] += step
                distance_text.set_text(f"Total length of wiring harness : {cumulative_distance[0]:.2f} cm")

            return line, distance_text

        anim = FuncAnimation(fig, update, frames=len(path), interval=800, repeat=False)
        plt.show()

    @staticmethod
    def export(distances: Dict[str, float], predecessors: Dict[str, Optional[str]],
               path: List[str], total_distance: float, output_dir: str):
        """Export results to a JSON file."""
        results = {
            "distances": distances,
            "predecessors": predecessors,
            "shortest_path": path,
            "total_distance_cm": total_distance
        }
        with open(os.path.join(output_dir, 'dijkstra_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        logging.info("Results exported.")

# ----------------------------- GUI Interface -----------------------------

class DijkstraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dijkstra Visualizer")

        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.start_node = tk.StringVar()
        self.target_node = tk.StringVar()

        self.build_gui()

    def build_gui(self):
        """Create GUI layout and controls."""
        tk.Label(self.root, text="Truck config (JSON)").grid(row=0, column=0, sticky='w')
        tk.Entry(self.root, textvariable=self.input_file, width=50).grid(row=0, column=1)
        tk.Button(self.root, text="Browse", command=self.browse_input).grid(row=0, column=2)

        tk.Label(self.root, text="Output Directory").grid(row=1, column=0, sticky='w')
        tk.Entry(self.root, textvariable=self.output_dir, width=50).grid(row=1, column=1)
        tk.Button(self.root, text="Browse", command=self.browse_output).grid(row=1, column=2)

        tk.Label(self.root, text="Start Node").grid(row=2, column=0, sticky='w')
        tk.Entry(self.root, textvariable=self.start_node, width=50).grid(row=2, column=1)

        tk.Label(self.root, text="Target Node").grid(row=3, column=0, sticky='w')
        tk.Entry(self.root, textvariable=self.target_node, width=50).grid(row=3, column=1)

        tk.Button(self.root, text="RUN", command=self.run_dijkstra).grid(row=4, column=1, pady=10)

    def browse_input(self):
        file_path = filedialog.askopenfilename(filetypes=[["JSON Files", "*.json"]])
        if file_path:
            self.input_file.set(file_path)

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def run_dijkstra(self):
        try:
            input_path = self.input_file.get()
            output_path = self.output_dir.get()
            start = self.start_node.get()
            target = self.target_node.get()

            if not all([input_path, output_path, start, target]):
                messagebox.showerror("Input Error", "All input fields must be filled.")
                return

            configure_logging(os.path.join(output_path, 'dijkstra_log.txt'))

            graph = Graph.load_from_file(input_path)
            solver = DijkstraSolver(graph)
            distances, predecessors = solver.solve(start)
            path = solver.reconstruct_path(predecessors, target)
            total_distance = distances.get(target, float('inf'))

            GraphVisualizer.export(distances, predecessors, path, total_distance, output_path)
            self.root.after(0, lambda: GraphVisualizer.animate(graph, path))
            messagebox.showinfo("Success", "Execution completed successfully.")

        except Exception as e:
            logging.exception("Error in Dijkstra run")
            messagebox.showerror("Error", str(e))

# ----------------------------- Entry Point -----------------------------

if __name__ == '__main__':
    root = tk.Tk()
    app = DijkstraApp(root)
    root.mainloop()

