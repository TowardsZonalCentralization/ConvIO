import json
import os
import heapq
import logging
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DijkstraVisualizer:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.graph_data = self.load_graph()

    def load_graph(self) -> Dict:
        """Load the graph data from a JSON file."""
        try:
            with open(self.input_file, 'r') as f:
                data = json.load(f)
            logging.info("Graph data successfully loaded from %s", self.input_file)
            return data
        except FileNotFoundError:
            logging.error("Input file not found: %s", self.input_file)
            raise
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from file: %s", self.input_file)
            raise

    def dijkstra(self, start_node: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Compute shortest paths using Dijkstra's algorithm."""
        distances = {node: float('inf') for node in self.graph_data['nodes']}
        predecessors = {node: None for node in self.graph_data['nodes']}
        distances[start_node] = 0
        pq = [(0, start_node)]

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            for neighbor, weight in self.graph_data['edges'].get(current_node, []):
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

        return distances, predecessors

    def construct_path(self, predecessors: Dict[str, str], target: str) -> List[str]:
        """Construct the shortest path from the start node to the target."""
        path = []
        while target:
            path.insert(0, target)
            target = predecessors[target]
        return path

    def visualize_graph(self, path: List[str], total_distance: float):
        """Visualize the node graph in 3D and highlight the shortest path."""
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')
            ax.set_zlabel('Z (cm)')

            # Adjust scale
            ax.set_box_aspect([1, 1, 1])

            pos_map = {node: tuple(coord) for node, coord in self.graph_data['coordinates'].items()}

            for node, (x, y, z) in pos_map.items():
                ax.scatter(x, y, z, color='blue', s=40)
                ax.text(x + 0.5, y + 0.5, z + 0.5, node, fontsize=8)

            for node, edges in self.graph_data['edges'].items():
                for neighbor, _ in edges:
                    x_vals = [pos_map[node][0], pos_map[neighbor][0]]
                    y_vals = [pos_map[node][1], pos_map[neighbor][1]]
                    z_vals = [pos_map[node][2], pos_map[neighbor][2]]
                    ax.plot(x_vals, y_vals, z_vals, color='gray', linewidth=0.8)

            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                x_vals = [pos_map[a][0], pos_map[b][0]]
                y_vals = [pos_map[a][1], pos_map[b][1]]
                z_vals = [pos_map[a][2], pos_map[b][2]]
                ax.plot(x_vals, y_vals, z_vals, color='red', linewidth=3)

            midpoint = pos_map[path[len(path)//2]]
            ax.text(midpoint[0], midpoint[1], midpoint[2] + 10, f"Distance: {total_distance:.2f} cm", fontsize=10, color='green')

            plt.title('Truck Chassis Node Graph with Shortest Path', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'graph_visualization.png'))
            plt.close()
            logging.info("Graph visualization saved.")
        except Exception as e:
            logging.error("Error during graph visualization: %s", e)
            raise

    def export_results(self, distances: Dict[str, float], predecessors: Dict[str, str], shortest_path: List[str], total_distance: float):
        """Export Dijkstra results to a JSON file."""
        try:
            results = {
                "distances": distances,
                "predecessors": predecessors,
                "shortest_path": shortest_path,
                "total_distance_cm": total_distance
            }
            output_path = os.path.join(self.output_dir, 'dijkstra_results.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            logging.info("Results exported to %s", output_path)
        except Exception as e:
            logging.error("Failed to export results: %s", e)
            raise

    def run(self):
        try:
            start_node = self.graph_data['start']
            target_node = self.graph_data['target']

            distances, predecessors = self.dijkstra(start_node)
            shortest_path = self.construct_path(predecessors, target_node)
            total_distance = distances[target_node]

            self.visualize_graph(shortest_path, total_distance)
            self.export_results(distances, predecessors, shortest_path, total_distance)

            messagebox.showinfo("Success", "Dijkstra algorithm executed successfully. Results saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Execution failed: {str(e)}")


class DijkstraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dijkstra Visualizer")

        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()

        tk.Label(root, text="Which configuration of the truck do you want to optimize?(Hint- Input a json file of the respective truck)").grid(row=0, column=0, sticky='w')
        tk.Entry(root, textvariable=self.input_file, width=50).grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_input).grid(row=0, column=2)

        tk.Label(root, text="Output Directory").grid(row=1, column=0, sticky='w')
        tk.Entry(root, textvariable=self.output_dir, width=50).grid(row=1, column=1)
        tk.Button(root, text="Browse", command=self.browse_output).grid(row=1, column=2)

        tk.Button(root, text="Run Dijkstra", command=self.run_dijkstra).grid(row=2, column=1, pady=10)

    def browse_input(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            self.input_file.set(file_path)

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def run_dijkstra(self):
        input_path = self.input_file.get()
        output_path = self.output_dir.get()

        if not input_path or not output_path:
            messagebox.showerror("Input Error", "Both input file and output directory must be selected.")
            return

        visualizer = DijkstraVisualizer(input_path, output_path)
        visualizer.run()


if __name__ == '__main__':
    root = tk.Tk()
    app = DijkstraApp(root)
    root.mainloop()
