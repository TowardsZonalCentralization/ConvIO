import json
import os
import heapq
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DijkstraVisualizer:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.output_json = os.path.join(self.output_dir, 'dijkstra_results.json')

        os.makedirs(self.output_dir, exist_ok=True)
        self.graph_data = self.load_graph()
        self.pos_map = {}

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

    def visualize_graph(self, path: List[str], total_distance: float) -> None:
        """Visualize the node graph in 3D and highlight the shortest path."""
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')
            ax.set_zlabel('Z (cm)')

            self.pos_map = {node: tuple(coord) for node, coord in self.graph_data['coordinates'].items()}

            # Determine uniform scale
            all_coords = list(self.pos_map.values())
            x_vals, y_vals, z_vals = zip(*all_coords)
            max_range = max(
                max(x_vals) - min(x_vals),
                max(y_vals) - min(y_vals),
                max(z_vals) - min(z_vals)
            ) / 2.0
            mid_x = (max(x_vals) + min(x_vals)) * 0.5
            mid_y = (max(y_vals) + min(y_vals)) * 0.5
            mid_z = (max(z_vals) + min(z_vals)) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            # Draw all nodes
            for node, (x, y, z) in self.pos_map.items():
                ax.scatter(x, y, z, color='blue', s=40)
                ax.text(x + 0.5, y + 0.5, z + 0.5, node, fontsize=8)

            # Draw all edges
            for node, edges in self.graph_data['edges'].items():
                for neighbor, _ in edges:
                    x_vals = [self.pos_map[node][0], self.pos_map[neighbor][0]]
                    y_vals = [self.pos_map[node][1], self.pos_map[neighbor][1]]
                    z_vals = [self.pos_map[node][2], self.pos_map[neighbor][2]]
                    ax.plot(x_vals, y_vals, z_vals, color='gray', linewidth=0.8)

            # Highlight shortest path
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                x_vals = [self.pos_map[a][0], self.pos_map[b][0]]
                y_vals = [self.pos_map[a][1], self.pos_map[b][1]]
                z_vals = [self.pos_map[a][2], self.pos_map[b][2]]
                ax.plot(x_vals, y_vals, z_vals, color='red', linewidth=3)

            # Annotate total distance
            midpoint = self.pos_map[path[len(path) // 2]]
            ax.text(midpoint[0], midpoint[1], midpoint[2] + 10, f"Distance: {total_distance:.2f} cm", fontsize=10, color='green')

            plt.title('Truck Chassis Node Graph with Shortest Path', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'graph_visualization.png'))
            plt.close()
            logging.info("Graph visualization saved.")
        except Exception as e:
            logging.error("Error during graph visualization: %s", e)
            raise

    def export_results(self, distances: Dict[str, float], predecessors: Dict[str, str], path: List[str], total_distance: float) -> None:
        """Export Dijkstra results to a JSON file."""
        try:
            results = {
                "distances": distances,
                "predecessors": predecessors,
                "shortest_path": path,
                "total_distance_cm": total_distance
            }
            with open(self.output_json, 'w') as f:
                json.dump(results, f, indent=4)
            logging.info("Results exported to %s", self.output_json)
        except Exception as e:
            logging.error("Failed to export results: %s", e)
            raise

    def run(self):
        """Execute the full pipeline."""
        try:
            start_node = self.graph_data['start']
            target_node = self.graph_data['target']

            distances, predecessors = self.dijkstra(start_node)
            path = self.construct_path(predecessors, target_node)
            total_distance = distances[target_node]

            self.visualize_graph(path, total_distance)
            self.export_results(distances, predecessors, path, total_distance)
        except Exception as e:
            logging.critical("Execution failed: %s", e)


if __name__ == '__main__':
    INPUT_FILE = 'dijkstra/node_graphs/example_nodegraph.json'
    OUTPUT_DIR = 'dijkstra/results'

    visualizer = DijkstraVisualizer(INPUT_FILE, OUTPUT_DIR)
    visualizer.run()
