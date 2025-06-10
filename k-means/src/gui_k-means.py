"""
K-means algorithm for clustering the I/Os and getting the optimal number and position of I/O Extenders, with a GUI for user interaction.
"""

import os
import logging
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import List, Optional

# ----------------------------- Logging Setup -----------------------------

def configure_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# ----------------------------- Clustering Logic -----------------------------

class KMeansClustering:
    def __init__(self, data: pd.DataFrame):
        self.original_data = data
        self.scaled_data = StandardScaler().fit_transform(data)
        self.labels: Optional[np.ndarray] = None
        self.centroids: Optional[np.ndarray] = None

    def run_elbow_method(self, max_k: int) -> List[float]:
        inertias = []
        for k in range(1, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(self.scaled_data)
            inertias.append(model.inertia_)
        return inertias

    def cluster(self, k: int) -> pd.DataFrame:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.labels = model.fit_predict(self.scaled_data)
        self.centroids = model.cluster_centers_

        label_map = {old_label: new_label + 1 for new_label, old_label in enumerate(np.argsort(self.centroids[:, 0]))}
        new_labels = np.array([label_map[label] for label in self.labels])
        new_centroids = np.array([self.centroids[old_label] for old_label in np.argsort(self.centroids[:, 0])])

        clustered_df = self.original_data.copy()
        clustered_df['Cluster'] = new_labels
        centroid_coords = [str(new_centroids[label - 1]) for label in new_labels]
        clustered_df['Centroid_Coords'] = centroid_coords

        return clustered_df.sort_values(by='Cluster'), new_centroids, new_labels

# ----------------------------- GUI Interface -----------------------------

class KMeansClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Clustering Tool")
        self.root.geometry("600x300")

        self.data: Optional[pd.DataFrame] = None
        self.clustering: Optional[KMeansClustering] = None

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.k_value_elbow = tk.StringVar(value="10")
        self.k_value_cluster = tk.StringVar(value="3")

        self.build_gui()

    def build_gui(self):
        padding = {'padx': 10, 'pady': 5}

        tk.Label(self.root, text="CSV File").grid(row=0, column=0, sticky='w', **padding)
        tk.Entry(self.root, textvariable=self.input_path, width=45).grid(row=0, column=1, **padding)
        tk.Button(self.root, text="Browse", command=self.browse_csv).grid(row=0, column=2, **padding)

        tk.Label(self.root, text="Save Output Folder").grid(row=1, column=0, sticky='w', **padding)
        tk.Entry(self.root, textvariable=self.output_path, width=45).grid(row=1, column=1, **padding)
        tk.Button(self.root, text="Browse", command=self.select_output_folder).grid(row=1, column=2, **padding)

        tk.Label(self.root, text="Max Clusters for Elbow").grid(row=2, column=0, sticky='w', **padding)
        tk.Entry(self.root, textvariable=self.k_value_elbow, width=10).grid(row=2, column=1, sticky='w', **padding)
        tk.Button(self.root, text="Run Elbow Method", command=self.run_elbow).grid(row=2, column=2, **padding)

        tk.Label(self.root, text="Clusters for Clustering").grid(row=3, column=0, sticky='w', **padding)
        tk.Entry(self.root, textvariable=self.k_value_cluster, width=10).grid(row=3, column=1, sticky='w', **padding)
        tk.Button(self.root, text="Run Clustering", command=self.run_clustering).grid(row=3, column=2, **padding)

        tk.Button(self.root, text="Exit", command=self.root.quit).grid(row=4, column=1, pady=20)

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.input_path.set(path)
            try:
                self.data = pd.read_csv(path)
                self.clustering = KMeansClustering(self.data)
                messagebox.showinfo("Success", "CSV loaded and scaled successfully.")
                logging.info("CSV loaded from %s", path)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                logging.error("Failed to load CSV: %s", e)

    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_path.set(folder)

    def run_elbow(self):
        if not self.clustering:
            messagebox.showwarning("No Data", "Load a CSV file first.")
            return
        try:
            max_k = int(self.k_value_elbow.get())
            inertias = self.clustering.run_elbow_method(max_k)
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, max_k + 1), inertias, marker='o')
            plt.title('Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.tight_layout()
            elbow_path = os.path.join(self.output_path.get(), "elbow_plot.png")
            plt.savefig(elbow_path)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logging.error("Elbow method failed: %s", e)

    def run_clustering(self):
        if not self.clustering:
            messagebox.showwarning("No Data", "Load a CSV file first.")
            return
        try:
            k = int(self.k_value_cluster.get())
            clustered_df, centroids, labels = self.clustering.cluster(k)

            sil_score = silhouette_score(self.clustering.scaled_data, labels)
            db_score = davies_bouldin_score(self.clustering.scaled_data, labels)

            if clustered_df.shape[1] > 3:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(self.clustering.scaled_data)
                centroids_2d = pca.transform(centroids)
            else:
                reduced = self.clustering.scaled_data
                centroids_2d = centroids

            if sil_score >= 0.5 and db_score <= 0.5:
                box_color = 'lightgreen'
            elif sil_score >= 0.3:
                box_color = 'khaki'
            else:
                box_color = 'lightcoral'

            plt.figure(figsize=(10, 6))
            plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=200, label='Centroids')

            for i, coord in enumerate(centroids_2d):
                plt.text(coord[0], coord[1], f'{i + 1}', fontsize=12, weight='bold', ha='center', va='center', color='white')

            metrics_text = f"Silhouette Score: {sil_score:.3f}\nDavies-Bouldin Score: {db_score:.3f}"
            plt.gca().text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, edgecolor='gray', alpha=0.8))

            plt.title(f'Cluster Visualization with k={k}')
            plt.legend()
            plt.tight_layout()
            cluster_plot_path = os.path.join(self.output_path.get(), "cluster_plot.png")
            plt.savefig(cluster_plot_path)
            plt.show()

            output_csv_path = os.path.join(self.output_path.get(), "clustered_data.csv")
            clustered_df.to_csv(output_csv_path, index=False)

            messagebox.showinfo("Success", f"Clustered data and plots saved to: {self.output_path.get()}")
            logging.info("Clustered data and plots saved in %s", self.output_path.get())

        except Exception as e:
            messagebox.showerror("Error", str(e))
            logging.error("Clustering failed: %s", e)

# ----------------------------- Entry Point -----------------------------

if __name__ == '__main__':
    root = tk.Tk()
    app = KMeansClusteringApp(root)
    root.mainloop()

