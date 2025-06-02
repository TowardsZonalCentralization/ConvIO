import os
import logging
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KMeansClusteringApp:
    def __init__(self, master):
        self.master = master
        master.title("K-Means Clustering Tool")
        master.geometry("400x300")

        self.data = None
        self.scaled_data = None
        self.labels = None

        # GUI elements
        tk.Label(master, text="K-Means Clustering GUI", font=('Helvetica', 14, 'bold')).pack(pady=10)

        tk.Button(master, text="Load CSV", command=self.load_csv).pack(pady=5)
        tk.Label(master, text="Max Clusters (K):").pack()
        self.k_entry = tk.Entry(master)
        self.k_entry.insert(0, "10")
        self.k_entry.pack(pady=5)

        tk.Button(master, text="Run Elbow Method", command=self.run_elbow).pack(pady=5)
        tk.Button(master, text="Cluster Data", command=self.cluster_data).pack(pady=5)
        tk.Button(master, text="Exit", command=master.quit).pack(pady=10)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            try:
                self.data = pd.read_csv(path)
                scaler = StandardScaler()
                self.scaled_data = scaler.fit_transform(self.data)
                messagebox.showinfo("Success", "Data loaded and scaled successfully.")
                logging.info("CSV loaded from %s", path)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                logging.error("Failed to load CSV: %s", e)

    def run_elbow(self):
        if self.scaled_data is None:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return

        try:
            max_k = int(self.k_entry.get())
            inertias = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(self.scaled_data)
                inertias.append(kmeans.inertia_)

            plt.figure(figsize=(8, 5))
            plt.plot(range(1, max_k + 1), inertias, marker='o')
            plt.title('Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.tight_layout()
            plt.show()

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for max K.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logging.error("Elbow method failed: %s", e)

    def cluster_data(self):
        if self.scaled_data is None:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return

        try:
            k = int(self.k_entry.get())
            kmeans = KMeans(n_clusters=k, random_state=42)
            self.labels = kmeans.fit_predict(self.scaled_data)
            original_df = self.data.copy()
            original_df['Cluster'] = self.labels

            # Plot original data with clusters
            plt.figure(figsize=(8, 5))
            for cluster_id in np.unique(self.labels):
                cluster = original_df[original_df['Cluster'] == cluster_id]
                plt.scatter(cluster.iloc[:, 0], cluster.iloc[:, 1], label=f'Cluster {cluster_id}')

            centroids = original_df.groupby('Cluster').mean().values
            plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5, label='Centroids')
            plt.title('K-Means Clustering')
            plt.xlabel(self.data.columns[0])
            plt.ylabel(self.data.columns[1])
            plt.legend()
            plt.tight_layout()
            plt.show()

            messagebox.showinfo("Success", f"Clustering completed with K={k}.")
            logging.info("KMeans clustering completed with K=%d", k)

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for clusters.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logging.error("Clustering failed: %s", e)

if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansClusteringApp(root)
    root.mainloop()
