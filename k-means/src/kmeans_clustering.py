import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import matplotlib

# For headless environments (no GUI)
matplotlib.use('Agg')


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least two columns for clustering.")
    return df.iloc[:, :2]  # Use only the first two columns


def elbow_method(data, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias


def plot_elbow(inertias, max_k=10, output_path='k-means/results/elbow_plot.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Elbow plot saved to {output_path}")
    plt.close()


def perform_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data = data.copy()
    data['Cluster'] = kmeans.fit_predict(data)
    return data, kmeans.cluster_centers_


def plot_clusters(data, centroids, output_path='k-means/results/cluster_plot.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    for cluster_id in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster_id]
        plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f"Cluster {cluster_id}")
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.6, label='Centroids')
    plt.title('K-Means Clustering Result')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Cluster plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    try:
        data_path = 'k-means/data/random_coordinates.csv'  # Path to uploaded file
        df = load_data(data_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")

        inertias = elbow_method(df)
        plot_elbow(inertias)

        optimal_k = 4  # Adjust after reviewing elbow_plot.png
        clustered_data, centroids = perform_kmeans(df, optimal_k)

        os.makedirs('results', exist_ok=True)
        clustered_data.to_csv('k-means/results/clustered_data.csv', index=False)
        print("Clustered data saved to k-means/results/clustered_data.csv")

        plot_clusters(clustered_data, centroids)

    except Exception as e:
        print(f"[ERROR] {e}")
