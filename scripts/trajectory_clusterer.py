"""
Description:
    Clusters kinematic aircraft trajectories into operational modes using TimeSeriesKMeans (DTW).

Author:
    Gerard Martínez Vilella
    April 16, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
import os

class TrajectoryClusterer:
    def __init__(self, data_path):
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset {self.data_path} not found.")
            
        # Load dataset of shape (N, features, seq_len)
        self.raw_tensor = np.load(self.data_path, allow_pickle=True)
        # Transpose to (N, seq_len, features) for tslearn compatibility
        self.tslearn_tensor = np.transpose(self.raw_tensor, (0, 2, 1))
        
    def perform_clustering(self, num_clusters, use_features_idx=(0, 1)):
        """
        Executes DTW KMeans on specified features (e.g. 0=track, 1=groundspeed).
        """
        print(f"Clustering {self.raw_tensor.shape[0]} trajectories into {num_clusters} groups...")
        target_data = self.tslearn_tensor[:, :, use_features_idx]
        
        kmeans = TimeSeriesKMeans(
            n_clusters=num_clusters, 
            metric="dtw", 
            max_iter=15, 
            n_init=10, 
            dtw_inertia=True
        )
        
        kmeans.fit(target_data)
        return kmeans.labels_, kmeans.cluster_centers_

    def generate_elbow_plot(self, max_clusters=10, use_features_idx=(0, 1, 2)):
        """
        Plots DTW inertia to visually determine the optimal number of clusters.
        """
        inertias = []
        cluster_range = range(1, max_clusters + 1)
        target_data = self.tslearn_tensor[:, :, use_features_idx]

        print("Calculating DTW inertias for Elbow plot...")
        for k in cluster_range:
            kmeans = TimeSeriesKMeans(
                n_clusters=k, 
                metric="dtw", 
                max_iter=5, 
                n_init=2, 
                dtw_inertia=True
            )
            kmeans.fit(target_data)
            inertias.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(cluster_range, inertias, marker='o', linestyle='--', color='b')
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('DTW Inertia', fontsize=12)
        plt.title('DTW KMeans Elbow Method Analysis', fontsize=14)
        plt.grid(True, alpha=0.5)
        plt.show()


if __name__ == '__main__':
    # Initialize the clusterer with our kinematic dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    input_base = "lszh_2019_runway14_kinematic_200pts"
    dataset_file = os.path.join(processed_dir, f"{input_base}.npy")
    
    if not os.path.exists(dataset_file):
        print(f"Test dataset {dataset_file} missing. Please run dataset_builder.py first.")
    else:
        clusterer = TrajectoryClusterer(dataset_file)
        
        # We cluster based on track and groundspeed (indices 0 and 1)
        target_cluster_counts = [2, 3, 4, 5]
        
        for k in target_cluster_counts:
            labels, centers = clusterer.perform_clustering(num_clusters=k, use_features_idx=(0, 1))
            
            output_filename = os.path.join(processed_dir, f"{input_base}_clust{k}.npz")
            np.savez(output_filename, X=clusterer.raw_tensor, y=labels)
            print(f"-> Saved assigned clusters to '{output_filename}' (Labels shape: {labels.shape})\n")