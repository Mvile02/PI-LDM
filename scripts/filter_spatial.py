import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from traffic.data import airports

def compute_positions_meters(X):
    """
    Computes local (x, y) coordinates in meters for trajectories using Dead Reckoning backwards
    from the runway anchor.
    X shape: (N, 4, 200) -> [track, groundspeed, altitude, elapsed_time]
    """
    N, features, points = X.shape
    
    X_m = np.zeros((N, points))
    Y_m = np.zeros((N, points))
    
    for i in range(N):
        track_deg = X[i, 0, :]
        gs_kts = X[i, 1, :]
        altitude = X[i, 2, :]
        elapsed_time = X[i, 3, :]
        
        # Convert gs to m/s
        gs_ms = gs_kts * 0.514444
        track_rad = np.radians(track_deg)
        dt = np.diff(elapsed_time)
        v_avg = (gs_ms[:-1] + gs_ms[1:]) / 2.0
        
        t1 = track_rad[:-1]
        t2 = track_rad[1:]
        diff = (t2 - t1 + np.pi) % (2*np.pi) - np.pi
        t_avg = t1 + diff/2.0
        
        dx = v_avg * np.sin(t_avg) * dt
        dy = v_avg * np.cos(t_avg) * dt
        
        x_m = np.zeros(points)
        y_m = np.zeros(points)
        
        # Determine landing vs departure (mostly landings here)
        start_alt = np.mean(altitude[:10])
        end_alt = np.mean(altitude[-10:])
        is_departure = end_alt > start_alt
        
        if is_departure:
            x_m[1:] = np.cumsum(dx)
            y_m[1:] = np.cumsum(dy)
        else:
            x_m[:-1] = -np.cumsum(dx[::-1])[::-1]
            y_m[:-1] = -np.cumsum(dy[::-1])[::-1]
            
        X_m[i, :] = x_m
        Y_m[i, :] = y_m
        
    return X_m, Y_m

def main():
    FILE_BASE = "LSZH_2019_R14_kinematic_200pts"
    AIRPORT_CODE = "LSZH"
    
    # Configuration
    N_CLUSTERS = 4
    
    # Threshold for deviation from the cluster center
    THRESHOLD_MEAN_M = 5000  
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X_file = os.path.join(base_dir, "data", "processed", f"{FILE_BASE}.npy")
    meta_file = os.path.join(base_dir, "data", "processed", f"{FILE_BASE}.csv")
    
    if not os.path.exists(X_file):
        print(f"Error: Dataset {X_file} not found. Please run dataset_builder.py first.")
        return
        
    print(f"Loading {X_file}...")
    X = np.load(X_file, allow_pickle=True).astype(np.float32)
    meta = pd.read_csv(meta_file) if os.path.exists(meta_file) else None
    
    print("Reconstructing spatial arrays in local meters...")
    X_m, Y_m = compute_positions_meters(X)
    
    # Flatten strictly corresponding points: (N, 200) + (N, 200) -> (N, 400)
    spatial_tensor = np.concatenate([X_m, Y_m], axis=1)
    
    print(f"Fitting KMeans (k={N_CLUSTERS}) purely on spatial paths to find the {N_CLUSTERS} golden routes...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(spatial_tensor)
    centers = kmeans.cluster_centers_  # shape (4, 400)
    
    print(f"Filtering trajectories with mean deviation > {THRESHOLD_MEAN_M}m...")
    distances = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        assigned_center = centers[labels[i]]
        
        # Point-wise Euclidean distance in meters along the path sequence
        x_center = assigned_center[:200]
        y_center = assigned_center[200:]
        
        dx = X_m[i] - x_center
        dy = Y_m[i] - y_center
        point_dists = np.sqrt(dx**2 + dy**2)
        
        # Penalize flights that consistently drive far off the center
        distances[i] = np.mean(point_dists)
        
    valid_mask = distances <= THRESHOLD_MEAN_M
    X_clean = X[valid_mask]
    labels_clean = labels[valid_mask]
    
    print("\n====================")
    print("Filtering complete:")
    print(f"Initial flights: {X.shape[0]}")
    print(f"Retained flights: {X_clean.shape[0]} ({(X_clean.shape[0]/X.shape[0]*100):.1f}%)")
    print(f"Discarded (Messy): {np.sum(~valid_mask)}")
    print("====================\n")
    
    # Save the cleaned dataset
    out_npy = os.path.join(base_dir, "data", "processed", f"{FILE_BASE}_spatial_{THRESHOLD_MEAN_M}m.npy")
    np.save(out_npy, X_clean)
    
    if meta is not None:
        meta_clean = meta.iloc[valid_mask].reset_index(drop=True)
        out_csv = os.path.join(base_dir, "data", "processed", f"{FILE_BASE}_spatial_{THRESHOLD_MEAN_M}m.csv")
        meta_clean.to_csv(out_csv, index=False)
        print(f"Cleaned dataset saved to: \n  {out_npy}\n  {out_csv}\n")
        
    # Save individual clusters separately
    print("Saving individual cluster patterns...")
    for k in range(N_CLUSTERS):
        mask_k = labels_clean == k
        count_k = np.sum(mask_k)
        if count_k > 0:
            X_k = X_clean[mask_k]
            out_npy_k = os.path.join(base_dir, "data", "clusters", f"{FILE_BASE}_spatial_{THRESHOLD_MEAN_M}m_c{k}.npy")
            np.save(out_npy_k, X_k)
            if meta is not None:
                meta_k = meta_clean.iloc[mask_k].reset_index(drop=True)
                out_csv_k = os.path.join(base_dir, "data", "clusters", f"{FILE_BASE}_spatial_{THRESHOLD_MEAN_M}m_c{k}.csv")
                meta_k.to_csv(out_csv_k, index=False)
            print(f"  -> Cluster {k}: {count_k} flights saved to ..._spatial_{THRESHOLD_MEAN_M}m_c{k}.npy")
    print()

    
    # ==== Plotting Verification ====
    print("Generating verification plot...")
    airport = airports[AIRPORT_CODE]
    anchor_lat, anchor_lon = airport.latitude, airport.longitude
    
    rwys = [r for r in airport.runways.list if "14" in r.name]
    if rwys:
        anchor_lat, anchor_lon = rwys[0].latitude, rwys[0].longitude
        
    m_per_deg_lat = 111320.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(anchor_lat))
    
    Lons_clean = anchor_lon + (X_m[valid_mask] / m_per_deg_lon)
    Lats_clean = anchor_lat + (Y_m[valid_mask] / m_per_deg_lat)
    
    plt.figure(figsize=(10, 10))
    plt.title(f'Tight STAR Filtering for {AIRPORT_CODE} (N={X_clean.shape[0]}) | Threshold: <{THRESHOLD_MEAN_M}m')
    
    # Vibrant colors dynamically assigned
    cmap = plt.cm.get_cmap('tab10', max(10, N_CLUSTERS))
    
    # Plot flights
    for k in range(N_CLUSTERS):
        idx = np.where(labels_clean == k)[0]
        # Only label one of each so legend isn't huge
        labeled = False
        for i in idx:
            plt.plot(Lons_clean[i], Lats_clean[i], color=cmap(k), alpha=0.08, linewidth=1,
                     label=f"Cluster {k} Trajectories" if not labeled else None)
            labeled = True
            
    # Plot bold red centerlines
    for k in range(N_CLUSTERS):
        Lons_c = anchor_lon + (centers[k, :200] / m_per_deg_lon)
        Lats_c = anchor_lat + (centers[k, 200:] / m_per_deg_lat)
        plt.plot(Lons_c, Lats_c, color='red', linewidth=3, zorder=5, 
                 label='Golden Centerline' if k==0 else None)
        
    plt.scatter(anchor_lon, anchor_lat, color='red', marker='x', s=100, zorder=6, label='RWY 14')
    
    plt.gca().set_aspect(1.0 / np.cos(np.radians(anchor_lat)))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_alpha(1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Output path
    out_dir = os.path.join(base_dir, "outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_file = os.path.join(out_dir, f"{FILE_BASE}_SPATIAL_{THRESHOLD_MEAN_M}m.png")
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    print(f"Verification plot saved to:\n  {plot_file}")
    #plt.show()

if __name__ == '__main__':
    main()
