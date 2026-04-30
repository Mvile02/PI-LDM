import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from traffic.data import airports

def compute_positions(X, meta, anchor_lat, anchor_lon):
    """
    Computes (lat, lon) for trajectories using Dead Reckoning backwards
    from the anchor point.
    X shape: (N, 4, 200) -> [track, groundspeed, altitude, elapsed_time]
    """
    N, features, points = X.shape
    
    # Pre-allocate arrays for Lat, Lon
    Lats = np.zeros((N, points))
    Lons = np.zeros((N, points))
    is_deps = np.zeros(N, dtype=bool)
    
    # 1 degree of latitude in meters approx
    m_per_deg_lat = 111320.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(anchor_lat))
    
    for i in range(N):
        track_deg = X[i, 0, :]
        gs_kts = X[i, 1, :]
        altitude = X[i, 2, :]
        elapsed_time = X[i, 3, :]
        
        # Convert gs to m/s (1 knot = 0.514444 m/s)
        gs_ms = gs_kts * 0.514444
        
        # Convert track to radians
        track_rad = np.radians(track_deg)
        
        # Calculate step increments (vectorized)
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
        
        # Detect if it's a departure or a landing based on altitude change
        start_alt = np.mean(altitude[:10])
        end_alt = np.mean(altitude[-10:])
        is_departure = end_alt > start_alt
        is_deps[i] = is_departure
        
        if is_departure:
            # For departures, the first point is at the anchor (0,0). Integrate forwards.
            x_m[1:] = np.cumsum(dx)
            y_m[1:] = np.cumsum(dy)
        else:
            # For landings, the last point is at the anchor (0,0). Integrate backwards.
            x_m[:-1] = -np.cumsum(dx[::-1])[::-1]
            y_m[:-1] = -np.cumsum(dy[::-1])[::-1]
            
        # Convert meters to lat/lon offsets
        Lons[i, :] = anchor_lon + (x_m / m_per_deg_lon)
        Lats[i, :] = anchor_lat + (y_m / m_per_deg_lat)
        
    return Lats, Lons, is_deps

def main():
    # --- CONFIGURATION AREA ---
    FILE_BASE = "LSZH_2019_R14_kinematic_200pts_spatial_5000m_c1"
    #FILE_BASE = "LSZH_2019_R14_kinematic_200pts_clust5_C2"
    AIRPORT_CODE = "LSZH"
    PLOT_MAP_BACKGROUND = False  # Set to True to overlay geographic map tiles (requires contextily)
    # --------------------------
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define possible locations for the .npy file
    search_paths = [
        os.path.join(base_dir, "data", "processed", f"{FILE_BASE}.npy"),
        os.path.join(base_dir, "data", "clusters", f"{FILE_BASE}.npy"),
        os.path.join(base_dir, "outputs", "trajectories", f"{FILE_BASE}.npy"),
        os.path.join(base_dir, "pi_ldm", "outputs", "trajectories", f"{FILE_BASE}.npy")
    ]
    
    X_file_npy = None
    for path in search_paths:
        if os.path.exists(path):
            X_file_npy = path
            break
            
    if X_file_npy is None:
        print(f"Error: {FILE_BASE}.npy not found in any standard directories.")
        return
        
    print(f"Loading data from: {X_file_npy}")
    X = np.load(X_file_npy, allow_pickle=True).astype(np.float32)
    
    labels = None # We are loading pure arrays now, not npz label structures
    
    # Attempt to locate corresponding metadata .csv in the exact same folder
    meta_file = X_file_npy.replace('.npy', '.csv')
    
    # Metadata is optional
    meta = None
    if os.path.exists(meta_file):
        meta = pd.read_csv(meta_file)
    else:
        print(f"Warning: Metadata .csv not found for {FILE_BASE}. Proceeding with kinematic analysis only.")
    
    # Get anchor point
    airport = airports[AIRPORT_CODE]
    if airport is None:
        print(f"Airport {AIRPORT_CODE} not found in database.")
        return
        
    anchor_lat, anchor_lon = airport.latitude, airport.longitude
    
    # Refine anchor: Default to Runway 14 if no specific runway is in filename
    mode = FILE_BASE.split("_")[-1]  # e.g., "landings" or "runway14"
    if "runway" not in mode.lower() or "sample" in FILE_BASE:
        mode = "runway14"  # Default fallback for samples
        print("Forced Runway 14 anchoring for sample/synthetic data.")
        
    anchor_name = "ARP"
    rwy_end_lat = None
    rwy_end_lon = None
    
    if "runway" in mode.lower():
        rwy_str = mode.replace("runway", "").upper()
        # Clean string to get purely the digits for reciprocal calculation
        rwy_digits = ''.join(filter(str.isdigit, rwy_str))
        
        # Find runway in airport's runways list
        rwys = [r for r in airport.runways.list if r.name == rwy_str]
        if rwys:
            anchor_lat, anchor_lon = rwys[0].latitude, rwys[0].longitude
            anchor_name = f"Runway {rwy_str}"
            print(f"Found {anchor_name} coordinates.")
            
            # Find opposite end
            if rwy_digits:
                recip = str((int(rwy_digits) + 18) % 36).zfill(2)
                rwy2 = [r for r in airport.runways.list if recip in r.name]
                if rwy2:
                    rwy_end_lat, rwy_end_lon = rwy2[0].latitude, rwy2[0].longitude
                    print(f"Found opposite end Runway {recip} coordinates to draw runway strip.")
        else:
            print(f"Runway {rwy_str} not found in database for {AIRPORT_CODE}. Falling back to ARP.")
            
    print(f"Anchoring trajectories to {AIRPORT_CODE} {anchor_name}: ({anchor_lat:.4f}, {anchor_lon:.4f})")
    
    Lats, Lons, is_deps = compute_positions(X, meta, anchor_lat, anchor_lon)
    
    # Plotting
    plt.figure(figsize=(10, 10))
    plt.title(f'Integrated Processed Trajectories for {AIRPORT_CODE} (N={len(X)})')
    
    # Plot flights
    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap('Set1', len(unique_labels))
        # Create a legend proxy dict
        legend_handles = {}
        for i in range(len(X)):
            cluster_id = labels[i]
            color = cmap(cluster_id)
            line, = plt.plot(Lons[i, :], Lats[i, :], color=color, alpha=0.3, linewidth=1.0)
            if cluster_id not in legend_handles:
                legend_handles[cluster_id] = line
                
        # Manually assign legends for clusters
        for cluster_id, line in sorted(legend_handles.items()):
            line.set_label(f'Cluster {cluster_id}')
    else:
        has_dep = False
        has_arr = False
        for i in range(len(X)):
            if is_deps[i]:
                color = 'red'
                label = 'Departure' if not has_dep else None
                has_dep = True
            else:
                color = 'blue'
                label = 'Landing' if not has_arr else None
                has_arr = True
                
            plt.plot(Lons[i, :], Lats[i, :], color=color, alpha=0.3, linewidth=1.0, label=label)
        
    # Plot Airport/Runway Marker
    plt.scatter(anchor_lon, anchor_lat, color='red', marker='x', s=100, label=f'{AIRPORT_CODE} {anchor_name}', zorder=5)
    
    # Plot real physical runway strip if possible
    if rwy_end_lat and rwy_end_lon:
        plt.plot([anchor_lon, rwy_end_lon], [anchor_lat, rwy_end_lat], 
                 color='magenta', linewidth=4, zorder=4, label='Physical Runway Strip')
    
    # Format map limits
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Geographic Map overlay
    if PLOT_MAP_BACKGROUND:
        try:
            import contextily as ctx
            print("Downloading map tiles and adding to plot...")
            # Use High-Resolution Satellite map for much better runway visibility
            ctx.add_basemap(plt.gca(), crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery)
        except ImportError:
            print("Warning: 'contextily' is not installed. Run 'pip install contextily' to see the map background.")
            plt.gca().set_aspect(1.0 / np.cos(np.radians(anchor_lat)))
    else:
        # Set aspect ratio suitable for given latitude
        plt.gca().set_aspect(1.0 / np.cos(np.radians(anchor_lat)))
    
    # Output path
    output_dir = os.path.join(base_dir, "outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    out_map = os.path.join(output_dir, f'map_{FILE_BASE}.png')
    
    plt.tight_layout()
    plt.savefig(out_map, dpi=200)
    print(f"Saved computed trajectory map to {out_map}")
    
    # Also save to pi_ldm/outputs/plots if appropriate
    pildm_dir = os.path.join(base_dir, "pi_ldm")
    if os.path.exists(pildm_dir):
        pildm_plot_dir = os.path.join(pildm_dir, "outputs", "plots")
        os.makedirs(pildm_plot_dir, exist_ok=True)
        pildm_out_map = os.path.join(pildm_plot_dir, f'map_{FILE_BASE}.png')
        plt.savefig(pildm_out_map, dpi=200)
        print(f"Also saved plot to {pildm_out_map}")

    plt.show()

if __name__ == "__main__":
    main()
