import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_trajectories(X_filepath, meta_filepath, num_samples=3, orig_dataset=None):
    # Load the tensors and metadata
    try:
        X = np.load(X_filepath, allow_pickle=True).astype(np.float32)
        print(f"Loaded tensor of shape: {X.shape}")
    except FileNotFoundError:
        print(f"Could not find the numpy data file at: {X_filepath}")
        return
        
    meta = None
    ac_types_mapping = None
    if os.path.exists(meta_filepath):
        meta = pd.read_csv(meta_filepath)
        # If typecode is numeric (synthetic data), try to load mapping from original dataset
        if 'typecode' in meta.columns and pd.api.types.is_numeric_dtype(meta['typecode']) and orig_dataset:
            base_dir_local = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for loc in ['processed', 'clusters']:
                orig_csv = os.path.join(base_dir_local, 'data', loc, f"{orig_dataset}.csv")
                if os.path.exists(orig_csv):
                    orig_df = pd.read_csv(orig_csv)
                    ac_types_mapping = sorted(orig_df['typecode'].astype(str).unique())
                    break

    print("Press 'n' in the plot window to see a new set of random trajectories.")
    
    # Create the figure once
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('Landing Kinematics over 200 Resampled Points', fontsize=18)

    def update_plot(event=None):
        if event is not None and event.key != 'n':
            return
        
        # Clear previous plots
        for ax in axes:
            ax.clear()
        
        # Choose random trajectories to visualize
        indices = np.random.choice(len(X), size=num_samples, replace=False)
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        for idx, c in zip(indices, colors[:num_samples]):
            track = X[idx, 0, :]
            groundspeed = X[idx, 1, :]
            altitude = X[idx, 2, :]
            x_axis = np.arange(200)
            
            callsign = meta.iloc[idx]['callsign'] if meta is not None and 'callsign' in meta.columns else f"Sample {idx}"
            ac_type_val = meta.iloc[idx]['typecode'] if meta is not None and 'typecode' in meta.columns else "Unknown"
            
            if ac_types_mapping is not None and isinstance(ac_type_val, (int, float, np.integer, np.floating)) and not pd.isna(ac_type_val):
                type_idx = int(ac_type_val)
                if 0 <= type_idx < len(ac_types_mapping):
                    ac_type_val = ac_types_mapping[type_idx]
                    
            label = f"{callsign} ({ac_type_val})"
            
            # Subplot 0: Altitude Profile
            axes[0].plot(x_axis, altitude, label=label, color=c, linewidth=2)
            axes[0].set_ylabel('Altitude (ft)', fontsize=16)
            axes[0].grid(True, linestyle='--', alpha=0.7)
            axes[0].legend()
            
            # Subplot 1: Groundspeed Profile
            axes[1].plot(x_axis, groundspeed, color=c, linewidth=2)
            axes[1].set_ylabel('Groundspeed (kts)', fontsize=16)
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            # Subplot 2: Track Angle
            axes[2].plot(x_axis, track, color=c, linewidth=2)
            axes[2].set_ylabel('Track Angle (deg)', fontsize=16)
            axes[2].set_xlabel('Resampled Waypoint Index', fontsize=16)
            axes[2].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save visualization to outputs/plots folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "outputs", "plots")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'trajectory_visualization.png')
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
        
        # Redraw
        fig.canvas.draw_idle()

    # Connect the event
    fig.canvas.mpl_connect('key_press_event', update_plot)
    
    # Initial draw
    update_plot()
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURATION AREA ---
    # The base name of the trajectories you want to plot (without extension)
    FILE_BASE = "LSZH_2019_R14_kinematic_200pts_spatial_5000m"
    
    # The base name of the original training dataset (to recover aircraft type names)
    ORIGINAL_DATASET_BASE = "LSZH_2019_R14_kinematic_200pts_spatial_5000m"
    # --------------------------

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define possible locations for the .npy file
    search_paths = [
        os.path.join(base_dir, "data", "processed", f"{FILE_BASE}.npy"),
        os.path.join(base_dir, "data", "clusters", f"{FILE_BASE}.npy"),
        os.path.join(base_dir, "outputs", "trajectories", f"{FILE_BASE}.npy"),
        os.path.join(base_dir, "pi_ldm", "outputs", "trajectories", f"{FILE_BASE}.npy")
    ]
    
    X_file = None
    for path in search_paths:
        if os.path.exists(path):
            X_file = path
            break
            
    if X_file is None:
        print(f"Error: {FILE_BASE}.npy not found in any standard directories.")
        
    meta_file = X_file.replace('.npy', '.csv')

    visualize_trajectories(X_file, meta_file, num_samples=3, orig_dataset=ORIGINAL_DATASET_BASE)

