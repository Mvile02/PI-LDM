import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_trajectories(X_filepath, meta_filepath, num_samples=3):
    # Load the tensors and metadata
    try:
        X = np.load(X_filepath)
        meta = pd.read_csv(meta_filepath)
        print(f"Loaded tensor of shape: {X.shape}")
    except FileNotFoundError:
        print(f"Could not find the data files at: {X_filepath} or {meta_filepath}")
        return

    print("Press 'n' in the plot window to see a new set of random trajectories.")
    
    # Create the figure once
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('PI-LDM Input Format: Landing Kinematics over 200 Resampled Points', fontsize=16)

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
            
            callsign = meta.iloc[idx]['callsign'] if 'callsign' in meta.columns else f"Sample {idx}"
            ac_type = meta.iloc[idx]['typecode'] if 'typecode' in meta.columns else "Unknown"
            label = f"{callsign} ({ac_type})"
            
            # Subplot 0: Altitude Profile
            axes[0].plot(x_axis, altitude, label=label, color=c, linewidth=2)
            axes[0].set_ylabel('Altitude (ft)')
            axes[0].grid(True, linestyle='--', alpha=0.7)
            axes[0].legend()
            
            # Subplot 1: Groundspeed Profile
            axes[1].plot(x_axis, groundspeed, color=c, linewidth=2)
            axes[1].set_ylabel('Groundspeed (kts)')
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            # Subplot 2: Track Angle
            axes[2].plot(x_axis, track, color=c, linewidth=2)
            axes[2].set_ylabel('Track Angle (deg)')
            axes[2].set_xlabel('Resampled Waypoint Index (N=0 to 199)')
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
    # The base name used in filter_data.py (without extension)
    FILE_BASE = "sample_trajectory3"
    # --------------------------

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X_file = os.path.join(base_dir, "data", "processed", f"{FILE_BASE}.npy")
    meta_file = os.path.join(base_dir, "data", "processed", f"{FILE_BASE}.csv")

    if not os.path.exists(X_file):
        # Fallback to outputs/trajectories
        X_file = os.path.join(base_dir, "outputs", "trajectories", f"{FILE_BASE}.npy")
        meta_file = os.path.join(base_dir, "outputs", "trajectories", f"{FILE_BASE}.csv")
    
    # Fallback for demo if the above doesn't exist yet
    if not os.path.exists(X_file):
        print("Such file does not exist")

    visualize_trajectories(X_file, meta_file, num_samples=3)

