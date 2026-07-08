import os
import argparse
import numpy as np
from scipy.signal import savgol_filter, medfilt
import matplotlib.pyplot as plt

def smooth_trajectories(input_filepath=None, output_filepath=None, window_length=15, polyorder=3, median_window=5):
    """
    Applies a Median filter followed by a Savitzky-Golay filter to smooth 
    the kinematic trajectories, ensuring they are derivable and removing noise, 
    kinks, or outliers that cause loops.
    
    Args:
        input_filepath: Path to the input .npy file
        output_filepath: Path to save the smoothed .npy file. If None, saves next to input with _smoothed suffix.
        window_length: Length of the Savitzky-Golay filter window (must be odd).
        polyorder: Order of the polynomial used to fit the samples.
        median_window: Window size for the median filter (must be odd). Set to 0 to disable.
    """
    if input_filepath is None:
        input_filepath = r"pi_ldm\outputs\trajectories\LSZH_2019_R14_kinematic_200pts_spatial_5000m_cond_synthetic_1000t.npy"

    print(f"Loading data from {input_filepath}...")
    try:
        X = np.load(input_filepath, allow_pickle=True).astype(np.float32)
        print(f"Loaded tensor of shape: {X.shape}")
    except FileNotFoundError:
        print(f"Could not find the numpy data file at: {input_filepath}")
        return None, None

    # Track angle needs to be unwrapped to prevent 360-degree
    # wrap-around jumps (e.g. 359 to 1) from being treated as massive outliers
    # by the filters, which causes ringing and spatial circles/loops.
    print("Unwrapping track angle to prevent circular wrapping artifacts...")
    track_rad = np.radians(X[:, 0, :])
    track_rad_unwrapped = np.unwrap(track_rad, axis=1)
    X[:, 0, :] = np.degrees(track_rad_unwrapped)

    if median_window > 0:
        print(f"Applying Median filter (kernel_size={median_window}) to remove outliers...")
        # medfilt expects a tuple of kernel sizes for each dimension
        X = medfilt(X, kernel_size=(1, 1, median_window))

    # Apply Savitzky-Golay filter along the last axis (time/waypoints)
    # The axis is 2 if shape is (N, Features, Points)
    print(f"Applying Savitzky-Golay filter (window_length={window_length}, polyorder={polyorder})...")
    X_smoothed = savgol_filter(X, window_length=window_length, polyorder=polyorder, axis=2)
    
    if output_filepath is None:
        base, ext = os.path.splitext(input_filepath)
        output_filepath = f"{base}_smoothed{ext}"
        
    np.save(output_filepath, X_smoothed)
    print(f"Saved smoothed trajectories to {output_filepath}")
    
    return X, X_smoothed

def visualize_comparison(X_orig, X_smoothed, num_samples=3):
    """
    Visualizes original vs smoothed trajectories.
    """
    print("Press 'n' in the plot window to see a new set of random trajectories.")
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle('Original vs Smoothed Trajectories', fontsize=16)

    def update_plot(event=None):
        if event is not None and event.key != 'n':
            return
            
        for ax in axes:
            ax.clear()
            
        indices = np.random.choice(len(X_orig), size=num_samples, replace=False)
        colors = ['blue', 'orange', 'green']
        
        for idx, c in zip(indices, colors[:num_samples]):
            # Features: 0=Track, 1=Groundspeed, 2=Altitude, 3=Vertical Speed (or similar)
            # We'll plot up to 4 features
            num_features = min(4, X_orig.shape[1])
            feature_names = ['Track Angle', 'Groundspeed', 'Altitude', 'Feature 4']
            
            x_axis = np.arange(X_orig.shape[2])
            
            for f_idx in range(num_features):
                axes[f_idx].plot(x_axis, X_orig[idx, f_idx, :], color=c, alpha=0.3, linestyle='--', label=f'Orig {idx}' if f_idx==0 else "")
                axes[f_idx].plot(x_axis, X_smoothed[idx, f_idx, :], color=c, linewidth=2, label=f'Smoothed {idx}' if f_idx==0 else "")
                axes[f_idx].set_ylabel(feature_names[f_idx])
                axes[f_idx].grid(True, linestyle=':', alpha=0.6)
                
            if num_features > 0:
                axes[0].legend(loc='upper right')
                
        axes[-1].set_xlabel('Resampled Waypoint Index')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', update_plot)
    update_plot()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smooth kinematic trajectories using a Savitzky-Golay filter.")
    parser.add_argument("--input_file", "-i", default=None, help="Path to the input .npy file")
    parser.add_argument("--output_file", "-o", default=None, help="Path to save the output file (optional)")
    parser.add_argument("--window", "-w", type=int, default=15, help="Window length for the SG filter (must be odd)")
    parser.add_argument("--polyorder", "-p", type=int, default=3, help="Polynomial order for the SG filter")
    parser.add_argument("--median_window", "-m", type=int, default=5, help="Window size for the median filter (must be odd). Set to 0 to disable.")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize before/after comparison")
    
    args = parser.parse_args()
    
    # Ensure window lengths are odd
    if args.window % 2 == 0:
        args.window += 1
        print(f"SG Window length must be odd. Adjusted to {args.window}.")
    if args.median_window > 0 and args.median_window % 2 == 0:
        args.median_window += 1
        print(f"Median Window length must be odd. Adjusted to {args.median_window}.")
        
    X_orig, X_smooth = smooth_trajectories(
        args.input_file, 
        args.output_file, 
        window_length=args.window, 
        polyorder=args.polyorder,
        median_window=args.median_window
    )
    
    if args.visualize and X_orig is not None:
        visualize_comparison(X_orig, X_smooth)
