import numpy as np
import os
import glob

def main():
    base_dir = r"c:\Users\usuario\Desktop\Delft\TFM\Code\data\processed"
    npy_files = glob.glob(os.path.join(base_dir, "*.npy"))

    if not npy_files:
        print("No .npy files found in the processed data directory.")
        return

    print("--- Final Track Angles of Extracted Trajectories ---")
    for npy_file in npy_files:
        filename = os.path.basename(npy_file)
        # Load the data
        X = np.load(npy_file)
        
        final_tracks = X[:, 0, -1]
        
        # Calculate magnetic track
        # LSZH magnetic declination is approx +3.6 degrees (East)
        # Magnetic Track = True Track - Magnetic Declination
        MAG_DEV = 3.6
        mag_tracks = (final_tracks - MAG_DEV) % 360
        
        # Round to nearest integer to group them
        rounded_mag_tracks = np.round(mag_tracks).astype(int)
        
        print(f"\n[File: {filename}]")
        print(f"Total trajectories: {len(X)}")
        
        # Print list of final track angles (True), wrapped if too long
        tracks_str = ", ".join([f"{t:.1f}" for t in final_tracks])
        print(f"All final track angles (True deg):")
        print(f"[{tracks_str}]")
        
        # Count occurrences
        from collections import Counter
        track_counts = Counter(rounded_mag_tracks)
        
        print("\n6 Most Common Track Appearances (Magnetic Heading -> Count):")
        # Top 4 most common
        for track, count in track_counts.most_common(6):
            # Show the corresponding true track approx for reference
            approx_true = (track + MAG_DEV) % 360
            print(f"  {track:03d} M (approx {approx_true:03.1f} T) : {count} times")

if __name__ == "__main__":
    main()
