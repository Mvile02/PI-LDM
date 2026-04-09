import numpy as np
import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_BASE = "X_LSZH_2026-03-01_1000_to_2026-03-01_1200_runway14"
X_file = os.path.join(base_dir, "data", "processed", f"{FILE_BASE}.npy")
meta_file = os.path.join(base_dir, "data", "processed", f"{FILE_BASE}.csv")

print(f"Loading validation samples from {X_file}...")
try:
    X = np.load(X_file)
    meta = pd.read_csv(meta_file)
    
    print(f"Tensor Shape: {X.shape}")
    print(f"Metadata Shape: {meta.shape}")
    
    # Print a single trajectory instance properties
    print(f"\nMetadata for flight 0: {meta.iloc[0].to_dict()}")
    print("Channels for flight 0:")
    print(f"  Track bounds: min={np.min(X[0, 0])}, max={np.max(X[0, 0])}")
    print(f"  Groundspeed bounds: min={np.min(X[0, 1])}, max={np.max(X[0, 1])}")
    print(f"  Altitude bounds: min={np.min(X[0, 2])}, max={np.max(X[0, 2])}")
    print(f"  Elapsed Time limits: min={np.min(X[0, 3])}, max={np.max(X[0, 3])}")
except Exception as e:
    print(f"Error viewing validation files: {e}")

