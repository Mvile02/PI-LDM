import numpy as np
import os

base_dir = r"c:\Users\usuario\Desktop\Delft\TFM\Code"
npy_file = os.path.join(base_dir, "data", "processed", "X_LSZH_2026-03-01_1000_to_2026-03-01_1200_runway14.npy")

if not os.path.exists(npy_file):
    print(f"File not found: {npy_file}")
    exit()

X = np.load(npy_file)
# X shape: (N, 4, 200)
# Channels: 0:track, 1:groundspeed, 2:altitude, 3:elapsed_time

# Check Groundspeed jumps (Channel 1)
gs = X[:, 1, :]
# Calculate jumps between resampled points
jumps = np.abs(np.diff(gs, axis=1))

max_jump = np.max(jumps)
print(f"Max Groundspeed jump between waypoints: {max_jump:.2f} kts")

# Find index of flight with largest jump
idx = np.unravel_index(np.argmax(jumps), jumps.shape)
print(f"Flight index with max jump: {idx[0]}")
print(f"Waypoint index: {idx[1]}")

# Check for jumps > 50 kts
high_jumps = np.sum(jumps > 50)
print(f"Number of jumps > 50 kts: {high_jumps}")
