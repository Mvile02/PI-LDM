import numpy as np
import os

path = r'c:\Users\usuario\Desktop\Delft\TFM\Code\data\processed\X_LSZH_2026-03-01_1000_to_2026-03-01_1200_runway14.npy'
if os.path.exists(path):
    data = np.load(path)
    print('Shape:', data.shape)
    # data is (N, 4, 200). 4 features: probably Track, Groundspeed, Altitude, Time (or similar)
    # The first dimension is batch/sample
    # The second dimension is feature
    # The third dimension is sequence
    for i in range(data.shape[1]):
        feat = data[:, i, :]
        print(f"Feature {i}: min={feat.min():.2f}, max={feat.max():.2f}, mean={feat.mean():.2f}, std={feat.std():.2f}")
else:
    print("File not found")
