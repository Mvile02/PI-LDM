import os
import numpy as np
import glob

data_dir = r'c:\Users\usuario\Desktop\Delft\TFM\Code\data\processed'
files = glob.glob(os.path.join(data_dir, '*.npy'))
for f in files:
    try:
        data = np.load(f)
        print(f"{os.path.basename(f)}: {data.shape}")
    except Exception as e:
        print(f"Error loading {f}: {e}")
