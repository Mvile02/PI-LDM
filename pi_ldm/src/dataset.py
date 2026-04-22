import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder

class AircraftTrajectoryDataset(Dataset):
    """
    Dataset for loading aircraft landing trajectories from .npy and .csv.
    Normalizes data to [-1, 1] range for diffusion models.
    """
    
    # Feature scaling constants based on domain knowledge and data analysis
    # Track: 0-360 -> [-1, 1] 
    # GS: 0-600 -> [-1, 1]   (Max observed ~500)
    # Alt: 0-40000 -> [-1, 1] (Max observed ~34000)
    # Time: 0-2500 -> [-1, 1] (Max observed ~2200)
    SCALES = {
        'track': 180.0,
        'gs': 300.0,
        'alt': 20000.0,
        'time': 1250.0
    }

    def __init__(self, data_dir, split='train', file_base=None):
        self.data_dir = data_dir
        self.split = split
        
        # If file_base is provided, use only that file series. 
        # Otherwise, load all available files in the directory.
        if file_base:
            files = [file_base]
        else:
            npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
            files = [os.path.basename(f).replace(".npy", "") for f in npy_files]
        
        all_X = []
        all_meta = []
        
        # Load all detected pairs
        for base in sorted(files):
            X_path = os.path.join(data_dir, f"{base}.npy")
            meta_path = os.path.join(data_dir, f"{base}.csv")
            
            if os.path.exists(X_path) and os.path.exists(meta_path):
                all_X.append(np.load(X_path))
                all_meta.append(pd.read_csv(meta_path))
            else:
                if file_base:
                    print(f"Error: Specified file not found: {base}")
        
        if not all_X:
            print(f"No data found in {data_dir}")
            self.X = np.zeros((1, 4, 200), dtype=np.float32)
            self.cond = np.zeros((1, 3), dtype=np.float32)
            self.X_norm = self.normalize(self.X)
            return

        # Concatenate all loaded data
        self.X = np.concatenate(all_X, axis=0)
        self.meta = pd.concat(all_meta, axis=0).reset_index(drop=True)
        
        # Apply Normalization to [-1, 1]
        self.X_norm = self.normalize(self.X)
        
        # Conditioning: Encode categorical constraints
        # 1. Runway / Airport (A)
        if 'airport' not in self.meta.columns:
            self.meta['airport'] = 'LSZH'
            
        self.le_airport = LabelEncoder()
        airport_idx = self.le_airport.fit_transform(self.meta['airport'].astype(str))
        
        # 2. Aircraft Type (T)
        self.le_type = LabelEncoder()
        type_idx = self.le_type.fit_transform(self.meta['typecode'].astype(str))
        
        # 3. Weather (W) - Mocked to 0
        weather_mock = np.zeros_like(airport_idx)
        
        self.cond = np.column_stack((airport_idx, type_idx, weather_mock)).astype(np.float32)
        print(f"Loaded dataset with {len(self.X)} samples from {len(all_X)} source files.")

    @staticmethod
    def normalize(X):
        """Scale 4 features to [-1, 1] range."""
        X_norm = X.copy().astype(np.float32)
        X_norm[:, 0, :] = (X[:, 0, :] / AircraftTrajectoryDataset.SCALES['track']) - 1.0
        X_norm[:, 1, :] = (X[:, 1, :] / AircraftTrajectoryDataset.SCALES['gs']) - 1.0
        X_norm[:, 2, :] = (X[:, 2, :] / AircraftTrajectoryDataset.SCALES['alt']) - 1.0
        X_norm[:, 3, :] = (X[:, 3, :] / AircraftTrajectoryDataset.SCALES['time']) - 1.0
        return X_norm

    @staticmethod
    def denormalize(X_norm):
        """Scale back to physical units from [-1, 1]."""
        is_torch = torch.is_tensor(X_norm)
        if is_torch:
            X = X_norm.clone()
        else:
            X = X_norm.copy()
            
        X[:, 0, :] = (X_norm[:, 0, :] + 1.0) * AircraftTrajectoryDataset.SCALES['track']
        X[:, 1, :] = (X_norm[:, 1, :] + 1.0) * AircraftTrajectoryDataset.SCALES['gs']
        X[:, 2, :] = (X_norm[:, 2, :] + 1.0) * AircraftTrajectoryDataset.SCALES['alt']
        X[:, 3, :] = (X_norm[:, 3, :] + 1.0) * AircraftTrajectoryDataset.SCALES['time']
        return X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return state tensor: (state_dim, seq_len) 
        # and condition tensor: (cond_dim)
        sample = torch.tensor(self.X_norm[idx], dtype=torch.float32)
        condition = torch.tensor(self.cond[idx], dtype=torch.float32)
        return sample, condition

def get_dataloaders(data_dir, batch_size=32, file_base=None):
    train_dataset = AircraftTrajectoryDataset(data_dir, file_base=file_base)
    # We use entire dataset for training in this simple implementation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, None
