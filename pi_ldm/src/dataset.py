import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

class AircraftTrajectoryDataset(Dataset):
    """
    Dataset for loading aircraft landing trajectories from .npy and .csv.
    """
    def __init__(self, data_dir, split='train', file_base='TEST'):
        self.data_dir = data_dir
        self.split = split
        
        X_path = os.path.join(data_dir, f"{file_base}.npy")
        meta_path = os.path.join(data_dir, f"{file_base}.csv")
        
        if os.path.exists(X_path) and os.path.exists(meta_path):
            self.X = np.load(X_path) # shape: (num_samples, 4, 200)
            self.meta = pd.read_csv(meta_path)
            
            # Simple conditioning: Encode categorical constraints (e.g. Aircraft Type, Airport)
            # In a real scenario, this would map to a continuous embedding space or one-hot vectors.
            # Here, we create a 3D condition vector representing [Runway/Airport, Aircraft, Weather]
            
            # 1. Runway / Airport (A)
            le_airport = LabelEncoder()
            airport_idx = le_airport.fit_transform(self.meta['airport'].astype(str))
            
            # 2. Aircraft Type (T)
            le_type = LabelEncoder()
            type_idx = le_type.fit_transform(self.meta['typecode'].astype(str))
            
            # 3. Weather (W) - Mocked to 0 for now as it's missing from the TEST subset
            weather_mock = np.zeros_like(airport_idx)
            
            # Combine into condition tensor: (batch, 3)
            # Typically these would be normalized or embedded.
            self.cond = np.column_stack((airport_idx, type_idx, weather_mock)).astype(np.float32)
            
        else:
            print(f"File not found: {X_path}")
            self.X = np.zeros((10, 4, 200), dtype=np.float32)
            self.cond = np.zeros((10, 3), dtype=np.float32)
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return state tensor: (state_dim, seq_len) 
        # and condition tensor: (cond_dim)
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        condition = torch.tensor(self.cond[idx], dtype=torch.float32)
        return sample, condition

def get_dataloaders(data_dir, batch_size=32, file_base='TEST'):
    # In practice, you would split your data into distinct 'train' and 'val' bases
    train_dataset = AircraftTrajectoryDataset(data_dir, split='train', file_base=file_base)
    # Mocking val dataset with the same for demonstration
    val_dataset = AircraftTrajectoryDataset(data_dir, split='val', file_base=file_base)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
