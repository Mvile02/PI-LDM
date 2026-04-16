"""
Description:
    Extraction and preprocessing of kinematic trajectory data for PI-LDM.
    Filters raw OpenSky data from Zurich (LSZH), extracting track, groundspeed, altitude, and elapsed time.

Author:
    Gerard Martínez Vilella
    April 16, 2026
"""

import os
import warnings
import numpy as np
import pandas as pd
from traffic.data.datasets import landing_zurich_2019
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


class DatasetBuilder:
    def __init__(self, target_points=200, features=None):
        if features is None:
            features = ['track', 'groundspeed', 'altitude', 'time']
        self.target_points = target_points
        self.features = features

    def compute_elapsed_time(self, flight_df):
        """Converts timestamp to zero-anchored elapsed time in seconds."""
        df = flight_df.copy()
        df['unix_ts'] = pd.to_datetime(df['timestamp'], utc=True).apply(lambda x: x.timestamp()).astype(int)
        df['time'] = df['unix_ts'] - df['unix_ts'].iloc[0]
        return df

    def resample_flight(self, df):
        """Uniformly samples the flight to hit the target number of points."""
        indices = np.linspace(0, len(df) - 1, self.target_points, dtype=int)
        return df.iloc[indices]

    def has_holding_pattern(self, df, angle_threshold=720):
        """Detects holding patterns via cumulative track angle changes."""
        return df['track_unwrapped'].diff().sum() >= angle_threshold

    def filter_flights(self, traffic_data, airport_code="LSZH"):
        """Categorizes flights into nominal, go-arounds, and holds."""
        categorized = {'nominal': [], 'go_around': [], 'holding': []}
        
        for flight in tqdm(traffic_data, desc="Categorizing flights (Checking go-arounds/holds)"):
            df = flight.data
            if flight.go_around(airport_code).has():
                categorized['go_around'].append(df)
            elif self.has_holding_pattern(df):
                categorized['holding'].append(df)
            else:
                categorized['nominal'].append(df)
                
        return categorized

    def separate_by_runway(self, flight_list):
        """Groups flights by destination runway."""
        runways = {}
        for df in flight_list:
            rwy = str(df['runway'].iloc[0])
            if rwy not in runways:
                runways[rwy] = []
            runways[rwy].append(df)
        return runways

    def build_tensor(self, flight_list):
        """Converts a list of flight dataframes into a (N, features, seq_len) numpy array."""
        processed_flights = []
        for df in tqdm(flight_list, desc="Building sequence tensors"):
            resampled = self.resample_flight(df)
            timed = self.compute_elapsed_time(resampled)
            
            # Check for NaNs
            if timed[self.features].isna().any().any():
                continue
                
            # Transpose to (features, seq_len)
            tensor_slice = timed[self.features].to_numpy().T
            processed_flights.append(tensor_slice)
            
        return np.stack(processed_flights) if processed_flights else np.array([])


if __name__ == '__main__':
    print("Loading Zurich 2019 dataset...")
    traffic_ds = landing_zurich_2019.between("2019-10-01", "2019-11-30").assign_id().unwrap().eval()
    
    builder = DatasetBuilder(target_points=200)
    
    print("Filtering traffic scenarios...")
    scenarios = builder.filter_flights(traffic_ds)
    
    # Extract nominal flights
    runways = builder.separate_by_runway(scenarios['nominal'])
    
    if '14' in runways:
        print(f"Building kinematic tensor for {len(runways['14'])} flights on Runway 14.")
        tensor_rwy14 = builder.build_tensor(runways['14'])
        
        # Determine output directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(base_dir, 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save array with a descriptive name
        output_base = "LSZH_2019_R14_kinematic_200pts"
        output_name = os.path.join(processed_dir, f"{output_base}.npy")
        
        np.save(output_name, tensor_rwy14)
        print(f"Saved {output_name} with shape: {tensor_rwy14.shape}")