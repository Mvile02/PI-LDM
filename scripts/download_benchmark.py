import os
import requests
import pandas as pd
from tqdm import tqdm

def download_benchmark():
    """
    Downloads the pre-processed LSZH Landing Trajectories dataset (Oct-Nov 2019)
    from Figshare.
    """
    # FIGSHARE metadata
    FILE_ID = "20291079"
    URL = f"https://ndownloader.figshare.com/files/{FILE_ID}"
    
    # Target directory and filename (Requested by User)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_path, "data", "raw")
    output_filename = "LSZH_2019_10_01_to_11_30_benchmark.parquet"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Target: {output_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"File already exists at {output_path}. Skipping download.")
    else:
        print(f"Downloading benchmark dataset from Figshare...")
        response = requests.get(URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        block_size = 1024 # 1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading")
        
        with open(output_path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        
        if total_size != 0 and t.n != total_size:
            print("ERROR: Something went wrong with the download.")
            return False
            
    # Verification
    print("\nVerifying data...")
    try:
        df = pd.read_parquet(output_path)
        print(f"Successfully loaded parquet file.")
        print(f"Total Trajectories/Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Quick check for dates
        if 'timestamp' in df.columns:
            # Handle different timestamp formats if necessary
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
            print(f"Date Range (from unix time): {df['time'].min()} to {df['time'].max()}")

    except Exception as e:
        print(f"Error during verification: {e}")
        return False
        
    print("\nBenchmark download and verification complete.")
    return True

if __name__ == "__main__":
    download_benchmark()
