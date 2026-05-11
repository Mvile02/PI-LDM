import os
import sys
import numpy as np
import pandas as pd

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_cluster.py [file.npz or .npy] [cluster_id_1] [cluster_id_2] ...")
        print("Example 1: python extract_cluster.py LSZH_2019_R14_kinematic_200pts_clust5.npz 2")
        print("Example 2: python extract_cluster.py LSZH_2019_R14_kinematic_200pts_spatial_5000m.npy 1 3")
        return

    file_name = sys.argv[1]
    cluster_ids = [int(arg) for arg in sys.argv[2:]]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    X_target = None
    meta_target = None
    
    if file_name.endswith(".npz"):
        # Old mode: .npz file contains X and y
        file_path = os.path.join(base_dir, "data", "clusters", file_name)
        if not os.path.exists(file_path):
            file_path = os.path.join(base_dir, "data", "processed", file_name)
            
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            return

        print(f"Loading clustered dataset: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        X = data['X'].astype(np.float32)
        y = data['y']
        
        mask = np.isin(y, cluster_ids)
        X_target = X[mask]
        
        meta_file = file_path.replace(".npz", ".csv")
        if os.path.exists(meta_file):
            print(f"Loading metadata from: {meta_file}")
            meta = pd.read_csv(meta_file)
            meta_target = meta.iloc[mask].reset_index(drop=True)
        else:
            print(f"Warning: Metadata file not found: {meta_file}")
        
    elif file_name.endswith(".npy"):
        # New mode: search for individual _c{id}.npy files
        base_name = file_name.replace(".npy", "")
        X_list = []
        meta_list = []
        for cid in cluster_ids:
            cluster_file = os.path.join(base_dir, "data", "clusters", f"{base_name}_c{cid}.npy")
            cluster_csv = os.path.join(base_dir, "data", "clusters", f"{base_name}_c{cid}.csv")
            
            if not os.path.exists(cluster_file):
                print(f"Error: Cluster {cid} file not found: {cluster_file}")
                return
            print(f"Loading cluster {cid} from: {cluster_file}")
            X_c = np.load(cluster_file, allow_pickle=True).astype(np.float32)
            X_list.append(X_c)
            
            if os.path.exists(cluster_csv):
                meta_c = pd.read_csv(cluster_csv)
                meta_list.append(meta_c)
            else:
                print(f"Warning: CSV file for cluster {cid} not found: {cluster_csv}")
            
        if X_list:
            X_target = np.concatenate(X_list, axis=0)
        if meta_list:
            meta_target = pd.concat(meta_list, ignore_index=True)
    else:
        print("Error: File must be .npz or .npy")
        return
        
    if X_target is None or len(X_target) == 0:
        print(f"Error: No trajectories found for clusters {cluster_ids}.")
        return
        
    cluster_str = "_".join([f"c{cid}" for cid in cluster_ids])
    print(f"Clusters {cluster_ids} have a total of {len(X_target)} unique trajectories.")
    
    # Save the resulting array as a clean npy compatible with PI-LDM
    if file_name.endswith(".npz"):
        output_filename = file_name.replace(".npz", f"_{cluster_str}.npy")
    else:
        output_filename = file_name.replace(".npy", f"_{cluster_str}.npy")
        
    output_path = os.path.join(base_dir, "data", "clusters", output_filename)
    
    np.save(output_path, X_target)
    
    if meta_target is not None:
        csv_output_path = output_path.replace(".npy", ".csv")
        meta_target.to_csv(csv_output_path, index=False)
        print(f"Success! Combined trajectories and metadata saved to:\n  {output_path}\n  {csv_output_path}")
    else:
        print(f"Success! Combined trajectories saved to: {output_path}")
        
    print(f"-> To train your model, set in train.py: FILE_BASE = '{output_filename.replace('.npy','')}'")

if __name__ == "__main__":
    main()
