import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Calculate aircraft diversity from a trajectory CSV.")
    parser.add_argument('--csv', type=str, help="Path to the CSV file or just the FILE_BASE.", default=None)
    parser.add_argument('--orig', type=str, help="Path to the original CSV file to decode numeric types.", default=None)
    args = parser.parse_args()

    # --- CONFIGURATION AREA ---
    # Default values if no arguments provided (same pattern as plot_npy.py)
    FILE_BASE = "LSZH_2019_R14_kinematic_200pts_spatial_5000m_cond_synthetic_1000t"
    ORIGINAL_DATASET_BASE = "LSZH_2019_R14_kinematic_200pts_spatial_5000m"
    # --------------------------

    csv_path = args.csv
    orig_dataset = args.orig if args.orig else ORIGINAL_DATASET_BASE
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # If no explicit CSV path was given, try to find it based on FILE_BASE
    if not csv_path:
        search_paths = [
            os.path.join(base_dir, "data", "processed", f"{FILE_BASE}.csv"),
            os.path.join(base_dir, "data", "clusters", f"{FILE_BASE}.csv"),
            os.path.join(base_dir, "outputs", "trajectories", f"{FILE_BASE}.csv"),
            os.path.join(base_dir, "pi_ldm", "outputs", "trajectories", f"{FILE_BASE}.csv")
        ]
        for p in search_paths:
            if os.path.exists(p):
                csv_path = p
                break
                
    if not csv_path or not os.path.exists(csv_path):
        print(f"Error: Could not find CSV file for '{FILE_BASE}'. Please provide a valid path via --csv.")
        return

    print(f"Analyzing diversity for: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)

    if 'typecode' not in df.columns:
        print("Error: 'typecode' column not found in the CSV.")
        return

    # If the typecode is purely numeric, map it back to string names using the original CSV
    if pd.api.types.is_numeric_dtype(df['typecode']) and orig_dataset:
        orig_csv = None
        for loc in ['processed', 'clusters']:
            p = os.path.join(base_dir, 'data', loc, f"{orig_dataset}.csv")
            if os.path.exists(p):
                orig_csv = p
                break
        
        if orig_csv:
            orig_df = pd.read_csv(orig_csv)
            ac_types_mapping = sorted(orig_df['typecode'].astype(str).unique())
            
            # Translate index back to string
            df['typecode'] = df['typecode'].apply(
                lambda x: ac_types_mapping[int(x)] if not pd.isna(x) and 0 <= int(x) < len(ac_types_mapping) else "Unknown"
            )
        else:
            print(f"Warning: Original dataset '{orig_dataset}' not found. Cannot map numeric typecodes.")

    # Calculate absolute counts and percentages
    counts = df['typecode'].value_counts()
    percentages = df['typecode'].value_counts(normalize=True) * 100

    print("\n--- Aircraft Type Diversity ---")
    print(f"Total trajectories: {len(df)}")
    print("-" * 34)
    print(f"{'Aircraft Type':<15} | {'Count':<7} | {'%':<5}")
    print("-" * 34)
    
    for ac_type, count in counts.items():
        pct = percentages[ac_type]
        print(f"{ac_type:<15} | {count:<7} | {pct:.2f}%")
    print("-" * 34)

    # Generate a nice Bar Plot and save it
    plt.figure(figsize=(10, 6))
    percentages.plot(kind='bar', color='#4A90E2', edgecolor='black')
    
    plt.title(f'Aircraft Type Distribution\nDataset: {os.path.basename(csv_path)}')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Aircraft Type')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_dir = os.path.join(base_dir, "outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"diversity_{os.path.basename(csv_path).replace('.csv', '.png')}")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved distribution plot to: {plot_path}")

if __name__ == "__main__":
    main()
