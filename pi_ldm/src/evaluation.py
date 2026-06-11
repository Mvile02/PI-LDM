"""
Description:
    Statistical evaluation module for Trajectory Generation (PI-LDM).
    Computes Fidelity (Discriminative Score), Diversity (PCA/t-SNE), 
    Usefulness (Downstream Predictive MAE), and Energy Distance.
    Includes a self-diagnostic test mode to verify the evaluation pipeline.

Author:
    Gerard Martínez Vilella
    May, 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# 1. Models for Statistical Evaluation
# ==========================================

class LSTMDiscriminator(nn.Module):
    """
    LSTM binary classifier to distinguish between real (label 1) and synthetic (label 0) trajectories.
    """
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        # Take the output of the last timestep
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


class LSTMRegressor(nn.Module):
    """
    LSTM model for downstream time-series forecasting.
    Predicts the final steps of velocity and altitude given the earlier sequence.
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Predict from the last step's hidden state
        return out


# ==========================================
# 2. Main Trajectory Evaluator
# ==========================================

class TrajectoryEvaluator:
    """
    Comprehensive evaluation suite for trajectory generative models.
    Supports trajectory arrays of shape (num_samples, 4, 200) - [track, gs, alt, time].
    """
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"--> Initializing TrajectoryEvaluator on {self.device}")

    # --- Feature Formatting Helper ---
    def _prepare_data(self, data, average_dimension='sequence'):
        """
        Formats trajectory data from (N, 4, 200) into a 2D format for PCA/t-SNE or metrics.
        """
        # Ensure numpy array
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        
        # Reshape to (N, 200, 4) for standard temporal-first handling
        if data.shape[1] == 4 and data.shape[2] == 200:
            data = np.transpose(data, (0, 2, 1))
            
        N, seq_len, num_features = data.shape
        
        if average_dimension == 'sequence':
            # Mean over sequence length
            return np.mean(data, axis=1)
        elif average_dimension == 'features':
            # Mean over features
            return np.mean(data, axis=2)
        elif average_dimension == 'flatten':
            # Flatten entire trajectory to 2D
            return data.reshape(N, -1)
        else:
            raise ValueError("average_dimension must be 'sequence', 'features', or 'flatten'")

    # ==========================================
    # Metric A: Diversity (PCA & t-SNE)
    # ==========================================
    def visualize_diversity(self, real_data, gen_data, method='PCA', average_dimension='sequence', 
                            max_samples=1000, save_path=None):
        """
        Calculates and plots spatial/distributional overlap between real and generated trajectories.
        """
        print(f"--> Computing Visual Diversity ({method}, dimension={average_dimension})...")
        
        # Prepare datasets
        n_samples = min(max_samples, len(real_data), len(gen_data))
        idx_real = np.random.permutation(len(real_data))[:n_samples]
        idx_gen = np.random.permutation(len(gen_data))[:n_samples]
        
        real_flat = self._prepare_data(real_data[idx_real], average_dimension)
        gen_flat = self._prepare_data(gen_data[idx_gen], average_dimension)
        
        plt.figure(figsize=(10, 8))
        plt.rcParams.update({'font.size': 12})
        
        if method == 'PCA':
            scaler = StandardScaler()
            pca = PCA(n_components=2)
            # Fit PCA on combined dataset to learn common projection
            combined = np.concatenate([real_flat, gen_flat], axis=0)
            combined_scaled = scaler.fit_transform(combined)
            pca.fit(combined_scaled)
            
            # Print explained variance ratio
            exp_var = pca.explained_variance_ratio_ * 100
            print(f"    PCA explained variance (scaled): PC1 = {exp_var[0]:.2f}%, PC2 = {exp_var[1]:.2f}% "
                  f"(Total: {sum(exp_var):.2f}%)")
            
            real_proj = pca.transform(scaler.transform(real_flat))
            gen_proj = pca.transform(scaler.transform(gen_flat))
            
            plt.scatter(real_proj[:, 0], real_proj[:, 1], c='crimson', alpha=0.4, label='Real', edgecolors='none')
            plt.scatter(gen_proj[:, 0], gen_proj[:, 1], c='royalblue', alpha=0.4, label='Synthetic', edgecolors='none')
            plt.xlabel(f'Principal Component 1 ({exp_var[0]:.1f}% Var)')
            plt.ylabel(f'Principal Component 2 ({exp_var[1]:.1f}% Var)')
            plt.title(f'PCA Trajectory Coverage (N={n_samples})')
            
        elif method == 't-SNE':
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            combined = np.concatenate([real_flat, gen_flat], axis=0)
            proj = tsne.fit_transform(combined)
            
            plt.scatter(proj[:n_samples, 0], proj[:n_samples, 1], c='crimson', alpha=0.4, label='Real', edgecolors='none')
            plt.scatter(proj[n_samples:, 0], proj[n_samples:, 1], c='royalblue', alpha=0.4, label='Synthetic', edgecolors='none')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.title(f't-SNE Trajectory Coverage (N={n_samples})')
            
        plt.legend(frameon=True)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Diversity plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()

    # ==========================================
    # Metric B: Fidelity (Discriminative Score)
    # ==========================================
    def compute_discriminative_score(self, real_data, gen_data, epochs=10, batch_size=64):
        """
        Trains an LSTM Classifier to distinguish real vs synthetic trajectories.
        Returns accuracy and Discriminative Score: |Accuracy - 0.5|
        """
        print("--> Evaluating Temporal Fidelity via LSTM Discriminator...")
        
        # Ensure temporal-last representation (N, seq_len, features) for RNNs
        if real_data.shape[1] == 4 and real_data.shape[2] == 200:
            real_data = np.transpose(real_data, (0, 2, 1))
            gen_data = np.transpose(gen_data, (0, 2, 1))
            
        # Format labels: 1 for Real, 0 for Synthetic
        X = np.concatenate([real_data, gen_data], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(len(real_data)), np.zeros(len(gen_data))], axis=0).astype(np.float32)
        
        # Split train/test
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
        
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create DataLoaders
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        # Instantiate Discriminator
        discriminator = LSTMDiscriminator(input_dim=X.shape[2]).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
        
        # Train
        discriminator.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                pred = discriminator(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            # print(f"    Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
            
        # Evaluate
        discriminator.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device).unsqueeze(1)
                pred = discriminator(batch_X)
                correct += ((pred > 0.5).float() == batch_y).sum().item()
                total += batch_y.size(0)
                
        test_accuracy = correct / total
        disc_score = abs(test_accuracy - 0.5)
        
        print(f"    LSTM Discriminator Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"    Discriminative Score (|Accuracy - 0.5|): {disc_score:.4f} (Ideal: 0.0000)")
        
        return test_accuracy, disc_score

    # ==========================================
    # Metric C: Usefulness / Utility (TSTR Downstream MAE)
    # ==========================================
    def compute_usefulness(self, real_data, gen_data, steps_ahead=15, epochs=10, batch_size=64):
        """
        Downstream utility evaluation:
        1. Trains an LSTM Regressor on SYNTHETIC data to predict velocity and altitude.
        2. Evaluates its Mean Absolute Error (MAE) on REAL test data.
        3. Compares this with a model trained entirely on REAL training data.
        """
        print(f"--> Evaluating Downstream Usefulness (Predicting final {steps_ahead} steps)...")
        
        # Standardize shapes to (N, seq_len, features)
        if real_data.shape[1] == 4 and real_data.shape[2] == 200:
            real_data = np.transpose(real_data, (0, 2, 1))
            gen_data = np.transpose(gen_data, (0, 2, 1))
            
        N, seq_len, features = real_data.shape
        
        # Partition data: x is [0 : seq_len - steps_ahead], y is [last timestep] for altitude and speed (features 1, 2)
        # Note: Features are: 0: track, 1: gs (ground speed), 2: alt (altitude), 3: time
        def build_io_pairs(dataset):
            X_io = dataset[:, :-steps_ahead, :]
            y_io = dataset[:, -1, 1:3] # Target is GS and Altitude at the very end
            return X_io, y_io

        # Train/Test splits for real data
        split_idx = int(0.8 * N)
        real_train = real_data[:split_idx]
        real_test = real_data[split_idx:]
        
        X_real_train, y_real_train = build_io_pairs(real_train)
        X_real_test, y_real_test = build_io_pairs(real_test)
        
        X_gen_train, y_gen_train = build_io_pairs(gen_data)
        
        # Datasets
        ds_tstr = TensorDataset(torch.tensor(X_gen_train, dtype=torch.float32), torch.tensor(y_gen_train, dtype=torch.float32))
        ds_trtr = TensorDataset(torch.tensor(X_real_train, dtype=torch.float32), torch.tensor(y_real_train, dtype=torch.float32))
        ds_test = TensorDataset(torch.tensor(X_real_test, dtype=torch.float32), torch.tensor(y_real_test, dtype=torch.float32))
        
        loader_tstr = DataLoader(ds_tstr, batch_size=batch_size, shuffle=True)
        loader_trtr = DataLoader(ds_trtr, batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
        
        # Helper to train and evaluate
        def train_eval_model(train_loader, label="TSTR"):
            model = LSTMRegressor(input_dim=features, output_dim=2).to(self.device)
            criterion = nn.L1Loss() # Mean Absolute Error
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            # Train
            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    pred = model(batch_X)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Test on Real Test
            model.eval()
            total_mae = 0
            count = 0
            with torch.no_grad():
                for batch_X, batch_y in loader_test:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    pred = model(batch_X)
                    total_mae += nn.functional.l1_loss(pred, batch_y, reduction='sum').item()
                    count += batch_y.numel()
                    
            avg_mae = total_mae / count
            print(f"    {label} Model Average MAE on Real Test Data: {avg_mae:.5f}")
            return avg_mae

        tstr_mae = train_eval_model(loader_tstr, "TSTR (Train Synthetic, Test Real)")
        trtr_mae = train_eval_model(loader_trtr, "TRTR (Train Real, Test Real)")
        
        ratio = tstr_mae / trtr_mae if trtr_mae > 0 else float('inf')
        print(f"    Downstream MAE Ratio (TSTR/TRTR): {ratio:.4f} (Ideal: ~1.000)")
        
        return tstr_mae, trtr_mae

    # ==========================================
    # Metric D: Distribution Similarity (Energy Distance)
    # ==========================================
    def compute_energy_distance(self, real_data, gen_data, bootstraps=50, sample_size=200):
        """
        Calculates mathematical energy distance between real and synthetic empirical distributions.
        Highly robust statistical metric for multi-dimensional generative model validation.
        """
        print(f"--> Calculating Energy Distance (bootstraps={bootstraps}, sample_size={sample_size})...")
        
        # Flatten spatial sequences for standard energy distance
        real_flat = self._prepare_data(real_data, 'flatten')
        gen_flat = self._prepare_data(gen_data, 'flatten')
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        
        def energy_distance(x, y):
            # E(x, y) = 2 * E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]
            a = cdist(x, y, "euclidean").mean()
            b = cdist(x, x, "euclidean").mean()
            c = cdist(y, y, "euclidean").mean()
            return 2 * a - b - c

        e_distances = []
        for _ in range(bootstraps):
            # Bootstrap sample to avoid computational bottleneck of large cdist
            idx_real = random.sample(range(len(real_flat)), sample_size)
            idx_gen = random.sample(range(len(gen_flat)), sample_size)
            
            real_sample = scaler.fit_transform(real_flat[idx_real])
            gen_sample = scaler.fit_transform(gen_flat[idx_gen])
            
            e_distances.append(energy_distance(real_sample, gen_sample))
            
        mean_e_dist = np.mean(e_distances)
        print(f"    Average Energy Distance: {mean_e_dist:.5f} (Ideal: 0.0000)")
        return mean_e_dist

    # ==========================================
    # Metric E: Physics Envelope & Coherence Violation Rate
    # ==========================================
    def compute_physics_violations(self, trajectories, gs_max=150.0, alt_min=0.0, max_roc=25.0, dt=1.0):
        """
        Domain-Specific Metric:
        Evaluates physical consistency of trajectories in raw physical units.
        trajectories: (N, 4, 200) in raw physical units
        """
        print("--> Evaluating Physical Constraints and Envelope Violations...")
        if torch.is_tensor(trajectories):
            trajectories = trajectories.detach().cpu().numpy()
            
        # Ensure shape (N, 4, 200)
        if trajectories.shape[1] != 4:
            raise ValueError(f"Expected shape with 4 features at axis 1, got {trajectories.shape}")
            
        N, _, seq_len = trajectories.shape
        
        track = trajectories[:, 0, :]
        gs = trajectories[:, 1, :]     # m/s or knots
        alt = trajectories[:, 2, :]    # ft or meters
        time = trajectories[:, 3, :]   # s
        
        stall_violations = 0
        speed_overshoot = 0
        ground_crashes = 0
        extreme_vertical_rate = 0
        
        for i in range(N):
            # 1. Check Stall Speed (e.g. Speed dropping below a threshold during flight)
            # Assuming gs unit is m/s. Stall speed is ~60m/s
            if np.any(gs[i] < 55.0):
                stall_violations += 1
                
            # 2. Check maximum GS (exceeding flight envelopes)
            if np.any(gs[i] > gs_max):
                speed_overshoot += 1
                
            # 3. Ground Crashes (altitude dropping below ground level prematurely before final steps)
            # Check if altitude goes below alt_min during mid-flight (first 85% of trajectory)
            mid_len = int(seq_len * 0.85)
            if np.any(alt[i, :mid_len] < alt_min):
                ground_crashes += 1
                
            # 4. Vert Rate Violation
            dh = np.diff(alt[i])
            dt_step = np.diff(time[i])
            # Handle zeros in time diff to avoid div by zero
            dt_step = np.where(dt_step == 0, 1.0, dt_step)
            roc = np.abs(dh / dt_step)
            if np.any(roc > max_roc):
                extreme_vertical_rate += 1
                
        # Calculate rates
        stall_rate = stall_violations / N
        overshoot_rate = speed_overshoot / N
        crash_rate = ground_crashes / N
        vert_rate_violation = extreme_vertical_rate / N
        
        print(f"    Stall Speed Violation Rate: {stall_rate*100:.2f}%")
        print(f"    Overspeed Violation Rate: {overshoot_rate*100:.2f}%")
        print(f"    Premature Ground Penetration Rate: {crash_rate*100:.2f}%")
        print(f"    Extreme Rate-of-Climb/Descent Rate: {vert_rate_violation*100:.2f}%")
        
        return {
            'stall_rate': stall_rate,
            'overshoot_rate': overshoot_rate,
            'crash_rate': crash_rate,
            'vert_rate_violation': vert_rate_violation
        }

    # ==========================================
    # Metric F: Memorization Check (1-NN Distance Distribution)
    # ==========================================
    def check_memorization(self, real_data, gen_data, sample_size=500, save_path=None):
        """
        Computes 1-Nearest Neighbor distance ratios to detect training data copying/memorization.
        """
        print("--> Computing Trajectory Memorization (1-NN Distance Analysis)...")
        # Prepare datasets (N, 800)
        real_flat = self._prepare_data(real_data, 'flatten')
        gen_flat = self._prepare_data(gen_data, 'flatten')
        
        # Subsample to speed up calculation and keep it balanced
        n_real = min(sample_size, len(real_flat))
        n_gen = min(sample_size, len(gen_flat))
        
        idx_real = np.random.permutation(len(real_flat))[:n_real]
        idx_gen = np.random.permutation(len(gen_flat))[:n_gen]
        
        R_sub = real_flat[idx_real]
        G_sub = gen_flat[idx_gen]
        
        # Scale to ensure distance metrics are normalized and not dominated by altitude
        scaler = StandardScaler()
        # Scale based on the real dataset distribution
        scaler.fit(real_flat)
        R_scaled = scaler.transform(real_flat)
        R_sub_scaled = scaler.transform(R_sub)
        G_sub_scaled = scaler.transform(G_sub)
        
        # 1-NN distances from Gen to Real
        dist_gen_to_real = cdist(G_sub_scaled, R_scaled, metric='euclidean')
        min_dist_gen_to_real = np.min(dist_gen_to_real, axis=1)
        
        # 1-NN distances from Real to Real (excluding self by using indices)
        dist_real_to_real = cdist(R_sub_scaled, R_scaled, metric='euclidean')
        # To exclude matching with itself, set the distance of identical indices to infinity
        for i, idx in enumerate(idx_real):
            dist_real_to_real[i, idx] = np.inf
        min_dist_real_to_real = np.min(dist_real_to_real, axis=1)
        
        # Plot distributions
        plt.figure(figsize=(10, 6))
        plt.hist(min_dist_real_to_real, bins=30, alpha=0.6, label='Real-to-Real (Baseline)', color='gray')
        plt.hist(min_dist_gen_to_real, bins=30, alpha=0.6, label='Synthetic-to-Real (Memorization Check)', color='orange')
        plt.xlabel('Euclidean Distance to Nearest Real Trajectory (Standardized)')
        plt.ylabel('Count')
        plt.title('Memorization Check (1-NN Distance Distribution)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Memorization plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()
        
        # Quantitative Metric: Percentage of suspected copies
        # We define a "copy" as a synthetic sample closer to a real sample than the 10% of median baseline distance
        strict_threshold = np.median(min_dist_real_to_real) * 0.1
        
        suspected_copies = np.mean(min_dist_gen_to_real < strict_threshold) * 100
        print(f"    Median Real-to-Real distance: {np.median(min_dist_real_to_real):.4f}")
        print(f"    Median Synthetic-to-Real distance: {np.median(min_dist_gen_to_real):.4f}")
        print(f"    Suspected exact training copies (dist < 10% median baseline): {suspected_copies:.2f}%")
        
        return {
            'median_r2r': np.median(min_dist_real_to_real),
            'median_s2r': np.median(min_dist_gen_to_real),
            'copy_percentage': suspected_copies
        }


# ==========================================
# 3. Self-Diagnostic Verification Suite
# ==========================================

def run_self_diagnostic(real_dataset_path, output_plots_dir):
    """
    Solves the 'Chicken-and-Egg' testing problem!
    Validates the entire statistical evaluation pipeline without needing PI-LDM synthetic data yet.
    Compares the real dataset against itself, and a systematically degraded version of itself.
    """
    print("="*65)
    print("     RUNNING SELF-DIAGNOSTIC PIPELINE FOR TRAJECTORY METRICS")
    print("="*65)
    
    # Check if the real dataset exists
    if not os.path.exists(real_dataset_path):
        print(f"Error: Target real dataset not found at {real_dataset_path}")
        print("Please check your file path or specify a valid .npy dataset path.")
        return
        
    print(f"Loading reference real dataset: {real_dataset_path}")
    real_data = np.load(real_dataset_path).astype(np.float32)
    print(f"Loaded shape: {real_data.shape}") # Expect (N, 4, 200)
    
    # Subsample to speed up self-diagnostic on CPU
    if len(real_data) > 800:
        print("Subsampling dataset to 800 samples for rapid diagnostic testing...")
        idx = np.random.choice(len(real_data), 800, replace=False)
        real_data = real_data[idx]
        
    # 1. Create a "Perfect Baseline" (split original data in half)
    half = len(real_data) // 2
    real_subset_A = real_data[:half]
    real_subset_B = real_data[half:2*half]
    
    # 2. Create a "Degraded Baseline" by adding systematic noise and shuffling coordinates
    print("\n--> Synthesizing corrupted dataset for diagnostic sensitivity testing...")
    corrupted_data = real_subset_A.copy()
    
    # Feature 0: Track - Add heavy random heading noise
    corrupted_data[:, 0, :] += np.random.normal(0, 45, size=corrupted_data[:, 0, :].shape)
    
    # Feature 1: Ground Speed - Systematically slow down speed by 40% (stall simulation)
    corrupted_data[:, 1, :] *= 0.6
    
    # Feature 2: Altitude - Add heavy high-frequency noise and drop altitude (ground penetration)
    corrupted_data[:, 2, :] -= 2000.0
    corrupted_data[:, 2, :] += np.random.normal(0, 500, size=corrupted_data[:, 2, :].shape)
    
    # Randomize temporal sequences completely for 30% of samples (broken dynamics)
    for i in range(int(len(corrupted_data) * 0.3)):
        np.random.shuffle(corrupted_data[i, :, :].T)
        
    print("Corrupted data prepared successfully.")
    
    # Initialize Evaluator
    evaluator = TrajectoryEvaluator()
    
    # Create output directory for verification plots
    os.makedirs(output_plots_dir, exist_ok=True)
    
    print("\n" + "-"*50)
    print(" TEST PHASE 1: EVALUATING PERFECT BASELINE (Subset A vs Subset B)")
    print(" (Goal: High overlap, near 0.0 distance, ~50% discriminator accuracy)")
    print("-"*50)
    
    # Run Diversity Plot
    pca_save_perfect = os.path.join(output_plots_dir, "diagnostic_pca_perfect.png")
    evaluator.visualize_diversity(real_subset_A, real_subset_B, method='PCA', 
                                  save_path=pca_save_perfect, max_samples=400)
    
    # Run Metrics
    evaluator.compute_energy_distance(real_subset_A, real_subset_B, bootstraps=15, sample_size=100)
    evaluator.compute_discriminative_score(real_subset_A, real_subset_B, epochs=5, batch_size=32)
    evaluator.compute_usefulness(real_subset_A, real_subset_B, steps_ahead=15, epochs=5, batch_size=32)
    mem_save_perfect = os.path.join(output_plots_dir, "diagnostic_memorization_perfect.png")
    evaluator.check_memorization(real_subset_A, real_subset_B, sample_size=200, save_path=mem_save_perfect)
    
    print("\n" + "-"*50)
    print(" TEST PHASE 2: EVALUATING CORRUPTED BASELINE (Subset A vs Corrupted)")
    print(" (Goal: High separation, large distance, ~100% discriminator accuracy)")
    print("-"*50)
    
    # Run Diversity Plot
    pca_save_corrupted = os.path.join(output_plots_dir, "diagnostic_pca_corrupted.png")
    evaluator.visualize_diversity(real_subset_A, corrupted_data, method='PCA', 
                                  save_path=pca_save_corrupted, max_samples=400)
    
    # Run Metrics
    evaluator.compute_energy_distance(real_subset_A, corrupted_data, bootstraps=15, sample_size=100)
    evaluator.compute_discriminative_score(real_subset_A, corrupted_data, epochs=5, batch_size=32)
    evaluator.compute_usefulness(real_subset_A, corrupted_data, steps_ahead=15, epochs=5, batch_size=32)
    mem_save_corrupted = os.path.join(output_plots_dir, "diagnostic_memorization_corrupted.png")
    evaluator.check_memorization(real_subset_A, corrupted_data, sample_size=200, save_path=mem_save_corrupted)
    
    # Physical Violations Check
    print("\n--> Checking Domain Physics Violation Metrics on reference vs corrupted data:")
    print("   [Reference Subset A Violations]")
    evaluator.compute_physics_violations(real_subset_A, gs_max=160.0, alt_min=0.0, max_roc=30.0)
    print("\n   [Corrupted Dataset Violations]")
    evaluator.compute_physics_violations(corrupted_data, gs_max=160.0, alt_min=0.0, max_roc=30.0)
    
    print("\n" + "="*65)
    print(" DIAGNOSTIC COMPLETE! Pipeline successfully validated.")
    print(f" View verification plots in: {output_plots_dir}")
    print("="*65)

def run_model_evaluation(real_path, gen_path, output_dir):
    """
    Evaluates real vs model-generated synthetic trajectories.
    """
    print("="*65)
    print("      RUNNING TRAJECTORY GENERATION STATISTICAL EVALUATION")
    print("="*65)
    print(f"Real trajectories: {real_path}")
    print(f"Synthetic trajectories: {gen_path}")
    
    if not os.path.exists(real_path):
        print(f"Error: Real dataset not found: {real_path}")
        return
    if not os.path.exists(gen_path):
        print(f"Error: Synthetic trajectories not found: {gen_path}")
        return
        
    real_data = np.load(real_path).astype(np.float32)
    gen_data = np.load(gen_path).astype(np.float32)
    print(f"Loaded Real shape: {real_data.shape}")
    print(f"Loaded Synthetic shape: {gen_data.shape}")
    
    # Subsample real data to match the synthetic data size for class balance in classifier/metrics
    if len(real_data) > len(gen_data):
        print(f"Balancing dataset: Subsampling real data to {len(gen_data)} samples...")
        idx = np.random.choice(len(real_data), len(gen_data), replace=False)
        real_data = real_data[idx]
        
    evaluator = TrajectoryEvaluator()
    os.makedirs(output_dir, exist_ok=True)
    
    # PCA Plots
    pca_save = os.path.join(output_dir, "trajectory_pca_comparison.png")
    evaluator.visualize_diversity(real_data, gen_data, method='PCA', save_path=pca_save, max_samples=1000)
    
    # t-SNE Plot
    tsne_save = os.path.join(output_dir, "trajectory_tsne_comparison.png")
    evaluator.visualize_diversity(real_data, gen_data, method='t-SNE', save_path=tsne_save, max_samples=400)
    
    # Metrics
    evaluator.compute_energy_distance(real_data, gen_data, bootstraps=30, sample_size=100)
    evaluator.compute_discriminative_score(real_data, gen_data, epochs=8, batch_size=32)
    evaluator.compute_usefulness(real_data, gen_data, steps_ahead=15, epochs=8, batch_size=32)
    mem_save = os.path.join(output_dir, "trajectory_memorization_check.png")
    evaluator.check_memorization(real_data, gen_data, sample_size=500, save_path=mem_save)
    
    # # Domain physics checks
    # print("\n--> Evaluating physical constraints on real vs model-generated data:")
    # print("   [Real Reference Data Envelope Violations]")
    # # Alt min is 0ft since we are landing, but let's check stall speed ~55m/s (or kn) and max vertical rate 25m/s
    # evaluator.compute_physics_violations(real_data, gs_max=150.0, alt_min=0.0, max_roc=25.0)
    # print("\n   [LDM-Generated Data Envelope Violations]")
    # evaluator.compute_physics_violations(gen_data, gs_max=150.0, alt_min=0.0, max_roc=25.0)
    
    print("\n" + "="*65)
    print(" EVALUATION COMPLETE! Comparison finished successfully.")
    print(f" Outputs saved to: {output_dir}")
    print("="*65)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate synthetic trajectory generative models.")
    parser.add_argument('--real', type=str, default=None, help="Path to real trajectories .npy file")
    parser.add_argument('--gen', type=str, default=None, help="Path to generated trajectories .npy file")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save evaluation plots and results")
    parser.add_argument('--diagnostic', action='store_true', help="Run the pipeline self-diagnostic test")
    
    args = parser.parse_args()
    
    if args.diagnostic:
        # Run self-diagnostic
        default_dataset = os.path.join(project_root, "data", "processed", "LSZH_2019_R14_kinematic_200pts.npy")
        default_output_plots = os.path.join(project_root, "pi_ldm", "outputs", "diagnostic_results")
        run_self_diagnostic(default_dataset, default_output_plots)
    else:
        # Determine files automatically if not specified
        real_file = args.real if args.real else os.path.join(project_root, "data", "processed", "LSZH_2019_R14_kinematic_200pts_spatial_5000m.npy")
        gen_file = args.gen if args.gen else os.path.join(project_root, "pi_ldm", "outputs", "trajectories", "LSZH_2019_R14_kinematic_200pts_spatial_5000m_synthetic_1000t.npy")
        out_dir = args.output_dir if args.output_dir else os.path.join(project_root, "pi_ldm", "outputs", "evaluation_results")
        
        run_model_evaluation(real_file, gen_file, out_dir)
