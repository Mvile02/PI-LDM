import math
import os
import sys

# Ensure the project root is in the system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import numpy as np
from pi_ldm.src.model import ConditionalUNet1D
from pi_ldm.src.physics import PhysicsLoss
from pi_ldm.src.dataset import AircraftTrajectoryDataset

# --- Colab / Drive Setup ---
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def mount_drive():
    if IN_COLAB:
        print("--> Environment detected: Google Colab")
        # Drive must be mounted manually in a Colab cell before running this script
        path = "/content/drive/MyDrive/TFM"
        if not os.path.exists(path):
             print(f"!! WARNING: Path not found in Drive: {path}")
             print("!! Ensure you have mounted Drive in a cell and the path exists.")
        return path
    else:
        print("--> Environment detected: Local PC")
        return os.getcwd()

BASE_DIR = mount_drive()
if IN_COLAB:
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "trajectories")
else:
    # Use project root structure
    MODELS_DIR = os.path.join(project_root, "pi_ldm", "models")
    OUTPUTS_DIR = os.path.join(project_root, "pi_ldm", "outputs", "trajectories")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
print(f"--> Models will be loaded from: {MODELS_DIR}")
print(f"--> Outputs will be saved to: {OUTPUTS_DIR}")

class PILDMSampler:
    """
    Implements Online Sampling (Inference) via Guided SDE.
    dz = [ f(z,t) - g(t)^2 \nabla_z \log p_t(z|c) + \eta \nabla_z \Phi(z) ] dt + g(t) dw
    """
    def __init__(self, model_path=None, state_dim=4, cond_dim=3, seq_len=200, timesteps=1000, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', ac_types=None, enable_physics=True):
        self.device = device
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.timesteps = timesteps
        self.ac_types = ac_types
        self.enable_physics = enable_physics
        
        self.model = ConditionalUNet1D(state_dim=state_dim, cond_dim=cond_dim).to(device)
        
        # Physics Guidance (Enabled by default)
        if self.enable_physics:
            self.physics_fn = PhysicsLoss(ac_types=ac_types).to(device)
        
        if model_path and os.path.exists(model_path):
            loaded_data = torch.load(model_path, map_location=device)
            # Handle both raw state_dicts (final models) and checkpoint dictionaries
            if "model_state_dict" in loaded_data:
                self.model.load_state_dict(loaded_data["model_state_dict"])
            else:
                self.model.load_state_dict(loaded_data)
            print(f"Loaded model from {model_path}")
        else:
            if self.enable_physics:
                raise FileNotFoundError(
                    f"Error: Physics-trained model weights not found at '{model_path}'. "
                    "Please train the model with physics active first to generate these weights."
                )
            else:
                raise FileNotFoundError(
                    f"Error: Standard model weights not found at '{model_path}'. "
                    "Please train the standard model first to generate these weights."
                )
            
        self.model.eval()

        # Same schedule as training
        beta_start, beta_end = 1e-4, 0.02
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def potential_function(self, x, cond=None):
        """
        Phi(z): Calculates exactly the Distance to Feasibility.
        Uses the Physics Loss as the penalty landscape.
        """
        if getattr(self, 'physics_fn', None) is None:
            return torch.tensor(0.0, device=self.device)
        # x is in normalized space [-1, 1]. Denormalize for physics calculation.
        x_phys = AircraftTrajectoryDataset.denormalize(x)
        trajectories = x_phys.transpose(1, 2)
        # We use the total physics loss as the unnormalized potential
        phi = self.physics_fn(trajectories, cond)
        return phi

    @torch.no_grad()
    def sample(self, cond, eta=0.01, enable_guidance=True, exact_grad=True):
        """
        Generates trajectories via denoising loop (Standard DDPM or Guided).
        cond: (batch, cond_dim)
        exact_grad: If True, backpropagates through the U-Net for exact gradients. 
                    If False, uses first-order approximation (faster, detach model).
        """
        batch_size = cond.shape[0]
        # Start from pure noise
        x_t = torch.randn((batch_size, self.state_dim, self.seq_len), device=self.device)
        
        for t_idx in reversed(range(0, self.timesteps)):
            time_tensor = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            
            # Predict noise epsilon_theta
            pred_noise = self.model(x_t, time_tensor, cond)
            
            # Physics Guidance Step
            if enable_guidance and self.enable_physics:
                with torch.enable_grad():
                    x_t_grad = x_t.clone().detach().requires_grad_(True)
                    alpha_hat_t = self.alpha_hat[t_idx]
                    
                    if exact_grad:
                        # Exact gradient (backpropagates through the U-Net)
                        x0_hat = (x_t_grad - math.sqrt(1 - alpha_hat_t) * self.model(x_t_grad, time_tensor, cond)) / math.sqrt(alpha_hat_t)
                    else:
                        # First-order approximation (bypasses U-Net backprop)
                        pred_noise_detached = pred_noise.detach()
                        x0_hat = (x_t_grad - math.sqrt(1 - alpha_hat_t) * pred_noise_detached) / math.sqrt(alpha_hat_t)
                    
                    phi = self.potential_function(x0_hat, cond)
                    
                    # Compute gradient w.r.t x_t \nabla_{x_t} \Phi(x_t)
                    # phi is now a tensor of shape (batch_size,), so we sum it to get a scalar for autograd
                    grad_phi = torch.autograd.grad(phi.sum(), x_t_grad)[0]
                
                # Apply the guidance term
                # Scale gradient to prevent it from destroying the generated trajectory
                grad_norm = torch.norm(grad_phi.reshape(batch_size, -1), dim=-1).view(-1, 1, 1) + 1e-8
                grad_phi_normalized = grad_phi / grad_norm
                
                # The guidance step should scale with the variance of the current timestep
                # to prevent large jumps at the end of the generation process
                guidance = eta * self.beta[t_idx] * grad_phi_normalized
            else:
                guidance = 0.0

            # Denoising step (Standard DDPM / Euler-Maruyama)
            alpha_t = self.alpha[t_idx]
            alpha_hat_t = self.alpha_hat[t_idx]
            
            z = torch.randn_like(x_t) if t_idx > 0 else 0.0
            
            # Update step
            x_t = (1 / math.sqrt(alpha_t)) * (x_t - pred_noise * (1 - alpha_t) / math.sqrt(1 - alpha_hat_t)) \
                  - guidance \
                  + math.sqrt(self.beta[t_idx]) * z
        
        # Denormalize output to physical space before returning
        x_final = AircraftTrajectoryDataset.denormalize(x_t)
        return x_final

def main():
    FILE_BASE = "LSZH_2019_R14_kinematic_200pts_spatial_5000m_c1"
    ENABLE_PHYSICS = True  # Set to False to load/sample from the standard non-physics model

    print("Initializing sampler...")
    
    # Determine the model name dynamically based on the ENABLE_PHYSICS flag
    if ENABLE_PHYSICS:
        model_name = f"{FILE_BASE}_physics"
        model_path = os.path.join(MODELS_DIR, f"{model_name}_final_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint_latest.pth")
    else:
        model_name = FILE_BASE
        model_path = os.path.join(MODELS_DIR, f"{model_name}_final_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint_latest.pth")
    
    # Load aircraft type list from dataset CSV metadata to align categories
    csv_path = os.path.join(project_root, "data", "processed", f"{FILE_BASE}.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(project_root, "data", "clusters", f"{FILE_BASE}.csv")
    
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        ac_types = sorted(df['typecode'].astype(str).unique())
        
        # Calculate realistic distribution (empirical probabilities)
        type_counts = df['typecode'].astype(str).value_counts(normalize=True)
        type_probs = [type_counts.get(ac, 0.0) for ac in ac_types]
    else:
        ac_types = None
        type_probs = None
        
    sampler = PILDMSampler(model_path=model_path, ac_types=ac_types, enable_physics=ENABLE_PHYSICS)
    
    num_samples = 1000
    save_file = f"{model_name}_synthetic_{num_samples}t"
    
    # Generate condition tensor based on realistic distribution
    if type_probs is not None:
        # Sample aircraft indices using the real probabilities from the dataset
        sampled_indices = np.random.choice(len(ac_types), size=num_samples, p=type_probs)
    else:
        sampled_indices = np.zeros(num_samples) # Fallback to index 0
        
    cond = torch.zeros((num_samples, 3), device=sampler.device)
    cond[:, 0] = 0.0  # Airport (e.g., LSZH is 0)
    cond[:, 1] = torch.tensor(sampled_indices, dtype=torch.float32, device=sampler.device)
    cond[:, 2] = 0.0  # Weather
    
    print(f"Generating {num_samples} trajectories with Physics Guidance={ENABLE_PHYSICS}...")
    trajectories = sampler.sample(cond, enable_guidance=ENABLE_PHYSICS)
    print("Generated shape:", trajectories.shape)

    # Save the generated trajectories
    output_dir = OUTPUTS_DIR
    
    # Convert to numpy and save
    traj_np = trajectories.detach().cpu().numpy()
    save_path = os.path.join(output_dir, save_file + ".npy")
    np.save(save_path, traj_np)
    print(f"Trajectories saved to {save_path}")

    # Save metadata for consistency with plot_map.py
    import pandas as pd
    # We use columns expected by dataset.py/plot_map.py
    meta_df = pd.DataFrame(cond.cpu().numpy(), columns=['airport', 'typecode', 'weather'])
    meta_save_path = os.path.join(output_dir, save_file + ".csv")
    meta_df.to_csv(meta_save_path, index=False)
    print(f"Metadata saved to {meta_save_path}")

if __name__ == "__main__":
    main()
