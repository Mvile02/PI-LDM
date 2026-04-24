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
IN_COLAB = 'google.colab' in sys.modules

def mount_drive():
    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        return "/content/drive/MyDrive/TFM"
    return os.getcwd()

BASE_DIR = mount_drive()
if IN_COLAB:
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "trajectories")
else:
    # Use local project structure
    MODELS_DIR = os.path.join(os.getcwd(), "pi_ldm", "models")
    OUTPUTS_DIR = os.path.join(os.getcwd(), "pi_ldm", "outputs", "trajectories")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

class PILDMSampler:
    """
    Implements Online Sampling (Inference) via Guided SDE.
    dz = [ f(z,t) - g(t)^2 \nabla_z \log p_t(z|c) + \eta \nabla_z \Phi(z) ] dt + g(t) dw
    """
    def __init__(self, model_path=None, state_dim=4, cond_dim=3, seq_len=200, timesteps=1000, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.timesteps = timesteps
        
        self.model = ConditionalUNet1D(state_dim=state_dim, cond_dim=cond_dim).to(device)
        self.physics_fn = PhysicsLoss().to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        else:
            print("No model path provided or file missing, using random weights.")
            
        self.model.eval()

        # Same schedule as training
        beta_start, beta_end = 1e-4, 0.02
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def potential_function(self, x):
        """
        Phi(z): Calculates exactly the Distance to Feasibility.
        Uses the Physics Loss as the penalty landscape.
        """
        # x is in normalized space [-1, 1]. Denormalize for physics calculation.
        x_phys = AircraftTrajectoryDataset.denormalize(x)
        trajectories = x_phys.transpose(1, 2)
        # We use the total physics loss as the unnormalized potential
        phi = self.physics_fn(trajectories)
        return phi

    @torch.no_grad()
    def sample(self, cond, eta=0.01, enable_guidance=True):
        """
        Generates trajectories via denoising loop with Physics Guidance Term.
        cond: (batch, cond_dim)
        """
        batch_size = cond.shape[0]
        # Start from pure noise
        x_t = torch.randn((batch_size, self.state_dim, self.seq_len), device=self.device)
        
        for t_idx in reversed(range(0, self.timesteps)):
            time_tensor = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            
            # Predict noise epsilon_theta
            pred_noise = self.model(x_t, time_tensor, cond)
            
            # Physics Guidance Step
            if enable_guidance:
                with torch.enable_grad():
                    x_t_grad = x_t.clone().detach().requires_grad_(True)
                    # We compute the potential on the predicted x0
                    alpha_hat_t = self.alpha_hat[t_idx]
                    x0_hat = (x_t_grad - math.sqrt(1 - alpha_hat_t) * self.model(x_t_grad, time_tensor, cond)) / math.sqrt(alpha_hat_t)
                    
                    phi = self.potential_function(x0_hat)
                    
                    # Compute gradient w.r.t x_t \nabla_{x_t} \Phi(x_t)
                    grad_phi = torch.autograd.grad(phi, x_t_grad)[0]
                
                # Apply the guidance term
                guidance = eta * grad_phi
            else:
                guidance = 0.0

            # Denoising step (Simplified DDPM / Euler-Maruyama step)
            alpha_t = self.alpha[t_idx]
            alpha_hat_t = self.alpha_hat[t_idx]
            
            z = torch.randn_like(x_t) if t_idx > 0 else 0.0
            
            # Update step including the guidance
            x_t = (1 / math.sqrt(alpha_t)) * (x_t - pred_noise * (1 - alpha_t) / math.sqrt(1 - alpha_hat_t)) \
                  - guidance \
                  + math.sqrt(self.beta[t_idx]) * z
        
        # Denormalize output to physical space before returning
        x_final = AircraftTrajectoryDataset.denormalize(x_t)
        return x_final

def main():
    print("Initializing sampler...")
    # Look for model in root
    model_path = os.path.join(MODELS_DIR, "final_model.pth")
    # Fallback to checkpoint if final doesn't exist
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "checkpoint_latest.pth")
        
    save_file = "synthetic_cluster_trajectories"
    
    sampler = PILDMSampler(model_path=model_path)
    
    # Generate x trajectories for a single condition (e.g., Airport: LSZH (0), Type: A320 (0), Weather: 0)
    num_samples = 30
    cond = torch.tensor([[0.0, 0.0, 0.0]], device=sampler.device).repeat(num_samples, 1)
    
    print(f"Generating {num_samples} trajectories without physics guidance...")
    trajectories = sampler.sample(cond, enable_guidance=False)
    print("Generated shape:", trajectories.shape)

    # Save the generated trajectories
    # output_dir = os.path.join(base_dir, "pi_ldm", "outputs", "trajectories") # Old path
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
