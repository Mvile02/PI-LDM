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
        trajectories = x.transpose(1, 2)
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

        return x_t

def main():
    print("Initializing sampler...")
    # Look for model in root
    base_dir = os.getcwd()
    model_path = os.path.join(base_dir, "pi_ldm", "models", "test_model.pth")
    
    sampler = PILDMSampler(model_path=model_path)
    cond = torch.tensor([[1.0, 0.5, -0.2]], device=sampler.device) # Dummy R, A, W
    print("Generating trajectories without physics guidance...")
    trajectory = sampler.sample(cond, enable_guidance=False)
    print("Generated shape:", trajectory.shape)

    # Save the generated trajectory
    output_dir = os.path.join(base_dir, "outputs", "trajectories")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy and save
    traj_np = trajectory.detach().cpu().numpy()
    save_path = os.path.join(output_dir, "sample_trajectory.npy")
    np.save(save_path, traj_np)
    print(f"Trajectory saved to {save_path}")

    # Save metadata for consistency with plot_map.py
    import pandas as pd
    meta_df = pd.DataFrame(cond.cpu().numpy(), columns=['R', 'A', 'W'])
    meta_save_path = os.path.join(output_dir, "sample_trajectory.csv")
    meta_df.to_csv(meta_save_path, index=False)
    print(f"Metadata saved to {meta_save_path}")

if __name__ == "__main__":
    main()
