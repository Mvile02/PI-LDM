import torch
from pi_ldm.src.model import ConditionalUNet1D
from pi_ldm.src.physics import PhysicsLoss
import math

class PILDMSampler:
    """
    Implements Online Sampling (Inference) via Guided SDE.
    dz = [ f(z,t) - g(t)^2 \nabla_z \log p_t(z|c) + \eta \nabla_z \Phi(z) ] dt + g(t) dw
    """
    def __init__(self, model_path=None, state_dim=6, cond_dim=3, seq_len=200, timesteps=1000, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.timesteps = timesteps
        
        self.model = ConditionalUNet1D(state_dim=state_dim, cond_dim=cond_dim).to(device)
        self.physics_fn = PhysicsLoss().to(device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
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
    sampler = PILDMSampler()
    cond = torch.tensor([[1.0, 0.5, -0.2]], device=sampler.device) # Dummy R, A, W
    print("Generating trajectories with physics guidance...")
    trajectory = sampler.sample(cond, eta=0.05)
    print("Generated shape:", trajectory.shape)

if __name__ == "__main__":
    main()
