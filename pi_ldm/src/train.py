import os
import sys

# Ensure the project root is in the system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import math
import glob
from pi_ldm.src.model import ConditionalUNet1D
from pi_ldm.src.physics import PhysicsLoss
from pi_ldm.src.dataset import get_dataloaders

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
else:
    # Use the local project structure
    MODELS_DIR = os.path.join(os.getcwd(), "pi_ldm", "models")

os.makedirs(MODELS_DIR, exist_ok=True)

class PILDMTrainer:
    def __init__(self, 
                 state_dim=4, 
                 cond_dim=3, 
                 timesteps=1000, 
                 beta_start=1e-4, 
                 beta_end=0.02,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.timesteps = timesteps
        
        # Diffusion schedules
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        # Model & Physics
        self.model = ConditionalUNet1D(state_dim=state_dim, cond_dim=cond_dim).to(device)
        self.physics_loss_fn = PhysicsLoss(dt=1.0, gamma1=1.0, gamma2=1.0, gamma3=1.0).to(device)
        
        # Weights for the physics-informed penalties
        self.lambda_physics = 0.1 
        self.lambda2_physics = 0.1
        self.lambda3_physics = 0.1

        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.mse_loss = nn.MSELoss()

    def save_checkpoint(self, epoch, loss, filename="checkpoint_latest.pth"):
        """Save model and optimizer state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        path = os.path.join(MODELS_DIR, filename)
        torch.save(checkpoint, path)
        # Also save a numbered checkpoint occasionally
        if epoch % 500 == 0:
            torch.save(checkpoint, os.path.join(MODELS_DIR, f"checkpoint_epoch_{epoch:05d}.pth"))
        print(f"--> Checkpoint saved: {path}")

    def load_checkpoint(self, filename="checkpoint_latest.pth"):
        """Load model and optimizer state if it exists"""
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            print(f"--> Loading checkpoint: {path}")
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'], checkpoint['loss']
        return 0, float('inf')

    def add_noise(self, x_0, t):
        """Forward diffusion process q(x_t | x_0)"""
        noise = torch.randn_like(x_0)
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise
        return x_t, noise

    def predict_x0(self, x_t, t, pred_noise):
        """Reverse process to get hat{x}_0 from x_t and predicted noise"""
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1)
        x0_hat = (x_t - torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t)
        return x0_hat

    def train_step(self, x_0, cond):
        self.optimizer.zero_grad()
        batch_size = x_0.shape[0]
        
        # 1. Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # 2. Add noise
        x_t, true_noise = self.add_noise(x_0, t)
        
        # 3. Predict noise
        pred_noise = self.model(x_t, t, cond)
        
        # 4. Standard Diffusion Loss
        loss_diff = self.mse_loss(pred_noise, true_noise)
        
        # 5. Physics-Informed Loss
        # We need the predicted trajectory x0_hat in physical space to compute physics loss
        x0_hat = self.predict_x0(x_t, t, pred_noise)
        
        # Denormalize to physical units before computing physics loss
        from pi_ldm.src.dataset import AircraftTrajectoryDataset
        x0_phys = AircraftTrajectoryDataset.denormalize(x0_hat)
        
        # Transpose from (batch, state_dim, seq_len) to (batch, seq_len, state_dim)
        trajectories = x0_phys.transpose(1, 2)
        
        loss_physics = self.physics_loss_fn(trajectories)
        
        # 6. Total Loss
        loss_total = loss_diff + self.lambda_physics * loss_physics
        
        loss_total.backward()
        self.optimizer.step()
        
        return loss_diff.item(), loss_physics.item(), loss_total.item()

def main():
    # Use current working directory as base
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data", "processed")
    FILE_BASE = "LSZH_2019_R14_kinematic_200pts_spatial_5000m_c1"
    
    # Load all available files
    train_loader, _ = get_dataloaders(data_dir, batch_size=32, file_base=FILE_BASE)
    
    trainer = PILDMTrainer()
    
    # --- Resume Training ---
    start_epoch, best_loss = trainer.load_checkpoint()
    
    print(f"Starting Training Loop from epoch {start_epoch}...")
    max_epochs = 20000  
    patience = 100       # Increased patience for long runs
    min_delta = 1e-5    
    
    early_stop_counter = 0
    checkpoint_interval = 50 # Save every 50 epochs
    
    for epoch in range(start_epoch, max_epochs):
        epoch_diff = 0
        epoch_phys = 0
        
        for batch_idx, (x_0, cond) in enumerate(train_loader):
            x_0 = x_0.to(trainer.device)
            cond = cond.to(trainer.device)
            
            l_diff, l_phys, l_tot = trainer.train_step(x_0, cond)
            epoch_diff += l_diff
            epoch_phys += l_phys
            
        avg_diff = epoch_diff / max(1, len(train_loader))
        avg_phys = epoch_phys / max(1, len(train_loader))
        avg_total = avg_diff + trainer.lambda_physics * avg_phys
            
        print(f"Epoch {epoch:04d} | Total: {avg_total:.5f} | Diff: {avg_diff:.5f} | Phys: {avg_phys:.5f}")
        
        # Convergence Check (Early Stopping)
        if best_loss - avg_total > min_delta:
            best_loss = avg_total
            early_stop_counter = 0
            # Could save best model weights here if desired
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\nConvergence reached! Early stopping triggered at epoch {epoch}.")
                print(f"Loss has not improved by more than {min_delta} for {patience} consecutive epochs.")
                break

        # Checkpoint Saving
        if epoch % checkpoint_interval == 0:
            trainer.save_checkpoint(epoch, avg_total)

    # Final Save
    output_path = os.path.join(MODELS_DIR, "final_model.pth")
    torch.save(trainer.model.state_dict(), output_path)
    print(f"Final model saved to {output_path}")

if __name__ == "__main__":
    main()
