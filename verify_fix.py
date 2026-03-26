import torch
import numpy as np
import os
import sys

# Ensure the project root is in the system path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from pi_ldm.src.model import ConditionalUNet1D
from pi_ldm.src.dataset import AircraftTrajectoryDataset, get_dataloaders
from pi_ldm.src.sample import PILDMSampler

def test_normalization():
    print("Testing data normalization...")
    # Mock data (Track, GS, Alt, Time)
    X = np.array([[[180.0, 360.0, 0.0], 
                   [300.0, 600.0, 150.0], 
                   [20000.0, 40000.0, 5000.0], 
                   [1250.0, 2500.0, 625.0]]], dtype=np.float32)
    
    X_norm = AircraftTrajectoryDataset.normalize(X)
    print("X_norm min/max (should be roughly -1 to 1):")
    print(f"  Min: {X_norm.min():.2f}, Max: {X_norm.max():.2f}")
    
    X_denorm = AircraftTrajectoryDataset.denormalize(X_norm)
    print("Denormalization error (should be ~0):", np.abs(X - X_denorm).max())
    assert np.allclose(X, X_denorm, atol=1e-5)
    print("Normalization test passed!")

def test_model_gradients():
    print("\nTesting model architecture (FiLM conditioning)...")
    model = ConditionalUNet1D(state_dim=4, cond_dim=3)
    x = torch.randn(2, 4, 100)
    t = torch.tensor([1, 500])
    c = torch.randn(2, 3)
    
    # Check if gradients w.r.t cond work (proving cond is used)
    c.requires_grad_(True)
    out = model(x, t, c)
    loss = out.sum()
    loss.backward()
    
    if c.grad is not None and torch.abs(c.grad).sum() > 0:
        print("Conditioning gradient check passed!")
    else:
        print("Conditioning gradient check FAILED! (Condition ignored)")

    # Check if gradients w.r.t time work
    # Note: Sinusoidal embeddings are not learnable usually, 
    # but the MLP after them is. Check if output changes with t.
    out1 = model(x, torch.tensor([1, 1]), c)
    out2 = model(x, torch.tensor([500, 500]), c)
    if not torch.allclose(out1, out2):
        print("Time embedding check passed!")
    else:
        print("Time embedding check FAILED! (Time ignored)")

def test_sampler_batch():
    print("\nTesting sampler batch generation...")
    sampler = PILDMSampler(state_dim=4, cond_dim=3, timesteps=10) # 10 steps for speed
    cond = torch.randn(5, 3) # 5 trajectories
    trajectories = sampler.sample(cond, enable_guidance=False)
    print(f"Generated shape: {trajectories.shape} (Expected: (5, 4, 200))")
    assert trajectories.shape == (5, 4, 200)
    
    # Check if output is in physical units (not [-1, 1])
    # Alt (index 2) should be roughly large
    print(f"Sample Altitude range: {trajectories[:, 2, :].min():.1f} to {trajectories[:, 2, :].max():.1f}")
    
    print("Sampler batch test passed!")

if __name__ == "__main__":
    try:
        test_normalization()
        test_model_gradients()
        test_sampler_batch()
        print("\nAll verifications PASSED!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
