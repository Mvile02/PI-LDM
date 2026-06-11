import unittest
import torch
from pi_ldm.src.physics import PhysicsLoss

class TestPhysicsLoss(unittest.TestCase):
    def setUp(self):
        # Initialize with A320 default properties
        self.loss_fn = PhysicsLoss(ac_types=['A320'])
        
    def test_eom_loss_requires_grad(self):
        # Mock sequence (batch_size, seq_len, state_dim) with 4 features:
        # [track, gs, alt, time]
        traj = torch.randn(2, 5, 4, requires_grad=True)
        # Add a realistic time channel to avoid division by zero or negative dt
        traj.data[:, :, 3] = torch.arange(5.0).view(1, 5)
        # Add positive altitude and speed values to avoid issues in atmospheric equations
        traj.data[:, :, 1] = 150.0 # knots
        traj.data[:, :, 2] = 5000.0 # feet
        
        loss = self.loss_fn(traj)
        
        self.assertTrue(loss.requires_grad, "Physics loss must maintain gradients.")
        
    def test_valid_trajectory_zero_loss(self):
        # A perfectly flyable straight-and-level trajectory at typical landing speed
        # track=0 deg, speed=150 kt, altitude=5000 ft, time=0s and 10s
        traj = torch.tensor([[[0.0, 150.0, 5000.0, 0.0], 
                              [0.0, 150.0, 5000.0, 10.0]]], requires_grad=True)
        
        loss = self.loss_fn(traj)
        self.assertEqual(loss.item(), 0.0, f"Valid trajectory should violate no physics, got loss {loss.item()}")

if __name__ == '__main__':
    unittest.main()
