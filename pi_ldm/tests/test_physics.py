import unittest
import torch
from pi_ldm.src.physics import PhysicsLoss

class TestPhysicsLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = PhysicsLoss()
        
    def test_eom_loss_requires_grad(self):
        # Mock sequence (batch_size, seq_len, state_dim)
        traj = torch.randn(2, 5, 6, requires_grad=True)
        loss = self.loss_fn(traj)
        
        self.assertTrue(loss.requires_grad, "Physics loss must maintain gradients.")
        
    def test_zero_violation(self):
        # We assume that for a perfectly flying trajectory (which currently returns 0 in skeleton), loss is 0
        traj = torch.zeros(1, 2, 6, requires_grad=True)
        loss = self.loss_fn(traj)
        self.assertEqual(loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
