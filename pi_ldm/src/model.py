import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalUNet1D(nn.Module):
    """
    1D U-Net Denoiser for Latent Diffusion Models with conditioning.
    Takes latent state z_t, diffusion time t, and conditions c={R,A,W}.
    """
    def __init__(self, state_dim, cond_dim, time_emb_dim=32, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim),
            nn.GELU()
        )

        # Simplified Encoder (downsampling)
        self.enc1 = nn.Conv1d(state_dim, hidden_dims[0], kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1)
        
        # Simplified Decoder (upsampling)
        self.dec1 = nn.ConvTranspose1d(hidden_dims[2], hidden_dims[1], kernel_size=4, padding=1, stride=2)
        # Concatenation of x_up (128) and x1 (64) is 192, not 256
        self.dec2 = nn.Conv1d(hidden_dims[1] + hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1)
        
        # Final output projection matches state dimension
        self.final_conv = nn.Conv1d(hidden_dims[0], state_dim, kernel_size=1)

    def forward(self, x, time, cond):
        """
        x: (batch, state_dim, seq_len)
        time: (batch,)
        cond: (batch, cond_dim) - runway, aircraft, weather variables
        """
        t_emb = self.time_mlp(time)
        c_emb = self.cond_mlp(cond)
        
        # Combine time and condition embeddings
        emb = t_emb + c_emb  # (batch, time_emb_dim)
        emb = emb.unsqueeze(-1) # (batch, time_emb_dim, 1) to broadcast over sequence

        # Encoding
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        
        # Bottleneck
        x_btn = F.relu(self.bottleneck(x2))
        
        # Ensure embedding dimension matches bottleneck if adding sequentially, 
        # or simplify by just combining at bottleneck:
        # (Assuming careful projection in a full implementation)
        
        # Decoding
        x_up = F.relu(self.dec1(x_btn))
        
        # Skip connection
        # Need to ensure dimensions align (padding if needed)
        if x_up.shape[-1] != x1.shape[-1]:
            x_up = F.interpolate(x_up, size=x1.shape[-1])
            
        x_concat = torch.cat([x_up, x1], dim=1)
        x_out = F.relu(self.dec2(x_concat))
        
        out = self.final_conv(x_out)
        return out
