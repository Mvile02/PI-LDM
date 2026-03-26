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

class ResidualBlock1D(nn.Module):
    """
    Residual Block with FiLM (Feature-wise Linear Modulation) conditioning.
    """
    def __init__(self, in_channels, out_channels, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(in_channels, 8), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # FiLM Conditioning: Projects embedding to scale and shift parameters
        self.emb_proj = nn.Linear(emb_dim, out_channels * 2)
        
        self.norm2 = nn.GroupNorm(min(out_channels, 8), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        """
        x: (batch, in_channels, seq_len)
        emb: (batch, emb_dim)
        """
        h = self.conv1(F.gelu(self.norm1(x)))
        
        # Apply FiLM Conditioning
        emb_out = self.emb_proj(emb).unsqueeze(-1) # (batch, out_channels * 2, 1)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.conv2(self.dropout(F.gelu(self.norm2(h))))
        return h + self.shortcut(x)

class ConditionalUNet1D(nn.Module):
    """
    1D U-Net Denoiser with FiLM conditioning at every layer.
    """
    def __init__(self, state_dim, cond_dim, time_emb_dim=64, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder (Downsampling path)
        self.enc1 = ResidualBlock1D(state_dim, hidden_dims[0], time_emb_dim)
        self.enc2 = ResidualBlock1D(hidden_dims[0], hidden_dims[1], time_emb_dim)
        self.down = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock1D(hidden_dims[1], hidden_dims[2], time_emb_dim)
        
        # Decoder (Upsampling path with skip connections)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = ResidualBlock1D(hidden_dims[2] + hidden_dims[1], hidden_dims[1], time_emb_dim)
        self.dec2 = ResidualBlock1D(hidden_dims[1] + hidden_dims[0], hidden_dims[0], time_emb_dim)
        
        # Final output projection
        self.final_conv = nn.Conv1d(hidden_dims[0], state_dim, kernel_size=1)

    def forward(self, x, time, cond):
        """
        x: (batch, state_dim, seq_len)
        time: (batch,)
        cond: (batch, cond_dim)
        """
        # Embed time and condition
        t_emb = self.time_mlp(time)
        c_emb = self.cond_mlp(cond)
        emb = t_emb + c_emb # (batch, time_emb_dim)
        
        # Encoding
        x1 = self.enc1(x, emb)             # (B, hidden_dims[0], L)
        x2 = self.enc2(x1, emb)            # (B, hidden_dims[1], L)
        x_low = self.down(x2)              # (B, hidden_dims[1], L/2)
        
        # Bottleneck
        x_btn = self.bottleneck(x_low, emb) # (B, hidden_dims[2], L/2)
        
        # Decoding
        x_up = self.up(x_btn)              # (B, hidden_dims[2], L)
        
        # Skip connection: concat along channel dimension
        x_dec1 = self.dec1(torch.cat([x_up, x2], dim=1), emb) # (B, hidden_dims[1], L)
        x_dec2 = self.dec2(torch.cat([x_dec1, x1], dim=1), emb) # (B, hidden_dims[0], L)
        
        out = self.final_conv(x_dec2)
        return out
