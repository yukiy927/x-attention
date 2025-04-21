import torch
import numpy as np
import matplotlib.pyplot as plt

def apply_rope(x, seq_len):
    """Applies RoPE to a [seq_len, dim] tensor."""
    dim = x.shape[-1]
    half_dim = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half_dim) / half_dim).to(x.device)
    positions = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, 1]
    angles = positions * freqs.unsqueeze(0)  # [seq_len, half_dim]

    # RoPE rotation: split and apply sin/cos
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated

# Parameters
seq_len = 2048
dim = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random (or unit) base query/key vector, shared across positions
base_q = torch.ones(dim, device=device)
base_k = torch.ones(dim, device=device)

# Apply RoPE position embeddings
Q = apply_rope(base_q.unsqueeze(0).repeat(seq_len, 1), seq_len)  # [seq_len, dim]
K0 = apply_rope(base_k.unsqueeze(0), 1)  # [1, dim] — position 0 key

# Compute attention scores between all Q[i] and K[0]
scores = (Q * K0).sum(dim=-1).cpu().numpy()  # shape: [seq_len]

# Plot the score
plt.figure(figsize=(10, 4))
plt.plot(scores)
plt.title("Attention score: query@i vs key@position=0 (RoPE)")
plt.xlabel("Query Position i")
plt.ylabel("Dot Product with Key@0")
plt.grid(True)
plt.tight_layout()
plt.show()
