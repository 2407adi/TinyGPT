# positional_embedding.py

import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (GPT-style).

    For each position t in [0, T), we add a learned vector p_t of size d_model.

    Input / Output shape: [B, T, d_model]
    """

    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        self.block_size = block_size
        # One learned vector per position 0..block_size-1
        self.pos_emb = nn.Embedding(block_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        returns: [B, T, d_model] with position info added
        """
        B, T, D = x.shape
        assert T <= self.block_size, "Sequence length > block_size"

        # positions: [0, 1, 2, ..., T-1]
        positions = torch.arange(T, device=x.device)  # [T]
        # lookup position embeddings -> [T, d_model], then add batch dim -> [1, T, d_model]
        pos_emb = self.pos_emb(positions).unsqueeze(0)  # [1, T, d_model]

        # broadcast over batch: [B, T, d_model] + [1, T, d_model]
        return x + pos_emb
