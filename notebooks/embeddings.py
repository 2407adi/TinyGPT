# src/embeddings.py

import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    """
    Combines:
    - Token embedding  : converts token IDs -> vectors
    
    - Positional embed : gives model sense of order in sequence

    - vocab_size: number of unique tokens; too small = poor text expressiveness, too large = slow/expensive embeddings.

    - d_model: hidden/embedding dimensionality; too small = low model capacity, too large = slow training + needs more data.

    - block_size: max context length; too small = short memory, too large = attention becomes expensive (O(n^2)).
    
    Output shape: [batch_size, block_size, d_model]
    """

    def __init__(self, vocab_size: int, d_model: int, block_size: int):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        self.block_size = block_size
        self.d_model = d_model

    def forward(self, idx: torch.Tensor):
        """
        idx: [batch_size, T] of token IDs, where T <= block_size
        """
        B, T = idx.shape
        assert T <= self.block_size

        tok_vectors = self.token_emb(idx)               # [B, T, d_model]

        pos = torch.arange(T, device=idx.device)        # [T]
        pos_vectors = self.pos_emb(pos)[None, :, :]     # [1, T, d_model]

        x = tok_vectors + pos_vectors                   # [B, T, d_model]
        return x
