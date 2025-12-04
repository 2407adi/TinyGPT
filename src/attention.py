# attention.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    Input:  x -> [B, T, d_model]
    Output: out -> [B, T, d_head]   (you can later project back to d_model)
    """

    def __init__(self, d_model: int, d_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)

        # Causal mask (lower-triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        returns: [B, T, d_head]
        """
        B, T, _ = x.shape

        # 1) Linear projections -> Q, K, V
        Q = self.W_q(x)  # [B, T, d_head]
        K = self.W_k(x)  # [B, T, d_head]
        V = self.W_v(x)  # [B, T, d_head]

        # 2) Scaled dot-product attention scores
        # scores[b, i, j] = dot(Q[b, i], K[b, j])
        scores = Q @ K.transpose(-2, -1)  # [B, T, T]
        scores = scores / math.sqrt(K.size(-1))

        # 3) Apply causal mask (no access to future tokens)
        # causal_mask[:T, :T] -> [T, T] for current sequence length
        mask = self.causal_mask[:T, :T]  # [T, T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 4) Softmax over allowed positions -> attention weights
        attn = F.softmax(scores, dim=-1)  # [B, T, T]
        attn = self.dropout(attn)

        # 5) Weighted sum of V -> context vectors
        out = attn @ V  # [B, T, d_head]

        return out
