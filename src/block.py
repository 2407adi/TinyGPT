import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SingleHeadSelfAttention

class FeedForwardMLP(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * hidden_mult
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x  # [B, T, d_model]

class TransformerBlock(nn.Module):
    """
    A single transformer block composed of:
      - Pre-LayerNorm
      - Single-head causal self-attention
      - Output projection (d_head -> d_model)
      - Residual add + dropout
      - Pre-LayerNorm
      - Feed-forward MLP
      - Residual add + dropout

    Input/Output shape: [B, T, d_model]
    """
    def __init__(
        self,
        d_model: int,
        d_head: int,
        block_size: int,
        mlp_hidden_mult: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SingleHeadSelfAttention(d_model=d_model, d_head=d_head, block_size=block_size, dropout=dropout)
        # project attention output back to d_model
        self.out_proj = nn.Linear(d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForwardMLP(d_model=d_model, hidden_mult=mlp_hidden_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        returns: [B, T, d_model]
        """
        # ----- Attention block (with pre-LN) -----
        residual = x  # [B, T, d_model]
        x_ln = self.ln1(x)  # [B, T, d_model]

        attn_out = self.attn(x_ln)  # [B, T, d_head]
        attn_out = self.out_proj(attn_out)  # [B, T, d_model]
        attn_out = self.dropout(attn_out)

        x = residual + attn_out  # [B, T, d_model] (first residual)

        # ----- Feed-forward block (with pre-LN) -----
        residual2 = x
        x_ln2 = self.ln2(x)  # [B, T, d_model]
        mlp_out = self.mlp(x_ln2)  # [B, T, d_model]
        mlp_out = self.dropout(mlp_out)

        x = residual2 + mlp_out  # [B, T, d_model] (second residual)

        return x
