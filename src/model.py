# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock  # same folder as model.py

class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        block_size: int,
        n_layers: int,
        d_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size
        self.d_head = d_head

        # 1) Token + positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding   = nn.Embedding(block_size, d_model)

        # 2) Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, dropout=dropout, d_head=d_head, block_size=block_size)
            for _ in range(n_layers)
        ])

        # 3) Final layernorm before LM head
        self.ln_f = nn.LayerNorm(d_model)

        # 4) LM head: maps hidden state → vocab logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # (Optional but common) Weight tying: share weights with token embedding
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx:      [batch_size, T] token indices
        targets:  [batch_size, T] next-token targets (optional)

        Returns:
          logits: [batch_size, T, vocab_size]
          loss:   scalar (if targets provided) else None
        """
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")

        # 1) Token + positional embeddings
        tok_emb = self.token_embedding(idx)  # [B, T, d_model]
        pos = torch.arange(T, device=idx.device)  # [T]
        pos_emb = self.pos_embedding(pos)[None, :, :]  # [1, T, d_model]

        x = tok_emb + pos_emb  # [B, T, d_model]

        # 2) Pass through N transformer blocks
        for block in self.blocks:
            x = block(x)

        # 3) Final layernorm
        x = self.ln_f(x)

        # 4) LM head
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Flatten for CE: [(B*T), vocab_size] vs [(B*T)]
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Simple autoregressive generation:
          start from idx [B, T], keep sampling next tokens.
        """
        for _ in range(max_new_tokens):
            # Crop to last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # Get logits for current sequence
            logits, _ = self(idx_cond)

            # Focus on last time step
            logits_last = logits[:, -1, :]  # [B, vocab_size]

            # Convert to probabilities & sample
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)

        return idx
