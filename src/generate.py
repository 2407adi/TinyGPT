# src/generate.py

import os
import torch
import torch.nn.functional as F

from .model import TinyGPT
from .char_tokenizer import CharTokenizer

# -----------------------
# 1) Paths & device
# -----------------------

# Where we saved things during training
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "tiny_gpt.pt")

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "char_tokenizer.json")
TOKENIZER_PATH = os.path.abspath(TOKENIZER_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


# -----------------------
# 2) Sampling helpers
# -----------------------

def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None):
    """
    logits: [vocab_size] (1D tensor) for the *next* token.
    temperature: >0. Lower = more greedy, higher = more random.
    top_k: if set, restrict sampling to top_k tokens.
    """
    # Scale by temperature
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    logits = logits / temperature

    # Optional top-k filtering
    if top_k is not None:
        # Keep only top_k logits, set the rest to -inf
        v, ix = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask[ix] = v
        logits = mask

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)  # [vocab_size]

    # Sample a token id
    next_id = torch.multinomial(probs, num_samples=1)  # [1]
    return next_id.item()


def generate_tokens(
    model: TinyGPT,
    start_ids: list[int],
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    """
    Autoregressively generate new tokens.

    model: trained TinyGPT model
    start_ids: list of token ids to start from
    max_new_tokens: how many tokens to append
    block_size: context window of the model
    """

    model.eval()
    # Current sequence of ids as a tensor [1, T]
    idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens (context window)
            idx_cond = idx[:, -block_size:]

            # Forward pass: logits [1, T, vocab_size]
            logits, _ = model(idx_cond)

            # Take logits at the last time step: [vocab_size]
            logits_last = logits[:, -1, :].squeeze(0)

            # Sample next token id
            next_id = sample_next_token(
                logits_last,
                temperature=temperature,
                top_k=top_k,
            )

            # Append to the sequence
            next_id_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
            idx = torch.cat([idx, next_id_tensor], dim=1)

    # Return as a simple Python list
    return idx.squeeze(0).tolist()


# -----------------------
# 3) Main: load & generate
# -----------------------

def main():
    print(f"Using device: {device}")

    # 3.1 Load tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
    vocab_size = tokenizer.vocab_size
    print(f"Loaded tokenizer with vocab size = {vocab_size}")

    # 3.2 Load checkpoint
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    config = ckpt["config"]
    print("Loaded checkpoint config:", config)

    # NOTE: d_head wasn't stored in your config dict – we know from train.py it was 128.
    d_model = config["d_model"]
    block_size = config["block_size"]
    n_layers = config["n_layers"]
    dropout = config["dropout"]
    d_head = d_model  # in your setup, you used d_head = d_model = 128

    # 3.3 Recreate model & load weights
    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        block_size=block_size,
        n_layers=n_layers,
        dropout=dropout,
        d_head=d_head,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded and ready for generation.\n")

    # 3.4 Define a prompt
    # You can later swap this for input() to take user prompt from CLI.
    prompt = "ROMEO:\n But, soft! what light through yonder window breaks?\n It is the east, and Juliet is the sun.\n"
    print(f"Prompt: {repr(prompt)}")

    # 3.5 Encode prompt to token ids
    start_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # 3.6 Generate
    max_new_tokens = 500
    temperature = 0.8
    top_k = 50

    print(
        f"Generating {max_new_tokens} new tokens "
        f"(temperature={temperature}, top_k={top_k})..."
    )

    out_ids = generate_tokens(
        model=model,
        start_ids=start_ids,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
        top_k=top_k,
    )

    # 3.7 Decode back to text
    out_text = tokenizer.decode(out_ids)
    print("\n================= GENERATED TEXT =================\n")
    print(out_text)
    print("\n==================================================\n")


if __name__ == "__main__":
    main()
