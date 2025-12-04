# src/train.py

import os
import torch
from torch import optim

from .model import TinyGPT
from .char_tokenizer import CharTokenizer
from .dataset import CharDataset

# -----------------------
# 1) Paths & basic config
# -----------------------

# Raw text file (Tiny Shakespeare or any corpus)

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tiny_shakespeare.txt")
data_path = os.path.abspath(data_path)

# Where to save/load tokenizer
tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "char_tokenizer.json")
tokenizer_path = os.path.abspath(tokenizer_path)

os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# -----------------------
# 2) Model & training hyperparams
# -----------------------

# Hyperparameter: context length (how many tokens the model sees)
block_size   = 128

# Model size hyperparams
d_model      = 128
d_head       = 128
n_layers     = 4
dropout      = 0.1

# Training hyperparams
batch_size   = 64
max_iters    = 5000
eval_interval = 500
eval_iters    = 100
learning_rate = 3e-4

# Where to save model checkpoints
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "tiny_gpt.pt")

# -----------------------
# 3) Load text and tokenizer
# -----------------------

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# If tokenizer is already saved, reuse it.
# Otherwise, build from corpus and save.
if os.path.exists(tokenizer_path):
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = CharTokenizer.load(tokenizer_path)
else:
    print("Building new tokenizer from corpus...")
    tokenizer = CharTokenizer(text)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

vocab_size = tokenizer.vocab_size
print(f"Vocab size: {vocab_size}")

# -----------------------
# 4) Encode corpus to token ids
# -----------------------

# Here we DON'T wrap the entire corpus in BOS/EOS;
# it's one long stream of characters.
ids = torch.tensor(
    tokenizer.encode(text, add_special_tokens=False),
    dtype=torch.long,
)

# -----------------------
# 5) Train/val split + datasets
# -----------------------

n = int(0.9 * len(ids))
train_ids = ids[:n]
val_ids   = ids[n:]

train_dataset = CharDataset(train_ids, block_size) #initializing a dataset object
val_dataset   = CharDataset(val_ids, block_size) #initializing a dataset object with methods __len__ and __getitem__

def get_batch(split: str):
    """
    Sample a random batch of (x, y) from train or val dataset.
    x, y: [batch_size, block_size]
    """
    dataset = train_dataset if split == "train" else val_dataset
    # Sample random starting positions
    ix = torch.randint(len(dataset), (batch_size,)) # creating a tensor of random indices between 0 and len(data) of size batch_size
    # for each random index, get the corresponding (x, y) from the dataset
    x = torch.stack([dataset[i][0] for i in ix])  # [B, T] # stacking the input sequences (x) from the dataset
    y = torch.stack([dataset[i][1] for i in ix])  # [B, T] # stacking the target sequences (y) from the dataset
    return x.to(device), y.to(device)

# -----------------------
# 6) Create model + optimizer
# -----------------------

model = TinyGPT(
    vocab_size=vocab_size,
    d_model=d_model,
    block_size=block_size,
    n_layers=n_layers,
    dropout=dropout,
    d_head=d_head,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# ---------------------------------
# 7) Helper to estimate train/val loss
# ---------------------------------

@torch.no_grad()
def estimate_loss():
    """
    Run the model on several batches of train & val data
    and return the average loss for each.
    """
    model.eval()
    out = {}

    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)

    model.train()
    return out

# -----------------------
# 8) Main training loop
# -----------------------

def main():
    print(f"Training on device: {device}")
    print(f"vocab_size={vocab_size}, block_size={block_size}")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    for step in range(1, max_iters + 1):
        # Periodic evaluation + checkpointing
        if step % eval_interval == 0 or step == 1:
            losses = estimate_loss()
            print(
                f"step {step:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "d_model": d_model,
                        "block_size": block_size,
                        "n_layers": n_layers,
                        "dropout": dropout,
                    },
                },
                ckpt_path,
            )
            print(f"Checkpoint saved to {ckpt_path}")

        # ---- One training step ----
        x, y = get_batch("train")   # [B, T], [B, T]
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Optional frequent print
        if step % 100 == 0:
            print(f"step {step:5d} | train batch loss {loss.item():.4f}")

    print("Training finished!")

if __name__ == "__main__":
    main()

