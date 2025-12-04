# TinyGPT

A small, educational character-level GPT implemented in PyTorch.
This repository contains a compact Transformer-based language model (TinyGPT) plus training and generation scripts designed for learning, experimentation, and quick iteration (e.g., on Tiny Shakespeare or your own character corpus).

Highlights
- Minimal, readable implementation of token/pos embeddings, transformer blocks, LM head, and training loop.
- Trainer (`src/train.py`) that builds a character tokenizer, trains and checkpoints the model.
- Generator (`src/generate.py`) that loads the tokenizer + checkpoint and autoregressively samples text.
- Good starting point for experiments and learning about transformers.

---

## Requirements

- Python 3.10+ (uses union types like `X | Y`)
- PyTorch (CPU or CUDA) — tested with torch 2.x but 1.13+ should work.
- Optional: GPU + CUDA for faster training.

Suggested install (CPU-only example):
```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For CUDA builds follow the official PyTorch install instructions for your CUDA version:
https://pytorch.org/get-started/locally/

---

## Quickstart

From the project root (the folder that contains the `TinyGPT/` directory):

Run training (default config in `src/train.py`):
```bash
# recommended: run as a module so relative imports work
python -m TinyGPT.src.train
```

Generate text using the trained model:
```bash
python -m TinyGPT.src.generate
```

Alternative (if you prefer running the script file directly):
```bash
# Make sure Python can resolve package relative imports:
PYTHONPATH=. python TinyGPT/src/train.py
PYTHONPATH=. python TinyGPT/src/generate.py
```

---

## Typical file locations / outputs

- `TinyGPT/data/tiny_shakespeare.txt` — default example corpus (character-level).
- `TinyGPT/artifacts/char_tokenizer.json` — tokenizer saved by training script.
- `TinyGPT/checkpoints/tiny_gpt.pt` — saved model checkpoint during training.
- `TinyGPT/src/` — source code:
	- `train.py` — training script
	- `generate.py` — generation / sampling script
	- `model.py` — TinyGPT model
	- `block.py` — transformer block implementation
	- `char_tokenizer.py` — simple char-level tokenizer
	- `dataset.py` — dataset helper for block/window sampling

---

## How training works (high-level)

- `src/train.py`:
	- Builds/loads a `CharTokenizer` from the corpus.
	- Encodes the corpus into token ids.
	- Creates `CharDataset` for train/val (90/10 split).
	- Instantiates `TinyGPT` with hyperparameters (see top of `train.py`).
	- Runs SGD (AdamW), prints periodic evals, and saves checkpoints to `checkpoints/tiny_gpt.pt`.

Key editable hyperparameters are at the top of `src/train.py`:
- `block_size` — context window length.
- `d_model`, `d_head`, `n_layers`, `dropout` — model size.
- `batch_size`, `max_iters`, `eval_interval`, `learning_rate` — training settings.

Tip: for quick smoke tests, reduce `max_iters` (e.g., 100) and `eval_iters` to run fast.

---

## How generation works (high-level)

- `src/generate.py`:
	- Loads `artifacts/char_tokenizer.json` and the checkpoint `checkpoints/tiny_gpt.pt`.
	- Reinstantiates `TinyGPT` using saved `config` from the checkpoint.
	- Encodes a prompt with the tokenizer and calls `generate_tokens`.
	- Sampling supports `temperature` and `top_k` filtering for controllable randomness.

Default generation parameters are set in `src/generate.py` (prompt, `max_new_tokens`, `temperature`, `top_k`).

---

## Example: quick local smoke test

1. Open `src/train.py` and set:
	 - `max_iters = 100`
	 - `eval_iters = 10`
2. From repo root:
```bash
python -m TinyGPT.src.train
```
3. After a short run, verify:
- `TinyGPT/artifacts/char_tokenizer.json` exists
- `TinyGPT/checkpoints/tiny_gpt.pt` exists
4. Run generation:
```bash
python -m TinyGPT.src.generate
```

---

## Model architecture (brief)

- Embeddings:
	- `token_embedding`: `nn.Embedding(vocab_size, d_model)`
	- `pos_embedding`: `nn.Embedding(block_size, d_model)`
- Transformer blocks:
	- Implemented in `src/block.py` and stacked `n_layers` times.
- Output:
	- Final `LayerNorm` + `lm_head` linear projecting to vocabulary size (weight-tied to token embedding).
- Loss:
	- Cross-entropy over flattened logits/targets for language modeling.

---

## Tips & troubleshooting

- "ModuleNotFoundError: attempted relative import with no known parent":
	- Run scripts as modules from the repository root: `python -m TinyGPT.src.train` or set `PYTHONPATH=.`.
- If `torch.cuda.is_available()` is True but you get CUDA errors, ensure you installed matching CUDA-enabled torch version.
- To reduce memory usage: reduce `batch_size` or `d_model` / `n_layers`.
- If tokenizer cannot be found when generating, ensure you have run training at least once so `artifacts/char_tokenizer.json` exists.
- Fine-tune sampling:
	- Lower `temperature` (e.g., 0.5) to make output more deterministic.
	- Use `top_k` to restrict sampling to top candidates (e.g., 40–100).

---

## Development & contribution

- The code is intentionally minimal and educational. Pull requests improving documentation, adding unit tests for the tokenizer/dataset, or adding README examples are welcome.
- Suggested small improvements:
	- Add a `requirements.txt` or `pyproject.toml` with pinned torch version.
	- Add CLI argument parsing to `train.py` and `generate.py`.
	- Add checkpoint naming by timestamp / step and resume training support.

---

## License

Use or adapt as you like. Add your preferred license here (e.g., MIT).
(If you want, I can add an MIT LICENSE file to the repo.)

---

## Quick checklist (to verify after first run)

- [ ] `python -m TinyGPT.src.train` starts, builds tokenizer, and prints training steps.
- [ ] `TinyGPT/artifacts/char_tokenizer.json` was created.
- [ ] `TinyGPT/checkpoints/tiny_gpt.pt` was saved.
- [ ] `python -m TinyGPT.src.generate` loads tokenizer & checkpoint and prints generated text.
