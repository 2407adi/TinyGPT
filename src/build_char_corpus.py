from pathlib import Path
import torch

from char_tokenizer import CharTokenizer, CharTokenizerConfig


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "tiny_shakespeare.txt"

    # 1) Load full corpus
    text = data_path.read_text(encoding="utf-8")
    print(f"Loaded corpus with {len(text)} characters.")

    # 2) Build tokenizer using full corpus
    config = CharTokenizerConfig(add_bos=True, add_eos=True)
    tokenizer = CharTokenizer(text, config=config)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 3) Encode the entire corpus into token ids
    # IMPORTANT: we usually don't want a single BOS/EOS around the whole thing
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Total tokens: {len(ids)}")

    # 4) Convert to torch tensors and do train/val split
    ids = torch.tensor(ids, dtype=torch.long)

    n = len(ids)
    n_train = int(0.9 * n)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    print(f"Train tokens: {len(train_ids)}, Val tokens: {len(val_ids)}")

    # 5) Save everything
    out_dir = root / "data"
    out_dir.mkdir(exist_ok=True)

    torch.save(train_ids, out_dir / "train_ids.pt")
    torch.save(val_ids, out_dir / "val_ids.pt")
    tokenizer.save(out_dir / "char_tokenizer.json")

    print("Saved:")
    print(" - data/train_ids.pt")
    print(" - data/val_ids.pt")
    print(" - data/char_tokenizer.json")


if __name__ == "__main__":
    main()
