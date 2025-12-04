import json
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class CharTokenizerConfig:
    """
    # @dataclass auto-generates __init__ etc., so you can easily do:
    # CharTokenizerConfig(add_bos=False, eos_token="</s>")
    # and get nice __repr__ too.
    """
    add_bos: bool = True
    add_eos: bool = True
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"

class CharTokenizer:
    """
    Very simple character-level tokenizer.

    - Builds a vocabulary from all unique characters in the training text
      + a few special tokens (BOS/EOS).
    - Exposes encode/decode methods:
        text -> list[int]
        list[int] -> text
    """

    def __init__(self, text: str, config: Optional[CharTokenizerConfig] = None):
        if config is None:
            config = CharTokenizerConfig()
        self.config = config

        # 1) Build base vocab from characters present in the text
        unique_chars = sorted(list(set(text)))
        # Example: ['\n', ' ', '!', '"', "'", '(', ')', ..., 'z']

        # 2) Define special tokens (we'll put them at the start)
        special_tokens = []
        if self.config.add_bos:
            special_tokens.append(self.config.bos_token)
        if self.config.add_eos:
            special_tokens.append(self.config.eos_token)

        # 3) Final vocab: special tokens + characters
        self.itos: Dict[int, str] = {}  # id -> string
        self.stoi: Dict[str, int] = {}  # string -> id

        idx = 0
        for tok in special_tokens:
            self.itos[idx] = tok
            self.stoi[tok] = idx
            idx += 1

        for ch in unique_chars:
            # Avoid collision if the corpus already contains strings
            # identical to special tokens (unlikely for these tokens).
            if ch in self.stoi:
                continue
            self.itos[idx] = ch
            self.stoi[ch] = idx
            idx += 1

        self.vocab_size = len(self.itos)

    # ------------- Public API ------------- #

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to a list of token ids.
        """
        tokens: List[int] = []

        if add_special_tokens and self.config.add_bos:
            tokens.append(self.stoi[self.config.bos_token])

        # Map each character to its id
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"Character {repr(ch)} not in vocabulary")
            tokens.append(self.stoi[ch])

        if add_special_tokens and self.config.add_eos:
            tokens.append(self.stoi[self.config.eos_token])

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a list of token ids back to text.
        """
        chars: List[str] = []
        for tid in token_ids:
            if tid not in self.itos:
                raise ValueError(f"Token id {tid} not in vocabulary")
            token_str = self.itos[tid]
            if skip_special_tokens and token_str in {
                self.config.bos_token,
                self.config.eos_token,
            }:
                continue
            chars.append(token_str)
        return "".join(chars)

    # ------------- Save / Load ------------- #

    def save(self, path: str) -> None:
        """
        Save tokenizer vocab + config to a JSON file.
        """
        data = {
            "itos": self.itos,
            "config": {
                "add_bos": self.config.add_bos,
                "add_eos": self.config.add_eos,
                "bos_token": self.config.bos_token,
                "eos_token": self.config.eos_token,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """
        Load tokenizer from JSON file.
        @classmethod → method that receives the class, not an instance.
        Used to create objects without needing an existing object.
        load() is an alternative constructor: CharTokenizer.load(path).
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config_dict = data["config"]
        config = CharTokenizerConfig(**config_dict)

        # Instantiate an empty tokenizer (we'll overwrite its vocab)
        dummy = cls(text="", config=config)
        dummy.itos = {int(k): v for k, v in data["itos"].items()}
        dummy.stoi = {v: int(k) for k, v in dummy.itos.items()}
        dummy.vocab_size = len(dummy.itos)
        return dummy
