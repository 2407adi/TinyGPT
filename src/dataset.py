# src/dataset.py

from typing import Tuple
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    Takes a 1D tensor of token ids and a block_size (context length),
    and returns (x, y) pairs for next-token prediction.

    x: ids[i : i+block_size]
    y: ids[i+1 : i+block_size+1]
    """

    def __init__(self, ids: torch.Tensor, block_size: int):
        assert ids.dim() == 1, "ids must be a 1D tensor"
        self.data = ids
        self.block_size = block_size

    def __len__(self) -> int:
        # Last index we can start from is len(data) - block_size - 1
        # But for convenience we do:
        # This gives us how many batches of size block_size we can get from a 1D tensor of length len(self.data)
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y
