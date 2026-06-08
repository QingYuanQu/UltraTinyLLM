import torch
from torch.utils.data import Dataset

class ArithmeticDataset(Dataset):
    def __init__(self, file_path, chars, max_len, repeat_factor=100):
        self.chars = chars
        self.data = []
        self.repeat_factor = repeat_factor
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.max_len = max_len
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [line.strip() for line in f if line.strip() and len(line.strip()) <= max_len]

    def __len__(self):
        return len(self.data) * self.repeat_factor

    def __getitem__(self, idx):
        real_idx = idx % len(self.data)
        seq = self.data[real_idx]
        tokens = [self.stoi[ch] for ch in seq]
        return torch.tensor(tokens, dtype=torch.long)