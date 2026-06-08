import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import ArithmeticDataset
from model import GPT
from utils import train, set_seed

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ArithmeticDataset("data/data.txt",sorted(list("0123456789+=")), max_len=6)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    model = GPT(vocab_size=dataset.vocab_size,
                embed_dim=32,
                num_heads=4,
                num_layers=3,
                max_len=6).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    set_seed(SEED)
    train(model, dataloader, optimizer, device, 4)

    torch.save(model.state_dict(), "model.pth")