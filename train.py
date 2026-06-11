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
    vocab = ["加", "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "等于"]
    dataset = ArithmeticDataset("data/data.txt", vocab, max_len=6)
    dataloader = DataLoader(dataset, batch_size=87, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    model = GPT(vocab_size=dataset.vocab_size,
                embed_dim=8,
                num_heads=1,
                num_layers=1,
                max_len=6).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    set_seed(SEED)
    train(model, dataloader, optimizer, device, 20)

    torch.save(model.state_dict(), "model.pth")