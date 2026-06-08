import torch

from datasets import ArithmeticDataset
from model import GPT
from utils import generate

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ArithmeticDataset("data/data.txt", sorted(list("0123456789+=")), max_len=6)
    model = GPT(vocab_size=dataset.vocab_size,
                embed_dim=8,
                num_heads=1,
                num_layers=1,
                max_len=6).to(device)

    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    with open('data/data.txt', 'r', encoding='utf-8') as f:
        test_lines = [line.strip() for line in f if line.strip()]

    correct = 0
    total = len(test_lines)

    print(f"\n开始测试 {total} 个样本...")
    print("=" * 60)

    for idx, line in enumerate(test_lines, 1):
        question = line[:-2]
        expected_answer = line.split('=')[1]
        answer = generate(model, device, dataset, question, max_new=2)
        is_correct = answer == expected_answer
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"[{idx}/{total}] {status} {question}={answer:3s}  (期望: {expected_answer})")

    # ===================== 统计结果 =====================
    print("=" * 60)
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n测试结果汇总:")
    print(f"  总样本数: {total}")
    print(f"  正确数量: {correct}")
    print(f"  错误数量: {total - correct}")
    print(f"  准确率: {accuracy:.2f}%")
