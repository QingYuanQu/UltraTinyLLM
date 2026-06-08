## GitHub Description（仓库简短描述）

> A minimal GPT from scratch (732 params) that learns single-digit addition — the tiniest language model to understand Transformers.

---


# UltraTinyLLM

> 能学会加法的最小 GPT — 仅约 20K 参数，从零构建一个完整的 Transformer 语言模型。

## 简介

UltraTinyLLM 是一个**从零手写**的极简 GPT 模型，用于学习两位数以内加法运算（`0+0` ~ `9+9`）。它剥离了所有工程复杂性，保留了一个 GPT 最核心的结构，是理解 Transformer 和语言模型训练的绝佳入门项目。

### 为什么做这个项目？

- **足够小**：约 20K 参数，任何设备都能秒级训练完成
- **足够完整**：包含 Tokenizer、数据集、模型、训练、推理、评估的完整流程
- **足够直观**：用"学会加法"这一可验证的任务，让黑盒模型的行为变得可解释

## 模型架构

```
Input: "3+5" → Token Embedding + Position Embedding
         ↓
   ┌─────────────────────┐
   │  TransformerBlock ×3 │
   │  ├─ LayerNorm        │
   │  ├─ Causal Attention │ (4 heads, dim=32)
   │  ├─ Residual Connect │
   │  ├─ LayerNorm        │
   │  ├─ FFN (GELU)       │
   │  └─ Residual Connect │
   └─────────────────────┘
         ↓
   LayerNorm → Linear Head
         ↓
Output: "08"
```

| 超参数 | 值                    |
|--------|----------------------|
| Embedding Dim | 8                    |
| Attention Heads | 1                    |
| Transformer Layers | 1                    |
| Max Sequence Length | 6                    |
| Vocabulary Size | 12 (`0-9`, `+`, `=`) |
| Total Parameters | 732                  |

## 项目结构

```
UltraTinyLLM/
├── model.py       # GPT 模型定义（TransformerBlock + GPT）
├── datasets.py    # 算术数据集加载器
├── train.py       # 训练入口
├── eval.py        # 评估脚本（逐样本测试 + 准确率统计）
├── utils.py       # 工具函数（训练循环、生成函数、随机种子）
├── data/
│   └── data.txt   # 训练数据（100 条加法算式）
└── model.pth      # 预训练权重
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch

```bash
pip install torch
```

### 训练

```bash
python train.py
```

输出示例：
```
Total parameters: 20,428
Epoch 1/4 | Average Loss: 1.8234
Epoch 2/4 | Average Loss: 0.4521
Epoch 3/4 | Average Loss: 0.0897
Epoch 4/4 | Average Loss: 0.0234
```

### 评估

```bash
python eval.py
```

输出示例：
```
开始测试 100 个样本...
============================================================
[1/100] ✓ 0+0=00  (期望: 00)
[2/100] ✓ 0+1=01  (期望: 01)
...
============================================================

测试结果汇总:
  总样本数: 100
  正确数量: 100
  错误数量: 0
  准确率: 100.00%
```

## 数据格式

训练数据位于 `data/data.txt`，每行一条加法算式，格式为 `A+B=CC`（结果补零至两位）：

```
0+0=00
0+1=01
...
9+9=18
```


## License

MIT

