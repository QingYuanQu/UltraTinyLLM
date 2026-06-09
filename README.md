# UltraTinyLLM

> 能学会加法的最小 GPT — 仅732个参数，从零构建一个完整的 Transformer 语言模型。

## 简介

UltraTinyLLM 是一个的极简 GPT 模型，用于学习两位数以内加法运算（`0+0` ~ `9+9`）。它剥离了所有工程复杂性，保留了一个 GPT 最核心的结构，是理解 Transformer 和语言模型训练的绝佳入门项目。

## 特点

- **足够小**：约 732个 参数，任何设备都能秒级训练完成
- **足够完整**：包含 Tokenizer、数据集、模型、训练、推理、评估的完整流程
- **足够直观**：用"学会加法"这一可验证的任务，让黑盒模型的行为变得可解释

## 模型参数

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
├── eval_user.py   # 测试用户输入
├── utils.py       # 工具函数（训练循环、生成函数、随机种子）
├── data/
│   └── data.txt   # 训练数据（100 条加法算式）
└── model.pth      # 预训练权重
```



## 快速开始

### 数据

训练数据位于 `data/data.txt`，每行一条加法算式，格式为 `A+B=CC`（结果补零至两位）：

```
零加零等于零零
零加一等于零一
...
九加九等于一八
```
### 训练

```bash
python train.py
```

### 推理

![eval.gif](data/eval.gif)


## License

MIT

