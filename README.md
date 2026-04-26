# QLoRA + SFT 原理讲解

用最小的模型和最精简的代码，从零理解 **QLoRA + 监督微调（SFT）** 的完整原理。

## 两份 Notebook，循序渐进

| Notebook | 模型 | 数据 | GPU | 时长 | 适合 |
|----------|------|------|-----|------|------|
| [qlora_sft_tutorial.ipynb](./qlora_sft_tutorial.ipynb) | GPT-2 124M | 8 条玩具数据 | 单卡 T4（或 CPU） | ~1 分钟 | 第一次接触 QLoRA |
| [qlora_sft_ddp_tutorial.ipynb](./qlora_sft_ddp_tutorial.ipynb) | Qwen2.5-1.5B | Alpaca 1000 条 | **双卡 T4 DDP** | ~10-15 分钟 | 进阶到多卡训练 |

### 一键在 Kaggle 运行

- 单卡版（选择 "GPU T4"）：[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/greathousesh/qlora-sft-tutorial/blob/main/qlora_sft_tutorial.ipynb)
- 双卡 DDP 版（选择 **"GPU T4 x2"**）：[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/greathousesh/qlora-sft-tutorial/blob/main/qlora_sft_ddp_tutorial.ipynb)

## 单卡版内容结构

| Step | 内容 |
|------|------|
| 0 | 安装依赖 |
| 1 | 手写 LoRALinear，理解低秩分解原理 |
| 2 | 演示 4-bit 量化（NF4）的精度损失 |
| 3 | 加载量化模型 + PEFT 挂载 LoRA 适配器 |
| 4 | 准备 SFT 数据集（Alpaca 格式） |
| 5 | Trainer 训练 |
| 6 | 保存 & 加载 LoRA 适配器 |
| 7 | 推理测试 |
| 8 | 合并权重用于部署 |

## 双卡 DDP 版内容结构

| Step | 内容 |
|------|------|
| 0 | 安装依赖 |
| 1 | 检查双 GPU 环境 |
| 2 | 理解 DDP（数据并行）vs 模型并行 |
| 3 | 定义自包含的训练函数（DDP 入口） |
| 4 | `notebook_launcher` 启动双卡训练 |
| 5 | 加载适配器做推理 |
| 6 | 合并 LoRA 权重并保存 |

## 环境要求

- Python 3.10+
- 主要依赖：`transformers>=4.40`, `peft>=0.10`, `bitsandbytes>=0.43`, `accelerate>=0.29`, `datasets>=2.18`
- 单卡版可在 CPU 上演示 LoRA 部分（4-bit 量化必须 GPU）
- 双卡 DDP 版需要 **2 张 GPU**（Kaggle 选 "GPU T4 x2"）

## 核心概念速查

```
原始权重 W (冻结, 4-bit 量化)
      ↓ 量化/反量化
  W_dequant (fp16)
      +
  B @ A (LoRA, 可训练, r << d)
      ↓
   输出 = x(W + BA)
```

**参数量对比（768×768 层，r=8）：**
- 原始：589,824 个参数
- LoRA：12,288 个参数（仅 **2.1%**）

**模型并行 vs 数据并行（DDP）：**

```
模型并行（device_map="auto"）          数据并行（DDP，本仓库双卡 notebook）
────────────────────────────          ──────────────────────────────────
GPU 0: [Layer 0~5]                     GPU 0: [完整模型副本]  ← batch[0:8]
GPU 1: [Layer 6~11]                    GPU 1: [完整模型副本]  ← batch[8:16]

适用：模型大到单卡装不下                 适用：模型能塞进单卡，要更高吞吐
缺点：跨卡通信开销，不能配 DDP           缺点：每卡都要装一份模型
```
