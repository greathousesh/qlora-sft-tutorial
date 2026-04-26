# QLoRA + SFT 原理讲解

用最小的模型（GPT-2）和最精简的代码，从零理解 **QLoRA + 监督微调（SFT）** 的完整原理。

## 一键在 Kaggle 运行

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/greathousesh/qlora-sft-tutorial/blob/main/qlora_sft_tutorial.ipynb)

> Kaggle 提供免费 T4 GPU（每周 30 小时），可运行完整 QLoRA 量化流程。打开后选加速器 **"GPU T4"** → Run All。

## 内容结构

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

## 环境要求

- Python 3.10+
- CUDA GPU（可选，无 GPU 时自动 fallback 到 CPU 演示 LoRA 部分）
- 主要依赖：`transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`

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
