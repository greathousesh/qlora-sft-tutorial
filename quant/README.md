# 量化交易从入门到 qlib

面向**懂编程的量化初学者**的系统课程，共 5 个 notebook，全部支持 Kaggle 一键运行（无需 GPU）。

## 课程结构

| Notebook | 主题 | 核心内容 | 依赖 |
|----------|------|----------|------|
| `01_quant_basics_data.ipynb` | 量化基础 + 数据探索 | OHLCV、收益率、Sharpe、相关性 | yfinance |
| `02_alpha_factors.ipynb` | Alpha 因子工程 | IC 分析、5 大因子族、因子衰减 | yfinance |
| `03_ml_models.ipynb` | 机器学习选股 | LightGBM、Walk-Forward 验证 | yfinance + lightgbm |
| `04_backtesting.ipynb` | 回测基础 | 多空组合、绩效指标、成本分析 | yfinance |
| `05_qlib_full_pipeline.ipynb` | qlib 完整工作流 | 表达式引擎、DataHandler、回测框架 | pyqlib |

## 学习路径

```
Vol.1 → Vol.2 → Vol.3 → Vol.4 → Vol.5
 数据    因子    ML模型   回测    qlib工程化
```

每节 notebook 独立可运行，数据从 yfinance 现下现用，无需预先准备。

## Kaggle 运行

点击每个 notebook 顶部的 **Open in Kaggle** 徽章，选择 CPU 运行，约 5-10 分钟执行完毕。

## 核心概念速查

| 概念 | 说明 |
|------|------|
| **IC (信息系数)** | 因子截面排名 vs 未来收益排名的 Spearman 相关，>0.02 认为有效 |
| **ICIR** | IC 均值 / IC 标准差，>0.5 认为稳定 |
| **Sharpe Ratio** | 年化收益 / 年化波动，>1.0 良好 |
| **最大回撤** | 峰值到谷底的最大跌幅 |
| **Walk-Forward** | 金融数据必须用时序验证，不可随机划分 |
| **Alpha** | 策略超越市场基准的超额收益 |
