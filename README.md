# AUKUS军事人工智能概念化比较分析：复现代码包

论文"共享技术与多元认知：AUKUS军事人工智能概念化比较分析"的定量分析复现代码与数据。

## 概述

本仓库包含论文定量分析部分的完整复现代码包。本研究受Rodriguez et al. (APSR, 2023)嵌入回归框架的启发，结合DeBERTa深度语境编码与线性变换矩阵，构建了适用于跨国军事人工智能话语语义比较的分析管线。

## 数据

- `data/raw/` — 美、英、澳三国国防部官方文本原始语料
- `data/intermediate/` — 目标术语表、锚定词表、术语出现记录
- `data/semantic_vectors_paragraph_global_A.parquet` — 最终分析数据集（37,866条观测 × 768维语义向量）

## 脚本

| 脚本 | 论文章节 | 说明 |
|------|---------|------|
| `modal_reproduce_deberta.py` | 方法 | DeBERTa语境编码 + A矩阵训练（Modal云GPU） |
| `manova_period_split.py` | H0（表3a） | Pre/Post-AUKUS分期MANOVA |
| `manova_h0_test.py` | 附录 | 不同主成分数量下MANOVA稳健性 |
| `manova_time_robustness.py` | H0 | 时间残差化MANOVA（年份固定效应） |
| `pc_regression_test.py` | H1（表3b） | 主成分回归国别效应 |
| `wild_cluster_bootstrap.py` | H2（表5） | 含线性时间趋势的DID + Wild Cluster Bootstrap标准误 |
| `did_robustness_full.py` | H2（表6、表9） | 6种规格稳健性检验 + 安慰剂检验 |
| `did_h3_test.py` | H2 | DID距离变化验证（Pre/Post欧氏距离） |
| `parallel_trends_test.py` | H2（图3） | 事件研究（Bootstrap置信区间） |
| `h4_nearest_neighbor.py` | H3（图4） | AUKUS签署后语义近邻分析 |
| `ablation_experiment.py` | 附录 | 向量空间消融实验（Raw U / Whitened U / Y=AU） |
| `data_completeness_check.py` | 附录 | 数据完整性与质量诊断 |

## 复现步骤

### 环境要求
```
Python 3.11+
numpy, scipy, pandas, pyarrow, scikit-learn, matplotlib, SciencePlots
torch, transformers（DeBERTa/GPT-2编码，建议使用Modal云GPU）
```

### 执行顺序
1. **DeBERTa语境编码**（需GPU）：
   ```bash
   python scripts/modal_reproduce_deberta.py
   ```
2. **统计分析**（本地即可）：
   ```bash
   python scripts/manova_period_split.py
   python scripts/pc_regression_test.py
   python scripts/wild_cluster_bootstrap.py
   python scripts/did_robustness_full.py
   python scripts/parallel_trends_test.py
   python scripts/h4_nearest_neighbor.py
   python scripts/ablation_experiment.py
   ```
3. **图表生成**：
   ```bash
   python scripts/generate_figures_tables.py
   ```

## 使用的模型

| 模型 | 用途 | 规格 |
|------|------|------|
| microsoft/deberta-v3-base | 语境语义编码（提取[MASK]位置隐藏状态） | 184M参数，768维 |
| GPT-2 | 词嵌入层作为语义字典（A矩阵训练 + 近邻分析） | 124M参数，768维，50,257词 |

## 关键参数

- 正则化参数 λ = 0.1（Ridge回归 + Procrustes正交先验）
- Bootstrap：1,000次重抽样，Rademacher权重，doc_id聚类（11,459个聚类单元）
- 随机种子：42
- DeBERTa最大序列长度：512，批次大小：32

## 参考文献

Rodriguez, P. L., Spirling, A., & Stewart, B. M. (2023). Embedding Regression: Models for Context-Specific Description and Inference. *American Political Science Review*, 117(4), 1255-1274.
