# 可选：上游 GPU 编码 / Optional: the upstream GPU encoding

本包的全部脚本从编码器输出 U（数据包 `encodings/` 与 `external_controls/encodings/`）开始，**不需要 GPU，也不需要重新编码**。本文件记录 U 的产生方式，供需要从文本重新编码的读者参考。重新编码所需的编码输入表（每个出现点的段落文本、目标词及其在段落中的位置、词性）在数据包的 `corpus/encoding_inputs/` 与 `external_controls/encoding_inputs/` 中（见 `data/README.md`）。

All scripts of the package start from the encoder outputs U in the data bundle; no GPU and no re-encoding are needed. This file documents how U was produced.

## 1. 编码器 / Encoders

| 编码器 Encoder | Hugging Face 模型 | 修订 Revision (commit) | 隐藏维度 | 编码环境 (torch / transformers / tokenizers) | 注意力实现 |
|---|---|---|---|---|---|
| DeBERTa-v3-base（基线 baseline） | `microsoft/deberta-v3-base` | `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` | 768 | 2.5.1 / 4.46.3 / 0.20.3 | 默认 |
| ModernBERT-large | `answerdotai/ModernBERT-large` | `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | 1024 | 2.6.0 / 4.48.3 / 0.21.0 | sdpa，`reference_compile=False` |
| RoBERTa-large | `FacebookAI/roberta-large` | `722cf37b1afa9454edce342e7895e588b6ff1d59` | 1024 | 2.5.1 / 4.48.3 / 0.21.0 | sdpa，`add_pooling_layer=False` |
| Ettin-encoder-1b | `jhu-clsp/ettin-encoder-1b` | `befd76be43d08b89ff9957012f3ff29d0842780b` | 1792 | 2.6.0 / 4.48.3 / 0.21.0 | sdpa，`reference_compile=False` |
| DeBERTa-v2-xlarge | `microsoft/deberta-v2-xlarge` | `1d134961d4db8e7e8eb1bc1ab81cb370244c57f7` | 1536 | 2.5.1 / 4.48.3 / 0.21.0 | eager |

各环境另含 `safetensors`（0.4.5 或 0.5.2）与 `sentencepiece` 0.2.0。GPU：NVIDIA A100；float32（无自动混合精度、无半精度）；确定性设置：cuDNN deterministic、关闭 TF32（`NVIDIA_TF32_OVERRIDE=0`）、`torch.set_float32_matmul_precision('highest')`、`torch.use_deterministic_algorithms(True, warn_only=True)`、设置 `CUBLAS_WORKSPACE_CONFIG`。

## 2. 每个出现点的编码 / Encoding of one occurrence

1. **遮蔽 Masking.** 在段落文本中把目标词替换为该编码器自己的遮蔽符（DeBERTa-v3 为 `[MASK]`）：
   - 词性为 `adjective` 的术语：在记录的起始位置（不区分大小写地核对）找到该形容词，否则取第一个整词匹配 `\b<term>\b`；把形容词连同其后的空白与下一个单词（`\w+`）一起替换为一个遮蔽符（其后无单词时只替换形容词）；找不到时不遮蔽。
   - 其他术语：若记录的起始位置处的文本与术语相同（不区分大小写）则替换该处，否则替换第一个整词匹配，都没有时不遮蔽。
2. **词元化 Tokenisation.** `tokenizer(text, max_length=512, padding="max_length", truncation=True)`。
3. **前向计算 Forward pass.** `AutoModel`、`eval()`、`no_grad()`，按输入顺序每批 32 行，`model(input_ids, attention_mask).last_hidden_state`。
4. **取向量 Pooling.** U = 最后一层隐状态在（截断后输入中）第一个遮蔽符位置的向量（float32）。基线编码器沿用原稿的规则：截断后没有遮蔽符的行取位置 1，有多个时取第一个（并计数）；替代编码器要求截断后恰有一个遮蔽符。

## 3. 共同支持 / Common support

五个编码器的词元化器分别以 512 个词元截断后，若某行在任一编码器下因截断而失去遮蔽符（截断后无遮蔽符、完整序列超过 512 个词元且完整序列中恰有一个遮蔽符），该行从共同支持中剔除。结果为 37,808 个目标出现点与全部 49,999 个锚点（`corpus/common_support_map.parquet`）。替代编码器只在共同支持行上编码与分析；基线编码器在全部 37,866 行上分析，并在第 5 号脚本中另报共同支持行上的结果。

## 4. 从 U 开始的计算 / From U onwards

U 之后的全部步骤（GPT-2 标签、A 矩阵、语义向量 Y、参考轴与所有检验）由 `scripts/00_prepare.py` 起的脚本在 CPU 上完成，见主 README。
