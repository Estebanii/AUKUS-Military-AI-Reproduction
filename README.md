# AUKUS 军事人工智能概念化研究：复现包

本复现包用于重现修订稿定量部分（第四至六节及相关脚注）中报告的全部数值结果，包括表 1–9 和图 1–4。计算在 CPU 上完成，约 10 分钟。数据（66 个文件，约 3.6 GB）作为本仓库的 GitHub Release 发布，标签 `rev1-data`。

## 运行环境

需要 bash、curl、tar、sha256sum（macOS 为 shasum）、[uv](https://docs.astral.sh/uv/)，以及 R 4.5（仅脚本 07 使用）。Windows 请在 WSL 或 Git Bash 中运行。

- Python 环境由 `environment/requirements.lock` 锁定（macOS arm64，Python 3.12.12）。Linux x86_64 改用 `environment/requirements.lock.linux-x86_64`（安装命令加 `--index-strategy unsafe-best-match`；版本相同，torch 为 CPU 版，未在 Linux 上实际运行过）。其他平台先用 `uv pip compile environment/requirements.in --generate-hashes` 生成本平台的锁定文件。
- R 包为 HonestDiD 0.2.8 与 jsonlite 2.0.0，由 `environment/install_R_packages.R` 安装。没有 R 时，脚本 07 加 `--allow-missing-r` 即可继续，HonestDiD 敏感性分析记为不可用，表 9 的数字不受影响。
- 参考机器为 macOS、Apple M3 Pro、36 GB 内存。峰值内存约 7 GB，磁盘约 10 GB（含数据包和下载的归档）。

## 运行步骤

```bash
bash get_data.sh                                   # 下载六个附件、核对 sha256、解压到 data/bundle/
uv venv .venv --python 3.12.12
VIRTUAL_ENV="$PWD/.venv" uv pip sync --require-hashes environment/requirements.lock
Rscript environment/install_R_packages.R
bash run_all.sh                                    # 依次运行 scripts/00–08，最后运行 compare.py
```

最后 `compare.py` 会逐项核对论文中的数字，正常情况下 394 项全部通过（PASS），退出码为 0。各脚本也可以按编号顺序单独运行，核对程序可以单独执行（`.venv/bin/python compare.py`）。脚本 07 默认直接使用数据包中保存的两项模拟记录；如果想重新计算并核对，运行 `bash run_all.sh --full`，大约多花 1 小时。

## 文件说明

```
get_data.sh               下载、核对并解压数据包
run_all.sh                依次运行全部脚本与 compare.py
scripts/                  编号脚本：00 测量准备（A 矩阵、语义向量、参考轴）；01 表 2；02 表 3；03 表 4–6、图 3；
                          04 图 4；05 表 8；06 表 7；07 表 9；08 图 1、2
code/replication/         计算代码；全部随机种子在 seeds.py 中设定，每个种子附有说明
compare.py, expected/     论文中 397 个数字的出处、文中数值与原始计算结果（完整精度）
PAPER_MAP.md / .csv       每个数字对应的脚本、输出文件、字段与随机种子
data/                     数据包清单与数据说明
environment/              Python 锁定文件、R 包清单与安装脚本
supplementary/            可选：表 7 置信区间的合成数据校准，不影响论文中的任何数字
optional_full_pipeline/   上游 GPU 编码的方法说明，不需要运行
results/                  运行后生成（约 1.2 GB）
```

## 结果核对

`compare.py` 把重新计算的结果按论文中的小数位数四舍五入后与文中数字比较（397 项中有 3 项是文字说明，只列出、不比较），同时报告它是否与当初写入论文的原始结果完全一致。在参考机器上有 390 项与原始结果完全一致，其余 4 项是图 1 坐标轴标签中的方差占比，当时没有保存完整精度的数值。在其他 CPU 或 BLAS 库上，最后几位小数可能略有差异，这不影响核对通过。输出的第一行记录本次运行的信息；如果 `results/` 中的结果不是由当前版本完整运行得到的，程序会给出 WARNING 并以退出码 2 退出，此时重新运行 `bash run_all.sh` 即可。

本包提到的脚注编号是全文的顺序号（稿件中每页用带圈数字重新编号），每条脚注的开头文字见 `PAPER_MAP.md`。原稿表 5、表 6 中的部分自助法结果使用同一个种子 42，修订稿改为按主成分分别使用 42、43、44，两种结果都列在 `expected/tables/tab6_placebo.csv` 中。

## 术语说明

原稿、`v1` 指首次投稿的分析及其结果。基线编码器指 DeBERTa-v3-base（代码中记为 `E0`），替代编码器指 ModernBERT-large、RoBERTa-large、Ettin-encoder-1b 与 DeBERTa-v2-xlarge（`E1`–`E4`）。参考轴是原稿语义向量（37,866 × 768）的前 88 个主成分。共同支持指五个编码器截断至 512 个词元后都保留被遮蔽词的 37,808 个观测点。U 为编码器在被遮蔽位置的隐状态，A 为将 U 映射到 GPT-2 词嵌入空间的矩阵（每个编码器一个，三国共用），Y = (U − U 均值) Aᵀ 为语义向量。
