# AUKUS 军事人工智能概念化研究：复现包（修订稿）

本包复现修订稿定量部分（第四、五、六节及相关脚注）中印出的全部数字，以及表 1–9、图 1–4。所有计算在 CPU 上完成：从数据包出发依次运行编号脚本，最后由 `compare.py` 逐项核对论文中印出的 397 个数字。

数据包（66 个文件，约 3.6 GB）作为本仓库的 GitHub Release 发布，标签 `rev1-data`：<https://github.com/Estebanii/AUKUS-Military-AI-Reproduction/releases/tag/rev1-data>。本仓库不使用 Git LFS，数据以普通 Release 附件分发。

## 快速开始

需要 bash、curl、tar、sha256sum 或 shasum；Python 环境用 [uv](https://docs.astral.sh/uv/) 安装；脚本 07 需要 R 4.5。Windows 请在 WSL 或 Git Bash 中运行。

1. 取得本仓库：在 GitHub 页面点"Code → Download ZIP"并解压，或 `git clone https://github.com/Estebanii/AUKUS-Military-AI-Reproduction.git`，进入得到的目录。
2. 下载并校验数据（下载六个附件、核对 sha256、解压到 `data/bundle/`、逐一核对 66 个文件；可重复运行）：

   ```bash
   bash get_data.sh
   ```

3. 环境（详见第 2 节）：

   ```bash
   uv venv .venv --python 3.12.12
   VIRTUAL_ENV="$PWD/.venv" uv pip sync --require-hashes environment/requirements.lock
   Rscript environment/install_R_packages.R
   ```

4. 运行（约 10 分钟）：

   ```bash
   bash run_all.sh
   ```

5. 核对（应报告 394 个计算值全部 PASS，退出码 0）：

   ```bash
   .venv/bin/python compare.py
   ```

从 ZIP 解压的目录没有 `.git`，运行记录中的提交号取自文件 `VERSION`，不需要 git。

## 1. 内容与目录

| 论文项目 | 脚本 | 说明 |
|---|---|---|
| 表 1、表 2 的样本数字 | `01_sample_table2.py` | 分国家、分时期的出现点、文档与起止年月 |
| 表 3（H1）、脚注 72、78 | `02_h1_manova_table3.py` | MANOVA（88 个主成分）；主成分数稳健性；替代编码器的 F 值 |
| 表 4–6、图 3（H2）、脚注 80 | `03_h2_did_tables4_6_fig3.py` | 子组 DiD、wild cluster bootstrap、平行趋势检验、安慰剂检验（含与不含时间趋势项）、事件研究图 |
| 图 4（H3）、脚注 83、84 | `04_h3_neighbours_fig4.py` | GPT-2 近邻词 |
| 表 8 | `05_cross_encoder_table8.py` | 四个替代编码器与同轴稳健性 |
| 表 7、脚注 81 | `06_distance_table7.py` | 美英距离的签署前后变化 |
| 表 9、脚注 85 | `07_external_control_table9.py` | 外部对照（新加坡、加拿大）；HonestDiD 敏感性用 R |
| 图 1、图 2 | `08_figures_1_2.py` | PCA 与 t-SNE 散点 |
| 全部 397 个印出数字 | `compare.py` | 逐项核对 |

`00_prepare.py` 在最前面运行：为五个编码器拟合 A 矩阵、计算语义向量 Y 与参考轴。每个印出数字对应的脚本、输出文件、字段与随机种子见 `PAPER_MAP.md`。

```
README.md                 本文件
PAPER_MAP.md / .csv       论文项目与印出数字的对照表
get_data.sh               下载、核对并解压数据包
run_all.sh                依次运行全部脚本与 compare.py
compare.py                核对全部印出数字
VERSION                   本包的提交号（ZIP 下载与 git archive 导出时写入）
expected/                 每个印出数字的位置、印出值与全精度参考值；表 5–9 的全精度参照
scripts/                  编号脚本 00–08 与运行记录脚本
code/replication/         计算代码；seeds.py 为全部随机种子的唯一设定处
environment/              Python 锁定文件（macOS arm64、Linux x86_64）、R 包清单与安装脚本
data/                     数据包清单、Release 附件清单与数据说明
supplementary/            可选的补充计算（表 7 区间的合成校准）
optional_full_pipeline/   上游 GPU 编码的说明（不需要运行）
results/                  运行后生成
```

## 2. 环境

参考环境：macOS 26.6（Apple M3 Pro，12 核）、36 GB 内存、Python 3.12.12。建议至少 16 GB 内存与 5 GB 可用磁盘，数据包约 3.6 GB 与下载的归档约 3.1 GB 另计。

**Python。** 用 uv 按锁定文件安装，版本与哈希已固定（42 个包，主要为 numpy 2.3.5、scipy 1.16.3、pandas 2.3.3、pyarrow 22.0.0、scikit-learn 1.7.2、torch 2.9.1、transformers 4.57.3、matplotlib 3.10.8）：

```bash
uv venv .venv --python 3.12.12
VIRTUAL_ENV="$PWD/.venv" uv pip sync --require-hashes environment/requirements.lock
```

`environment/requirements.lock` 适用于 macOS arm64。Linux x86_64 改用 `environment/requirements.lock.linux-x86_64`（版本相同，torch 为 CPU 版），命令为 `uv pip sync --require-hashes --index-strategy unsafe-best-match environment/requirements.lock.linux-x86_64`；该文件已解析并带哈希核对安装，但未在 Linux 上运行过。其他平台先用 `uv pip compile environment/requirements.in --generate-hashes --python-version 3.12 --python-platform <平台> -o environment/requirements.lock.local` 生成本平台的锁定文件再安装。新装环境中第一次 `import torch` 可能需要 1–3 分钟。

**R（仅脚本 07 需要）。** R 4.5.0，包 HonestDiD 0.2.8 与 jsonlite 2.0.0。`Rscript environment/install_R_packages.R` 安装这两个版本（CRAN 二进制包，Linux 上从源代码编译；CRAN 更新后从存档安装同一版本）。没有 R 时脚本 07 会停止；加 `--allow-missing-r` 可以继续，此时 HonestDiD 敏感性记为不可用，表 9 与脚注 85 的数字不受影响。

**字体。** 图的中文标签需要系统中文字体（如 STHeiti、PingFang SC、Noto Sans CJK SC），没有时用英文标签，数据不变。

## 3. 数据包

数据包 `data_bundle_rev1`（66 个文件，3,568,098,322 字节）的内容与各列说明见 `data/README.md`。Release `rev1-data` 的附件按顶层目录各打一个归档，另附逐文件清单：

| 附件 | 内容 | 大小 |
|---|---|---|
| `corpus.tar.gz` | 语料、原始文件、编码输入表（10 个文件） | 276 MB |
| `encodings.tar.gz` | 五个编码器的编码输出（20 个文件） | 1.96 GB |
| `external_controls.tar.gz` | 外部对照的语料、编码与设计记录（23 个文件） | 381 MB |
| `gpt2.tar.gz` | GPT-2 模型文件（6 个文件） | 464 MB |
| `original_outputs.tar.gz` | 原稿分析的结果文件（7 个文件） | 25 KB |
| `MANIFEST.sha256` | 66 个文件的逐文件清单 | 8 KB |

`bash get_data.sh` 把附件下载到 `data/downloads/`，按 `data/RELEASE_ASSETS.sha256` 核对，解压到 `data/bundle/`，再按 `data/MANIFEST.sha256` 逐一核对 66 个文件；中断后续传，重复运行时跳过已核对的部分。

手工方式：从 Release 页面下载六个附件到一个目录，在该目录运行 `sha256sum -c /path/to/package/data/RELEASE_ASSETS.sha256`（macOS 用 `shasum -a 256 -c`），把五个归档解压到同一目录并放入 `MANIFEST.sha256`，然后把该目录软链接为 `data/bundle`，或设置 `REPLICATION_DATA` 指向它。校验全部文件：

```bash
cd /path/to/data_bundle_rev1
grep -v '^#' MANIFEST.sha256 | awk '{print $1 "  " $3}' | shasum -a 256 -c --quiet && echo "all files OK"
```

## 4. 运行

```bash
bash run_all.sh
```

`run_all.sh` 先记录本次运行（`results/run_info.json`），再依次运行 `scripts/00_prepare.py` 至 `scripts/08_figures_1_2.py`，最后运行 `compare.py`，任何一步失败即停止；结果写入 `results/`，日志在 `results/logs/`。也可以按编号顺序逐个运行。

| 选项或环境变量 | 作用 |
|---|---|
| `bash run_all.sh --full` | 脚本 07 为五个编码器重算表 9 推断方法的两项设计模拟并与数据包中的存档逐位核对，总时间多约 1 小时 |
| `scripts/07_external_control_table9.py --allow-missing-r` | 没有 R 时继续运行 |
| `REPLICATION_DATA` | 数据包目录（默认 `data/bundle`） |
| `REPLICATION_RESULTS` | 结果目录（默认 `results/`） |
| `REPLICATION_BLAS_THREADS` | BLAS 线程数（默认脚本 07 为 12，其余为 4） |

`supplementary/distance_calibration.py` 用合成数据校准表 7 的区间，可选，不影响任何印出数字。

## 5. 核对论文数字

`compare.py` 核对修订稿定量部分与表 1–9 中登记的 397 个数字：394 个计算值按论文的印出精度与重算值比较，3 个标签只列出不检验；另报每个数字是否与论文所据结果逐位相同。第一行打印本次运行的记录（运行编号、提交号、代码指纹、数据包清单）；若 `results/` 不是本包当前版本的一次完整运行，开头与结尾打印 `WARNING:` 并以退出码 2 结束；有任何 FAIL 时退出码为 1。

参考运行：394 个计算值全部 PASS，其中 390 个逐位相同，其余 4 个是图 1 坐标轴上的方差占比，没有存档的全精度值。394 个计算值中 367 个依赖本次运行的结果，另外 27 个是分析参数、数据包常数与一个原稿存档值，直接按代码参数或数据包核对。

近邻词表、跨编码器与外部对照的定性表述不在这 397 项之内，列于 `PAPER_MAP.md` 末尾"文字判断"一节，并注明支持它们的结果字段。

**脚注定位。** 修订稿的脚注按页用带圈数字编号，本包所说的"脚注 n"指该脚注在全文中的顺序号。本包引用的脚注如下：

| 顺序号 | 所在节 | 开头文字 |
|---|---|---|
| 1 | 正文之前 | 本文的复现代码和数据可在https://github.c… |
| 61 | （三）数据与语义测量 | 目标术语的选择采用自动化发现算法，共筛选出56个军事人工… |
| 70 | （三）数据与语义测量 | 转换矩阵 A 基于100个高频锚词（非目标术语）训练而成… |
| 72 | （四）实证策略与统计推断 | 稳健性检验表明在3至99个主成分范围内结论均不变。… |
| 75 | （四）实证策略与统计推断 | H2的因果推断主要集中在英国与美国的比较层面。澳大利亚的… |
| 78 | （二）基线差异：概念化差异的存在检验（H1） | 由于澳大利亚Pre-AUKUS仅有844条观测（其中60… |
| 80 | （三）AUKUS协定签署的影响（H2） | PC3上英国主项与交互项同为负。该安慰剂结果对是否纳入趋… |
| 81 | （三）AUKUS协定签署的影响（H2） | 该距离从两国均值差的平方范数中扣除以文档聚类方差估计的抽… |
| 83 | （四）AUKUS 三国军事人工智能概念化的具体语义（H3） | 为消除GPT-2字节对编码分词机制产生的子词片段干扰（如… |
| 84 | （四）AUKUS 三国军事人工智能概念化的具体语义（H3） | 近邻分析基于AUKUS签署后（2021年9月至2025年… |
| 85 | （五）稳健性检验 | 新加坡的样本窗口为2014年1月至2025年11月，加拿… |

## 6. 随机种子

全部随机种子与抽样次数只在 `code/replication/seeds.py` 设定。

| 步骤 | 种子 | 抽样次数 | 输出 |
|---|---|---|---|
| H2 主模型的 wild cluster bootstrap（表 4） | 42、43、44（PC1–PC3） | 1,000 | `results/h2/wcb.json` |
| 稳健性模型与安慰剂检验（表 6） | 42、43、44 | 1,000 | `results/h2/did.json` |
| 含时间趋势项的安慰剂检验（脚注 80） | 42、43、44 | 1,000 | `results/h2/placebo_with_trend.json` |
| 平行趋势事件研究（表 5、图 3） | 42、43、44 | 1,000 | `results/h2/parallel_trends.json` |
| 三国事件研究的 Wald 检验（2017–2024） | 42 | 1,000 | `results/h2/did.json` |
| 替代编码器的主模型与同轴模型（表 8） | 42、43、44 | 1,000 | `results/alternatives/*/`、`results/cross_encoder/` |
| 表 8 的描述性重跑 | 20260929 | 9,999 | `results/alternatives/*/rerun_descriptive.json` |
| 距离变化的枢轴自助法（表 7） | 20260929 | 2,000 | `results/distance/` |
| 外部对照的月块自助法（表 9） | 20260928 | 9,999 | `results/external/*/estimate/` |
| 外部对照推断的开发模拟、独立验证模拟、覆盖与尺寸模拟 | 20260928、20260929、20260928 | 每种情形 2,000 | `results/external/*/design/` |
| 图 1 的 PCA、图 2 的 t-SNE | 42 | — | `results/figures/` |
| 分析用的 PCA、A 矩阵、MANOVA、近邻词、HonestDiD | 确定性计算 | — | — |

原稿表 5、表 6 的部分自助法结果使用共同种子 42，修订稿与本包按主成分分别使用 42、43、44；表 6 两种口径的值并列于 `expected/tables/tab6_placebo.csv`。

在锁定环境与默认线程数下，重跑得到的每个有全精度参考值的数字都与论文所据结果逐位相同。脚本在载入 numpy 之前固定 BLAS 线程数（脚本 07 为 12，其余为 4，与原结果一致）。在其他 CPU、BLAS 库或操作系统上，结果可能在末几位不同；`compare.py` 按印出精度比较，这类差异不造成 FAIL，只在 `differs by …` 中列出。

## 7. 未纳入的部分

- 表 9 推断方法的两项设计模拟默认读取数据包中的存档记录，`--full` 重算并逐位核对。
- GPU 编码：编码器输出与编码输入表都在数据包中，重新编码不是必需步骤，方法见 `optional_full_pipeline/README.md`。
- 原分析流程中不进入修订稿的部分没有纳入。

## 8. 运行时间与内存

Apple M3 Pro（12 核、36 GB）上一次参考运行的时间与峰值内存。峰值内存随系统负载波动，脚本 00 在多次运行中为 5–7 GB。

| 步骤 | 时间（秒） | 峰值内存（GB） |
|---|---|---|
| 建立 Python 环境 | 73 | — |
| 校验数据包 | 8 | — |
| `00_prepare.py` | 38 | 约 7 |
| `01_sample_table2.py` | 0.1 | 0.1 |
| `02_h1_manova_table3.py` | 46 | 1.5 |
| `03_h2_did_tables4_6_fig3.py` | 13 | 1.3 |
| `04_h3_neighbours_fig4.py` | 7 | 2.0 |
| `05_cross_encoder_table8.py` | 134 | 5.5 |
| `06_distance_table7.py` | 6 | 2.5 |
| `07_external_control_table9.py` | 208 | 3.9 |
| `08_figures_1_2.py` | 92 | 1.6 |
| `run_all.sh` 合计 | 562 | 约 7 |
| 脚本 07 的 `--full` 重算 | 3,408 | 5.1 |

磁盘：`results/` 约 1.2 GB，虚拟环境约 0.8 GB。

## 9. 常见问题

- `data bundle not found`：运行 `bash get_data.sh`，或设置 `REPLICATION_DATA`。
- `could not download …`：Release 不可达或网络中断，检查后重新运行 `bash get_data.sh`，已核对的附件会跳过，中断的下载会续传。
- 某个附件的 sha256 不符：下载不完整，脚本已删除该文件，重新运行即可。
- 从 ZIP 解压后脚本没有执行权限：用 `bash get_data.sh`、`bash run_all.sh`。
- `run scripts/00_prepare.py first`：按编号顺序运行，或直接用 `bash run_all.sh`。
- 脚本 07 提示需要 R：安装 R 与 HonestDiD（第 2 节），或加 `--allow-missing-r`。
- 内存不足：峰值约 7 GB，关闭其他程序或逐个运行脚本。
- `compare.py` 打印 `WARNING:` 并以退出码 2 结束：`results/` 缺少运行记录，或由另一版本的代码、另一数据包写成，重新运行 `bash run_all.sh`。

## 10. 术语

| 术语 | 含义 |
|---|---|
| 原稿、`v1` | 首次投稿所用的分析及其结果 |
| 基线编码器 | DeBERTa-v3-base，修订稿主分析所用，代码中写作 `E0` |
| 替代编码器 | ModernBERT-large、RoBERTa-large、Ettin-encoder-1b、DeBERTa-v2-xlarge，代码中写作 `E1`–`E4` |
| 参考轴 | 原稿语义向量（37,866 × 768）的前 88 个主成分；各编码器的 Y 投影到参考轴上可直接比较 |
| 共同支持 | 五个编码器截断到 512 个词元后都保留被遮蔽词的 37,808 个目标出现点 |
| U、A、Y | U 为编码器在被遮蔽位置的隐状态；A 为把 U 映射到 GPT-2 词嵌入空间的矩阵，每个编码器一个、三国共用；Y = (U − U 均值) Aᵀ 为语义向量 |
| θ、δ_UK | 外部对照中英美相对变化 θ = δ_UK − δ_US，δ_g 为成员国 g 相对对照国的签署前后变化 |
