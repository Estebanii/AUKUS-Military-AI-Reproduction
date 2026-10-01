# AUKUS 军事人工智能概念化研究：复现包（修订稿） / Replication package (revised manuscript)

本包复现修订稿定量部分（第四、五、六节及相关脚注）中印出的全部数字，以及表 1–9、图 1–4。所有计算都在 CPU 上完成，从单独分发的数据包开始，依次运行编号脚本即可；最后由 `compare.py` 逐项核对论文中印出的 397 个数字。数据包（66 个文件）约 3.6 GB（3,568,098,322 字节），作为本仓库的 GitHub Release 发布，标签 `rev1-data`：<https://github.com/Estebanii/AUKUS-Military-AI-Reproduction/releases/tag/rev1-data>，下载与校验见第 4 节。

## 最快路径 / Quick start

数据约 3.6 GB（3,568,098,322 字节，66 个文件），作为本仓库 GitHub Release `rev1-data` 的普通附件下载，不是 Git LFS。需要 bash、curl、tar 与 sha256sum 或 shasum；Python 环境用 [uv](https://docs.astral.sh/uv/) 安装；脚本 07 需要 R（第 3 节）。Windows 请在 WSL 或 Git Bash 中运行。The data (about 3.6 GB, 66 files) are plain assets of the GitHub Release `rev1-data` of this repository, not Git LFS. On Windows use WSL or Git Bash.

1. 取得本仓库 / Get the repository：在 GitHub 页面点 “Code → Download ZIP” 并解压（或 `git clone https://github.com/Estebanii/AUKUS-Military-AI-Reproduction.git`），进入得到的目录。
2. 下载并校验数据 / Get the data（下载、核对 sha256、解压到 `data/bundle/`、逐一核对 66 个文件；可重复运行，已完成的部分会跳过）：

   ```bash
   bash get_data.sh
   ```

3. 环境 / Environment（详见第 3 节；锁定文件适用于 macOS arm64，Linux x86_64 的锁定文件与命令见第 3 节；R 4.5 需已安装）：

   ```bash
   uv venv .venv --python 3.12.12
   VIRTUAL_ENV="$PWD/.venv" uv pip sync --require-hashes environment/requirements.lock
   Rscript environment/install_R_packages.R
   ```

4. 运行 / Run（约 10 分钟）：

   ```bash
   bash run_all.sh
   ```

5. 核对 / Check（应报告 394 个计算值全部 PASS，退出码 0；第一行是本次运行的记录）：

   ```bash
   .venv/bin/python compare.py
   ```

不用 git 也可以：从 ZIP 解压的目录没有 `.git`，运行记录中的提交号取自文件 `VERSION`。手工下载数据（逐个附件、大小与 sha256）见第 4 节。Without git, the run record takes the commit id from the file `VERSION`; manual download of the data: section 4.

## English summary

This package reproduces every number printed in the quantitative part of the revised manuscript (Tables 1–9, Figures 1–4 and the related footnotes and text). All computations run on a CPU from a separately distributed, checksummed data bundle (`data_bundle_rev1`, 66 files, about 3.6 GB, 3,568,098,322 bytes), published as the GitHub Release `rev1-data` of this repository (<https://github.com/Estebanii/AUKUS-Military-AI-Reproduction/releases/tag/rev1-data>; download and checks in section 4). This repository does not use Git LFS; the data are plain Release assets, and `bash get_data.sh` downloads, checks and unpacks them (see "Quick start"). Install the pinned Python environment (section 3), place the bundle (section 4), run `./run_all.sh` (about 10 minutes, section 5), and `compare.py` lists the 397 registered printed items, checks the 394 computed values among them against the recomputed results at the printed precision (the other 3 are labels, listed without testing) and reports whether each is also identical at full precision (section 6). Of the 394 computed items, 367 depend on the run; the other 27 (23 analysis parameters, 3 bundle constants, namely the vector width 768, and 1 archived value of the original submission) are verified by `compare.py` without results, so on an empty results directory exactly these 27 pass and 367 fail. Every random seed is set in one file, `code/replication/seeds.py`; section 7 lists each stochastic step with its seed, number of draws and the outputs it affects. In the pinned environment every recomputed number that has a full-precision reference (390 of the 394 computed values) is bit-identical to the results behind the manuscript; the other 4 are Figure 1's axis percentages, which have no archived full-precision value and match at the printed precision. The GPU encoding of the texts is documented but not required (`optional_full_pipeline/`).

## 目录 / Contents

- 最快路径 / Quick start（见上）

1. 内容概览
2. 目录结构
3. 环境
4. 数据包
5. 运行
6. 核对论文数字
7. 随机种子与可重复性 / Random seeds and reproducibility
8. 未纳入的部分
9. 运行时间与内存
10. 常见问题
11. 术语

## 1. 内容概览 / What is reproduced

| 论文项目 | 脚本 | 说明 |
|---|---|---|
| 表 2、第四节第三部分与第五节第一部分的样本数字；表 1 的样本量 | `01_sample_table2.py` | 分国家、分时期的出现点、文档与起止年月 |
| 表 3（H1）、脚注 72、78 | `02_h1_manova_table3.py` | MANOVA（88 个主成分）；检验 6 的主成分数稳健性（k = 3–99）；全样本三国 MANOVA 的主成分数与时段稳健性；替代编码器的 F 值 |
| 表 4–6、图 3（H2）、脚注 80 | `03_h2_did_tables4_6_fig3.py` | 子组 DiD、野聚类自助法、平行趋势检验、安慰剂检验（表 6 不含时间趋势项；另报含时间趋势项的安慰剂）、事件研究图 |
| 图 4（H3）、脚注 83、84 | `04_h3_neighbours_fig4.py` | GPT-2 近邻词 |
| 表 8 | `05_cross_encoder_table8.py` | 四个替代编码器（共同支持样本）与同轴稳健性 |
| 表 7、脚注 81 | `06_distance_table7.py` | 美英距离的签署前后变化（枢轴自助法） |
| 表 9、脚注 85 | `07_external_control_table9.py` | 外部对照（新加坡、加拿大）；事件研究的 HonestDiD 敏感性用 R |
| 图 1、图 2 | `08_figures_1_2.py` | PCA 与 t-SNE 散点 |
| 全部 397 个印出数字 | `compare.py` | 逐项核对 |

`00_prepare.py` 在最前面运行：它为五个编码器拟合 A 矩阵并计算语义向量 Y，并计算参考轴（原稿语义向量的前 88 个主成分，见第 11 节）。表 1 的其余数字（文档聚类数、向量维数、时间趋势起点）分别来自脚本 03 的结果、数据包与分析参数。逐项对照（论文位置 → 脚本 → 输出文件 → 字段 → 随机种子）见 `PAPER_MAP.md` 与 `PAPER_MAP.csv`。

## 2. 目录结构 / Layout

```
README.md                 本文件
PAPER_MAP.md / .csv       论文项目与印出数字的对照表
run_all.sh                依次运行全部脚本与 compare.py
get_data.sh               下载、核对并解压数据包到 data/bundle（第 4 节）
VERSION                   本包的提交号（GitHub 的 Download ZIP 与 git archive 导出时写入；git 克隆中为占位符）
compare.py                核对全部印出数字
expected/paper_numbers.csv  每个印出数字的位置、印出值、精度、结果文件与字段、全精度参考值
expected/tables/          表 5–9 的全精度参照（印出值与重算值；表 6 另列原稿值）
scripts/00_...08_*.py     编号脚本（按论文项目组织）
scripts/record_run.py     记录运行（results/run_info.json，供 compare.py 核对结果是否来自本次运行）
code/replication/         计算代码（纯函数，读普通文件，写 results/）
  seeds.py                全部随机种子与抽样次数（唯一设定处）
  params.py               分析参数（主成分数、自助法、各检验的设定）
  external/               外部对照检验（表 9）的代码、参数与 HonestDiD 的 R 脚本
environment/              Python 依赖锁定文件（含哈希；macOS arm64 与 Linux x86_64）、R 包清单与 R 包安装脚本
data/                     数据包清单 MANIFEST.sha256、Release 附件的 sha256 清单 RELEASE_ASSETS.sha256 与数据说明 README.md
supplementary/            可选的补充计算（距离区间的合成校准）
optional_full_pipeline/   上游 GPU 编码的说明（不需要运行）
results/                  运行后生成（不在版本库中）
```

## 3. 环境 / Environment

参考环境：macOS 26.6（Apple M3 Pro，arm64，12 核），36 GB 内存；Python 3.12.12；numpy / scipy 使用 Apple Accelerate。建议至少 16 GB 内存与 5 GB 可用磁盘（数据包约 3.6 GB 另计；下载的归档约 3.1 GB，解压后可删除）。

**Python.** 用 [uv](https://docs.astral.sh/uv/)（参考版本 0.9.2）按锁定文件安装，所有包的版本与哈希都已固定：

```bash
cd /path/to/package                # 本包根目录（ZIP 解压或 git clone 得到的目录）
uv venv .venv --python 3.12.12
VIRTUAL_ENV="$PWD/.venv" uv pip sync --require-hashes environment/requirements.lock
.venv/bin/python -c "import numpy, scipy, sklearn, pandas, torch, transformers; print('ok')"
```

新装环境中第一次 `import torch` 可能需要 1–3 分钟，之后很快。The first `import torch` after a fresh install can take 1–3 minutes.

锁定文件由 `environment/requirements.in` 生成（42 个包，主要为 numpy 2.3.5、scipy 1.16.3、pandas 2.3.3、pyarrow 22.0.0、scikit-learn 1.7.2、torch 2.9.1（仅 CPU 用途：GPT-2 嵌入）、transformers 4.57.3、matplotlib 3.10.8）。`environment/requirements.lock` 适用于 macOS arm64（参考环境）。Linux x86_64 用 `environment/requirements.lock.linux-x86_64`（版本相同，torch 为 PyTorch CPU 索引的 2.9.1+cpu）：

```bash
uv venv .venv --python 3.12.12
VIRTUAL_ENV="$PWD/.venv" uv pip sync --require-hashes --index-strategy unsafe-best-match environment/requirements.lock.linux-x86_64
```

该文件已针对 Linux x86_64 解析并带哈希核对安装，但未在 Linux 上运行：与参考结果逐位相同只在 macOS arm64 上核验过，在其他平台上以印出精度上的一致为准（第 7 节）。Resolved and installed (with hash checks) for Linux x86_64, not run on Linux: bit-identity was verified on macOS arm64 only; agreement at the printed precision is the criterion elsewhere. 其他平台（Windows 建议改用 WSL 与 Linux 锁定文件）先生成本平台的锁定文件再安装，例如：

```bash
uv pip compile environment/requirements.in --generate-hashes --python-version 3.12 \
    --python-platform x86_64-pc-windows-msvc -o environment/requirements.lock.local
VIRTUAL_ENV="$PWD/.venv" uv pip sync --require-hashes environment/requirements.lock.local
```

结果在末位可能不同，见第 7 节。

**R（仅脚本 07 需要）.** R 4.5.0，包 HonestDiD 0.2.8 与 jsonlite 2.0.0（依赖清单见 `environment/R_packages.txt`）。脚本在 PATH 中找 `Rscript`（找不到时用 `/usr/local/bin/Rscript`）。

用 `environment/install_R_packages.R` 安装参考版本 HonestDiD 0.2.8 与 jsonlite 2.0.0（安装到用户库；需要单独的库目录时先设 `R_LIBS_USER` 并建立该目录）。这两个版本仍是 CRAN 当前版本时，脚本安装 CRAN 的二进制包（macOS、Windows 不需要编译器；Linux 从源代码编译）；CRAN 有更新版本后，脚本用 `remotes::install_version` 从 CRAN 存档安装这两个确切版本。装到的版本不是 0.2.8 与 2.0.0 时脚本报错。HonestDiD 的依赖包按 CRAN 当前版本安装。

```bash
Rscript environment/install_R_packages.R
Rscript -e 'cat(as.character(packageVersion("HonestDiD")), as.character(packageVersion("jsonlite")), "\n")'   # 应为 0.2.8 2.0.0
```

从源代码安装需要编译工具链：Fortran 编译器（gfortran）、GMP 头文件（`gmp.h`）以及 Rust 工具链（`rustc`、`cargo`；HonestDiD 的依赖需要），例如用 Homebrew `brew install gcc gmp rust`，或 CRAN 为 macOS 提供的工具链。若装到的 HonestDiD 版本不是 0.2.8，事件研究的 HonestDiD 敏感性结果可能不同；表 9 与脚注 85 的数字不用 HonestDiD，不受影响。

没有 R 时脚本 07 会停止并提示；加 `--allow-missing-r` 可以继续，此时事件研究的 HonestDiD 敏感性记为“不可用”，表 9 与脚注 85 的数字不受影响（它们不用 HonestDiD）。

**图中的中文.** 图的中文标签需要系统中有中文字体（如 macOS 自带的 STHeiti、PingFang SC，或 Noto Sans CJK SC）；没有时图用英文标签，图中数据不变。

## 4. 数据包 / Data bundle

**数据位置 / Where the data are.** 数据包 `data_bundle_rev1`（66 个文件）约 3.6 GB（3,568,098,322 字节），作为本仓库的 GitHub Release 发布，标签 `rev1-data`：<https://github.com/Estebanii/AUKUS-Military-AI-Reproduction/releases/tag/rev1-data>。内容、来源与各列说明见 `data/README.md`。

Release 附件为每个顶层目录一个归档（均小于 2 GB；归档内的路径相对于数据包根目录）与逐文件清单 / The release assets are one archive per top-level directory of the bundle (each under 2 GB; paths inside are relative to the bundle root) and the file list:

| 附件 Asset | 内容 Content | 字节 Bytes | sha256 |
|---|---|---|---|
| `corpus.tar.gz` | `corpus/`（10 个文件 files） | 275,619,865 | `91bb2120ffcc8c79b106d9379d24807a7cf71a068223e929b3018220436738b8` |
| `encodings.tar.gz` | `encodings/`（20 个文件 files） | 1,963,729,540 | `e97e6ea68f135dcff192203f5ee3b56c19683e9e6b3da0344596c77918b807b3` |
| `external_controls.tar.gz` | `external_controls/`（23 个文件 files） | 380,916,536 | `142ac1ca9aeaa1d0dd5f886478bc5af1d8b5449375197b0f0a4761b85d1dd45e` |
| `gpt2.tar.gz` | `gpt2/`（6 个文件 files） | 463,976,185 | `6ee3671430e8db0456da72f02b3f7761d1ecd796ed7f60d1352a583667598e4a` |
| `original_outputs.tar.gz` | `original_outputs/`（7 个文件 files） | 24,562 | `5b6a4156f5d8a84b3faaae76dceca157ac99913e7c5202ac100d6779f24b8281` |
| `MANIFEST.sha256` | 逐文件清单（66 个文件）/ file list (66 files) | 8,042 | `6f1f783a09d9efa1628078a313ff3f8d9e2b5d1ba12d9d76fb2a73ec085f5a6c` |

**一条命令 / One command.** 在本包根目录运行 `bash get_data.sh`：它从 Release `rev1-data` 把上表六个附件下载到 `data/downloads/`（中断后续传，失败自动重试），按 `data/RELEASE_ASSETS.sha256`（与上表相同）核对每个附件，解压到 `data/bundle/`，再按 `data/MANIFEST.sha256` 逐一核对 66 个文件并打印结果；重复运行时跳过已存在且核对无误的部分。它只需要 bash、curl、tar 与 sha256sum 或 shasum（不需要 Python、git、Git LFS 或 GitHub 账号）；Windows 请在 WSL 或 Git Bash 中运行。完成后数据包即在默认位置 `data/bundle`，无需再放置。`get_data.sh` downloads the six assets with resume and retries, checks each against `data/RELEASE_ASSETS.sha256`, unpacks them into `data/bundle/` and checks all 66 files; it is idempotent and needs only bash, curl, tar and sha256sum or shasum.

**手工下载（备用）/ Manual download (fallback).** 下载、核对归档并解压到同一目录；之后按下文校验全部 66 个文件 / Download, check the archives and extract them into one directory; then run the checksum command below (66 files):

```bash
mkdir -p rev1-data data_bundle_rev1 && cd rev1-data
BASE=https://github.com/Estebanii/AUKUS-Military-AI-Reproduction/releases/download/rev1-data
for f in corpus.tar.gz encodings.tar.gz external_controls.tar.gz gpt2.tar.gz original_outputs.tar.gz MANIFEST.sha256; do curl -fLO "$BASE/$f"; done
shasum -a 256 -c <<'EOF'
91bb2120ffcc8c79b106d9379d24807a7cf71a068223e929b3018220436738b8  corpus.tar.gz
e97e6ea68f135dcff192203f5ee3b56c19683e9e6b3da0344596c77918b807b3  encodings.tar.gz
142ac1ca9aeaa1d0dd5f886478bc5af1d8b5449375197b0f0a4761b85d1dd45e  external_controls.tar.gz
6ee3671430e8db0456da72f02b3f7761d1ecd796ed7f60d1352a583667598e4a  gpt2.tar.gz
5b6a4156f5d8a84b3faaae76dceca157ac99913e7c5202ac100d6779f24b8281  original_outputs.tar.gz
6f1f783a09d9efa1628078a313ff3f8d9e2b5d1ba12d9d76fb2a73ec085f5a6c  MANIFEST.sha256
EOF
for a in corpus encodings external_controls gpt2 original_outputs; do tar -xzf "$a.tar.gz" -C ../data_bundle_rev1; done
cp MANIFEST.sha256 ../data_bundle_rev1/
```

手工下载时的放置方式，二选一 / Placement after a manual download, either:

```bash
ln -s /path/to/data_bundle_rev1 data/bundle          # 或 or
export REPLICATION_DATA=/path/to/data_bundle_rev1
```

校验全部文件的 sha256（约 10 秒）。第一行应输出 66；任何文件不符时 `shasum` 列出 `FAILED` 并以非零状态退出，不会输出 `all files OK`：

```bash
(cd /path/to/data_bundle_rev1 && grep -vc '^#' MANIFEST.sha256 &&
 grep -v '^#' MANIFEST.sha256 | awk '{print $1 "  " $3}' | shasum -a 256 -c --quiet && echo "all files OK")
diff <(grep -v '^#' /path/to/data_bundle_rev1/MANIFEST.sha256) <(grep -v '^#' data/MANIFEST.sha256) && echo "manifest matches"
```

Linux、WSL 与 Git Bash 上把 `shasum -a 256 -c` 换成 `sha256sum -c`（macOS 上请用 `shasum -a 256 -c`：macOS 自带的 `sha256sum` 不从标准输入读清单）。On Linux, WSL and Git Bash use `sha256sum -c` instead of `shasum -a 256 -c`. 上面手工下载命令中核对六个附件的 `shasum -a 256 -c` 同理 / The same applies to the check of the six assets in the manual download above.

## 5. 运行 / Running

一次运行全部（结果写入 `results/`，日志在 `results/logs/`）：

```bash
./run_all.sh
```

`run_all.sh` 使用 `.venv/bin/python`（可用环境变量 `PYTHON` 指定），先运行 `scripts/record_run.py` 记录本次运行（`results/run_info.json`），再依次运行 `scripts/00_prepare.py` 至 `scripts/08_figures_1_2.py`，最后运行 `compare.py`；任何一步失败即停止。也可以逐个运行（顺序同上，后面的脚本读取前面的结果）：

```bash
.venv/bin/python scripts/record_run.py
.venv/bin/python scripts/00_prepare.py
.venv/bin/python scripts/01_sample_table2.py
...
.venv/bin/python compare.py
```

依赖关系：00 为全部脚本提供 A、Y 与参考轴；06 读取 02–05 的结果做硬核对；07 读取 03 与 05 的朝向与元数据；compare.py 读取全部结果。

选项与环境变量：

| 名称 | 作用 |
|---|---|
| `./run_all.sh --full` | 脚本 07 另外重算两项设计模拟（推断选择；覆盖与尺寸，每个编码器 10–15 分钟），并对五个编码器都跑完整链条；重算的设计记录与数据包中的存档逐位核对。总时间多约 1 小时。 |
| `scripts/07_external_control_table9.py --recompute-design` / `--all-encoders` / `--allow-missing-r` | 同上，可单独使用 |
| `scripts/08_figures_1_2.py --encoder deberta-v3-base` | 用基线编码器的 Y（而非原语义向量）画图 1、2 |
| `REPLICATION_DATA` | 数据包目录（默认 `data/bundle`） |
| `REPLICATION_RESULTS` | 结果目录（默认 `results/`） |
| `REPLICATION_GPT2` | GPT-2 文件目录（默认数据包中的 `gpt2/`；文件的 sha256 会被核对） |
| `REPLICATION_BLAS_THREADS` | BLAS 线程数（默认：脚本 07 为 12，其余为 4；见第 7 节）。脚本总是显式设定 `VECLIB_MAXIMUM_THREADS`、`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`，继承的这些变量不起作用；每个脚本的日志第一行打印实际线程数 |

补充（可选，不影响任何印出数字）：`supplementary/distance_calibration.py` 用合成数据校准表 7 的区间与读法（只读取数据包中出现点的元数据——国家、年月、文档——以得到真实的格结构，不读取任何向量；88 维约 12 分钟；`--k 1`、`--k 3`、`--stress` 约 1–2 分钟；`--variant e|f` 各约 70–80 分钟），结果写入 `results/supplementary/`。

## 6. 核对论文数字 / Checking the manuscript's numbers

```bash
.venv/bin/python compare.py
```

**覆盖范围 / Coverage.** `compare.py` 核对修订稿定量部分（第四、五、六节及其脚注）与表 1–9 中登记的 397 个标量数字（其中 3 个是标签，只列出不检验）。印出的标签（年份、“95%”、“5%”）、设计常数以及文字判断（脚注 80 关于 PC1 在两种安慰剂规格下均不显著的判断；第五节第四、五部分：近邻词表的五个编码器、四个替代编码器中的三个、前四位、多数编码器、三种情况等，以及跨编码器检验与外部对照的定性表述）不在这 397 项之内，列于 `PAPER_MAP.md` 末尾的“文字判断 / Textual claims”一节，并注明支持它们的结果字段；“近三倍”（MDE 0.0934 ÷ 主结果 0.0333 = 2.81）作为派生量列在同一节。compare.py checks the 397 registered scalar items of the manuscript's quantitative sections and Tables 1–9; labels (years, "95%", "5%"), design constants and the qualitative claims (footnote 80 on PC1; Sections V.4 and V.5: word lists, cross-encoder and external-control statements) are listed in `PAPER_MAP.md`, "Textual claims", with the results fields that support them.

**结果来自哪一次运行 / Which run.** `run_all.sh` 先写 `results/run_info.json`：运行编号（UTC 开始时间 + 本包提交号的前 12 位 + 数据包清单 sha256 的前 12 位）、完整的提交号及其来源、代码指纹（`code/`、`scripts/`、`run_all.sh` 与锁定文件的 sha256）与数据包清单的 sha256。提交号在 git 克隆中取 `git rev-parse HEAD`（来源 `git`）；从 GitHub 下载的 ZIP 或 `git archive` 导出的目录没有 `.git`，提交号取自文件 `VERSION`（导出时由 git 写入，来源 `VERSION`）。`compare.py` 第一行打印这条记录，形如：

```
Results of run <UTC 时间>-<提交号前 12 位>-<清单前 12 位> (package commit <提交号> from <git 或 VERSION>, code <代码指纹前 12 位>, data manifest <清单 sha256>)
```

若该文件缺失，或其中的提交号、代码指纹或数据包清单与当前这份复现包不同，`compare.py` 在开头与结尾打印 `WARNING:`，并以退出码 2 结束（同时有 FAIL 时为 1）：这说明 `results/` 不是这份复现包的一次完整运行，应重新运行 `./run_all.sh`。没有 `.git` 本身不会引起警告。compare.py prints the run record first (the commit is read from git, or from the file VERSION in a ZIP download or `git archive` export) and, if the record is missing or its commit, code fingerprint or data manifest differs from this copy of the package, warns at the top and the bottom and exits with status 2; a missing .git alone is not a problem.

`expected/paper_numbers.csv` 列出定量部分印出的全部 397 个数字（位置、印出值、印出精度），以及每个数字对应的结果文件与 JSON 字段（或数据包文件、分析参数、由结果计算的表达式）和论文所据结果的全精度参考值。`compare.py` 对每个数字：

- 按论文的舍入规则（对最短十进制表示四舍五入，印出的小数位数；百分数乘 100）把重算值与印出值比较，得到 PASS / FAIL；印出 `<0.001` 的 p 值检验重算值小于 0.001；表 4 的显著性星号按自助法 p 值核对；
- 另报重算值与全精度参考值是否逐位相同（`exact`）。

输出：每个论文项目一行（`PASS  Table 4  48/48 items; 48 identical ...`），失败项逐条列出；明细写入 `results/compare/items.csv` 与 `summary.csv`；有任何 FAIL 时退出码为 1。3 个印出的是标签而不是计算值（脚注 75、78 中份额所属的年份“2021”，脚注 85 的“2021年1月至8月”），只列出不检验。

参考运行的结果：394 个计算的数字全部 PASS，其中 390 个与参考值逐位相同；其余 4 个是图 1 坐标轴上的方差占比，按图本身的 PCA（原语义向量，脚本 08）核对，原图没有存档的全精度数值，因此没有全精度参考值（基线编码器分析用的 PCA-3 给出 7.001706% 与 5.429895%，图本身为 7.001905% 与 5.429906%，印出精度下都是 7.0% 与 5.4%）；3 个标签未检验。论文印出值与重算值之间没有不一致。

394 个计算值中，367 个依赖本次运行的结果；另外 27 个不读取 `results/`：23 个分析参数（安慰剂断点年份、方向阈值、外部对照的样本窗口、近邻词个数、时间趋势起点）、3 个数据包常数（基线编码器隐状态宽度 768）与 1 个原稿存档值（表 8 注引用的原稿 PC1 英国×签署后系数），`compare.py` 直接按代码参数或数据包核对它们；在空的结果目录上运行 `compare.py`，恰好这 27 项 PASS、其余 367 项 FAIL（退出码 1）。 Of the 394 computed items, 367 depend on the run; the other 27 (23 analysis parameters, 3 bundle constants, namely the vector width 768, and 1 archived value of the original submission) are verified by `compare.py` without results, so on an empty results directory exactly these 27 pass and 367 fail.

`expected/tables/` 另以全精度列出表 5–9 的各格（印出值、重算值；表 6 另列原稿首次投稿时的值：PC2 的两个 p 值与 PC3 的 2021 年 p 值与重算值不同，即第 7 节所说的早先种子方案的结果，修订稿印出的是重算值），供对照阅读；其中每个印出的数字都已包含在 `paper_numbers.csv` 中，由 `compare.py` 逐项核对。

### 脚注定位 / Locating footnotes

修订稿的脚注以带圈数字（①②③…）编号，并且每页重新从①开始，因此无法在稿件中按“脚注 85”这样的编号直接查找。本包各处（`expected/paper_numbers.csv`、`PAPER_MAP.md`、`PAPER_MAP.csv`、`compare.py` 的输出）所说的“脚注 n / Footnote n”，指该脚注在全文中的顺序号（从第一个脚注起连续计数）。下表列出本包引用的每个脚注所在的节及其开头文字（前 28 个字符），可据此在稿件中定位。

The manuscript numbers its footnotes with circled numerals that restart on every page. The package refers to a footnote by its sequential position in the whole manuscript; the table gives the section and the opening words of every footnote the package cites.

| 顺序号 No. | 所在节 Section | 开头文字 Opening words |
|---|---|---|
| 1 | 正文之前 (front matter) | 本文的复现代码和数据可在https://github.c… |
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

## 7. 随机种子与可重复性 / Random seeds and reproducibility

全部随机种子与抽样次数只在 `code/replication/seeds.py` 设定，其余代码从那里读取（`code/replication/params.py` 与 `code/replication/external/params.py` 引用它的常量）。All seeds and resampling counts are set in `code/replication/seeds.py` only.

| 步骤 Step | 种子 Seed | 抽样次数 Draws | 设定处 Where set | 影响的输出 Output affected |
|---|---|---|---|---|
| H2 主模型的野聚类自助法（Rademacher 权重，按文档聚类；表 4 的标准误、p 值、区间与星号） | 42、43、44（PC1、PC2、PC3；`numpy.random.RandomState`） | 1,000 | `WCB_SEED_BASE`、`WCB_DRAWS` | `results/h2/wcb.json`、`same_axis.json` |
| 稳健性模型（年份固定效应、2017 年起、时间趋势）与安慰剂检验（表 6） | 42、43、44（按主成分） | 1,000 | `WCB_SEED_BASE`、`WCB_DRAWS` | `results/h2/did.json` |
| 含时间趋势项的安慰剂检验（签署前样本，虚拟断点 2020、2021；主模型加虚拟断点；脚注 80） | 42、43、44（按主成分） | 1,000 | `WCB_SEED_BASE`、`WCB_DRAWS` | `results/h2/placebo_with_trend.json` |
| 平行趋势事件研究（美英，2014–2024；表 5、图 3）的自助法协方差与 Wald 检验 | 42、43、44（按主成分） | 1,000 | `WCB_SEED_BASE`、`WCB_DRAWS` | `results/h2/parallel_trends.json` |
| 稳健性部分的三国事件研究 Wald 检验（2017–2024） | 42（三个主成分相同，沿用原设定） | 1,000 | `EVENT_STUDY_WALD_SEED`、`WCB_DRAWS` | `results/h2/did.json`（`model_6_proper_wald`） |
| 替代编码器的主模型与同轴模型（表 8 的两个 Holm 检验族） | 42、43、44（按主成分） | 1,000 | `WCB_SEED_BASE`、`WCB_DRAWS` | `results/alternatives/*/wcb.json`、`same_axis.json`、`results/cross_encoder/cross_encoder.json` |
| 表 8 中 PC1 英国×签署后的描述性重跑（p = (k + 1)/(B + 1)，与主结果并列，不替代） | 20260929 | 9,999 | `RERUN_SEED`、`RERUN_DRAWS` | `results/alternatives/*/rerun_descriptive.json`、`cross_encoder.json` |
| 同轴稳健性（安慰剂与签署前 Wald 检验在参考轴与各编码器自身轴上；事后补充的描述，论文未印出其数字） | 42、43、44 | 1,000 | `WCB_SEED_BASE`、`WCB_DRAWS` | `results/cross_encoder/same_axis_robustness/*.json` |
| 距离变化的枢轴自助法（在国家 × 时期的格内按文档有放回重抽；表 7、脚注 81） | 20260929（`numpy.random.default_rng`，固定抽样顺序） | 2,000 | `DISTANCE_SEED`、`DISTANCE_DRAWS` | `results/distance/*.json` |
| 外部对照的循环月块自助法（块长 6，另报 3、12；表 9 的描述性行、虚拟断点诊断、签署前斜率、安慰剂、敏感性样本；目标对比不可估计的抽样由其后从同一生成器抽出的补充抽样替换，最多占 1%） | 20260928（PCG64） | 9,999（另抽 199 次补充抽样 = 2 × ⌊0.01 × 9,999⌋ + 1） | `EXTERNAL_SEED`、`EXTERNAL_BLOCK_BOOTSTRAP_DRAWS`、`EXTERNAL_MAX_FAILURE_SHARE` | `results/external/*/estimate/*.json` |
| 外部对照推断的选择与校准：八个候选推断在五种误差情形下的开发模拟，得出校准的 CR2 临界值（表 9 的 p 值与区间所用的 C8）；MDE 用这些临界值，本身不另抽样 | 20260928（`SeedSequence([种子, 样本序号])` 按情形派生） | 每种情形 2,000 次 | `EXTERNAL_SEED`、`EXTERNAL_SIMULATION_REPLICATIONS` | `design/inference_selection.json`、`design/mde.json`（脚注 85 的 0.277 SD）、表 9 |
| 选中推断的独立验证模拟 | 20260929 | 每种情形 2,000 次 | `EXTERNAL_VERIFICATION_SEED`、`EXTERNAL_SIMULATION_REPLICATIONS` | `design/inference_selection.json` |
| 覆盖与尺寸模拟（事件研究敏感性与签署前斜率的判定标签；其中 HonestDiD 在约束内的描述性检查取新加坡窗口 iid 情形的前 50 次） | 20260928 | 每种情形 2,000 次；HonestDiD 检查 50 次 | `EXTERNAL_SEED`、`EXTERNAL_SIMULATION_REPLICATIONS`、`EXTERNAL_HONESTDID_CHECK_REPLICATIONS` | `design/simulation.json` |
| 外部对照中并列报告的文档聚类野聚类自助法 | 42、43、44 | 1,000 | `WCB_SEED_BASE`、`WCB_DRAWS` | `estimate/main.json`（`wcb_v1`） |
| 图 1 的 PCA | `random_state=42`（沿用原作图脚本；在锁定的 scikit-learn 下，此形状选用 `covariance_eigh` 求解器，结果与种子无关） | — | `FIGURE_PCA_RANDOM_STATE` | `results/figures/figure1_pca.*` |
| 图 2 的 t-SNE（先 PCA 到 50 维；困惑度 30，1,000 次迭代，Barnes–Hut） | PCA 与 t-SNE 均为 `random_state=42` | — | `FIGURE_PCA_RANDOM_STATE`、`TSNE_RANDOM_STATE` | `results/figures/figure2_tsne.*` |
| 分析用的全部 PCA（参考轴、88 个主成分、检验 6 的主成分数网格、各编码器的前 3 个主成分） | 无：`svd_solver='full'`（LAPACK 完整奇异值分解），确定性 | — | `params.py` 的 `pca` | 全部 |
| A 矩阵、MANOVA、近邻词、HonestDiD（R） | 无：确定性计算 | — | — | — |
| 锚点抽样（上游，一次性完成，不重跑）：49,999 个锚点按国家比例分层抽取 | `random_state=42`（`pandas.DataFrame.sample`） | — | `ANCHOR_SAMPLE_RANDOM_STATE` | 数据包 `corpus/anchor_occurrences.parquet` |
| 补充：距离区间的合成校准（`supplementary/distance_calibration.py`） | 数据：`SeedSequence([20260930, 设定序号])`（Δ_sq = 0 的设定从 0 起，非零设定从 100 起，压力设定从 200 起），方向向量 `default_rng([20260930, 999])`；自助法 20260929 | 每个设定 500 次（`--reps` 的默认值）；B = 2,000 | `CALIBRATION_DATA_SEED`、`CALIBRATION_REPS`、`CALIBRATION_DIRECTION_KEY`、`CALIBRATION_NONZERO_INDEX_OFFSET`、`CALIBRATION_STRESS_INDEX_OFFSET`、`DISTANCE_SEED`、`DISTANCE_DRAWS` | `results/supplementary/distance_calibration*.json` |

**逐位可重复 / Bit-identical reruns.** 在锁定环境（第 3 节）与默认的 BLAS 线程数下，重跑得到的每个有全精度参考值的数字都与论文所据的结果逐位相同：参考运行中 `compare.py` 报告有全精度参考值的 390 个计算值全部 `exact`（另 4 个为图 1 坐标轴的方差占比，没有全精度参考值，见第 6 节）。浮点运算的结果与线程数有关，因此脚本在载入 numpy 之前固定 BLAS 线程数：原结果的主分析用 4 个线程计算，外部对照检验（脚本 07）用 12 个线程计算，脚本默认沿用（`scripts/_common.py` 在载入 numpy 之前显式设定线程变量，覆盖继承的设置，并在日志第一行打印实际线程数；可用 `REPLICATION_BLAS_THREADS` 覆盖）。

**跨平台 / Other platforms.** 在不同的 CPU、BLAS 库、线程数或操作系统上，矩阵运算的舍入顺序不同，结果可能在末几位不同；自助法的抽样本身由种子决定，不受影响。`compare.py` 在论文的印出精度上比较，因此这类差异不会造成 FAIL；它另报每个数字是否逐位相同，差异会显示为 `differs by …`。

**原稿表 5、表 6 的说明 / Note on the original submission's Tables 5–6.** 首次投稿表 5、表 6 的部分自助法结果使用共同种子 42；修订稿与本包对 PC1、PC2、PC3 分别使用 42、43、44。表 6 的首次投稿值与修订稿值并列于 `expected/tables/tab6_placebo.csv`。

## 8. 未纳入的部分 / What is not included

- **表 9 的设计模拟。** 默认运行中，脚本 07 以数据包中存档的两项设计模拟记录（推断选择；覆盖与尺寸模拟）为条件，重算 MDE 与全部估计；`./run_all.sh --full` 重算这两项模拟（每个编码器 10–15 分钟），并与存档逐位核对。
- **GPU 编码。** 编码器输出 U 与编码输入表都在数据包中；重新编码不是必需步骤，编码方法与设置见 `optional_full_pipeline/README.md`。
- **不在论文中的分析。** 原分析流程中不进入修订稿的部分（外部对照的迁移诊断、编码器适用性预试、外部对照检验族的 Holm 汇总报告）没有纳入；外部对照中表 9 所需的全部估计都已纳入。
- **记录字段。** 原结果文件中的运行记录与来源哈希等字段不属于计算结果，本包的输出不含这些字段；数值字段全部保留。

## 9. 运行时间与内存 / Runtime and memory

下表给出 Apple M3 Pro（12 核、36 GB）上一次参考运行的时间和峰值内存。峰值内存为单个脚本进程的最大常驻内存，记录于 `results/logs/runtime.tsv`，随系统负载波动：脚本 00 在多次从空目录的运行中为 5.1–7.1 GB，表中按约 7 GB 给出；其余脚本各次运行相差不到 0.5 GB。

| 步骤 Step | 时间 Time (s) | 峰值内存 Peak memory (GB) |
|---|---|---|
| 建立 Python 环境（`uv venv` + `uv pip sync`；uv 已缓存安装包，首次下载另需网络与时间） | 73 | — |
| 校验数据包（66 个文件的 sha256） | 8 | — |
| `00_prepare.py` | 38 | 约 7 |
| `01_sample_table2.py` | 0.1 | 0.1 |
| `02_h1_manova_table3.py` | 46 | 1.5 |
| `03_h2_did_tables4_6_fig3.py` | 13 | 1.3 |
| `04_h3_neighbours_fig4.py` | 7 | 2.0 |
| `05_cross_encoder_table8.py` | 134 | 5.5 |
| `06_distance_table7.py` | 6 | 2.5 |
| `07_external_control_table9.py` | 208 | 3.9 |
| `08_figures_1_2.py` | 92 | 1.6 |
| `compare.py` 与各脚本的启动 | 约 20 | — |
| `./run_all.sh` 合计 | 562 | 约 7 |
| `07_external_control_table9.py --recompute-design --all-encoders`（`--full` 中的脚本 07） | 3,408 | 5.1 |

磁盘：`results/` 约 1.1 GB（几乎全部是 `results/intermediate/` 中各编码器的 A 与 Y），虚拟环境约 0.9 GB。

## 10. 常见问题 / Troubleshooting

- `data bundle not found`：运行 `bash get_data.sh`，或设置 `REPLICATION_DATA`、建立 `data/bundle` 链接（第 4 节）。
- `get_data.sh` 报告 `could not download <地址>`：Release `rev1-data` 尚未发布、地址不可达或网络中断；在浏览器中打开报告的地址、检查网络后重新运行 `bash get_data.sh`（已核对的附件保留，中断的下载会续传），退出码非零。
- `get_data.sh` 报告某个附件的 sha256 不符：下载不完整或被改动，脚本已删除该文件；重新运行即可（已核对的附件会跳过）。
- 从 ZIP 解压后 `./get_data.sh` 或 `./run_all.sh` 提示没有执行权限：改用 `bash get_data.sh`、`bash run_all.sh`。
- `run scripts/00_prepare.py first` / `run the script that writes it first`：按编号顺序运行，或直接用 `./run_all.sh`。
- `GPT-2 file ... sha256` 不符：数据包的 `gpt2/` 被改动或不完整，重新校验数据包。
- 脚本 07 提示需要 R：安装 R 与 HonestDiD（第 3 节），或加 `--allow-missing-r`。
- 内存不足：峰值内存约 7 GB（脚本 00，见第 9 节）。建议至少 16 GB 内存；关闭其他程序，或逐个运行脚本。
- `compare.py` 报告 `differs by …` 但 PASS：见第 7 节“跨平台”；不影响论文数字。
- `compare.py` 打印 `WARNING:` 并以退出码 2 结束：`results/run_info.json` 缺失，或 `results/` 由另一版本的代码、另一数据包或未完成的运行写成；重新运行 `./run_all.sh`（第 6 节）。

## 11. 术语 / Glossary

| 术语 | 含义 |
|---|---|
| 原稿 / original submission、`v1` | 首次投稿所用的分析。结果文件与代码中的 `v1` 均指它（如 `v1_sample`：原稿的全样本口径）。 |
| 基线编码器 / baseline encoder | DeBERTa-v3-base（重新编码），修订稿主分析所用；代码中也写作 `E0`。 |
| 替代编码器 / alternative encoders | ModernBERT-large、RoBERTa-large、Ettin-encoder-1b、DeBERTa-v2-xlarge（`E1`–`E4`）。 |
| 参考轴 / reference axes | 原稿（首次投稿）的语义向量（数据包 `Y_vector_global`，由 `data.original_vectors` 读取；37,866 × 768）的前 88 个主成分，不是重新编码的基线编码器的主成分；各编码器的 Y 投影到参考轴上可直接比较（“同轴”）。The first 88 principal components of the original submission's vectors, not of the re-encoded baseline. |
| 共同支持 / common support | 五个编码器截断到 512 个词元后都保留被遮蔽词的 37,808 个目标出现点。 |
| U、A、Y | U：编码器在被遮蔽位置的隐状态；A：把 U 映射到 GPT-2 词嵌入空间的矩阵（每个编码器一个，三国共用，即全局统一的 A；用 100 个锚词的 49,999 个出现点拟合）；Y = (U − U 均值) Aᵀ：语义向量。 |
| M1、θ、δ_UK | 外部对照的主模型；θ = δ_UK − δ_US（英美相对变化），δ_g 为成员国 g 相对对照国的签署前后变化。 |
