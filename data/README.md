# 数据包说明 / Data bundle

复现所需的全部数据在一个单独分发的数据包 `data_bundle_rev1` 中（66 个文件）约 3.6 GB（3,568,098,322 字节），作为本仓库的 GitHub Release 发布，标签 `rev1-data`：<https://github.com/Estebanii/AUKUS-Military-AI-Reproduction/releases/tag/rev1-data>。本目录保存数据包的逐文件清单 `MANIFEST.sha256`（sha256、字节数、相对路径）与本说明。

The data are distributed separately as the bundle `data_bundle_rev1` (66 files, 3,568,098,322 bytes), published as the GitHub Release `rev1-data` of this repository (<https://github.com/Estebanii/AUKUS-Military-AI-Reproduction/releases/tag/rev1-data>). This directory holds its file list with checksums, `MANIFEST.sha256`, and this note.

## 1. 获取、放置与校验 / Download, placement and verification

Release `rev1-data` 的附件为每个顶层目录一个归档（均小于 2 GB；归档内的路径相对于数据包根目录）与逐文件清单 / One archive per top-level directory (each under 2 GB) and the file list:

| 附件 Asset | 内容 Content | 字节 Bytes | sha256 |
|---|---|---|---|
| `corpus.tar.gz` | `corpus/`（10 个文件 files） | 275,619,865 | `91bb2120ffcc8c79b106d9379d24807a7cf71a068223e929b3018220436738b8` |
| `encodings.tar.gz` | `encodings/`（20 个文件 files） | 1,963,729,540 | `e97e6ea68f135dcff192203f5ee3b56c19683e9e6b3da0344596c77918b807b3` |
| `external_controls.tar.gz` | `external_controls/`（23 个文件 files） | 380,916,536 | `142ac1ca9aeaa1d0dd5f886478bc5af1d8b5449375197b0f0a4761b85d1dd45e` |
| `gpt2.tar.gz` | `gpt2/`（6 个文件 files） | 463,976,185 | `6ee3671430e8db0456da72f02b3f7761d1ecd796ed7f60d1352a583667598e4a` |
| `original_outputs.tar.gz` | `original_outputs/`（7 个文件 files） | 24,562 | `5b6a4156f5d8a84b3faaae76dceca157ac99913e7c5202ac100d6779f24b8281` |
| `MANIFEST.sha256` | 逐文件清单（66 个文件）/ file list (66 files) | 8,042 | `6f1f783a09d9efa1628078a313ff3f8d9e2b5d1ba12d9d76fb2a73ec085f5a6c` |

本仓库不使用 Git LFS；数据全部作为 Release 附件以普通 HTTPS 下载分发 / This repository does not use Git LFS; the data are plain Release assets. 最简单的方式是在本包根目录运行 `bash get_data.sh`：下载上表的附件（中断后续传）、按 `data/RELEASE_ASSETS.sha256` 核对、解压到 `data/bundle/` 并按清单逐一核对 66 个文件；只需要 bash、curl、tar 与 sha256sum 或 shasum，Windows 请用 WSL 或 Git Bash。`bash get_data.sh` does all of this (resume, retries, checks; idempotent).

手工方式：下载、核对并把全部归档解压到同一目录（命令见主 README 第 4 节），再把 `MANIFEST.sha256` 放入该目录，然后运行下面的校验（66 个文件）。Extract all archives into one directory, add `MANIFEST.sha256`, then run the check below (66 files; commands in the main README, section 4).

放置，二选一 / Placement, either:

- 把数据包目录放到（或软链接到）`data/bundle`：`ln -s /path/to/data_bundle_rev1 data/bundle`；或
- 设置环境变量 `REPLICATION_DATA=/path/to/data_bundle_rev1`。

校验（在数据包目录内运行；第一行应为 66；任何文件不符时 `shasum` 列出 `FAILED` 并以非零状态退出）/ Verify (inside the bundle directory; the first line must be 66; a mismatch prints `FAILED` and exits non-zero):

```bash
cd /path/to/data_bundle_rev1
grep -vc '^#' MANIFEST.sha256
grep -v '^#' MANIFEST.sha256 | awk '{print $1 "  " $3}' | shasum -a 256 -c --quiet && echo "all files OK"
# Linux、WSL、Git Bash：... | sha256sum -c --quiet && echo "all files OK"
diff <(grep -v '^#' MANIFEST.sha256) <(grep -v '^#' /path/to/package/data/MANIFEST.sha256) && echo "manifest matches the package"
```

## 2. 内容 / Contents

| 路径 Path | 内容 Content |
|---|---|
| `corpus/occurrences.parquet` | 原稿的 37,866 个目标术语出现点（美、英、澳国防部门文档，11,459 篇）；行序即全部分析的行序。共 16 列：`occurrence_id`（出现点编号）；`term` 与 `matched_term`（术语及其在文中的匹配形式）；`article_id` 与 `doc_id`（文档编号，`doc_id` 为自助法的聚类单元）；`country`、`year`、`month`；`post_aukus`（2021 年 9 月起为 1）；`ai_similarity` 与 `pos_type`（该术语在 `terms.csv` 中的相似度得分与词性，逐行重复）；`text_block`（段落文本）；`source_type`（来源类型）；`concept_id` 与 `concept_label`（概念编号与术语标签）；`Y_vector_global`（原稿的语义向量，768 维，float64；参考轴与原稿图 1–2 的基础）。The 37,866 target-term occurrences of the original submission (16 columns: identifiers, term, document, country, year, month, Post, term attributes, paragraph text, and the original semantic vectors `Y_vector_global`). |
| `corpus/raw/{us,uk,au}.csv` | 原稿的原始语料：美国、英国、澳大利亚国防部门网站的文档，与首次投稿公开发布的 `data/raw/*.csv` 逐字节相同（us 18,047 条、uk 9,320 条、au 6,936 条 CSV 记录；字段内含换行，`wc -l` 得到的行数更大）。4 列：`title`（标题）；`publish_date`（发布日期，沿用各网站格式：美国为“Nov. 24, 2025”式，英、澳为“2025-10-06”式）；`content`（正文；美国 108 行、英国 1 行为空）；`type`（文档类型：news、release、contract、speech、statement、unknown）。`corpus/occurrences.parquet` 由原稿流程从这些文档中抽取；本包的计算从 `occurrences.parquet` 开始，不重跑抽取。The raw corpus of the original submission (byte-identical to its published `data/raw/*.csv`); columns `title`, `publish_date` (site format), `content`, `type`. |
| `corpus/encoding_inputs/{targets,anchors}.parquet` | 编码器读取的输入表：37,866 个目标出现点与 49,999 个锚点，行序与 `encodings/<encoder>/*_rows.parquet` 的 `row_key` 一致。9 列：`row_index`（行号）；`row_key`（目标为 `t:<出现点编号>`，锚点为 `a:<源行>`）；`source`（target / anchor）；`source_row`；`occurrence_id`；`text`（送入编码器的文本：目标行为所在段落，锚点行为锚词所在的文本片段）；`term`（目标词或锚词）；`start_char`（该词在 `text` 中的起始字符位置）；`pos_type`（词性）。基线编码器读取全部行；替代编码器读取按 `common_support_map.parquet` 选出的 37,808 个目标行（`row_index` 重新从 0 编号）与全部锚点。各行各列与编码时的输入逐一相同。The tables the encoders read (paragraph text, target word, start character, POS); identical in every row and column to the encoding inputs. |
| `corpus/terms.csv` | 56 个目标术语及其类别、词性与各国出现次数。The 56 target terms with category, part of speech and counts by country. |
| `corpus/anchor_words.json` | A 矩阵的 100 个锚词。该文件的 description 字段沿用原稿分析的措辞“country-specific matrix A”，实际使用的是全局统一的 A 矩阵（见 README 术语表与正文），文件内容未改。The 100 anchor words; the description field keeps the original analysis's wording ("country-specific matrix A"), while one global matrix A is used (README glossary); the file is unchanged. |
| `corpus/anchor_occurrences.parquet` | A 矩阵训练所用的 49,999 个锚词出现点（从全部锚词出现点中按国家比例分层抽取 50,000 × 各国占比并取整，共 49,999 个；抽样一次性完成，种子见 README 第 7 节），行序即锚点编码的行序。The 49,999 anchor occurrences used to fit A. |
| `corpus/common_support_map.parquet` | 五个编码器的共同支持：37,808 个目标出现点与 49,999 个锚点在原行序中的位置（截断到 512 个词元后，任一编码器的输入中不含被遮蔽词的行被剔除）。The common-support rows of the five encoders. |
| `encodings/<encoder>/{targets,anchors}_U.npy` 与 `_rows.parquet` | 五个编码器（DeBERTa-v3-base 为基线；ModernBERT-large、RoBERTa-large、Ettin-encoder-1b、DeBERTa-v2-xlarge 为替代编码器）在被遮蔽位置的最后一层隐状态 U（float32），及每行的行键与遮蔽记录。基线编码器为全部 37,866 / 49,999 行，替代编码器为共同支持行。编码方法见 `optional_full_pipeline/README.md`。Hidden states at the masked position (the encoder output; GPU encoding is not re-run). |
| `external_controls/documents.parquet` | 外部对照检验（表 9）的文档表：美、英、澳、新加坡、加拿大五国的 47,726 篇文档（标题、正文、发布日期、体裁、来源网址、归档时间戳、是否保留等）；三个成员国的行提供发布日期与体裁，新加坡与加拿大的行是对照国语料。The document table of the external-control analysis (all five countries). |
| `external_controls/targets_meta.parquet` | 新加坡与加拿大的 13,318 个目标术语出现点的元数据（国家、文档、年、月、术语、体裁）。Metadata of the 13,318 control-country occurrences. |
| `external_controls/encoding_inputs/targets.parquet` | 新加坡与加拿大 13,318 个目标出现点的编码输入表（列同 `corpus/encoding_inputs/`；`row_key` 为 `t:SG:<编号>` 或 `t:CA:<编号>`，行序与 `targets_meta.parquet` 及 `external_controls/encodings/` 一致）。The encoding inputs of the control-country occurrences. |
| `external_controls/encodings/<encoder>/targets_U.npy` 与 `_rows.parquet` | 这些出现点在五个编码器下的隐状态 U 与行键。Their encodings. |
| `external_controls/design_records/<encoder>/{inference_selection,simulation}.json` | 表 9 推断方法的设计模拟记录（只用设计：行、国家、月份、文档、权重，不读取结果变量）：候选推断的开发与独立验证模拟及选择轨迹、覆盖与尺寸模拟。它们是 `selection.run` 与 `simulation.run` 的返回值；脚本 07 默认直接读取，`--recompute-design` 会重新计算并逐位核对（每个编码器 10–15 分钟）。The design-only simulation records of Table 9 (recomputable with `--recompute-design`). |
| `gpt2/` | GPT-2 模型文件（Hugging Face `openai-community/gpt2`，修订 `607a30d783dfa663caf39e06633721c8d4cfcd7e`；只用其词元化器与输入词嵌入矩阵，构造锚词标签与第 H3 部分的近邻词）。代码会逐一核对六个文件的 sha256（`code/replication/params.py`）。也可用 `REPLICATION_GPT2` 指向另一份相同文件。The GPT-2 files (tokenizer and input embeddings only), hash-checked by the code. |
| `original_outputs/*.json` | 原稿（首次投稿）分析的 7 个结果文件，逐一说明见下表。The 7 result files of the original submission's analysis (see the table below). |

`original_outputs/` 各文件 / The files of `original_outputs/`（文件名沿用原稿分析的编号：`did_h3_verification.json` 的 H3 指原稿的距离假说，`h4_nearest_neighbor_results.json` 的 H4 指原稿的近邻词分析，即修订稿的 H3；七个文件与首次投稿公开发布的数据文件逐字节相同，其中的说明性字段为原稿分析自带 / file names follow the numbering of the original analysis; the seven files are byte-identical to the data files published with the original submission, including their descriptive fields):

| 文件 File | 内容 Content | 读取者 Read by |
|---|---|---|
| `wild_cluster_bootstrap_results.json` | 原稿主模型（H2）的 OLS 与 wild cluster bootstrap 系数、标准误与 p 值（PC1–PC3）。Original main-model coefficients with OLS and wild-cluster-bootstrap inference. | 脚本 05（原稿 PC1 英国×签署后系数，用于方向与效应比较）；`compare.py`（表 8 注中的基线估计）。Script 05; `compare.py`. |
| `did_h3_verification.json` | 原稿三国签署前后的欧氏距离及其一致性核对结果。Original pre/post Euclidean distances of the three countries. | 脚本 06（原稿距离的逐位复现核对）。Script 06 (exact reproduction check). |
| `did_robustness_results.json` | 原稿稳健性模型（年份固定效应、时间趋势、安慰剂、2017 年后样本、事件研究 Wald 检验）的结果，含第 7 节所说早先种子方案下的 p 值。Original robustness models, including the p values of the earlier seed scheme (README section 7). | 不被脚本读取，供对照。Reference only. |
| `h4_nearest_neighbor_results.json` | 原稿近邻词分析（即修订稿的 H3；签署前后分期、GPT-2 词表过滤）的结果。Original nearest-neighbour words. | 不被脚本读取，供对照。Reference only. |
| `manova_period_split.json` | 原稿分期 MANOVA（检验 1–6）的结果。Original period-split MANOVA. | 不被脚本读取，供对照。Reference only. |
| `manova_robustness_results.json` | 原稿不同主成分数下的 MANOVA 结果。Original MANOVA at several numbers of components. | 不被脚本读取，供对照。Reference only. |
| `manova_time_robustness.json` | 原稿按共同时间范围重做的 MANOVA 结果。Original MANOVA on common time ranges. | 不被脚本读取，供对照。Reference only. |

## 3. 未包含的数据 / Not included

- 原始网页抓取文件（网页 HTML 等）：`corpus/raw/*.csv` 与 `external_controls/documents.parquet` 由其整理而来。The raw scraped web files (the raw CSV files and the document table are derived from them).

## 4. 文本内容 / Text content

`corpus/raw/*.csv`（`title`、`content`）、`corpus/occurrences.parquet`（`text_block`）、两个 `encoding_inputs/` 表（`text`）与 `external_controls/documents.parquet`（`title`、`content`）含有美国、英国、澳大利亚、新加坡与加拿大政府网站公开文档的原文片段，仅供复现研究使用；转载或再分发请遵守各来源网站的使用条款。The bundle contains excerpts of public government web documents, provided for replication only.
