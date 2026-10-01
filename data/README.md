# 数据包说明

复现所需的全部数据在数据包 `data_bundle_rev1` 中（66 个文件，约 3.6 GB），作为本仓库的 GitHub Release 发布，标签 `rev1-data`：<https://github.com/Estebanii/AUKUS-Military-AI-Reproduction/releases/tag/rev1-data>。本目录保存数据包的逐文件清单 `MANIFEST.sha256`（sha256、字节数、相对路径）、Release 附件清单 `RELEASE_ASSETS.sha256` 与本说明。

## 1. 获取与校验

在本包根目录运行 `bash get_data.sh`：下载 Release 的六个附件、按 `RELEASE_ASSETS.sha256` 核对、解压到 `data/bundle/`，再按 `MANIFEST.sha256` 逐一核对 66 个文件。手工下载与放置方式见主 README 第 3 节。本仓库不使用 Git LFS。

| 附件 | 内容 | 大小 |
|---|---|---|
| `corpus.tar.gz` | `corpus/`（10 个文件） | 276 MB |
| `encodings.tar.gz` | `encodings/`（20 个文件） | 1.96 GB |
| `external_controls.tar.gz` | `external_controls/`（23 个文件） | 381 MB |
| `gpt2.tar.gz` | `gpt2/`（6 个文件） | 464 MB |
| `original_outputs.tar.gz` | `original_outputs/`（7 个文件） | 25 KB |
| `MANIFEST.sha256` | 逐文件清单（66 个文件） | 8 KB |

在数据包目录内校验全部文件（macOS 用 `shasum -a 256 -c`，Linux 用 `sha256sum -c`）：

```bash
cd /path/to/data_bundle_rev1
grep -v '^#' MANIFEST.sha256 | awk '{print $1 "  " $3}' | shasum -a 256 -c --quiet && echo "all files OK"
```

## 2. 内容

| 路径 | 内容 |
|---|---|
| `corpus/occurrences.parquet` | 原稿的 37,866 个目标术语出现点（美、英、澳国防部门文档，11,459 篇），行序即全部分析的行序。16 列：`occurrence_id`（出现点编号）；`term` 与 `matched_term`（术语及其在文中的匹配形式）；`article_id` 与 `doc_id`（文档编号，`doc_id` 为自助法的聚类单元）；`country`、`year`、`month`；`post_aukus`（2021 年 9 月起为 1）；`ai_similarity` 与 `pos_type`（该术语在 `terms.csv` 中的相似度得分与词性）；`text_block`（段落文本）；`source_type`（来源类型）；`concept_id` 与 `concept_label`（概念编号与术语标签）；`Y_vector_global`（原稿的 768 维语义向量，参考轴与图 1、图 2 由它计算） |
| `corpus/raw/{us,uk,au}.csv` | 原稿的原始语料：美国、英国、澳大利亚国防部门网站的文档，与首次投稿公开发布的 `data/raw/*.csv` 逐字节相同（us 18,047 条、uk 9,320 条、au 6,936 条 CSV 记录；字段内含换行，`wc -l` 得到的行数更大）。4 列：`title`（标题）；`publish_date`（发布日期，沿用各网站格式）；`content`（正文；美国 108 条、英国 1 条为空）；`type`（文档类型：news、release、contract、speech、statement、unknown）。`corpus/occurrences.parquet` 由原稿流程从这些文档中抽取 |
| `corpus/encoding_inputs/{targets,anchors}.parquet` | 编码器读取的输入表：37,866 个目标出现点与 49,999 个锚点，行序与 `encodings/<encoder>/*_rows.parquet` 的 `row_key` 一致。9 列：`row_index`；`row_key`（目标为 `t:<出现点编号>`，锚点为 `a:<源行>`）；`source`（target 或 anchor）；`source_row`；`occurrence_id`；`text`（送入编码器的文本）；`term`（目标词或锚词）；`start_char`（该词在 `text` 中的起始字符位置）；`pos_type`（词性）。基线编码器读取全部行，替代编码器读取 `common_support_map.parquet` 中的共同支持行。各行各列与编码时的输入逐一相同 |
| `corpus/terms.csv` | 56 个目标术语及其类别、词性与各国出现次数 |
| `corpus/anchor_words.json` | A 矩阵的 100 个锚词。该文件的 description 字段沿用原稿分析的措辞"country-specific matrix A"，实际使用的是全局统一的 A 矩阵（见主 README 术语表），文件内容未改 |
| `corpus/anchor_occurrences.parquet` | A 矩阵训练所用的 49,999 个锚词出现点（按国家比例分层抽取，一次性完成），行序即锚点编码的行序 |
| `corpus/common_support_map.parquet` | 五个编码器的共同支持：37,808 个目标出现点与 49,999 个锚点在原行序中的位置 |
| `encodings/<encoder>/{targets,anchors}_U.npy` 与 `_rows.parquet` | 五个编码器在被遮蔽位置的最后一层隐状态 U（float32）及每行的行键与遮蔽记录。基线编码器为全部行，替代编码器为共同支持行。编码方法见 `optional_full_pipeline/README.md` |
| `external_controls/documents.parquet` | 外部对照检验（表 9）的文档表：美、英、澳、新加坡、加拿大五国的 47,726 篇文档（标题、正文、发布日期、体裁、来源网址、归档时间戳、是否保留等） |
| `external_controls/targets_meta.parquet` | 新加坡与加拿大的 13,318 个目标术语出现点的元数据（国家、文档、年、月、术语、体裁） |
| `external_controls/encoding_inputs/targets.parquet` | 这 13,318 个出现点的编码输入表（列同 `corpus/encoding_inputs/`；`row_key` 为 `t:SG:<编号>` 或 `t:CA:<编号>`） |
| `external_controls/encodings/<encoder>/targets_U.npy` 与 `_rows.parquet` | 这些出现点在五个编码器下的隐状态 U 与行键 |
| `external_controls/design_records/<encoder>/{inference_selection,simulation}.json` | 表 9 推断方法的设计模拟记录（只用设计：行、国家、月份、文档、权重，不读取结果变量）。脚本 07 默认直接读取，`--recompute-design` 重新计算并逐位核对 |
| `gpt2/` | GPT-2 模型文件（Hugging Face `openai-community/gpt2`，修订 `607a30d783dfa663caf39e06633721c8d4cfcd7e`），只用其词元化器与输入词嵌入矩阵；代码会核对六个文件的 sha256 |
| `original_outputs/*.json` | 原稿（首次投稿）分析的 7 个结果文件，见下表 |

`original_outputs/` 的文件名沿用原稿分析的编号：`did_h3_verification.json` 的 H3 指原稿的距离假说，`h4_nearest_neighbor_results.json` 的 H4 指原稿的近邻词分析，即修订稿的 H3。七个文件与首次投稿公开发布的数据文件逐字节相同，其中的说明性字段为原稿分析自带。

| 文件 | 内容 | 读取者 |
|---|---|---|
| `wild_cluster_bootstrap_results.json` | 原稿主模型（H2）的 OLS 与 wild cluster bootstrap 系数、标准误与 p 值 | 脚本 05（原稿 PC1 英国×签署后系数）；`compare.py`（表 8 注中的基线估计） |
| `did_h3_verification.json` | 原稿三国签署前后的欧氏距离及其一致性核对结果 | 脚本 06（原稿距离的逐位复现核对） |
| `did_robustness_results.json` | 原稿稳健性模型（年份固定效应、时间趋势、安慰剂、2017 年后样本、事件研究 Wald 检验）的结果，含早先种子方案下的 p 值 | 供对照 |
| `h4_nearest_neighbor_results.json` | 原稿近邻词分析的结果 | 供对照 |
| `manova_period_split.json` | 原稿分期 MANOVA（检验 1–6）的结果 | 供对照 |
| `manova_robustness_results.json` | 原稿不同主成分数下的 MANOVA 结果 | 供对照 |
| `manova_time_robustness.json` | 原稿按共同时间范围重做的 MANOVA 结果 | 供对照 |

## 3. 未包含的数据

原始网页抓取文件（网页 HTML 等）：`corpus/raw/*.csv` 与 `external_controls/documents.parquet` 由其整理而来。

## 4. 文本内容

`corpus/raw/*.csv`、`corpus/occurrences.parquet`（`text_block`）、两个 `encoding_inputs/` 表（`text`）与 `external_controls/documents.parquet`（`title`、`content`）含有美国、英国、澳大利亚、新加坡与加拿大政府网站公开文档的原文片段，仅供复现研究使用；转载或再分发请遵守各来源网站的使用条款。
