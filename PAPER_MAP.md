# 论文项目对照表 / Paper map

每个印出的数字（逐项）见 `PAPER_MAP.csv`：论文位置 → 脚本 → 输出文件 → 字段（JSON 指针）→ 随机种子；`compare.py` 按 `expected/paper_numbers.csv` 逐项核对。全部随机种子集中在 `code/replication/seeds.py`（见 README 第 7 节）。本表按论文项目汇总。
Every printed number is listed in `PAPER_MAP.csv` (location -> script -> output file -> field -> seeds). All seeds are set in `code/replication/seeds.py`.

覆盖范围 / Coverage：`compare.py` 核对修订稿定量部分（第四、五、六节及其脚注）与表 1–9 中登记的 397 个标量数字（其中 3 个是标签，只列出不检验）。印出的标签（年份、“95%”、“5%”等）、设计常数以及文字判断（脚注 80 关于 PC1 的判断，第五节第四、五部分关于近邻词表、跨编码器与外部对照的定性表述）不在这 397 项之内，列于本文件末尾“文字判断 / Textual claims”一节，并注明支持它们的结果字段。
`compare.py` checks the 397 registered scalar items of the manuscript's quantitative sections (IV-VI with their footnotes) and Tables 1-9 (3 of them are labels, listed but not tested). Labels (years, "95%", "5%"), design constants and the qualitative claims (footnote 80 on PC1; Sections V.4 and V.5: word lists, cross-encoder and external-control statements) are listed under "Textual claims" at the end of this file with the results fields that support them.

| 论文项目 Item | 内容 Content | 印出的计算值 Computed printed numbers | 脚本 Script(s) | 输出文件 Output files | 随机种子 Seeds |
|---|---|---|---|---|---|
| Table 1 | 变量定义与样本量 / variables and sample sizes | 12 | scripts/01_sample_table2.py, scripts/03_h2_did_tables4_6_fig3.py | code/replication/params.py; data bundle: encodings/deberta-v3-base/targets_U.npy; results/h2/wcb.json; results/table2_sample.json | 无 none |
| Table 2 | 分国家、分时期的出现点与文档数 / occurrences by country and period | 16 | scripts/01_sample_table2.py | results/table2_sample.json | 无 none |
| Figure 1 | PCA 散点（PC1 × PC2，按国家与签署前后）/ PCA scatter | 4 | scripts/08_figures_1_2.py | results/figures/figures_1_2.json; results/figures/figure1_pca.{png,pdf,csv} | FIGURE_PCA_RANDOM_STATE (no effect with the covariance_eigh solver) (code/replication/seeds.py) |
| Figure 2 | t-SNE 散点（无印出数字）/ t-SNE scatter (no printed numbers) | 0 | scripts/08_figures_1_2.py | results/figures/figure2_tsne.{png,pdf,csv} | FIGURE_PCA_RANDOM_STATE, TSNE_RANDOM_STATE (code/replication/seeds.py) |
| Table 3 | H1：MANOVA（签署前美英，2014 年起，88 个主成分）/ H1 MANOVA | 8 | scripts/01_sample_table2.py, scripts/02_h1_manova_table3.py, scripts/03_h2_did_tables4_6_fig3.py | results/h1/deberta-v3-base/h1.json; results/h1/deberta-v3-base/manova.json; results/h2/meta.json; results/table2_sample.json | 无 none |
| Table 4 | H2：子组 DiD 与野聚类自助法 / H2 subgroup DiD, wild cluster bootstrap | 48 | scripts/03_h2_did_tables4_6_fig3.py | code/replication/params.py; results/h2/did.json; results/h2/wcb.json | WCB_SEED_BASE (42/43/44 by PC), WCB_DRAWS (code/replication/seeds.py) |
| Figure 3 | H2：事件研究系数（美英，2014–2024，基年 2021）/ event-study coefficients | 1 | scripts/03_h2_did_tables4_6_fig3.py | results/h2/parallel_trends.json; results/figures/figure3_event_study.{png,pdf,csv} | WCB_SEED_BASE (42/43/44 by PC), WCB_DRAWS (code/replication/seeds.py) |
| Table 5 | H2：平行趋势联合 Wald 检验 / parallel-trends Wald tests | 15 | scripts/03_h2_did_tables4_6_fig3.py | results/h2/parallel_trends.json; results/h2/wcb.json | WCB_SEED_BASE (42/43/44 by PC), WCB_DRAWS (code/replication/seeds.py) |
| Table 6 | H2：安慰剂检验（虚拟断点 2020、2021）/ placebo tests | 17 | scripts/03_h2_did_tables4_6_fig3.py | code/replication/params.py; results/h2/did.json; results/h2/wcb.json | WCB_SEED_BASE (42/43/44 by PC), WCB_DRAWS (code/replication/seeds.py) |
| Table 7 | 美英距离的签署前后变化（枢轴自助法）/ change of the US-UK distance | 84 | scripts/03_h2_did_tables4_6_fig3.py, scripts/06_distance_table7.py | data bundle: encodings/deberta-v3-base/targets_U.npy; results/distance/deberta-v3-base.json; results/distance/summary.json; results/h2/meta.json | DISTANCE_SEED, DISTANCE_DRAWS (code/replication/seeds.py) |
| Figure 4 | H3：各国均值向量的 GPT-2 近邻词 / nearest GPT-2 words | 1 | scripts/04_h3_neighbours_fig4.py | code/replication/params.py; results/h3/h3.json; results/figures/figure4_neighbours.{png,pdf,csv} | 无 none |
| Table 8 | 四个替代编码器（共同支持样本）/ four alternative encoders | 59 | scripts/05_cross_encoder_table8.py | code/replication/params.py; data bundle: original_outputs/wild_cluster_bootstrap_results.json; results/cross_encoder/cross_encoder.json | WCB_SEED_BASE, WCB_DRAWS; RERUN_SEED, RERUN_DRAWS (code/replication/seeds.py) |
| Table 9 | 外部对照（新加坡、加拿大）/ external controls (Singapore, Canada) | 25 | scripts/07_external_control_table9.py | results/external/ModernBERT-large/estimate/main.json; results/external/deberta-v2-xlarge/estimate/main.json; results/external/deberta-v3-base/design/mde.json; results/external/deberta-v3-base/estimate/summary.json; results/external/ettin-encoder-1b/estimate/main.json; results/external/roberta-large/estimate/main.json | EXTERNAL_SEED, EXTERNAL_VERIFICATION_SEED, EXTERNAL_BLOCK_BOOTSTRAP_DRAWS, EXTERNAL_SIMULATION_REPLICATIONS (code/replication/seeds.py) |
| Footnote 61 | 位置 （三）数据与语义测量：“目标术语的选择采用自动化发现算法，共筛选出56个军事人工…” | 1 | scripts/01_sample_table2.py | results/table2_sample.json | 无 none |
| Footnote 70 | 位置 （三）数据与语义测量：“转换矩阵 A 基于100个高频锚词（非目标术语）训练而成…” | 2 | scripts/00_prepare.py | results/intermediate/deberta-v3-base_A_fit.json | 无 none |
| Footnote 72 | 位置 （四）实证策略与统计推断：“稳健性检验表明在3至99个主成分范围内结论均不变。…” | 2 | scripts/02_h1_manova_table3.py | results/h1/deberta-v3-base/h1_test6_pc_robustness.json | 无 none |
| Footnote 75 | 位置 （四）实证策略与统计推断：“H2的因果推断主要集中在英国与美国的比较层面。澳大利亚的…” | 2 | scripts/01_sample_table2.py | results/table2_sample.json | 无 none |
| Footnote 78 | 位置 （二）基线差异：概念化差异的存在检验（H1）：“由于澳大利亚Pre-AUKUS仅有844条观测（其中60…” | 6 | scripts/01_sample_table2.py, scripts/02_h1_manova_table3.py | results/h1/deberta-v3-base/manova.json; results/table2_sample.json | 无 none |
| Footnote 80 | 位置 （三）AUKUS协定签署的影响（H2）：“PC3上英国主项与交互项同为负。该安慰剂结果对是否纳入趋…” | 2 | scripts/03_h2_did_tables4_6_fig3.py | results/h2/placebo_with_trend.json | WCB_SEED_BASE (42/43/44 by PC), WCB_DRAWS (code/replication/seeds.py) |
| Footnote 81 | 位置 （三）AUKUS协定签署的影响（H2）：“该距离从两国均值差的平方范数中扣除以文档聚类方差估计的抽…” | 1 | scripts/06_distance_table7.py | results/distance/deberta-v3-base.json | DISTANCE_SEED, DISTANCE_DRAWS (code/replication/seeds.py) |
| Footnote 83 | 位置 （四）AUKUS 三国军事人工智能概念化的具体语义（H3）：“为消除GPT-2字节对编码分词机制产生的子词片段干扰（如…” | 2 | scripts/04_h3_neighbours_fig4.py | results/h3/h3.json | 无 none |
| Footnote 84 | 位置 （四）AUKUS 三国军事人工智能概念化的具体语义（H3）：“近邻分析基于AUKUS签署后（2021年9月至2025年…” | 6 | scripts/01_sample_table2.py, scripts/04_h3_neighbours_fig4.py | code/replication/params.py; results/h3/h3.json; results/table2_sample.json | 无 none |
| Footnote 85 | 位置 （五）稳健性检验：“新加坡的样本窗口为2014年1月至2025年11月，加拿…” | 7 | scripts/07_external_control_table9.py | code/replication/external/params.py; results/external/deberta-v3-base/design/mde.json; results/external/deberta-v3-base/estimate/event_study.json | EXTERNAL_SEED, EXTERNAL_VERIFICATION_SEED, EXTERNAL_BLOCK_BOOTSTRAP_DRAWS, EXTERNAL_SIMULATION_REPLICATIONS (code/replication/seeds.py) |
| Section IV.3 text |  | 12 | scripts/01_sample_table2.py | results/table2_sample.json | 无 none |
| Section IV.4 text |  | 6 | scripts/01_sample_table2.py, scripts/02_h1_manova_table3.py, scripts/03_h2_did_tables4_6_fig3.py | code/replication/params.py; results/h1/deberta-v3-base/h1.json; results/h2/meta.json; results/h2/wcb.json; results/table2_sample.json | WCB_SEED_BASE (42/43/44 by PC), WCB_DRAWS (code/replication/seeds.py) |
| Section V.1 text |  | 4 | scripts/01_sample_table2.py | results/table2_sample.json | 无 none |
| Section V.2 text |  | 12 | scripts/01_sample_table2.py, scripts/02_h1_manova_table3.py | results/h1/<each alternative encoder>/h1.json; results/h1/deberta-v3-base/manova.json; results/table2_sample.json | 无 none |
| Section V.3 text |  | 29 | scripts/01_sample_table2.py, scripts/03_h2_did_tables4_6_fig3.py, scripts/06_distance_table7.py | code/replication/params.py; results/distance/deberta-v3-base.json; results/distance/summary.json; results/h2/parallel_trends.json; results/h2/wcb.json; results/table2_sample.json | DISTANCE_SEED, DISTANCE_DRAWS; WCB_SEED_BASE (42/43/44 by PC), WCB_DRAWS (code/replication/seeds.py) |
| Section V.4 text |  | 7 | scripts/04_h3_neighbours_fig4.py | code/replication/params.py; results/h3/h3.json | 无 none |
| Section V.5 text |  | 2 | scripts/07_external_control_table9.py | code/replication/params.py; results/external/deberta-v3-base/estimate/summary.json | EXTERNAL_SEED, EXTERNAL_VERIFICATION_SEED, EXTERNAL_BLOCK_BOOTSTRAP_DRAWS, EXTERNAL_SIMULATION_REPLICATIONS (code/replication/seeds.py) |
| Section VI text |  | 1 | scripts/01_sample_table2.py | results/table2_sample.json | 无 none |

说明 / Notes:

- `(parameter)`：印出的是固定的分析参数（如安慰剂断点年份、方向阈值 0.30、外部对照样本窗口），`compare.py` 核对它与代码中的参数一致。Printed analysis parameters are checked against the code.
- `(data bundle)`：印出的是数据本身的属性（基线编码器隐状态宽度 768）或原稿结果文件中的数值。
- 脚注编号为全文顺序号；稿件中的脚注以带圈数字每页重新编号，“位置”列出所在节与开头文字（见 README 第 6 节“脚注定位”）。
- `(label)`：印出的是标签文字（如“2021年1月至8月”），不是计算值，只列出不检验（共 3 项，不计入上表该列；该列合计 394，加 3 个标签共 397 项）。
- 表 1–6 与图 3、4 用基线编码器（DeBERTa-v3-base 重新编码）的结果；图 1、2 的散点按原稿以原语义向量绘制，图 1 坐标轴上的方差占比按图本身的 PCA 核对；原图没有存档的全精度数值，这 4 项因此没有全精度参考值（基线编码器分析用的 PCA-3 在印出精度下相同）。

## 文字判断 / Textual claims

以下判断不是单个标量，`compare.py` 不逐项核对；表中列出支持它们的结果字段与参考运行中的取值。The claims below are not single scalars; the results fields that support them and their values in the reference run are listed.

| 位置 Location | 判断 Claim | 结果字段 Results fields | 参考运行 Reference run |
|---|---|---|---|
| 脚注 80（五（三）） | PC1 的英国安慰剂交互项在两种规格下（含与不含线性时间趋势项）均不显著 | results/h2/placebo_with_trend.json /2020/PC1/UK_x_fake_post/boot_p、/2021/PC1/UK_x_fake_post/boot_p（含趋势项）；results/h2/did.json /model_4a_placebo_2020/PC1/UK_x_fake_post/boot_p、/model_4b_placebo_2021/PC1/UK_x_fake_post/boot_p（不含趋势项，表 6） | 含趋势项：p = 0.681（2020）、0.201（2021）；不含趋势项：p = 0.569（2020）、0.584（2021） |
| 五（四）第 2 段 | 三国共同的前 15 个近邻词中有 8 个：engineering、engineers、equipment、experimentation、analytics、optimization、technology、testing | results/h3/h3.json /post_aukus_analysis/common_words_top15 | 8 个，词项同文 |
| 五（四）第 3 段 | 美国的独特近邻词：customization、debugging、labs、science、technical | results/h3/h3.json /post_aukus_analysis/unique_words_top15/US | 同文 |
| 五（四）第 3 段 | 只有 science 在五个编码器下都保留在美国近邻词前列 | results/h3/h3.json 与 results/alternatives/<encoder>/h3.json /post_aukus_analysis/nearest_words/US（前 15 位） | science 的名次：10 / 7 / 8 / 8 / 12（基线、ModernBERT-large、RoBERTa-large、Ettin-encoder-1b、DeBERTa-v2-xlarge） |
| 五（四）第 3 段 | 定制化、调试、实验室三词在替代编码器下均不在任何一国的前 15 位，技术性（technical）仅在 ModernBERT-large 中进入澳大利亚的前 15 位 | results/alternatives/<encoder>/h3.json /post_aukus_analysis/nearest_words（前 15 位，三国） | customization、debugging、labs 在四个替代编码器下均不在任何国家前 15 位；technical 只在 ModernBERT-large 下进入前 15 位（澳大利亚第 14 位，不在美英前 15 位） |
| 五（四）第 3 段 | 替代编码器下新出现的美国独特词为 info 以及 research、programming、mathematics 等 | results/alternatives/<encoder>/h3.json /post_aukus_analysis/unique_words_top15/US | info（4/4）；research（ModernBERT-large）、programming（RoBERTa-large、Ettin-encoder-1b）、mathematics（Ettin-encoder-1b）；另有 disinformation、knowledge、science（DeBERTa-v2-xlarge） |
| 五（四）第 4 段 | 英国的独特近邻词：expertise、technological、technologies | results/h3/h3.json /post_aukus_analysis/unique_words_top15/UK | 同文 |
| 五（四）第 4 段 | 在四个替代编码器中的三个下，英国前十五位的独特近邻词仍包含 technological、tech、capabilities | results/alternatives/<encoder>/h3.json /post_aukus_analysis/unique_words_top15/UK | ModernBERT-large：cyber、technological；RoBERTa-large：physics、tech；Ettin-encoder-1b：capabilities、tech；DeBERTa-v2-xlarge：bandwidth、spac（不含） |
| 五（四）第 4 段 | equipment 在五个编码器下都位居英国近邻词的前四位，在美国近邻词中靠后或不在前十五位之内 | results/h3/h3.json 与 results/alternatives/<encoder>/h3.json /post_aukus_analysis/nearest_words/UK、/US | 英国名次 3 / 3 / 3 / 3 / 4；美国名次 12 / 12 / 15 / 16 / 29 |
| 五（四）第 5 段 | 澳大利亚的独特近邻词：acquisition、downtime、engineer、experiments、maintenance | results/h3/h3.json /post_aukus_analysis/unique_words_top15/AU | 同文 |
| 五（四）第 5 段 | maintenance 在多数替代编码器下仍是澳大利亚的独特近邻词 | results/alternatives/<encoder>/h3.json /post_aukus_analysis/unique_words_top15/AU | ModernBERT-large、RoBERTa-large、DeBERTa-v2-xlarge 是；Ettin-encoder-1b 否 |
| 五（四）第 6 段 | 三国各自前 15 个近邻词的共同词签署前 10 个、签署后 8 个（这两个数字由 compare.py 核对）；替代编码器下共同词个数有下降、持平和略升三种情况 | results/h3/h3.json 与 results/alternatives/<encoder>/h3.json /pre_aukus_analysis/common_words_top15、/post_aukus_analysis/common_words_top15（词数） | 基线 10→8；ModernBERT-large 10→10；RoBERTa-large 8→9；Ettin-encoder-1b 8→8；DeBERTa-v2-xlarge 11→8 |
| 五（四）第 6 段 | maintenance 签署前同时出现在三国近邻词列表中，签署后退出美英列表、成为澳大利亚的独特近邻词；这一点在多数编码器中成立 | /pre_aukus_analysis/common_words_top15；/post_aukus_analysis/unique_words_top15/AU；/post_aukus_analysis/nearest_words/US、/UK | 签署后为澳大利亚独特词：5 个编码器中的 4 个（Ettin-encoder-1b 除外）；签署前为三国共同词：只有基线 |
| 五（五）第 1 段 | 在各编码器自行拟合的第一主成分上，以 2020 年为假断点的安慰剂检验均拒绝零假设，而参考轴上的同一检验均未拒绝 | results/alternatives/<encoder>/did.json /model_4a_placebo_2020/PC1/UK_x_fake_post/boot_p（自行拟合）；results/cross_encoder/same_axis_robustness/<encoder>.json /same_axis/did/model_4a_placebo_2020/PC1/UK_x_fake_post/boot_p（参考轴） | 自行拟合：p = 0.013 / 0.024 / 0.044 / 0.018；参考轴：p = 0.861 / 0.632 / 0.583 / 0.460（ModernBERT-large、RoBERTa-large、Ettin-encoder-1b、DeBERTa-v2-xlarge） |
| 五（五）第 1 段 | 各编码器自行拟合的第一主成分中有两个与参考轴方向不可比 | results/cross_encoder/cross_encoder.json /rows/<encoder>/orientation/abs_cos_pc1（表 8 的 \|cos\|）；判据 \|cos\| < 0.30（code/replication/params.py `orientation.primary_threshold`） | \|cos\|：ModernBERT-large 0.021、DeBERTa-v2-xlarge 0.234（不可比）；RoBERTa-large 0.367、Ettin-encoder-1b 0.457 |
| 五（五）第 1 段 | 参考轴方向上四个替代编码器的交互项与安慰剂检验结果均与基线一致 | results/cross_encoder/cross_encoder.json /families/same_axis (PC1, Holm over 4)/holm（表 8）；results/cross_encoder/same_axis_robustness/<encoder>.json /same_axis/did/model_4a_placebo_2020/PC1/UK_x_fake_post/boot_p；results/h2/wcb.json、results/h2/did.json（基线） | 交互项 0.0434 / 0.0505 / 0.0674 / 0.0537，与基线（0.0333）同号，Holm p 均 <0.001；2020 年安慰剂 p：0.861 / 0.632 / 0.583 / 0.460，与基线（0.569）同为未拒绝 |
| 五（五）第 2 段 | 相对非成员国，美国变动的点估计的绝对值大于英国 | results/external/deberta-v3-base/estimate/summary.json /rows（δ_UK、δ_US，表 9 A 栏） | 新加坡：\|δ_US\| = 0.0244 > \|δ_UK\| = 0.0142；加拿大：\|δ_US\| = 0.0706 > \|δ_UK\| = 0.0294 |
| 五（五）第 2 段 | 这一设计能可靠检出的最小效应近三倍于主结果（派生量） | results/external/deberta-v3-base/design/mde.json /mde/SG/estimands/theta/ref_PC1/mde ÷ results/h2/wcb.json /comparison/PC1/UK_x_post/coef | 0.0934 / 0.0333 = 2.81 |
| 五（五）第 2、3 段 | 外部对照提供的信息是方向一致，且没有出现反向证据；外部对照的点估计与主结果同向 | results/external/deberta-v3-base/estimate/summary.json /rows（θ，表 9 A 栏）；results/external/<encoder>/estimate/main.json（表 9 B 栏） | θ：新加坡 0.0387 [−0.0319, 0.1092]，加拿大 0.0412 [−0.0395, 0.1218]；B 栏 0.0527 / 0.0599 / 0.0826 / 0.0646；全部为正，与主结果 0.0333 同号，没有显著为负的估计 |

印出的标签与设计常数 / Labels and design constants:

| 印出内容 Printed | 含义 Meaning | 定义处 Defined in |
|---|---|---|
| 95%（区间） | 全部置信区间的水平 | code/replication/params.py `wcb.interval_level`；距离与外部对照见各自参数 |
| 5%（显著性水平） | 检验水平 α = 0.05 | code/replication/params.py `families.alpha`；code/replication/external/params.py `bootstrap.alpha`、`mde.alpha` |
| 年份与月份（2014 年起、2021 年 9 月起为签署后、断点 2020 与 2021 等） | 样本起点、签署月份、安慰剂断点 | code/replication/params.py `wcb.trend_origin_year`、`manova.test6.min_year`、`did_robustness.placebo_years`；数据包 `post_aukus` |
| 设计常数（88 个主成分、B = 1,000 / 2,000 / 9,999、100 个锚词、λ = 0.1、前 15 位等） | 分析设定 | code/replication/params.py、code/replication/seeds.py、code/replication/external/params.py |
