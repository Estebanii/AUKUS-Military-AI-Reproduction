#!/usr/bin/env python3
"""
论文图表生成脚本 — 中文权威期刊风格
==========================================
基于最新验证数据生成全部图表，输出到 part/figures/ 和 part/tables/

图: SciencePlots主题 + 中文字体(宋体/黑体)
表: 三线表 + 中文字体 + PDF/PNG双输出

数据来源:
- final_fuxian/outputs/*.json (全部已交叉验证)
- final_fuxian/data/semantic_vectors_paragraph_global_A.parquet
- final_fuxian/figures/parallel_trends/event_study_coefficients.csv
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ============================================================
# 路径设置
# ============================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
FIG_DIR = PROJECT_ROOT / 'chatu' / 'figures'
TAB_DIR = PROJECT_ROOT / 'chatu' / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 样式配置
# ============================================================
try:
    import scienceplots
    plt.style.use(['science', 'no-latex', 'grid'])
    print("Using SciencePlots style")
except ImportError:
    print("SciencePlots not available, using default style")
    plt.rcParams.update({
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

# 中文字体
FONT_TITLE = fm.FontProperties(family='STHeiti', size=11, weight='bold')
FONT_BODY = fm.FontProperties(family='Songti SC', size=9)
FONT_HEADER = fm.FontProperties(family='STHeiti', size=9, weight='bold')
FONT_AXIS = fm.FontProperties(family='Songti SC', size=9)
FONT_LEGEND = fm.FontProperties(family='Songti SC', size=8)
FONT_NOTE = fm.FontProperties(family='Songti SC', size=7)

# 配色（色盲友好 Tableau10 风格）
COLORS = {
    'US': '#4E79A7', 'UK': '#E15759', 'AU': '#59A14F',
    'pre': '#BAB0AC', 'post': '#4E79A7',
    'common': '#76B7B2', 'sig': '#E15759', 'ns': '#BAB0AC',
}

DPI = 300

# ============================================================
# 数据加载
# ============================================================
def load_all_data():
    """加载全部数据"""
    data = {}
    data['manova'] = json.load(open(OUTPUT_DIR / 'manova_period_split.json'))
    data['robustness'] = json.load(open(OUTPUT_DIR / 'did_robustness_results.json'))
    data['bootstrap'] = json.load(open(OUTPUT_DIR / 'wild_cluster_bootstrap_results.json'))
    data['neighbors'] = json.load(open(OUTPUT_DIR / 'h4_nearest_neighbor_results.json'))
    data['event_csv'] = pd.read_csv(
        PROJECT_ROOT / 'figures' / 'parallel_trends' / 'event_study_coefficients.csv'
    )
    return data


def load_parquet():
    """加载原始parquet（仅图1/2需要）"""
    path = DATA_DIR / 'semantic_vectors_paragraph_global_A.parquet'
    if not path.exists():
        print(f"WARNING: {path} not found, skipping PCA/t-SNE figures")
        return None
    return pd.read_parquet(path)


def save_fig(fig, name, output_dir=None):
    """保存PDF+PNG"""
    out = output_dir or FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight', pad_inches=0.1)
    fig.savefig(out / f'{name}.png', dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out / name}.pdf/.png")


# ============================================================
# 三线表渲染器
# ============================================================
class ThreeLineTable:
    """中文三线表（宋体正文+黑体表头）"""

    def __init__(self, title, headers, rows, notes=None, row_height=0.35):
        self.title = title
        self.headers = headers
        self.rows = rows
        self.notes = notes or []
        self.n_cols = len(headers)
        self.row_height = row_height

    def render(self, name):
        row_h = self.row_height
        header_h = 0.45
        title_h = 0.55
        note_h = 0.22 * max(1, len(self.notes))
        pad = 0.15
        total_h = title_h + header_h + row_h * len(self.rows) + note_h + 2 * pad
        fig_w = 7.5

        fig, ax = plt.subplots(figsize=(fig_w, total_h))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, total_h)
        ax.axis('off')

        col_w = [1.0 / self.n_cols] * self.n_cols
        y = total_h - pad

        # 标题
        ax.text(0.5, y - title_h / 2, self.title, ha='center', va='center',
                fontproperties=FONT_TITLE)
        y -= title_h

        # 顶线
        ax.plot([0.02, 0.98], [y, y], color='black', linewidth=1.5)

        # 表头
        x = 0.02
        for j, h in enumerate(self.headers):
            cx = x + col_w[j] / 2
            ax.text(cx, y - header_h / 2, h, ha='center', va='center',
                    fontproperties=FONT_HEADER)
            x += col_w[j]
        y -= header_h

        # 中线
        ax.plot([0.02, 0.98], [y, y], color='black', linewidth=0.75)

        # 数据行
        for row in self.rows:
            x = 0.02
            for j, cell in enumerate(row):
                cx = x + col_w[j] / 2
                ax.text(cx, y - row_h / 2, str(cell), ha='center', va='center',
                        fontproperties=FONT_BODY)
                x += col_w[j]
            y -= row_h

        # 底线
        ax.plot([0.02, 0.98], [y, y], color='black', linewidth=1.5)

        # 脚注
        for i, note in enumerate(self.notes):
            y -= 0.22
            ax.text(0.02, y, note, ha='left', va='top',
                    fontproperties=FONT_NOTE, wrap=True)

        fig.tight_layout()
        save_fig(fig, name, output_dir=TAB_DIR)


# ============================================================
# 格式化工具
# ============================================================
def fmt_p(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    elif p < 0.1: return '\u2020'
    return ''

def fmt_coef(c, p):
    return f'{c:.4f}{fmt_p(p)}'


# ============================================================
# 图1: PCA分布（双面板）
# ============================================================
def gen_figure_1(df):
    print("Generating Figure 1: PCA distribution...")
    Y = np.vstack(df['Y_vector_global'].values)
    pca = PCA(n_components=3, random_state=42)
    Y_pca = pca.fit_transform(Y)

    countries = df['country'].values
    post = df['post_aukus'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # 左: 按国家
    for c, label in [('US', '美国'), ('UK', '英国'), ('AU', '澳大利亚')]:
        mask = countries == c
        ax1.scatter(Y_pca[mask, 0], Y_pca[mask, 1], c=COLORS[c],
                   alpha=0.15, s=3, label=label, rasterized=True)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontproperties=FONT_AXIS)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontproperties=FONT_AXIS)
    ax1.set_title('(a) 按国家', fontproperties=FONT_TITLE)
    ax1.legend(prop=FONT_LEGEND, markerscale=3)

    # 右: 按时期
    for p_val, label, color, marker in [(0, 'AUKUS前', COLORS['pre'], 'o'),
                                         (1, 'AUKUS后', COLORS['post'], 's')]:
        mask = post == p_val
        ax2.scatter(Y_pca[mask, 0], Y_pca[mask, 1], c=color,
                   alpha=0.15, s=3, label=label, marker=marker, rasterized=True)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontproperties=FONT_AXIS)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontproperties=FONT_AXIS)
    ax2.set_title('(b) 按AUKUS签署前后', fontproperties=FONT_TITLE)
    ax2.legend(prop=FONT_LEGEND, markerscale=3)

    fig.suptitle('图1  三国军事人工智能概念在语义嵌入空间中的PCA分布',
                 fontproperties=fm.FontProperties(family='STHeiti', size=11), y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig1_pca')


# ============================================================
# 图2: t-SNE分布
# ============================================================
def gen_figure_2(df):
    print("Generating Figure 2: t-SNE distribution (this may take ~5 min)...")
    Y = np.vstack(df['Y_vector_global'].values)
    pca = PCA(n_components=50, random_state=42)
    Y_50 = pca.fit_transform(Y)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, n_jobs=-1)
    Y_tsne = tsne.fit_transform(Y_50)

    countries = df['country'].values
    fig, ax = plt.subplots(figsize=(6, 5))

    for c, label in [('US', '美国'), ('UK', '英国'), ('AU', '澳大利亚')]:
        mask = countries == c
        ax.scatter(Y_tsne[mask, 0], Y_tsne[mask, 1], c=COLORS[c],
                   alpha=0.2, s=3, label=label, rasterized=True)

    ax.set_xlabel('t-SNE 1', fontproperties=FONT_AXIS)
    ax.set_ylabel('t-SNE 2', fontproperties=FONT_AXIS)
    ax.set_title('图2  三国军事人工智能概念在语义空间中的t-SNE分布',
                 fontproperties=fm.FontProperties(family='STHeiti', size=10))
    ax.legend(prop=FONT_LEGEND, markerscale=3)
    fig.tight_layout()
    save_fig(fig, 'fig2_tsne')


# ============================================================
# 图3: 事件研究（Bootstrap CI）
# ============================================================
def gen_figure_3(data):
    print("Generating Figure 3: Event study with Bootstrap CI...")
    df = data['event_csv']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for idx, pc in enumerate(['PC1', 'PC2', 'PC3']):
        ax = axes[idx]
        pc_data = df[df['pc'] == pc]

        cd = pc_data[pc_data['country'] == 'UK'].sort_values('year')
        years = cd['year'].values
        coefs = cd['coef'].values
        ci_lo = cd['ci_lower'].values
        ci_hi = cd['ci_upper'].values

        ax.fill_between(years, ci_lo, ci_hi, alpha=0.15, color=COLORS['UK'])
        ax.errorbar(years, coefs, yerr=[coefs - ci_lo, ci_hi - coefs],
                   fmt='o-', color=COLORS['UK'], label='英国',
                   markersize=4, capsize=2, linewidth=1)

        ax.plot(2021, 0, 'kD', markersize=6, zorder=10)
        ax.axvline(x=2021, color='red', ls='--', lw=1.2, alpha=0.6)
        ax.axhline(y=0, color='black', lw=0.4, alpha=0.4)
        ax.set_xlabel('年份', fontproperties=FONT_AXIS)
        ax.set_ylabel('系数', fontproperties=FONT_AXIS)
        ax.set_title(pc, fontproperties=FONT_TITLE)
        if idx == 0:
            ax.legend(prop=FONT_LEGEND, loc='best')
        ax.set_xticks([y for y in range(2014, 2025) if y != 2021])
        ax.tick_params(labelsize=6)

    fig.suptitle('图3  事件研究分析：英国相对于美国的年度系数（基准年2021）',
                 fontproperties=fm.FontProperties(family='STHeiti', size=10), y=1.03)
    fig.tight_layout()
    save_fig(fig, 'fig3_event_study')


# ============================================================
# 图4: Post-AUKUS近邻词柱状图
# ============================================================
def gen_figure_4(data):
    print("Generating Figure 4: Post-AUKUS nearest neighbors...")
    nn = data['neighbors']['post_aukus_analysis']['nearest_words']
    common = set(data['neighbors']['post_aukus_analysis']['common_words_top15'])
    unique = data['neighbors']['post_aukus_analysis']['unique_words_top15']
    top_n = 15

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    clabels = {'US': '美国', 'UK': '英国', 'AU': '澳大利亚'}

    for idx, c in enumerate(['US', 'UK', 'AU']):
        ax = axes[idx]
        words_data = nn[c][:top_n]
        words = list(reversed([w['word'] for w in words_data]))
        sims = list(reversed([w['similarity'] for w in words_data]))
        unique_set = set(unique.get(c, []))

        bar_colors = []
        for w in words:
            if w in unique_set:
                bar_colors.append(COLORS[c])
            elif w in common:
                bar_colors.append(COLORS['common'])
            else:
                bar_colors.append('#D4D4D4')

        bars = ax.barh(range(len(words)), sims, color=bar_colors, height=0.7,
                       edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=8)
        ax.set_xlabel('余弦相似度', fontproperties=FONT_AXIS)
        ax.set_title(clabels[c], fontproperties=FONT_TITLE)
        ax.set_xlim(0.19, max(sims) + 0.02)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['common'], label='三国共有'),
        Patch(facecolor=COLORS['US'], label='美国独有'),
        Patch(facecolor=COLORS['UK'], label='英国独有'),
        Patch(facecolor=COLORS['AU'], label='澳大利亚独有'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               prop=FONT_LEGEND, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('图4  AUKUS签署后三国军事人工智能概念化的语义近邻词（Top-15）',
                 fontproperties=fm.FontProperties(family='STHeiti', size=10), y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig4_neighbors')


# ============================================================
# 表1: 变量定义与操作化
# ============================================================
def gen_table_1(data):
    print("Generating Table 1: Variable definitions...")
    ThreeLineTable(
        '表1  变量定义与操作化',
        ['变量', '类型', '操作化定义', '描述统计'],
        [
            ['语义表征 (Y)', '因变量',
             '768维语义向量，Y=A\u00b7U\n(DeBERTa-v3-base, 全局A矩阵变换)',
             'N=37,866'],
            ['国家 (Country)', '核心自变量',
             'D_UK（英国=1）、D_AU（澳大利亚=1）\n美国为基准组',
             'US:20,360\nUK:11,224\nAU:6,282'],
            ['AUKUS协议签订\n(Post-AUKUS)', '调节变量',
             '二值变量\n2021年9月及之后取值为1',
             '前期:20,784\n后期:17,082'],
            ['时间 (t)', '控制变量',
             '线性时间趋势，t=year-2014\n（仅用于H2模型）',
             '范围:2004-2025'],
            ['国家\u00d7AUKUS\n交互项', '交互变量',
             'D_UK\u00d7Post, D_AU\u00d7Post\n（仅用于H2模型）',
             '用于检验H2'],
            ['文档标识\n(Doc ID)', '聚类变量',
             '文本来源的唯一标识符\n用于Wild Cluster Bootstrap聚类',
             '聚类数:11,459'],
        ],
        [
            '注：语义表征由DeBERTa-v3-base模型编码，经全局转换矩阵A变换生成。',
            '时间变量和交互项仅用于H2双重差分模型，H1基线检验不包含这些变量。',
        ],
        row_height=0.55,
    ).render('tab1_variables')


# ============================================================
# 表2: 样本分布
# ============================================================
def gen_table_2(data):
    print("Generating Table 2: Sample distribution...")
    n = data['bootstrap']['n_samples']
    ThreeLineTable(
        '表2  样本分布与基本统计',
        ['国家', '术语出现次数', '占比', 'AUKUS前', 'AUKUS后'],
        [
            ['美国（US）', '20,360', '53.8%', '13,232', '7,128'],
            ['英国（UK）', '11,224', '29.6%', '6,708', '4,516'],
            ['澳大利亚（AU）', '6,282', '16.6%', '844', '5,438'],
            ['合计', '37,866', '100%', '20,784', '17,082'],
        ],
        ['注：数据为三国国防部官方文本中军事人工智能相关术语的段落级出现次数。']
    ).render('tab2_sample')


# ============================================================
# 表3a: MANOVA
# ============================================================
def gen_table_3a(data):
    print("Generating Table 3: MANOVA...")
    m = data['manova']
    r = m['results']
    # r[8] = Test 6: Pre-AUKUS US vs UK 2014+重叠时段
    ThreeLineTable(
        '表3  国家差异整体检验（MANOVA）',
        ['检验', '样本期', 'N', 'F统计量', 'p值', '结论'],
        [
            ['Pre-AUKUS US vs UK', '2014-2021.08', f"{r[8]['n']:,}", f"{r[8]['F']:.2f}", '<0.001', '支持H1***'],
        ],
        [
            "注：MANOVA采用Wilks' Lambda检验，Rao's F近似，使用88个主成分（保留约99%方差）。",
            "分析样本限制在美英数据均有覆盖的重叠时段（2014年起），以确保跨国比较的时间可比性。",
            "H1为描述性存在检验，不做年份固定效应控制。",
        ]
    ).render('tab3a_manova')


# ============================================================
# 表4: DID主回归
# ============================================================
def gen_table_4(data):
    print("Generating Table 4: DID regression...")
    bc = data['bootstrap']['comparison']
    rob = data['robustness']['model_3_time_trend']

    var_keys = ['intercept', 'UK', 'AU', 'time', 'post_aukus', 'UK_x_post', 'AU_x_post']
    var_labels = ['截距', 'D_UK（英国）', 'D_AU（澳大利亚）',
                  't（时间趋势）', 'Post（AUKUS后）',
                  'D_UK\u00d7Post', 'D_AU\u00d7Post']

    rows = []
    for vk, vl in zip(var_keys, var_labels):
        coef_cells = [vl]
        se_cells = ['']
        for pc in ['PC1', 'PC2', 'PC3']:
            d = bc[pc][vk]
            coef_cells.append(fmt_coef(d['coef'], d['p_bootstrap']))
            se_cells.append(f"({d['se_bootstrap']:.4f})")
        rows.append(coef_cells)
        rows.append(se_cells)

    # R² row
    r2_row = ['R\u00b2']
    for pc in ['PC1', 'PC2', 'PC3']:
        r2_row.append(f"{rob[pc]['_meta']['r_squared']:.4f}")
    rows.append(r2_row)

    ThreeLineTable(
        '表4  主成分回归分析结果（DID模型）',
        ['变量', 'PC1', 'PC2', 'PC3'],
        rows,
        [
            '注：系数为OLS估计值，括号内为Wild Cluster Bootstrap标准误。',
            '模型包含线性时间趋势控制变量，t = year \u2212 2014。',
            '***p<0.001, **p<0.01, *p<0.05, \u2020p<0.1（Bootstrap p值）。',
            'N=37,866，聚类数=11,459。基准组=美国，基准期=AUKUS前。',
        ]
    ).render('tab4_did_regression')


# ============================================================
# 表5: 平行趋势Wald
# ============================================================
def gen_table_5(data):
    print("Generating Table 5: Parallel trends Wald...")
    # 事件研究Wald检验结果（US+UK, 2014-2024, parallel_trends_test.py）
    wald_results = [
        ['PC1', '2.25', '7', '0.945', '满足'],
        ['PC2', '13.29', '7', '0.065', '满足（边际）'],
        ['PC3', '12.25', '7', '0.093', '满足（边际）'],
    ]
    ThreeLineTable(
        '表5  平行趋势检验（Wald联合检验）',
        ['主成分', 'Wald χ²', 'df', 'p值', '结论'],
        wald_results,
        [
            '注：W=β\'inv(Σ)β，Σ为Wild Cluster Bootstrap完整协方差矩阵',
            '（doc_id聚类，1,000次，Rademacher权重）。',
            '事件研究模型（US+UK, 2014-2024），检验AUKUS签署前7个年度UK×Year交互项是否联合为零。',
            '基准年=2021。',
        ]
    ).render('tab5_parallel_trends')


# ============================================================
# 表6: 安慰剂检验
# ============================================================
def gen_table_6(data):
    print("Generating Table 6: Placebo test...")
    p2020 = data['robustness']['model_4a_placebo_2020']
    p2021 = data['robustness']['model_4b_placebo_2021']
    rows = []
    for pc in ['PC1', 'PC2', 'PC3']:
        uk_2020 = p2020[pc]['UK_x_fake_post']
        uk_2021 = p2021[pc]['UK_x_fake_post']
        rows.append([
            pc, 'UK',
            f"{uk_2020['coef']:.4f}", f"{uk_2020['boot_p']:.3f}",
            f"{uk_2021['coef']:.4f}", f"{uk_2021['boot_p']:.3f}",
            '通过' if uk_2020['boot_p'] > 0.05 and uk_2021['boot_p'] > 0.05 else '未通过'
        ])
    ThreeLineTable(
        '表6  安慰剂检验结果',
        ['主成分', '国家', '2020系数', '2020 p值', '2021系数', '2021 p值', '结论'],
        rows,
        [
            '注：安慰剂检验仅使用AUKUS签署前数据，以虚假治疗时点重新估计DID模型。',
            '若安慰剂系数显著，说明该"效应"在AUKUS签署前即已存在。',
            'p值为Bootstrap p值（1,000次，Rademacher权重，doc_id聚类）。',
        ]
    ).render('tab6_placebo')


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("论文图表生成 — 中文权威期刊风格")
    print("=" * 60)

    # 加载数据
    print("\nLoading JSON data...")
    data = load_all_data()

    # 生成表格（不需要parquet）
    print("\n--- Generating Tables ---")
    gen_table_1(data)
    gen_table_2(data)
    gen_table_3a(data)
    gen_table_4(data)
    gen_table_5(data)
    gen_table_6(data)

    # 生成图（图3/4不需要parquet）
    print("\n--- Generating Figures ---")
    gen_figure_3(data)
    gen_figure_4(data)

    # 图1/2需要parquet
    df = load_parquet()
    if df is not None:
        gen_figure_1(df)
        gen_figure_2(df)
    else:
        print("Skipping Figure 1/2 (no parquet data)")

    print("\n" + "=" * 60)
    print(f"Output: {FIG_DIR} and {TAB_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
