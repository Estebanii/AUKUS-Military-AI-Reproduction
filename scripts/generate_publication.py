#!/usr/bin/env python3
"""
生成出版质量的表格和图表
===============================
为论文"共享技术与多元认知：AUKUS军事人工智能概念化比较分析"生成
可直接插入Word正式稿的专业三线表和统计图表。

输出: chatu/tables/*.pdf,*.png  +  chatu/figures/*.pdf,*.png

Usage:
    python generate_publication.py              # 生成全部
    python generate_publication.py --tables     # 仅表格
    python generate_publication.py --figures    # 仅图表
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# ============================================================
# Section 1: 环境设置
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # review/
REPO_ROOT = PROJECT_ROOT.parent   # llm-ER/

# matplotlib缓存目录（避免fontconfig警告）
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.mpl_cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# PDF嵌入TrueType（确保中文字体正确嵌入）
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.unicode_minus'] = False

# 字体配置
FONT_BODY = fm.FontProperties(family='Songti SC', size=9)
FONT_HEADER = fm.FontProperties(family='STHeiti', size=9)
FONT_TITLE = fm.FontProperties(family='STHeiti', size=11)
FONT_NOTE = fm.FontProperties(family='Songti SC', size=7.5)
FONT_AXIS = fm.FontProperties(family='STHeiti', size=10)
FONT_LEGEND = fm.FontProperties(family='Songti SC', size=9)
FONT_SMALL = fm.FontProperties(family='Songti SC', size=7)
FONT_SMALL_HEADER = fm.FontProperties(family='STHeiti', size=7)

# 颜色
COLORS = {
    'US': '#1f77b4',
    'UK': '#ff7f0e',
    'AU': '#2ca02c',
    'pre': '#7f7f7f',
    'post': '#d62728',
    'common': '#999999',
}

# 路径
OUTPUT_ROOT = PROJECT_ROOT / 'chatu'
TABLES_DIR = OUTPUT_ROOT / 'tables'
FIGURES_DIR = OUTPUT_ROOT / 'figures'

DATA_DIR = PROJECT_ROOT / 'data'
BOOTSTRAP_JSON = PROJECT_ROOT / 'outputs' / 'wild_cluster_bootstrap_results.json'
ROBUSTNESS_JSON = PROJECT_ROOT / 'outputs' / 'did_robustness_results.json'
MANOVA_JSON = PROJECT_ROOT / 'outputs' / 'manova_period_split.json'
NEIGHBOR_JSON = PROJECT_ROOT / 'outputs' / 'h4_nearest_neighbor_results.json'
EVENT_CSV = PROJECT_ROOT / 'figures' / 'parallel_trends' / 'event_study_coefficients.csv'
PARQUET_FILE = DATA_DIR / 'semantic_vectors_paragraph_global_A.parquet'

logger = logging.getLogger(__name__)

# ============================================================
# Section 2: ThreeLineTable 类
# ============================================================

class ThreeLineTable:
    """
    渲染出版质量的三线表（中文学术标准）。

    结构:
        ─────────────────── (顶线, linewidth=1.5)
        表头行 (黑体 STHeiti)
        ─────────────────── (中线, linewidth=0.75)
        数据行 (宋体 Songti SC)
        ─────────────────── (底线, linewidth=1.5)
        脚注 (小号宋体)
    """

    def __init__(self, title, col_headers, data_rows, notes=None,
                 col_widths=None, col_aligns=None, small=False):
        self.title = title
        self.col_headers = col_headers
        self.data_rows = data_rows
        self.notes = notes or []
        self.n_cols = len(col_headers)
        self.n_rows = len(data_rows)
        self.small = small  # 用于附录等长表格

        if col_widths is None:
            self.col_widths = [1.0 / self.n_cols] * self.n_cols
        else:
            total = sum(col_widths)
            self.col_widths = [w / total for w in col_widths]

        if col_aligns is None:
            self.col_aligns = ['left'] + ['center'] * (self.n_cols - 1)
        else:
            self.col_aligns = col_aligns

    def render(self, save_prefix, dpi=300):
        """渲染并保存为 PDF 和 PNG。"""
        font_body = FONT_SMALL if self.small else FONT_BODY
        font_header = FONT_SMALL_HEADER if self.small else FONT_HEADER
        font_title = FONT_TITLE
        font_note = FONT_NOTE

        base_row_h = 0.28 if self.small else 0.35
        header_h = 0.45  # extra space for multi-line headers
        title_h = 0.55
        note_line_h = 0.22
        note_h = note_line_h * max(1, len(self.notes))
        pad = 0.15

        # Calculate per-row heights based on max lines in any cell
        row_heights = []
        for row in self.data_rows:
            max_lines = max(str(cell).count('\n') + 1 for cell in row)
            row_heights.append(base_row_h * max(1, max_lines * 0.75))
        total_data_h = sum(row_heights) if row_heights else 0

        # Header may also have multi-line content
        max_header_lines = max((str(h).count('\n') + 1 for h in self.col_headers), default=1)
        header_h = max(header_h, base_row_h * max_header_lines * 0.75)

        fig_height = title_h + header_h + total_data_h + note_h + 2 * pad
        fig_width = 7.5

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        left = 0.03
        right = 0.97
        tw = right - left

        # y坐标（从上到下递减），all in normalized [0,1]
        total = fig_height
        y_title = 1.0 - pad / total
        y_top = y_title - title_h / total
        y_header_center = y_top - header_h / total / 2
        y_mid = y_top - header_h / total
        y_cursor = y_mid  # tracks current row top

        y_note_start = y_mid - total_data_h / total - 0.01
        y_bot = y_note_start - 0.005

        # 标题
        ax.text(0.5, y_title, self.title,
                fontproperties=font_title, ha='center', va='top')

        # 顶线
        ax.plot([left, right], [y_top, y_top], 'k-', linewidth=1.5, clip_on=False)

        # 表头
        xpos = self._x_positions(left, tw)
        for j, h in enumerate(self.col_headers):
            ax.text(xpos[j], y_header_center, h,
                    fontproperties=font_header, ha=self.col_aligns[j], va='center')

        # 中线
        ax.plot([left, right], [y_mid, y_mid], 'k-', linewidth=0.75, clip_on=False)

        # 数据行
        for i, row in enumerate(self.data_rows):
            rh_norm = row_heights[i] / total
            y_center = y_cursor - rh_norm / 2
            for j, cell in enumerate(row):
                ax.text(xpos[j], y_center, str(cell),
                        fontproperties=font_body, ha=self.col_aligns[j], va='center',
                        linespacing=1.2)
            y_cursor -= rh_norm

        # 底线
        ax.plot([left, right], [y_bot, y_bot], 'k-', linewidth=1.5, clip_on=False)

        # 脚注
        y_note = y_bot - 0.015
        note_step = note_line_h / total
        for k, note in enumerate(self.notes):
            ax.text(left, y_note - k * note_step, note,
                    fontproperties=font_note, ha='left', va='top', color='#333333')

        fig.savefig(f'{save_prefix}.pdf', bbox_inches='tight', pad_inches=0.08)
        fig.savefig(f'{save_prefix}.png', dpi=dpi, bbox_inches='tight', pad_inches=0.08)
        plt.close(fig)
        logger.info(f"  Saved: {save_prefix}.pdf/.png")

    def _x_positions(self, left, tw):
        positions = []
        cum = left
        for j in range(self.n_cols):
            cw = self.col_widths[j] * tw
            if self.col_aligns[j] == 'left':
                positions.append(cum + 0.005)
            elif self.col_aligns[j] == 'right':
                positions.append(cum + cw - 0.005)
            else:
                positions.append(cum + cw / 2)
            cum += cw
        return positions


# ============================================================
# Section 3: DataLoader（懒加载）
# ============================================================

class DataLoader:
    def __init__(self):
        self._cache = {}

    def _load_json(self, key, path):
        if key not in self._cache:
            with open(path) as f:
                self._cache[key] = json.load(f)
        return self._cache[key]

    def bootstrap(self):
        return self._load_json('bootstrap', BOOTSTRAP_JSON)

    def robustness(self):
        return self._load_json('robustness', ROBUSTNESS_JSON)

    def manova(self):
        return self._load_json('manova', MANOVA_JSON)

    def neighbors(self):
        return self._load_json('neighbors', NEIGHBOR_JSON)

    def event_study(self):
        if 'event' not in self._cache:
            self._cache['event'] = pd.read_csv(EVENT_CSV)
        return self._cache['event']

    def parquet(self):
        if 'parquet' not in self._cache:
            logger.info("  Loading parquet (474MB)...")
            self._cache['parquet'] = pd.read_parquet(PARQUET_FILE)
        return self._cache['parquet']


# ============================================================
# Section 4: 辅助函数
# ============================================================

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '\u2020'
    return ''

def fmt_coef(coef, p, dec=4):
    return f"{coef:+.{dec}f}{sig_stars(p)}"

def fmt_coef_nosign(coef, p, dec=4):
    return f"{coef:.{dec}f}{sig_stars(p)}"

def fmt_p(p):
    if p < 0.001: return '<0.001***'
    if p < 0.01: return f'{p:.3f}**'
    if p < 0.05: return f'{p:.3f}*'
    return f'{p:.3f}'

def fmt_p_plain(p):
    if p < 0.001: return '<0.001'
    return f'{p:.3f}'

def verify_fonts():
    ok = True
    for name, fp in [('Songti SC', FONT_BODY), ('STHeiti', FONT_HEADER)]:
        resolved = fm.findfont(fp, fallback_to_default=False)
        if 'DejaVu' in resolved or 'default' in resolved.lower():
            logger.warning(f"WARNING: {name} not found -> {resolved}")
            ok = False
        else:
            logger.info(f"  Font OK: {name} -> {Path(resolved).name}")
    return ok


# ============================================================
# Section 5: 表格生成函数
# ============================================================

def gen_table_1(data):
    """表1: 变量定义与操作化"""
    headers = ['变量', '类型', '操作化定义', '描述统计']
    rows = [
        ['语义表征 (Y)', '因变量',
         '768维语义向量，Y=A\u00b7U\n(DeBERTa-v3-base, 全局A矩阵变换)',
         'N=37,866'],
        ['国家 (Country)', '核心自变量',
         'D_UK（英国=1）、D_AU（澳大利亚=1）\n美国为基准组',
         'US:20,360\nUK:11,224\nAU:6,282'],
        ['AUKUS协议签订\n(Post-AUKUS)', '调节变量',
         '二值变量\n2021年9月及之后取值为1',
         '前期:20,784\n后期:17,082'],
        ['国家\u00d7AUKUS\n交互项', '交互变量',
         'UK\u00d7Post-AUKUS\nAU\u00d7Post-AUKUS',
         '用于检验H2'],
        ['文档标识\n(Doc ID)', '聚类变量',
         '文本来源的唯一标识符',
         '聚类数:11,459'],
    ]
    ThreeLineTable(
        '表 1  变量定义与操作化', headers, rows,
        col_widths=[0.18, 0.12, 0.42, 0.22],
        col_aligns=['left', 'center', 'left', 'center'],
    ).render(str(TABLES_DIR / 'table_1_variables'))


def gen_table_2(data):
    """表2: 样本分布与基本统计"""
    headers = ['国家', '术语出现次数', '占比', 'AUKUS前', 'AUKUS后']
    rows = [
        ['美国 (US)', '20,360', '53.8%', '13,232', '7,128'],
        ['英国 (UK)', '11,224', '29.6%', '6,708', '4,516'],
        ['澳大利亚 (AU)', '6,282', '16.6%', '844', '5,438'],
        ['合计', '37,866', '100%', '20,784', '17,082'],
    ]
    notes = ['注：数据基于56个目标术语在三国国防部语料中的出现频次。']
    ThreeLineTable(
        '表 2  样本分布与基本统计', headers, rows, notes,
        col_widths=[0.22, 0.22, 0.12, 0.20, 0.20],
        col_aligns=['left', 'right', 'right', 'right', 'right'],
    ).render(str(TABLES_DIR / 'table_2_sample_stats'))


def gen_table_3a(data):
    """表3: 国家差异整体检验（MANOVA）— Pre-AUKUS US vs UK 重叠时段"""
    m = data.manova()
    r = m['results'][8]  # Test 6: Pre-AUKUS US vs UK 2014+重叠时段
    headers = ['检验', '样本期', 'N', 'F统计量', 'p值', '结论']
    rows = [
        ['Pre-AUKUS US vs UK', '2014-2021.08', f"{r['n']:,}", f"{r['F']:.2f}", '<0.001', '支持H1***'],
    ]
    notes = [
        '注：MANOVA采用Wilks\' Lambda检验，Rao\'s F近似，使用88个主成分（保留约99%方差）。',
        '分析样本限制在美英数据均有覆盖的重叠时段（2014年起），以确保跨国比较的时间可比性。',
        'H1为描述性存在检验，不做年份固定效应控制。',
    ]
    ThreeLineTable(
        '表 3  国家差异整体检验（MANOVA）', headers, rows, notes,
        col_widths=[0.20, 0.18, 0.12, 0.15, 0.12, 0.18],
        col_aligns=['left', 'center', 'center', 'center', 'center', 'center'],
    ).render(str(TABLES_DIR / 'table_3a_manova'))



def gen_table_4(data):
    """表4: 三国在语义轴上的位置分布"""
    headers = ['国家', '样本量', 'US-UK轴', 'US-AU轴', 'UK-AU轴', 'PC1均值']
    rows = [
        ['美国 (US)', '20,360', '0.240', '0.223', '-0.003', '-0.354'],
        ['英国 (UK)', '11,224', '-0.099', '-0.068', '0.046', '-0.300'],
        ['澳大利亚 (AU)', '6,282', '-0.071', '-0.139', '-0.140', '-0.277'],
    ]
    notes = ['注：语义轴由各国对平均向量差异定义。']
    ThreeLineTable(
        '表 4  三国在语义轴上的位置分布', headers, rows, notes,
        col_widths=[0.20, 0.13, 0.16, 0.16, 0.16, 0.16],
        col_aligns=['left', 'right', 'center', 'center', 'center', 'center'],
    ).render(str(TABLES_DIR / 'table_4_semantic_axes'))


def gen_table_5(data):
    """表5: 主成分回归分析结果（含线性时间趋势的DID模型）"""
    boot = data.bootstrap()['comparison']
    rob = data.robustness()['model_3_time_trend']

    headers = ['变量', 'PC1', 'PC2', 'PC3']
    var_keys = ['intercept', 'UK', 'AU', 'time', 'post_aukus', 'UK_x_post', 'AU_x_post']
    var_labels = ['Intercept（截距）', 'D_UK（英国）', 'D_AU（澳大利亚）',
                  't（时间趋势）', 'Post（AUKUS后）',
                  'D_UK \u00d7 Post（英国\u00d7AUKUS后）',
                  'D_AU \u00d7 Post（澳大利亚\u00d7AUKUS后）']

    rows = []
    for vk, vl in zip(var_keys, var_labels):
        coef_row = [vl]
        se_row = ['']
        for pc in ['PC1', 'PC2', 'PC3']:
            d = boot[pc][vk]
            coef_row.append(fmt_coef_nosign(d['coef'], d['p_bootstrap']))
            se_row.append(f"({d['se_bootstrap']:.4f})")
        rows.append(coef_row)
        rows.append(se_row)

    # R²行
    r2_row = ['R\u00b2']
    for pc in ['PC1', 'PC2', 'PC3']:
        r2_row.append(f"{rob[pc]['_meta']['r_squared']:.4f}")
    rows.append(r2_row)

    notes = [
        '注：系数为OLS估计值，括号内为Wild Cluster Bootstrap标准误。',
        '模型包含线性时间趋势控制变量，时间变量 t = year − 2014。',
        '***p<0.001, **p<0.01, *p<0.05, \u2020p<0.1（Bootstrap p值）。',
        'N=37,866，聚类数=11,459。基准组=美国，基准期=AUKUS前。',
    ]
    ThreeLineTable(
        '表 5  主成分回归分析结果（DID模型）', headers, rows, notes,
        col_widths=[0.30, 0.23, 0.23, 0.23],
        col_aligns=['left', 'center', 'center', 'center'],
    ).render(str(TABLES_DIR / 'table_5_regression'))


def gen_table_6(data):
    """表6: H2假设检验——DID交互项"""
    boot = data.bootstrap()['comparison']
    rob = data.robustness()['model_3_time_trend']

    headers = ['交互项', 'PC1系数', 'PC2系数\u2020', 'PC3系数\u2021', '总体模式']

    uk_row = ['D_UK \u00d7 Post（英国\u00d7AUKUS后）']
    au_row = ['D_AU \u00d7 Post（澳大利亚\u00d7AUKUS后）']
    for pc in ['PC1', 'PC2', 'PC3']:
        uk_d = boot[pc]['UK_x_post']
        au_d = boot[pc]['AU_x_post']
        uk_row.append(fmt_coef(uk_d['coef'], uk_d['p_bootstrap']))
        au_row.append(fmt_coef(au_d['coef'], au_d['p_bootstrap']))
    uk_row.append('PC1+PC2分化（‡PC3为既有趋势）')
    au_row.append('数据不足')

    r2_row = ['R\u00b2']
    for pc in ['PC1', 'PC2', 'PC3']:
        r2_row.append(f"{rob[pc]['_meta']['r_squared']:.4f}")
    r2_row.append('')

    rows = [uk_row, au_row, r2_row]
    notes = [
        '注：正系数表示相对于美国分化，负系数表示向美国趋同。模型含线性时间趋势。',
        '\u2020PC2在聚类稳健Wald下满足平行趋势(p=0.076)，OLS下违反(p=0.005)。',
        '\u2021PC3安慰剂检验未通过，该效应不应归因于AUKUS因果效应。',
        'Bootstrap: 1,000次, Rademacher权重, 11,459个doc_id聚类。',
    ]
    ThreeLineTable(
        '表 6  H2假设检验：DID交互项分析结果', headers, rows, notes,
        col_widths=[0.28, 0.15, 0.15, 0.15, 0.22],
        col_aligns=['left', 'center', 'center', 'center', 'center'],
    ).render(str(TABLES_DIR / 'table_6_did_interaction'))


def gen_table_7(data):
    """表7: 平行趋势联合检验（UK only, 2014-2024, 完整Bootstrap协方差Wald）"""
    # 事件研究Wald检验结果（parallel_trends_test.py, US+UK, 2014-2024）
    wald_results = [
        ['PC1', '2.25', '7', '0.945', '满足'],
        ['PC2', '13.29', '7', '0.065', '满足（边际）'],
        ['PC3', '12.25', '7', '0.093', '满足（边际）'],
    ]
    headers = ['主成分', 'Wald χ²', 'df', 'p值', '结论']
    notes = [
        '注：W=β\'Σ⁻¹β，Σ为Wild Cluster Bootstrap完整协方差矩阵',
        '（doc_id聚类，1,000次，Rademacher权重）。',
        '事件研究模型（US+UK, 2014-2024），检验AUKUS签署前7个年度UK×Year交互项是否联合为零。',
        '基准年=2021。',
    ]
    ThreeLineTable(
        '表 7  平行趋势联合检验结果', headers, wald_results, notes,
        col_widths=[0.15, 0.20, 0.10, 0.20, 0.25],
        col_aligns=['center'] * 5,
    ).render(str(TABLES_DIR / 'table_7_parallel_trends'))


def gen_table_9(data):
    """表9: 安慰剂检验结果"""
    plac = data.robustness()['summary']['placebo']

    headers = ['安慰剂时点', 'PC1 UK系数', 'PC1 p值',
               'PC2 UK系数', 'PC2 p值', 'PC3 UK系数', 'PC3 p值']
    rows = []
    for label, key in [('2020年', 'placebo_2020'), ('2021年', 'placebo_2021')]:
        row = [label]
        for pc in ['PC1', 'PC2', 'PC3']:
            d = plac[pc][key]
            row.append(f"{d['UK_coef']:.4f}")
            row.append(fmt_p_plain(d['UK_p']))
        rows.append(row)

    notes = [
        '注：仅使用AUKUS前数据（N=20,784），设置虚假处理时点。',
        'PC1、PC2在所有安慰剂时点均不显著（p>0.4），支持因果解释。',
        'PC3在两个时点均显著（2020: p<0.001, 2021: p=0.013），',
        '  表明PC3的效应为AUKUS前既有趋势，不应归因于AUKUS。',
    ]
    ThreeLineTable(
        '表 9  安慰剂检验结果', headers, rows, notes,
        col_widths=[0.16, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
        col_aligns=['left', 'center', 'center', 'center', 'center', 'center', 'center'],
    ).render(str(TABLES_DIR / 'table_9_placebo'))


def gen_appendix_b(data):
    """附录B: 56个目标术语列表"""
    terms = [
        ('artificial intelligence', 508, 'AI核心'),
        ('cyber', 398, 'AI应用'),
        ('autonomous', 365, '自主系统'),
        ('unmanned', 361, '自主系统'),
        ('autonomy', 207, '自主系统'),
        ('autonomous systems', 206, '自主系统'),
        ('lethal', 162, '伦理治理'),
        ('uncrewed', 157, '自主系统'),
        ('drones', 156, '自主系统'),
        ('unmanned aerial', 147, '自主系统'),
        ('machine learning', 143, 'AI核心'),
        ('surveillance', 142, 'AI应用'),
        ('robotics', 123, '自主系统'),
        ('drone', 122, '自主系统'),
        ('lethality', 110, '伦理治理'),
        ('robotic', 96, '自主系统'),
        ('targeting', 95, 'AI应用'),
        ('safety', 87, '伦理治理'),
        ('uas', 85, '自主系统'),
        ('automated', 84, 'AI应用'),
        ('intelligence ai', 81, 'AI应用'),
        ('artificial intelligence ai', 81, 'AI核心'),
        ('algorithms', 77, 'AI核心'),
        ('automation', 74, 'AI应用'),
        ('unmanned aircraft', 72, '自主系统'),
        ('unmanned aerial systems', 70, '自主系统'),
        ('reconnaissance', 67, 'AI应用'),
        ('target', 57, 'AI应用'),
        ('tactics', 56, 'AI应用'),
        ('unmanned systems', 50, '自主系统'),
        ('cyber capabilities', 48, 'AI应用'),
        ('uncrewed aerial', 45, '自主系统'),
        ('ai capabilities', 44, 'AI应用'),
        ('isr', 35, 'AI应用'),
        ('robots', 33, '自主系统'),
        ('uncrewed systems', 32, '自主系统'),
        ('algorithm', 32, 'AI核心'),
        ('uavs', 31, '自主系统'),
        ('ai systems', 29, 'AI应用'),
        ('unmanned aerial system', 27, '自主系统'),
        ('responsible ai', 27, '伦理治理'),
        ('robot', 27, '自主系统'),
        ('use of artificial intelligence', 26, 'AI应用'),
        ('uncrewed aerial systems', 24, '自主系统'),
        ('uav', 23, '自主系统'),
        ('ai and autonomy', 22, 'AI核心'),
        ('human-machine teaming', 21, 'AI应用'),
        ('the drone', 20, '自主系统'),
        ('autonomous underwater', 19, '自主系统'),
        ('ai and machine learning', 19, 'AI核心'),
        ('ai models', 18, 'AI核心'),
        ('a drone', 17, '自主系统'),
        ('drone technology', 17, '自主系统'),
        ('uncrewed aircraft', 16, '自主系统'),
        ('autonomous capabilities', 14, '自主系统'),
        ('ai algorithms', 13, 'AI核心'),
    ]
    headers = ['术语', '总频率', '类别']
    rows = [[t, str(f), c] for t, f, c in terms]
    notes = [
        '注：术语经语义过滤（与32个核心概念的余弦相似度>0.35）、',
        '频率筛选（总频率\u226510）和人工审核后确定。',
    ]
    ThreeLineTable(
        '附录 B  56个目标术语列表', headers, rows, notes,
        col_widths=[0.50, 0.18, 0.22],
        col_aligns=['left', 'right', 'center'],
        small=True,
    ).render(str(TABLES_DIR / 'appendix_b_terms'))


# ============================================================
# Section 6: 图表生成函数
# ============================================================

def gen_figure_1(data):
    """图1: PCA分布散点图（双面板）"""
    from sklearn.decomposition import PCA

    df = data.parquet()
    logger.info("  Extracting Y vectors...")
    Y = np.vstack(df['Y_vector_global'].values)

    logger.info("  Running PCA (768D -> 3D)...")
    pca = PCA(n_components=3, random_state=42)
    Y_pca = pca.fit_transform(Y)
    ev = pca.explained_variance_ratio_

    countries = df['country'].values
    post = df['post_aukus'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # (a) 按国家
    for c, color, label in [('US', COLORS['US'], '美国'),
                             ('UK', COLORS['UK'], '英国'),
                             ('AU', COLORS['AU'], '澳大利亚')]:
        mask = countries == c
        ax1.scatter(Y_pca[mask, 0], Y_pca[mask, 1],
                    c=color, alpha=0.12, s=2, label=label, rasterized=True)
    ax1.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontproperties=FONT_AXIS)
    ax1.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontproperties=FONT_AXIS)
    ax1.set_title('(a) 按国家分布', fontproperties=FONT_TITLE)
    ax1.legend(prop=FONT_LEGEND, markerscale=6, framealpha=0.8)

    # (b) 按时期
    pre_mask = ~post.astype(bool)
    post_mask = post.astype(bool)
    ax2.scatter(Y_pca[pre_mask, 0], Y_pca[pre_mask, 1],
                c=COLORS['pre'], alpha=0.12, s=2, label='AUKUS前',
                marker='o', rasterized=True)
    ax2.scatter(Y_pca[post_mask, 0], Y_pca[post_mask, 1],
                c=COLORS['post'], alpha=0.12, s=2, label='AUKUS后',
                marker='o', rasterized=True)
    ax2.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontproperties=FONT_AXIS)
    ax2.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontproperties=FONT_AXIS)
    ax2.set_title('(b) 按AUKUS时期分布', fontproperties=FONT_TITLE)
    ax2.legend(prop=FONT_LEGEND, markerscale=6, framealpha=0.8)

    fig.suptitle('图 1  三国军事人工智能概念在语义嵌入空间中的PCA分布',
                 fontproperties=FONT_TITLE, y=1.02)
    fig.tight_layout()

    save = str(FIGURES_DIR / 'figure_1_pca')
    fig.savefig(f'{save}.pdf', bbox_inches='tight', pad_inches=0.1, dpi=150)
    fig.savefig(f'{save}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"  Saved: {save}.pdf/.png")


def gen_figure_2(data):
    """图2: t-SNE分布散点图"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    df = data.parquet()
    Y = np.vstack(df['Y_vector_global'].values)

    logger.info("  PCA 768D -> 50D...")
    pca = PCA(n_components=50, random_state=42)
    Y50 = pca.fit_transform(Y)

    logger.info("  Running t-SNE (50D -> 2D, ~5-10 min)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, verbose=1, n_jobs=-1)
    Y2 = tsne.fit_transform(Y50)

    countries = df['country'].values
    fig, ax = plt.subplots(figsize=(8, 8))
    for c, color, label in [('US', COLORS['US'], '美国'),
                             ('UK', COLORS['UK'], '英国'),
                             ('AU', COLORS['AU'], '澳大利亚')]:
        mask = countries == c
        ax.scatter(Y2[mask, 0], Y2[mask, 1],
                   c=color, alpha=0.15, s=2, label=label, rasterized=True)
    ax.set_xlabel('t-SNE 维度1', fontproperties=FONT_AXIS)
    ax.set_ylabel('t-SNE 维度2', fontproperties=FONT_AXIS)
    ax.set_title('图 2  三国军事人工智能概念在语义空间中的t-SNE分布',
                 fontproperties=FONT_TITLE)
    ax.legend(prop=FONT_LEGEND, markerscale=6, framealpha=0.8)

    save = str(FIGURES_DIR / 'figure_2_tsne')
    fig.savefig(f'{save}.pdf', bbox_inches='tight', pad_inches=0.1, dpi=150)
    fig.savefig(f'{save}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"  Saved: {save}.pdf/.png")


def gen_figure_3(data):
    """图3: 事件研究平行趋势图（三面板，UK only, 2014-2024）"""
    df = data.event_study()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for idx, pc in enumerate(['PC1', 'PC2', 'PC3']):
        ax = axes[idx]
        sub = df[(df['pc'] == pc) & (df['country'] == 'UK')].sort_values('year')
        years = sub['year'].values.astype(float)
        coefs = sub['coef'].values
        ci_lo = sub['ci_lower'].values
        ci_hi = sub['ci_upper'].values

        ax.fill_between(years, ci_lo, ci_hi, alpha=0.15, color=COLORS['UK'])
        ax.errorbar(years, coefs,
                    yerr=[coefs - ci_lo, ci_hi - coefs],
                    fmt='o-', color=COLORS['UK'], label='英国',
                    markersize=4, capsize=2, linewidth=1)

        # 基准年
        ax.plot(2021, 0, 'kD', markersize=6, zorder=10)
        ax.axvline(x=2021, color='red', ls='--', lw=1.2, alpha=0.6)
        ax.axhline(y=0, color='black', lw=0.4, alpha=0.4)

        ax.set_xlabel('年份', fontproperties=FONT_AXIS)
        ax.set_ylabel('系数', fontproperties=FONT_AXIS)
        ax.set_title(pc, fontproperties=FONT_TITLE)
        if idx == 0:
            ax.legend(prop=FONT_LEGEND, loc='best', framealpha=0.8)

        # x轴刻度
        ax.set_xticks([y for y in range(2014, 2025) if y != 2021])
        ax.tick_params(labelsize=7)

    fig.suptitle('图 3  事件研究分析：英国相对于美国的年度系数（2014-2024，基准年2021）',
                 fontproperties=fm.FontProperties(family='STHeiti', size=10), y=1.03)
    fig.tight_layout()

    save = str(FIGURES_DIR / 'figure_3_event_study')
    fig.savefig(f'{save}.pdf', bbox_inches='tight', pad_inches=0.1)
    fig.savefig(f'{save}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info(f"  Saved: {save}.pdf/.png")


def gen_figure_4(data):
    """图4: 语义近邻词柱状图（Post-AUKUS）"""
    nn = data.neighbors()['post_aukus_analysis']['nearest_words']
    top_n = 15

    country_words = {}
    for c in ['US', 'UK', 'AU']:
        country_words[c] = [(w['word'], w['similarity']) for w in nn[c][:top_n]]

    # 共有/独有
    word_sets = {c: set(w for w, _ in words) for c, words in country_words.items()}
    common = word_sets['US'] & word_sets['UK'] & word_sets['AU']

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    clabels = {'US': '美国', 'UK': '英国', 'AU': '澳大利亚'}

    for idx, c in enumerate(['US', 'UK', 'AU']):
        ax = axes[idx]
        words = list(reversed(country_words[c][:top_n]))
        labels = [w for w, _ in words]
        values = [s for _, s in words]
        bar_colors = [COLORS['common'] if w in common else COLORS[c] for w in labels]

        bars = ax.barh(range(len(labels)), values, color=bar_colors, height=0.7,
                       edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel('余弦相似度', fontproperties=FONT_AXIS)
        ax.set_title(clabels[c], fontproperties=FONT_TITLE, fontsize=12)

        # 设置合适的x范围
        min_v = min(values)
        max_v = max(values)
        ax.set_xlim(min_v - 0.005, max_v + 0.008)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['common'], label='三国共有'),
        Patch(facecolor=COLORS['US'], label='美国独有'),
        Patch(facecolor=COLORS['UK'], label='英国独有'),
        Patch(facecolor=COLORS['AU'], label='澳大利亚独有'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               prop=FONT_LEGEND, bbox_to_anchor=(0.5, -0.01), frameon=True)

    fig.suptitle('图 4  AUKUS签署后三国概念化的语义近邻词（Top-15，经BPE过滤）',
                 fontproperties=FONT_TITLE, y=1.01)
    fig.tight_layout()

    save = str(FIGURES_DIR / 'figure_4_nearest_neighbors')
    fig.savefig(f'{save}.pdf', bbox_inches='tight', pad_inches=0.12)
    fig.savefig(f'{save}.png', dpi=300, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    logger.info(f"  Saved: {save}.pdf/.png")


# ============================================================
# Section 7: Main
# ============================================================

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='生成出版质量的表格和图表')
    parser.add_argument('--tables', action='store_true', help='仅生成表格')
    parser.add_argument('--figures', action='store_true', help='仅生成图表')
    args = parser.parse_args()

    gen_all = not args.tables and not args.figures

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== 验证字体 ===")
    verify_fonts()

    data = DataLoader()

    if args.tables or gen_all:
        logger.info("\n=== 生成表格 ===")
        table_funcs = [
            ('表1: 变量定义', gen_table_1),
            ('表2: 样本分布', gen_table_2),
            ('表3a: MANOVA', gen_table_3a),
            ('表4: 语义轴位置', gen_table_4),
            ('表5: DID回归', gen_table_5),
            ('表6: H2交互项', gen_table_6),
            ('表7: 平行趋势Wald', gen_table_7),
            ('表9: 安慰剂检验', gen_table_9),
            ('附录B: 56术语', gen_appendix_b),
        ]
        for name, func in table_funcs:
            logger.info(f"\n{name}...")
            try:
                func(data)
            except Exception as e:
                logger.error(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

    if args.figures or gen_all:
        logger.info("\n=== 生成图表 ===")
        figure_funcs = [
            ('图1: PCA分布', gen_figure_1),
            ('图2: t-SNE分布', gen_figure_2),
            ('图3: 事件研究', gen_figure_3),
            ('图4: 近邻词', gen_figure_4),
        ]
        for name, func in figure_funcs:
            logger.info(f"\n{name}...")
            try:
                func(data)
            except Exception as e:
                logger.error(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

    logger.info(f"\n=== 完成 ===")
    logger.info(f"输出目录: {OUTPUT_ROOT}")


if __name__ == '__main__':
    main()
