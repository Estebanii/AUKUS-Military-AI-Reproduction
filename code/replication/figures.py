"""Figures 1-4 (the plotting code of the original figure script). Every figure is written as PNG and PDF, and the
plotted numbers are written beside it as CSV (the figures are compared through these CSV files, not as images).

Chinese labels need a CJK font; the first available of :data:`CJK_FONTS` is used. Without one the labels are drawn
in English (the plotted data are the same).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, output, seeds

COLORS = {'US': '#1f77b4', 'UK': '#ff7f0e', 'AU': '#2ca02c', 'pre': '#4C72B0', 'post': '#DD8452', 'common': '#9E9E9E'}
CJK_FONTS = ('STHeiti', 'Songti SC', 'PingFang SC', 'Heiti SC', 'Noto Sans CJK SC', 'Noto Serif CJK SC',
             'Source Han Sans SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei')
Z = 1.96


def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in CJK_FONTS if f in available), None)
    if font:
        plt.rcParams['font.sans-serif'] = [font] + list(plt.rcParams['font.sans-serif'])
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt, font is not None


def _label(cjk: bool, zh: str, en: str) -> str:
    return zh if cjk else en


def _save(fig, stem: str) -> None:
    base = data.results_dir() / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')


COUNTRY_ZH = {'US': ('美国', 'United States'), 'UK': ('英国', 'United Kingdom'), 'AU': ('澳大利亚', 'Australia')}


def figure1_pca(Y, meta, stem: str) -> dict:
    """PCA-3 (random_state from :mod:`replication.seeds`) of Y: PC1 x PC2 by country and by period, with centroids."""
    from sklearn.decomposition import PCA
    plt, cjk = _setup()
    model = PCA(n_components=3, random_state=seeds.FIGURE_PCA_RANDOM_STATE)
    S = model.fit_transform(Y)
    ev = model.explained_variance_ratio_
    countries, post = meta['country'].to_numpy(), meta['post_aukus'].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for c in ('AU', 'UK', 'US'):
        m = countries == c
        ax1.scatter(S[m, 0], S[m, 1], c=COLORS[c], alpha=0.25, s=4, edgecolors='none',
                    label=_label(cjk, *COUNTRY_ZH[c]), rasterized=True)
        ax1.scatter(S[m, 0].mean(), S[m, 1].mean(), c=COLORS[c], s=120, marker='X', edgecolors='white',
                    linewidths=1.2, zorder=10)
    for value, zh, en, color, marker in ((0, 'AUKUS前', 'before AUKUS', COLORS['pre'], 'o'),
                                         (1, 'AUKUS后', 'after AUKUS', COLORS['post'], 's')):
        m = post == value
        ax2.scatter(S[m, 0], S[m, 1], c=color, marker=marker, alpha=0.25, s=4, edgecolors='none',
                    label=_label(cjk, zh, en), rasterized=True)
        ax2.scatter(S[m, 0].mean(), S[m, 1].mean(), c=color, s=120, marker='X', edgecolors='white', linewidths=1.2,
                    zorder=10)
    for ax, zh, en in ((ax1, '(a) 按国家', '(a) by country'), (ax2, '(b) 按AUKUS签署前后', '(b) before / after')):
        ax.set_xlabel(f'PC1 ({ev[0] * 100:.1f}%)')
        ax.set_ylabel(f'PC2 ({ev[1] * 100:.1f}%)')
        ax.set_title(_label(cjk, zh, en))
        ax.legend(markerscale=2.5)
    fig.suptitle(_label(cjk, '图1  三国军事人工智能概念在语义嵌入空间中的PCA分布',
                        'Figure 1  PCA of the semantic vectors'), y=1.02)
    fig.tight_layout()
    _save(fig, stem)
    plt.close(fig)
    centroids = []
    for key, col, values in (('country', countries, ('US', 'UK', 'AU')), ('post_aukus', post, (0, 1))):
        for v in values:
            m = col == v
            centroids.append({'group': key, 'value': v, 'n': int(m.sum()), 'pc1_mean': float(S[m, 0].mean()),
                              'pc2_mean': float(S[m, 1].mean())})
    output.write_text(f'{stem}.csv', pd.DataFrame(centroids).to_csv(index=False))
    return {'explained_variance_ratio': [float(v) for v in ev], 'centroids': centroids}


def figure2_tsne(Y, meta, stem: str) -> dict:
    """PCA-50 then t-SNE (perplexity 30, 1,000 iterations; random states from :mod:`replication.seeds`)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    plt, cjk = _setup()
    Y50 = PCA(n_components=50, random_state=seeds.FIGURE_PCA_RANDOM_STATE).fit_transform(Y)
    T = TSNE(n_components=2, perplexity=30, random_state=seeds.TSNE_RANDOM_STATE, max_iter=1000,
             n_jobs=-1).fit_transform(Y50)
    countries = meta['country'].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in ('AU', 'UK', 'US'):
        m = countries == c
        ax.scatter(T[m, 0], T[m, 1], c=COLORS[c], alpha=0.6, s=6, edgecolors='none',
                   label=_label(cjk, *COUNTRY_ZH[c]), rasterized=True)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(_label(cjk, '图2  三国军事人工智能概念在语义空间中的t-SNE分布', 'Figure 2  t-SNE of the semantic vectors'))
    ax.legend(markerscale=2.5)
    fig.tight_layout()
    _save(fig, stem)
    plt.close(fig)
    frame = pd.DataFrame({'country': countries, 'tsne1': T[:, 0], 'tsne2': T[:, 1]})
    output.write_text(f'{stem}.csv', frame.to_csv(index=False))
    return {'rows': int(len(T))}


def figure3_event_study(parallel_trends: dict, stem: str) -> None:
    """UK x year coefficients (base year 2021) with b +/- 1.96 x bootstrap SE, PC1-PC3."""
    plt, cjk = _setup()
    rows = []
    for pc in ('PC1', 'PC2', 'PC3'):
        coefs = parallel_trends['coefficients'][pc]
        for name, c in coefs.items():
            if name.startswith('UK_x_'):
                year = int(name.split('_')[-1])
                rows.append({'pc': pc, 'year': year, 'coef': c['coef'], 'se': c['se'],
                             'ci_lower': c['coef'] - Z * c['se'], 'ci_upper': c['coef'] + Z * c['se']})
    frame = pd.DataFrame(rows).sort_values(['pc', 'year']).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, pc in zip(axes, ('PC1', 'PC2', 'PC3')):
        d = frame[frame['pc'] == pc]
        years, b = d['year'].to_numpy(), d['coef'].to_numpy()
        lo, hi = d['ci_lower'].to_numpy(), d['ci_upper'].to_numpy()
        ax.fill_between(years, lo, hi, alpha=0.15, color=COLORS['UK'])
        ax.errorbar(years, b, yerr=[b - lo, hi - b], fmt='o-', color=COLORS['UK'], markersize=4, capsize=2,
                    linewidth=1, label=_label(cjk, '英国', 'United Kingdom'))
        ax.plot(2021, 0, 'kD', markersize=6, zorder=10)
        ax.axvline(x=2021, color='red', ls='--', lw=1.2, alpha=0.6)
        ax.axhline(y=0, color='black', lw=0.4, alpha=0.4)
        ax.set_xlabel(_label(cjk, '年份', 'year'))
        ax.set_ylabel(_label(cjk, '系数', 'coefficient'))
        ax.set_title(pc)
        ax.set_xticks([y for y in range(2014, 2025) if y != 2021])
        ax.tick_params(labelsize=6)
    axes[0].legend(loc='best')
    fig.suptitle(_label(cjk, '图3  事件研究分析：英国相对于美国的年度系数（基准年2021）',
                        'Figure 3  Event study: UK relative to US by year (base year 2021)'), y=1.03)
    fig.tight_layout()
    _save(fig, stem)
    plt.close(fig)
    output.write_text(f'{stem}.csv', frame.to_csv(index=False))


def figure4_neighbours(h3: dict, stem: str, top_n: int = 15) -> None:
    """Top-15 neighbour words after the signing per country; bars coloured by common / unique."""
    from matplotlib.patches import Patch
    plt, cjk = _setup()
    post = h3['post_aukus_analysis']
    nn, unique = post['nearest_words'], post['unique_words_top15']
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    rows = []
    for ax, c in zip(axes, ('US', 'UK', 'AU')):
        items = nn[c][:top_n]
        words = [w['word'] for w in items][::-1]
        sims = [w['similarity'] for w in items][::-1]
        colors = [COLORS[c] if w in set(unique.get(c, [])) else COLORS['common'] for w in words]
        ax.barh(range(len(words)), sims, color=colors, height=0.7, edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=8)
        ax.set_xlabel(_label(cjk, '余弦相似度', 'cosine similarity'))
        ax.set_title(_label(cjk, *COUNTRY_ZH[c]))
        ax.set_xlim(0.19, max(sims) + 0.02)
        for rank, w in enumerate(items, 1):
            rows.append({'country': c, 'rank': rank, 'word': w['word'], 'similarity': w['similarity'],
                         'unique_to_country': w['word'] in set(unique.get(c, []))})
    handles = [Patch(facecolor=COLORS['common'], label=_label(cjk, '共有', 'common'))] + [
        Patch(facecolor=COLORS[c], label=_label(cjk, f'{COUNTRY_ZH[c][0]}独有', f'{COUNTRY_ZH[c][1]} only'))
        for c in ('US', 'UK', 'AU')]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(_label(cjk, '图4  AUKUS签署后三国军事人工智能概念化的语义近邻词（Top-15）',
                        'Figure 4  Nearest neighbour words after the signing (top 15)'), y=1.02)
    fig.tight_layout(rect=(0, 0.07, 1, 1))   # the bottom strip holds the legend, below the x-axis labels
    _save(fig, stem)
    plt.close(fig)
    output.write_text(f'{stem}.csv', pd.DataFrame(rows).to_csv(index=False))
