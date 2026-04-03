#!/usr/bin/env python3
"""
H1 基线差异假说：Pre-AUKUS 子样本 PC 回归

在 AUKUS 签署前的数据上进行 PCA 降维和国别回归，
检验三国对军事 AI 概念化的基线差异模式。

输出：final_fuxian/outputs/h1_baseline_regression.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from scipy import stats

# ============================================================
# 路径
# ============================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'semantic_vectors_paragraph_global_A.parquet'
OUTPUT_PATH = PROJECT_ROOT / 'outputs' / 'h1_baseline_regression.json'

# ============================================================
# Wild Cluster Bootstrap（与主分析一致的推断方法）
# ============================================================
def wild_cluster_bootstrap(Y_pc, X, cluster_ids, n_bootstrap=1000, seed=42):
    """Wild Cluster Bootstrap（Rademacher 权重，中心化尾部概率 p 值）"""
    rng = np.random.RandomState(seed)
    n = len(Y_pc)

    # OLS
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ Y_pc
    residuals = Y_pc - X @ beta_hat
    fitted = X @ beta_hat

    # OLS SE & t-stats
    sigma2 = np.sum(residuals ** 2) / (n - X.shape[1])
    se_ols = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stat_ols = beta_hat / se_ols
    p_ols = 2 * (1 - stats.t.cdf(np.abs(t_stat_ols), n - X.shape[1]))

    # 聚类信息
    unique_clusters = np.unique(cluster_ids)
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    # Bootstrap
    bootstrap_betas = np.zeros((n_bootstrap, X.shape[1]))
    for b in range(n_bootstrap):
        weights = rng.choice([-1, 1], size=len(unique_clusters))
        sample_weights = np.empty(n)
        for j, c in enumerate(unique_clusters):
            sample_weights[cluster_to_idx[c]] = weights[j]

        Y_star = fitted + sample_weights * residuals
        bootstrap_betas[b] = XtX_inv @ X.T @ Y_star

    se_bootstrap = np.std(bootstrap_betas, axis=0, ddof=1)
    centered = bootstrap_betas - beta_hat[np.newaxis, :]
    p_bootstrap = np.mean(
        np.abs(centered) >= np.abs(beta_hat[np.newaxis, :]), axis=0
    )

    return {
        'beta': beta_hat,
        'se_ols': se_ols,
        'se_bootstrap': se_bootstrap,
        'p_ols': p_ols,
        'p_bootstrap': p_bootstrap,
        't_stat_ols': t_stat_ols,
        'bootstrap_betas': bootstrap_betas,
    }


def run_regression(Y_pca, country_labels, cluster_ids, feature_label, n_bootstrap=1000, seed=42):
    """对 3 个 PC 分别跑回归，返回结构化结果"""
    countries_in_data = sorted(set(country_labels))
    has_au = 'AU' in countries_in_data

    # 构造设计矩阵
    intercept = np.ones(len(Y_pca))
    uk_dummy = (country_labels == 'UK').astype(float)
    features = [intercept, uk_dummy]
    feature_names = ['intercept', 'UK']

    if has_au:
        au_dummy = (country_labels == 'AU').astype(float)
        features.append(au_dummy)
        feature_names.append('AU')

    X = np.column_stack(features)

    results = {}
    for pc_idx in range(min(3, Y_pca.shape[1])):
        y = Y_pca[:, pc_idx]
        pc_name = f'PC{pc_idx + 1}'

        boot = wild_cluster_bootstrap(y, X, cluster_ids, n_bootstrap=n_bootstrap, seed=seed + pc_idx)

        pc_result = {}
        for i, fname in enumerate(feature_names):
            pc_result[fname] = {
                'coef': float(boot['beta'][i]),
                'se_ols': float(boot['se_ols'][i]),
                'se_bootstrap': float(boot['se_bootstrap'][i]),
                'p_ols': float(boot['p_ols'][i]),
                'p_bootstrap': float(boot['p_bootstrap'][i]),
                't_stat_ols': float(boot['t_stat_ols'][i]),
            }

        # R²
        y_hat = X @ boot['beta']
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        pc_result['_meta'] = {
            'r_squared': float(r_squared),
            'n_obs': int(len(y)),
            'n_clusters': int(len(np.unique(cluster_ids))),
            'feature_names': feature_names,
        }
        results[pc_name] = pc_result

    return results


def main():
    print("=" * 60)
    print("H1 基线差异假说：Pre-AUKUS PC 回归")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_parquet(DATA_PATH)
    pre_df = df[df['post_aukus'] == False].copy()
    print(f"Pre-AUKUS 总样本: {len(pre_df)}")

    for c in ['US', 'UK', 'AU']:
        n = len(pre_df[pre_df['country'] == c])
        print(f"  {c}: {n}")

    # ============================================================
    # Pre-AUKUS PCA（独立于全样本）
    # ============================================================
    print("\n执行 Pre-AUKUS PCA...")
    Y_pre = np.vstack(pre_df['Y_vector_global'].values)

    pca_pre = PCA(n_components=3, random_state=42)
    Y_pca_pre = pca_pre.fit_transform(Y_pre)

    var_ratios = pca_pre.explained_variance_ratio_
    print(f"  PC1: {var_ratios[0]:.4f}, PC2: {var_ratios[1]:.4f}, PC3: {var_ratios[2]:.4f}")
    print(f"  累计: {sum(var_ratios):.4f}")

    # ============================================================
    # PCA 轴稳定性验证（与全样本 PCA 的余弦相似度）
    # ============================================================
    print("\n验证 PCA 轴稳定性...")
    Y_full = np.vstack(df['Y_vector_global'].values)
    pca_full = PCA(n_components=3, random_state=42)
    pca_full.fit(Y_full)

    axis_stability = {}
    for i in range(3):
        cos_sim = np.abs(np.dot(pca_pre.components_[i], pca_full.components_[i]))
        # 取绝对值因为 PCA 轴方向可能翻转
        axis_stability[f'PC{i+1}'] = float(cos_sim)
        print(f"  PC{i+1} 余弦相似度: {cos_sim:.4f}")

    # ============================================================
    # 主分析：US-UK only（Pre-AUKUS）
    # ============================================================
    print("\n" + "=" * 60)
    print("主分析：Pre-AUKUS US vs UK")
    print("=" * 60)

    usuk_mask = pre_df['country'].isin(['US', 'UK']).values
    Y_pca_usuk = Y_pca_pre[usuk_mask]
    country_usuk = pre_df.loc[usuk_mask, 'country'].values
    cluster_usuk = pre_df.loc[usuk_mask, 'doc_id'].values

    print(f"样本量: {len(Y_pca_usuk)}")
    print(f"聚类数: {len(np.unique(cluster_usuk))}")

    main_results = run_regression(Y_pca_usuk, country_usuk, cluster_usuk,
                                  'pre_aukus_us_uk', n_bootstrap=1000, seed=42)

    for pc in ['PC1', 'PC2', 'PC3']:
        uk = main_results[pc]['UK']
        r2 = main_results[pc]['_meta']['r_squared']
        print(f"  {pc}: β_UK={uk['coef']:.4f}, boot_p={uk['p_bootstrap']:.3f}, R²={r2:.4f}")

    # ============================================================
    # 探索性分析：含 AU（Pre-AUKUS 三国）
    # ============================================================
    print("\n" + "=" * 60)
    print("探索性分析：Pre-AUKUS 三国（含 AU）")
    print("=" * 60)

    country_all = pre_df['country'].values
    cluster_all = pre_df['doc_id'].values

    print(f"样本量: {len(Y_pca_pre)}")
    print(f"聚类数: {len(np.unique(cluster_all))}")

    exploratory_results = run_regression(Y_pca_pre, country_all, cluster_all,
                                         'pre_aukus_three_country', n_bootstrap=1000, seed=42)

    for pc in ['PC1', 'PC2', 'PC3']:
        uk = exploratory_results[pc]['UK']
        au = exploratory_results[pc]['AU']
        r2 = exploratory_results[pc]['_meta']['r_squared']
        print(f"  {pc}: β_UK={uk['coef']:.4f}(p={uk['p_bootstrap']:.3f}), "
              f"β_AU={au['coef']:.4f}(p={au['p_bootstrap']:.3f}), R²={r2:.4f}")

    # ============================================================
    # 稳健性：年份固定效应
    # ============================================================
    print("\n" + "=" * 60)
    print("稳健性检验：Pre-AUKUS US-UK + 年份FE")
    print("=" * 60)

    # 构造年份哑变量（US-UK子样本）
    years_usuk = pre_df.loc[usuk_mask, 'year'].values
    unique_years = sorted(set(years_usuk))
    base_year = unique_years[0]  # 最早年份为基准
    print(f"基准年份: {base_year}, 年份范围: {unique_years[0]}-{unique_years[-1]}")

    intercept = np.ones(len(Y_pca_usuk))
    uk_dummy = (country_usuk == 'UK').astype(float)
    year_dummies = []
    year_names = []
    for y in unique_years[1:]:  # 跳过基准年
        year_dummies.append((years_usuk == y).astype(float))
        year_names.append(f'year_{y}')

    X_fe = np.column_stack([intercept, uk_dummy] + year_dummies)
    fe_feature_names = ['intercept', 'UK'] + year_names

    fe_results = {}
    for pc_idx in range(3):
        y = Y_pca_usuk[:, pc_idx]
        pc_name = f'PC{pc_idx + 1}'

        boot = wild_cluster_bootstrap(y, X_fe, cluster_usuk, n_bootstrap=1000, seed=42 + pc_idx)

        # 只取 UK 系数
        uk_idx = 1
        fe_results[pc_name] = {
            'UK': {
                'coef': float(boot['beta'][uk_idx]),
                'se_bootstrap': float(boot['se_bootstrap'][uk_idx]),
                'p_bootstrap': float(boot['p_bootstrap'][uk_idx]),
            },
            '_meta': {
                'n_obs': int(len(y)),
                'n_year_dummies': len(year_names),
                'base_year': int(base_year),
            }
        }
        uk_fe = fe_results[pc_name]['UK']
        print(f"  {pc_name}: β_UK={uk_fe['coef']:.4f}, boot_p={uk_fe['p_bootstrap']:.3f}")

    # ============================================================
    # 输出
    # ============================================================
    output = {
        'method': 'H1 Baseline Differentiation: Pre-AUKUS PC Regression',
        'description': 'Pre-AUKUS子样本PCA + 国别回归，检验AUKUS签署前的基线跨国差异',
        'pca_basis': 'Pre-AUKUS (independent of full sample)',
        'n_bootstrap': 1000,
        'seed_base': 42,
        'pre_aukus_pca': {
            'n_components': 3,
            'explained_variance_ratio': {
                f'PC{i+1}': float(var_ratios[i]) for i in range(3)
            },
            'cumulative_variance': float(sum(var_ratios)),
            'n_samples_fit': int(len(Y_pre)),
        },
        'pca_axis_stability': {
            'description': 'Cosine similarity between Pre-AUKUS PCA and Full-Sample PCA axes',
            **axis_stability,
        },
        'main_analysis': {
            'description': 'Pre-AUKUS US vs UK (主分析)',
            'sample': 'US + UK, Pre-AUKUS only',
            **main_results,
        },
        'exploratory_analysis': {
            'description': 'Pre-AUKUS 三国含AU (探索性)',
            'sample': 'US + UK + AU, Pre-AUKUS only',
            'note': 'AU Pre-AUKUS仅844条观测(60%集中于2021)，结果仅供参考',
            **exploratory_results,
        },
        'robustness_year_fe': {
            'description': 'Pre-AUKUS US-UK + 年份固定效应',
            **fe_results,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
