#!/usr/bin/env python3
"""
消融实验：Raw U vs Whitened U vs Y=AU
======================================
比较三种向量空间下的核心统计结果：
1. Raw U — DeBERTa原始嵌入（各向异性）
2. Whitened U — 白化后的DeBERTa嵌入（校正各向异性）
3. Y = AU — A矩阵变换后的GPT-2空间嵌入（本研究使用的方法）

对每种空间计算：
- 各向异性指标（平均余弦方向相似度）
- MANOVA F值（全时段，时间残差化）
- DID核心系数（UK×post PC1-3，含Bootstrap p值）
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

def load_data():
    df = pd.read_parquet(DATA_DIR / 'semantic_vectors_paragraph_global_A.parquet')
    U = np.vstack(df['U_vector'].values)
    Y = np.vstack(df['Y_vector_global'].values)
    return df, U, Y

def whiten(X):
    """ZCA白化：校正各向异性，保持原始空间方向"""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 避免除以零
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    W = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    return X_centered @ W

def measure_anisotropy(X, label):
    """测量各向异性：平均余弦方向相似度"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.maximum(norms, 1e-10)
    mean_dir = X_norm.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
    mean_cos = float((X_norm @ mean_dir).mean())

    # PC1解释方差比
    cov = np.cov(X_norm, rowvar=False)
    vals = np.linalg.eigvalsh(cov)
    pc1_ratio = float(vals[-1] / vals.sum())

    print(f"  {label}: mean_cos={mean_cos:.4f}, PC1_ratio={pc1_ratio:.4f}")
    return {'mean_cos': mean_cos, 'pc1_ratio': pc1_ratio}

def residualize_year_fe(X, years):
    """年份固定效应残差化"""
    unique_years = np.unique(years)
    design = np.ones((len(years), 1))
    for y in unique_years[1:]:
        design = np.column_stack([design, (years == y).astype(float)])
    XtX_inv = np.linalg.pinv(design.T @ design)
    beta = XtX_inv @ design.T @ X
    return X - design @ beta

def compute_manova_f(X_resid, labels):
    """Wilks' Lambda → Rao's F"""
    unique = np.unique(labels)
    n_groups = len(unique)
    n_total, n_vars = X_resid.shape

    grand_mean = X_resid.mean(axis=0)
    B = np.zeros((n_vars, n_vars))
    W = np.zeros((n_vars, n_vars))

    for c in unique:
        mask = labels == c
        data = X_resid[mask]
        diff = (data.mean(axis=0) - grand_mean).reshape(-1, 1)
        B += mask.sum() * (diff @ diff.T)
        centered = data - data.mean(axis=0)
        W += centered.T @ centered

    sign_W, logdet_W = np.linalg.slogdet(W)
    sign_T, logdet_T = np.linalg.slogdet(W + B)
    if sign_W <= 0 or sign_T <= 0:
        return np.nan

    lambda_wilks = np.exp(logdet_W - logdet_T)
    p, k, n = n_vars, n_groups, n_total

    # Rao's F
    t = np.sqrt((p**2 * (k-1)**2 - 4) / (p**2 + (k-1)**2 - 5)) if (p**2 + (k-1)**2 - 5) > 0 else 1.0
    df1 = p * (k - 1)
    a = n - 1 - (p + k) / 2
    b = (df1 - 2) / 2
    df2 = a * t - b

    lambda_t = lambda_wilks ** (1.0 / t) if t != 0 else lambda_wilks
    F = ((1 - lambda_t) / lambda_t) * (df2 / df1)
    return float(F)

def compute_did_bootstrap(X_pca, df, n_bootstrap=500, seed=42):
    """DID回归 + Bootstrap（简化版，500次以节省时间）"""
    n = len(df)
    D_UK = (df['country'].values == 'UK').astype(float)
    D_AU = (df['country'].values == 'AU').astype(float)
    time_var = (df['year'].values - 2014).astype(float)
    post = df['post_aukus'].values.astype(float)

    design = np.column_stack([
        np.ones(n), D_UK, D_AU, time_var, post,
        D_UK * post, D_AU * post
    ])
    feature_names = ['intercept', 'UK', 'AU', 'time', 'post', 'UK_x_post', 'AU_x_post']

    clusters = df['doc_id'].values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {}
    for i, c in enumerate(clusters):
        if c not in cluster_to_idx:
            cluster_to_idx[c] = []
        cluster_to_idx[c].append(i)
    cluster_ids = list(unique_clusters)

    XtX_inv = np.linalg.pinv(design.T @ design)

    results = {}
    rng = np.random.RandomState(seed)

    for pc_idx in range(3):
        y = X_pca[:, pc_idx]
        beta = XtX_inv @ design.T @ y
        residuals = y - design @ beta

        # Bootstrap
        boot_betas = np.zeros((n_bootstrap, len(feature_names)))
        for b in range(n_bootstrap):
            weights = rng.choice([-1, 1], size=n_clusters)
            sample_weights = np.empty(n)
            for j, c in enumerate(cluster_ids):
                sample_weights[cluster_to_idx[c]] = weights[j]
            Y_star = design @ beta + residuals * sample_weights
            boot_betas[b] = XtX_inv @ design.T @ Y_star

        se_boot = np.std(boot_betas, axis=0, ddof=1)
        centered = boot_betas - beta[np.newaxis, :]
        p_boot = np.mean(np.abs(centered) >= np.abs(beta[np.newaxis, :]), axis=0)

        pc_name = f'PC{pc_idx+1}'
        results[pc_name] = {}
        for i, name in enumerate(feature_names):
            results[pc_name][name] = {
                'coef': float(beta[i]),
                'se_boot': float(se_boot[i]),
                'p_boot': float(p_boot[i])
            }

    return results

def run_ablation(label, vectors, df):
    """对一种向量空间运行完整消融分析"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    countries = df['country'].values
    years = df['year'].values

    # 1. 各向异性
    print("\n1. 各向异性测量:")
    aniso = measure_anisotropy(vectors, label)

    # 2. PCA → MANOVA（时间残差化）
    print("\n2. MANOVA（88 PCs, 时间残差化）:")
    pca88 = PCA(n_components=88)
    V88 = pca88.fit_transform(vectors)
    V88_resid = residualize_year_fe(V88, years)
    F = compute_manova_f(V88_resid, countries)
    print(f"  F={F:.2f}")

    # 3. DID（3 PCs）
    print("\n3. DID回归（3 PCs, Bootstrap 500次）:")
    pca3 = PCA(n_components=3)
    V3 = pca3.fit_transform(vectors)
    var_explained = pca3.explained_variance_ratio_
    print(f"  PCA方差: {var_explained}")

    did = compute_did_bootstrap(V3, df, n_bootstrap=500)
    for pc in ['PC1', 'PC2', 'PC3']:
        uk = did[pc]['UK_x_post']
        print(f"  {pc} UK×post: coef={uk['coef']:.4f}, p={uk['p_boot']:.3f}")

    return {
        'anisotropy': aniso,
        'manova_F': F,
        'pca_variance': [float(v) for v in var_explained],
        'did': did
    }

def main():
    print("消融实验：Raw U vs Whitened U vs Y=AU")
    print("=" * 70)

    df, U, Y = load_data()
    print(f"数据: N={len(df)}, U shape={U.shape}, Y shape={Y.shape}")

    # 白化U
    print("\n白化U向量中...")
    U_white = whiten(U)
    print(f"  白化完成: shape={U_white.shape}")

    # 运行三组消融
    results = {}
    results['raw_U'] = run_ablation("Raw U (DeBERTa原始)", U, df)
    results['whitened_U'] = run_ablation("Whitened U (白化后DeBERTa)", U_white, df)
    results['Y_AU'] = run_ablation("Y = AU (A矩阵变换, GPT-2空间)", Y, df)

    # 汇总对比表
    print("\n" + "=" * 70)
    print("消融实验汇总")
    print("=" * 70)

    print(f"\n{'指标':<30} {'Raw U':<18} {'Whitened U':<18} {'Y=AU':<18}")
    print("-" * 84)

    for key, label in [('mean_cos', '各向异性(mean_cos↓更好)'), ('pc1_ratio', 'PC1方差占比')]:
        vals = [results[s]['anisotropy'][key] for s in ['raw_U', 'whitened_U', 'Y_AU']]
        print(f"{label:<30} {vals[0]:<18.4f} {vals[1]:<18.4f} {vals[2]:<18.4f}")

    vals = [results[s]['manova_F'] for s in ['raw_U', 'whitened_U', 'Y_AU']]
    print(f"{'MANOVA F(残差化)':<30} {vals[0]:<18.2f} {vals[1]:<18.2f} {vals[2]:<18.2f}")

    for pc in ['PC1', 'PC2', 'PC3']:
        coefs = [results[s]['did'][pc]['UK_x_post']['coef'] for s in ['raw_U', 'whitened_U', 'Y_AU']]
        ps = [results[s]['did'][pc]['UK_x_post']['p_boot'] for s in ['raw_U', 'whitened_U', 'Y_AU']]
        print(f"{f'{pc} UK×post coef':<30} {coefs[0]:<18.4f} {coefs[1]:<18.4f} {coefs[2]:<18.4f}")
        print(f"{f'{pc} UK×post p':<30} {ps[0]:<18.3f} {ps[1]:<18.3f} {ps[2]:<18.3f}")

    # 保存结果
    output = {
        'experiment': 'Ablation: Raw U vs Whitened U vs Y=AU',
        'n_samples': len(df),
        'n_bootstrap': 500,
        'results': {}
    }
    for space_name, space_results in results.items():
        output['results'][space_name] = {
            'anisotropy': space_results['anisotropy'],
            'manova_F': space_results['manova_F'],
            'pca_variance': space_results['pca_variance'],
            'did_UK_x_post': {
                pc: {
                    'coef': space_results['did'][pc]['UK_x_post']['coef'],
                    'p_boot': space_results['did'][pc]['UK_x_post']['p_boot']
                } for pc in ['PC1', 'PC2', 'PC3']
            }
        }

    with open(OUTPUT_DIR / 'ablation_results.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUTPUT_DIR / 'ablation_results.json'}")

if __name__ == '__main__':
    main()
