#!/usr/bin/env python3
"""
MANOVA 时间稳健性检验
=====================

回应审稿人可能的质疑：三国数据时间范围不同（UK从2002, US从2014, AU从2017），
MANOVA是否受到时间混淆的影响？

检验方案：
1. 受限样本MANOVA：仅使用2017+数据（三国均有数据）
2. 时间残差MANOVA：先回归掉年份固定效应，再对残差做MANOVA
3. 对比结果，判断时间混淆的影响程度
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


def compute_wilks_lambda_f(Y, labels):
    """Wilks' Lambda → Rao's F approximation"""
    unique = np.unique(labels)
    n_groups = len(unique)
    n_total = len(labels)
    n_vars = Y.shape[1]

    groups = {c: Y[labels == c] for c in unique}
    grand_mean = np.mean(Y, axis=0)

    B = np.zeros((n_vars, n_vars))
    for c, data in groups.items():
        diff = (np.mean(data, axis=0) - grand_mean).reshape(-1, 1)
        B += len(data) * (diff @ diff.T)

    W = np.zeros((n_vars, n_vars))
    for c, data in groups.items():
        centered = data - np.mean(data, axis=0)
        W += centered.T @ centered

    # Wilks' Lambda via log-determinant for numerical stability
    try:
        sign_W, logdet_W = np.linalg.slogdet(W)
        sign_T, logdet_T = np.linalg.slogdet(W + B)
        if sign_W <= 0 or sign_T <= 0:
            return np.nan, np.nan, {}
        lambda_wilks = np.exp(logdet_W - logdet_T)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, {}

    p, k, n = n_vars, n_groups, n_total

    if (p**2 + (k-1)**2 - 5) > 0:
        t = np.sqrt((p**2 * (k-1)**2 - 4) / (p**2 + (k-1)**2 - 5))
    else:
        t = 1

    df1 = p * (k - 1)
    # Note: (p*(k-1)-2)/2 is NOT multiplied by t (verified against statsmodels)
    df2 = t * (n - 1 - (p + k) / 2) - (p * (k - 1) - 2) / 2

    if t > 0 and df2 > 0:
        lambda_t = lambda_wilks ** (1/t)
        F = ((1 - lambda_t) / lambda_t) * (df2 / df1)
        p_value = stats.f.sf(F, df1, df2)
    else:
        F, p_value = np.nan, np.nan

    return F, p_value, {
        'wilks_lambda': lambda_wilks, 'df1': df1, 'df2': df2,
        'n_groups': n_groups, 'n_samples': n_total, 'n_vars': n_vars
    }


def residualize_year_fe(Y, years):
    """Remove year fixed effects from Y by regressing on year dummies"""
    unique_years = sorted(np.unique(years))
    n = len(years)

    # Build year dummy matrix (leave one out for identification)
    X = np.ones((n, 1))  # intercept
    for y in unique_years[1:]:  # skip first year as reference
        X = np.column_stack([X, (years == y).astype(float)])

    # OLS: Y = X @ beta + residual
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    residuals = Y - X @ beta

    return residuals


def main():
    print("=" * 70)
    print("MANOVA 时间稳健性检验")
    print("=" * 70)
    print("问题：三国数据时间范围不同，MANOVA是否受时间混淆影响？")

    # Load data
    data_path = DATA_DIR / 'semantic_vectors_paragraph_global_A.parquet'
    df = pd.read_parquet(data_path)
    Y_matrix = np.vstack(df['Y_vector_global'].values)
    country_labels = df['country'].values
    years = df['year'].values

    print(f"\n全样本: N={len(df)}")
    print(f"各国年份范围:")
    for c in ['US', 'UK', 'AU']:
        mask = country_labels == c
        print(f"  {c}: {years[mask].min()}-{years[mask].max()}, N={mask.sum()}")

    # Use 88 PCs ≈ 99% variance
    n_pc = 88
    pca = PCA(n_components=n_pc)
    Y_pca_full = pca.fit_transform(Y_matrix)
    print(f"\nPCA: {n_pc} components, 累积方差={np.sum(pca.explained_variance_ratio_)*100:.1f}%")

    # ============================================================
    # Test 1: Original full sample (baseline)
    # ============================================================
    print("\n" + "=" * 70)
    print("检验1：全样本 MANOVA（基线）")
    print("=" * 70)

    F_full, p_full, det_full = compute_wilks_lambda_f(Y_pca_full, country_labels)
    print(f"  N = {len(Y_pca_full)}")
    print(f"  F = {F_full:.2f}, p = {p_full:.2e}")

    # ============================================================
    # Test 2: Restricted sample (2017+ only)
    # ============================================================
    print("\n" + "=" * 70)
    print("检验2：受限样本 MANOVA（2017+，三国均有数据）")
    print("=" * 70)

    mask_2017 = years >= 2017
    Y_restricted = Y_pca_full[mask_2017]
    labels_restricted = country_labels[mask_2017]
    years_restricted = years[mask_2017]

    print(f"  N = {mask_2017.sum()}")
    for c in ['US', 'UK', 'AU']:
        m = labels_restricted == c
        print(f"  {c}: N={m.sum()}, years={years_restricted[m].min()}-{years_restricted[m].max()}")

    F_restricted, p_restricted, det_restricted = compute_wilks_lambda_f(Y_restricted, labels_restricted)
    print(f"  F = {F_restricted:.2f}, p = {p_restricted:.2e}")
    print(f"  vs 全样本 F = {F_full:.2f} (变化 {(F_restricted - F_full)/F_full*100:+.1f}%)")

    # ============================================================
    # Test 3: Year FE residuals MANOVA (full sample)
    # ============================================================
    print("\n" + "=" * 70)
    print("检验3：时间残差 MANOVA（全样本，回归掉年份FE后）")
    print("=" * 70)

    Y_resid_full = residualize_year_fe(Y_pca_full, years)
    F_resid_full, p_resid_full, det_resid_full = compute_wilks_lambda_f(Y_resid_full, country_labels)
    print(f"  N = {len(Y_resid_full)}")
    print(f"  F = {F_resid_full:.2f}, p = {p_resid_full:.2e}")
    print(f"  vs 全样本原始 F = {F_full:.2f} (变化 {(F_resid_full - F_full)/F_full*100:+.1f}%)")

    # ============================================================
    # Test 4: Year FE residuals MANOVA (restricted 2017+)
    # ============================================================
    print("\n" + "=" * 70)
    print("检验4：时间残差 MANOVA（2017+样本，回归掉年份FE后）")
    print("=" * 70)

    Y_resid_restricted = residualize_year_fe(Y_restricted, years_restricted)
    F_resid_restricted, p_resid_restricted, _ = compute_wilks_lambda_f(Y_resid_restricted, labels_restricted)
    print(f"  N = {mask_2017.sum()}")
    print(f"  F = {F_resid_restricted:.2f}, p = {p_resid_restricted:.2e}")
    print(f"  vs 全样本原始 F = {F_full:.2f} (变化 {(F_resid_restricted - F_full)/F_full*100:+.1f}%)")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("汇总比较")
    print("=" * 70)
    print(f"{'检验':<35} {'N':>7} {'F':>10} {'p值':>12} {'结论':<10}")
    print("-" * 80)

    tests = [
        ("1. 全样本（基线）", len(Y_pca_full), F_full, p_full),
        ("2. 受限样本（2017+）", mask_2017.sum(), F_restricted, p_restricted),
        ("3. 时间残差（全样本）", len(Y_resid_full), F_resid_full, p_resid_full),
        ("4. 时间残差（2017+）", mask_2017.sum(), F_resid_restricted, p_resid_restricted),
    ]

    results_json = []
    for name, n, f, p in tests:
        conclusion = "支持H1***" if p < 0.001 else ("支持H1**" if p < 0.01 else "支持H1*" if p < 0.05 else "差异不显著")
        print(f"{name:<35} {n:>7} {f:>10.2f} {p:>12.2e} {conclusion:<10}")
        results_json.append({
            'test': name, 'n': int(n), 'F': float(f),
            'p': float(p), 'conclusion': conclusion
        })

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    f_min = min(r['F'] for r in results_json)
    if all(r['p'] < 0.001 for r in results_json):
        print("所有四种检验均支持H1（p < 0.001）。")
        print(f"最保守的F统计量为 {f_min:.2f}，仍远高于临界值。")
        print("三国在军事AI概念化上的差异不是时间混淆的产物。")
    else:
        print("部分检验结果存在差异，需要进一步审慎解读。")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'manova_time_robustness.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'method': 'MANOVA Time Robustness Check',
            'description': '检验MANOVA结果是否受三国数据时间范围不同的影响',
            'n_pc': n_pc,
            'results': results_json
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
