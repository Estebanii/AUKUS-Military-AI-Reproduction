#!/usr/bin/env python3
"""
MANOVA 分期检验
===============

按 Pre-AUKUS / Post-AUKUS 分期进行 MANOVA，检验：
1. AUKUS签署前，三国概念化差异是否已经存在（基线差异）
2. AUKUS签署后，差异是否持续/增强
3. 对比两个时期的F值，量化AUKUS对差异程度的影响

回应方法论质疑：全时段MANOVA的显著性是否被post-AUKUS分化效应驱动
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from datetime import datetime

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

    # Wilks' Lambda via log-determinant for numerical stability (88-dim matrices)
    try:
        sign_W, logdet_W = np.linalg.slogdet(W)
        sign_T, logdet_T = np.linalg.slogdet(W + B)
        if sign_W <= 0 or sign_T <= 0:
            return np.nan, np.nan, {}
        lambda_wilks = np.exp(logdet_W - logdet_T)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, {}

    p, k, n = n_vars, n_groups, n_total

    # Rao's F approximation (Rencher & Christensen 2012, eq 6.10)
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
        p_value = stats.f.sf(F, df1, df2)  # sf() avoids 1-cdf() precision loss
    else:
        F, p_value = np.nan, np.nan

    # Wilks' Lambda complement (approximate effect size measure)
    eta_sq = 1 - lambda_wilks

    return F, p_value, {
        'wilks_lambda': float(lambda_wilks), 'df1': float(df1), 'df2': float(df2),
        'n_groups': int(n_groups), 'n_samples': int(n_total), 'n_vars': int(n_vars),
        'partial_eta_sq': float(eta_sq)
    }


def residualize_year_fe(Y, years):
    """Remove year fixed effects from Y"""
    unique_years = sorted(np.unique(years))
    n = len(years)
    X = np.ones((n, 1))
    for y in unique_years[1:]:
        X = np.column_stack([X, (years == y).astype(float)])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    return Y - X @ beta


def main():
    print("=" * 70)
    print("MANOVA 分期检验：Pre-AUKUS vs Post-AUKUS")
    print("=" * 70)

    # Load data
    data_path = DATA_DIR / 'semantic_vectors_paragraph_global_A.parquet'
    df = pd.read_parquet(data_path)
    Y_matrix = np.vstack(df['Y_vector_global'].values)
    country_labels = df['country'].values
    years = df['year'].values
    post_aukus = df['post_aukus'].values.astype(bool)

    print(f"\n全样本: N={len(df)}")
    print(f"\n分期样本量:")
    for period, mask in [("Pre-AUKUS", ~post_aukus), ("Post-AUKUS", post_aukus)]:
        print(f"\n  {period}:")
        for c in ['US', 'UK', 'AU']:
            cmask = (country_labels == c) & mask
            if cmask.sum() > 0:
                print(f"    {c}: N={cmask.sum()}, years={years[cmask].min()}-{years[cmask].max()}")
            else:
                print(f"    {c}: N=0")

    # PCA on full sample (consistent basis); 88 PCs ≈ 99% variance
    n_pc = 88
    pca = PCA(n_components=n_pc)
    Y_pca_full = pca.fit_transform(Y_matrix)
    print(f"\nPCA: {n_pc} components (fit on full sample for consistency)")

    results = []

    # ============================================================
    # Test 1: Full sample (reference baseline)
    # ============================================================
    print("\n" + "=" * 70)
    print("检验1：全样本 MANOVA（参照基线）")
    F, p, det = compute_wilks_lambda_f(Y_pca_full, country_labels)
    print(f"  N={len(Y_pca_full)}, F={F:.2f}, p={p:.2e}")
    results.append({"test": "1. 全样本（原始）", "n": int(len(Y_pca_full)),
                     "F": float(F), "p": float(p), "conclusion": f"支持H1***" if p < 0.001 else "..."})

    # Test 1b: Full sample + year FE residualized
    Y_resid_full = residualize_year_fe(Y_pca_full, years)
    F_r, p_r, _ = compute_wilks_lambda_f(Y_resid_full, country_labels)
    print(f"  (时间残差) F={F_r:.2f}, p={p_r:.2e}")
    results.append({"test": "1b. 全样本（时间残差）", "n": int(len(Y_pca_full)),
                     "F": float(F_r), "p": float(p_r), "conclusion": f"支持H1***" if p_r < 0.001 else "..."})

    # ============================================================
    # Test 2: Pre-AUKUS only
    # ============================================================
    print("\n" + "=" * 70)
    print("检验2：Pre-AUKUS MANOVA（基线差异检验）")
    pre_mask = ~post_aukus
    Y_pre = Y_pca_full[pre_mask]
    c_pre = country_labels[pre_mask]
    y_pre = years[pre_mask]

    F_pre, p_pre, det_pre = compute_wilks_lambda_f(Y_pre, c_pre)
    print(f"  N={len(Y_pre)}, F={F_pre:.2f}, p={p_pre:.2e}")
    for c in ['US', 'UK', 'AU']:
        print(f"    {c}: N={(c_pre == c).sum()}")
    results.append({"test": "2. Pre-AUKUS（原始）", "n": int(len(Y_pre)),
                     "F": float(F_pre), "p": float(p_pre),
                     "n_per_country": {c: int((c_pre == c).sum()) for c in ['US', 'UK', 'AU']},
                     "conclusion": f"支持H1***" if p_pre < 0.001 else ("支持H1*" if p_pre < 0.05 else "差异不显著")})

    # Pre-AUKUS + year FE
    Y_pre_resid = residualize_year_fe(Y_pre, y_pre)
    F_pre_r, p_pre_r, _ = compute_wilks_lambda_f(Y_pre_resid, c_pre)
    print(f"  (时间残差) F={F_pre_r:.2f}, p={p_pre_r:.2e}")
    results.append({"test": "2b. Pre-AUKUS（时间残差）", "n": int(len(Y_pre)),
                     "F": float(F_pre_r), "p": float(p_pre_r),
                     "conclusion": f"支持H1***" if p_pre_r < 0.001 else ("支持H1*" if p_pre_r < 0.05 else "差异不显著")})

    # ============================================================
    # Test 3: Post-AUKUS only
    # ============================================================
    print("\n" + "=" * 70)
    print("检验3：Post-AUKUS MANOVA（AUKUS后差异检验）")
    post_mask = post_aukus
    Y_post = Y_pca_full[post_mask]
    c_post = country_labels[post_mask]
    y_post = years[post_mask]

    F_post, p_post, det_post = compute_wilks_lambda_f(Y_post, c_post)
    print(f"  N={len(Y_post)}, F={F_post:.2f}, p={p_post:.2e}")
    for c in ['US', 'UK', 'AU']:
        print(f"    {c}: N={(c_post == c).sum()}")
    results.append({"test": "3. Post-AUKUS（原始）", "n": int(len(Y_post)),
                     "F": float(F_post), "p": float(p_post),
                     "n_per_country": {c: int((c_post == c).sum()) for c in ['US', 'UK', 'AU']},
                     "conclusion": f"支持H1***" if p_post < 0.001 else ("支持H1*" if p_post < 0.05 else "差异不显著")})

    # Post-AUKUS + year FE
    Y_post_resid = residualize_year_fe(Y_post, y_post)
    F_post_r, p_post_r, _ = compute_wilks_lambda_f(Y_post_resid, c_post)
    print(f"  (时间残差) F={F_post_r:.2f}, p={p_post_r:.2e}")
    results.append({"test": "3b. Post-AUKUS（时间残差）", "n": int(len(Y_post)),
                     "F": float(F_post_r), "p": float(p_post_r),
                     "conclusion": f"支持H1***" if p_post_r < 0.001 else ("支持H1*" if p_post_r < 0.05 else "差异不显著")})

    # ============================================================
    # Test 4: Pre-AUKUS restricted to 2017+ (all 3 countries)
    # ============================================================
    print("\n" + "=" * 70)
    print("检验4：Pre-AUKUS 2017+（三国均有数据的时期）")
    pre_2017_mask = (~post_aukus) & (years >= 2017)
    Y_pre17 = Y_pca_full[pre_2017_mask]
    c_pre17 = country_labels[pre_2017_mask]
    y_pre17 = years[pre_2017_mask]

    F_pre17, p_pre17, _ = compute_wilks_lambda_f(Y_pre17, c_pre17)
    print(f"  N={len(Y_pre17)}, F={F_pre17:.2f}, p={p_pre17:.2e}")
    for c in ['US', 'UK', 'AU']:
        print(f"    {c}: N={(c_pre17 == c).sum()}")
    results.append({"test": "4. Pre-AUKUS 2017+（原始）", "n": int(len(Y_pre17)),
                     "F": float(F_pre17), "p": float(p_pre17),
                     "n_per_country": {c: int((c_pre17 == c).sum()) for c in ['US', 'UK', 'AU']},
                     "conclusion": f"支持H1***" if p_pre17 < 0.001 else ("支持H1*" if p_pre17 < 0.05 else "差异不显著")})

    # ============================================================
    # Test 5: US vs UK only (pre-AUKUS, avoids AU sample issue)
    # ============================================================
    print("\n" + "=" * 70)
    print("检验5：Pre-AUKUS US vs UK（避免AU样本量问题）")
    pre_usuk = (~post_aukus) & ((country_labels == 'US') | (country_labels == 'UK'))
    Y_usuk = Y_pca_full[pre_usuk]
    c_usuk = country_labels[pre_usuk]

    F_usuk, p_usuk, _ = compute_wilks_lambda_f(Y_usuk, c_usuk)
    print(f"  N={len(Y_usuk)}, F={F_usuk:.2f}, p={p_usuk:.2e}")
    for c in ['US', 'UK']:
        print(f"    {c}: N={(c_usuk == c).sum()}")
    results.append({"test": "5. Pre-AUKUS US vs UK", "n": int(len(Y_usuk)),
                     "F": float(F_usuk), "p": float(p_usuk),
                     "n_per_country": {c: int((c_usuk == c).sum()) for c in ['US', 'UK']},
                     "conclusion": f"支持H1***" if p_usuk < 0.001 else ("支持H1*" if p_usuk < 0.05 else "差异不显著")})

    # ============================================================
    # Test 6: US vs UK, Pre-AUKUS, overlapping years only (2014+)
    # ============================================================
    print("\n" + "=" * 70)
    print("检验6：Pre-AUKUS US vs UK 重叠时段（2014+）")
    pre_usuk_overlap = (~post_aukus) & ((country_labels == 'US') | (country_labels == 'UK')) & (years >= 2014)
    Y_usuk_ov = Y_pca_full[pre_usuk_overlap]
    c_usuk_ov = country_labels[pre_usuk_overlap]

    F_usuk_ov, p_usuk_ov, _ = compute_wilks_lambda_f(Y_usuk_ov, c_usuk_ov)
    print(f"  N={len(Y_usuk_ov)}, F={F_usuk_ov:.2f}, p={p_usuk_ov:.2e}")
    for c in ['US', 'UK']:
        print(f"    {c}: N={(c_usuk_ov == c).sum()}")
    results.append({"test": "6. Pre-AUKUS US vs UK（2014+重叠）", "n": int(len(Y_usuk_ov)),
                     "F": float(F_usuk_ov), "p": float(p_usuk_ov),
                     "n_per_country": {c: int((c_usuk_ov == c).sum()) for c in ['US', 'UK']},
                     "conclusion": f"支持H1***" if p_usuk_ov < 0.001 else ("支持H1*" if p_usuk_ov < 0.05 else "差异不显著")})

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("汇总：分期MANOVA结果对比")
    print("=" * 70)
    print(f"\n{'检验':<35} {'N':>8} {'F':>10} {'p':>12} {'结论':>12}")
    print("-" * 80)
    for r in results:
        print(f"  {r['test']:<33} {r['n']:>8} {r['F']:>10.2f} {r['p']:>12.2e} {r['conclusion']:>12}")

    print(f"\n关键对比:")
    print(f"  Pre-AUKUS F = {results[2]['F']:.2f} vs Post-AUKUS F = {results[4]['F']:.2f}")
    print(f"  F变化率 = {(results[4]['F'] - results[2]['F']) / results[2]['F'] * 100:+.1f}%")
    print(f"  Pre-AUKUS(时间残差) F = {results[3]['F']:.2f} vs Post-AUKUS(时间残差) F = {results[5]['F']:.2f}")

    # Save results
    output = {
        "method": "MANOVA Period Split Analysis",
        "description": "分期MANOVA：检验Pre/Post-AUKUS差异是否均显著，以及F值变化",
        "n_pc": n_pc,
        "aukus_date": "2021-09 (month-level coding)",
        "results": results,
        "interpretation": {
            "pre_aukus_significant": results[2]['p'] < 0.05,
            "post_aukus_significant": results[4]['p'] < 0.05,
            "f_ratio_post_over_pre": float(results[4]['F'] / results[2]['F']) if results[2]['F'] > 0 else None,
            "f_ratio_post_over_pre_residualized": float(results[5]['F'] / results[3]['F']) if results[3]['F'] > 0 else None
        }
    }

    output_path = OUTPUT_DIR / 'manova_period_split.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
