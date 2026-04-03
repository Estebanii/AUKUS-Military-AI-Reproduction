"""
Comprehensive DID Robustness Analysis
=====================================

Addresses all DID methodology issues identified in the audit:

1. Main DID (baseline, standard specification without year controls)
2. Year Fixed Effects robustness (TWFE DID)
3. Linear time trend robustness (matches original paper equation)
4. Placebo tests at fake treatment dates (2020, 2021-Jan using pre-AUKUS data)
5. Restricted sample (2017+, when all three countries have data)
6. Proper Wald test for parallel trends (full covariance matrix)

All models use Wild Cluster Bootstrap (1,000 iterations, Rademacher weights,
doc_id clustering) for inference.

"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

# ============================================================
# Path Setup
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Data Loading & PCA
# ============================================================

def load_data() -> pd.DataFrame:
    """Load semantic vectors data with global A matrix transformation."""
    data_path = DATA_DIR / 'semantic_vectors_paragraph_global_A.parquet'
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} records")

    required_cols = ['country', 'year', 'month', 'Y_vector_global', 'post_aukus', 'doc_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def prepare_pca(df: pd.DataFrame, n_components: int = 3) -> Tuple[np.ndarray, PCA]:
    """Run PCA on Y_vector_global, return PC scores and fitted PCA object."""
    Y = np.stack(df['Y_vector_global'].values)
    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(Y)

    for i in range(n_components):
        logger.info(f"  PC{i+1} explained variance: {pca.explained_variance_ratio_[i]:.6f}")
    return pc_scores, pca


# ============================================================
# Wild Cluster Bootstrap (optimized, vectorized)
# ============================================================

def wild_cluster_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Wild Cluster Bootstrap with Rademacher weights.
    Vectorized inner loop for performance.

    Returns dict with: ols_coef, ols_se, ols_p, boot_se, boot_p, se_ratio
    """
    n, k = X.shape

    # OLS estimates
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ (X.T @ y)
    residuals = y - X @ beta_hat

    # OLS standard errors
    s2 = np.sum(residuals ** 2) / (n - k)
    ols_se = np.sqrt(np.diag(XtX_inv) * s2)
    ols_t = beta_hat / ols_se
    ols_p = 2 * (1 - stats.t.cdf(np.abs(ols_t), df=n - k))

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Cluster setup: map each observation to an integer cluster index
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    obs_cluster_idx = np.array([cluster_to_idx[c] for c in clusters])

    # Pre-compute X'X_inv @ X' for fast bootstrap
    XtX_inv_Xt = XtX_inv @ X.T  # (k, n)
    fitted = X @ beta_hat

    # Bootstrap
    rng = np.random.RandomState(seed)
    bootstrap_betas = np.zeros((n_bootstrap, k))

    for b in range(n_bootstrap):
        # Rademacher weights per cluster
        cluster_weights = rng.choice([-1.0, 1.0], size=n_clusters)
        # Vectorized: map cluster weights to observation weights
        obs_weights = cluster_weights[obs_cluster_idx]
        # Perturbed response
        y_star = fitted + obs_weights * residuals
        # Bootstrap OLS
        bootstrap_betas[b] = XtX_inv_Xt @ y_star

    # Bootstrap SE
    boot_se = np.std(bootstrap_betas, axis=0, ddof=1)

    # Bootstrap p-values: proportion of |β*-β̂| >= |β̂|
    centered = bootstrap_betas - beta_hat
    boot_p = np.mean(np.abs(centered) >= np.abs(beta_hat), axis=0)

    se_ratio = boot_se / np.where(ols_se > 0, ols_se, 1e-20)

    return {
        'ols_coef': beta_hat,
        'ols_se': ols_se,
        'ols_p': ols_p,
        'boot_se': boot_se,
        'boot_p': boot_p,
        'se_ratio': se_ratio,
        'r_squared': r_squared,
        'n_obs': n,
        'n_clusters': n_clusters
    }


# ============================================================
# Model Builders
# ============================================================

def build_main_model(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Model 1: Standard DID (no year controls).
    Y = b0 + b_UK·D_UK + b_AU·D_AU + b_post·Post + b_UK_post·(UK×Post) + b_AU_post·(AU×Post) + ε
    """
    D_UK = (df['country'] == 'UK').astype(float).values
    D_AU = (df['country'] == 'AU').astype(float).values
    post = df['post_aukus'].astype(float).values
    n = len(df)

    X = np.column_stack([
        np.ones(n), D_UK, D_AU, post,
        D_UK * post, D_AU * post
    ])
    names = ['intercept', 'UK', 'AU', 'post_aukus', 'UK_x_post', 'AU_x_post']
    return X, names


def build_year_fe_model(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Model 2: TWFE DID with year fixed effects.
    Y = b0 + b_UK·D_UK + b_AU·D_AU + Σγ_y·Year_y + b_UK_post·(UK×Post) + b_AU_post·(AU×Post) + ε

    post_aukus main effect dropped (absorbed by year FE).
    Reference year: 2021.
    """
    D_UK = (df['country'] == 'UK').astype(float).values
    D_AU = (df['country'] == 'AU').astype(float).values
    post = df['post_aukus'].astype(float).values
    n = len(df)

    # Year FE (drop 2021 as reference)
    years_present = sorted(df['year'].unique())
    ref_year = 2021
    year_dummies = []
    year_names = []
    for y in years_present:
        if y != ref_year:
            year_dummies.append((df['year'] == y).astype(float).values)
            year_names.append(f'year_{y}')

    X = np.column_stack(
        [np.ones(n), D_UK, D_AU] + year_dummies + [D_UK * post, D_AU * post]
    )
    names = ['intercept', 'UK', 'AU'] + year_names + ['UK_x_post', 'AU_x_post']
    return X, names


def build_time_trend_model(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Model 3: DID with linear time trend.
    Y = b0 + b_UK·D_UK + b_AU·D_AU + b_time·t + b_post·Post
        + b_UK_post·(UK×Post) + b_AU_post·(AU×Post) + ε

    time = (year - 2014), normalized to start at 0 for US data.
    """
    D_UK = (df['country'] == 'UK').astype(float).values
    D_AU = (df['country'] == 'AU').astype(float).values
    post = df['post_aukus'].astype(float).values
    time_var = (df['year'] - 2014).astype(float).values
    n = len(df)

    X = np.column_stack([
        np.ones(n), D_UK, D_AU, time_var, post,
        D_UK * post, D_AU * post
    ])
    names = ['intercept', 'UK', 'AU', 'time', 'post_aukus', 'UK_x_post', 'AU_x_post']
    return X, names


def build_placebo_model(
    df: pd.DataFrame,
    fake_cutoff_year: int
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Placebo test: use only pre-AUKUS data with a fake treatment date.
    fake_post = 1 if year >= fake_cutoff_year.
    """
    pre_df = df[df['post_aukus'] == 0].copy()
    pre_df['fake_post'] = (pre_df['year'] >= fake_cutoff_year).astype(float)

    D_UK = (pre_df['country'] == 'UK').astype(float).values
    D_AU = (pre_df['country'] == 'AU').astype(float).values
    fake_post = pre_df['fake_post'].values
    n = len(pre_df)

    X = np.column_stack([
        np.ones(n), D_UK, D_AU, fake_post,
        D_UK * fake_post, D_AU * fake_post
    ])
    names = ['intercept', 'UK', 'AU', 'fake_post', 'UK_x_fake_post', 'AU_x_fake_post']
    return X, names, pre_df


# ============================================================
# Proper Wald Test for Parallel Trends
# ============================================================

def proper_wald_test(X: np.ndarray, y: np.ndarray, test_indices: List[int]) -> Dict:
    """
    Proper Wald test using full OLS covariance matrix: W = β̂'V⁻¹β̂
    where V is the covariance sub-matrix of the tested coefficients.
    Also computes the simplified version W = Σ(β/SE)² for comparison.
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)
    residuals = y - X @ beta_hat
    s2 = np.sum(residuals ** 2) / (n - k)
    V_full = XtX_inv * s2

    beta_test = beta_hat[test_indices]
    V_test = V_full[np.ix_(test_indices, test_indices)]

    # Proper Wald: W = β'V⁻¹β
    V_test_inv = np.linalg.inv(V_test)
    wald_proper = float(beta_test @ V_test_inv @ beta_test)

    # Simplified Wald: W = Σ(β/SE)²
    se_test = np.sqrt(np.diag(V_test))
    wald_simplified = float(np.sum((beta_test / se_test) ** 2))

    df_test = len(test_indices)
    p_proper = float(1 - stats.chi2.cdf(wald_proper, df_test))
    p_simplified = float(1 - stats.chi2.cdf(wald_simplified, df_test))

    return {
        'wald_proper': wald_proper,
        'p_proper': p_proper,
        'wald_simplified': wald_simplified,
        'p_simplified': p_simplified,
        'df': df_test
    }


def cluster_robust_wald_test(
    X: np.ndarray, y: np.ndarray, clusters: np.ndarray,
    test_indices: List[int], n_bootstrap: int = 1000, seed: int = 42
) -> Dict:
    """
    Cluster-robust Wald test using Wild Cluster Bootstrap covariance.
    W = β̂'V_boot⁻¹β̂ where V_boot is the bootstrap covariance sub-matrix.
    Ensures inferential consistency with the main DID regressions that also
    use Wild Cluster Bootstrap for standard errors.
    """
    n, k = X.shape

    # OLS estimates
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)
    residuals = y - X @ beta_hat
    fitted = X @ beta_hat

    # Cluster setup
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    obs_cluster_idx = np.array([cluster_to_idx[c] for c in clusters])

    # Pre-compute for fast bootstrap
    XtX_inv_Xt = XtX_inv @ X.T

    # Bootstrap
    rng = np.random.RandomState(seed)
    bootstrap_betas = np.zeros((n_bootstrap, k))

    for b in range(n_bootstrap):
        cluster_weights = rng.choice([-1.0, 1.0], size=n_clusters)
        obs_weights = cluster_weights[obs_cluster_idx]
        y_star = fitted + obs_weights * residuals
        bootstrap_betas[b] = XtX_inv_Xt @ y_star

    # Bootstrap covariance matrix
    V_boot_full = np.cov(bootstrap_betas.T)  # (k, k)

    # Extract sub-matrix for tested coefficients
    beta_test = beta_hat[test_indices]
    V_boot_test = V_boot_full[np.ix_(test_indices, test_indices)]

    # Cluster-robust Wald: W = β'V_boot⁻¹β
    V_boot_test_inv = np.linalg.inv(V_boot_test)
    wald_cluster = float(beta_test @ V_boot_test_inv @ beta_test)

    # Also compute OLS Wald for comparison
    s2 = np.sum(residuals ** 2) / (n - k)
    V_ols_full = XtX_inv * s2
    V_ols_test = V_ols_full[np.ix_(test_indices, test_indices)]
    V_ols_test_inv = np.linalg.inv(V_ols_test)
    wald_ols = float(beta_test @ V_ols_test_inv @ beta_test)

    df_test = len(test_indices)
    p_cluster = float(1 - stats.chi2.cdf(wald_cluster, df_test))
    p_ols = float(1 - stats.chi2.cdf(wald_ols, df_test))

    return {
        'wald_cluster_robust': wald_cluster,
        'p_cluster_robust': p_cluster,
        'wald_ols': wald_ols,
        'p_ols': p_ols,
        'df': df_test,
        'n_clusters': n_clusters
    }


def run_event_study_with_proper_wald(
    df: pd.DataFrame,
    pc_scores: np.ndarray,
    base_year: int = 2021,
    min_year: int = 2017,
    max_year: int = 2024,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Event study design with cluster-robust Wald test for parallel trends.
    Matches the original parallel_trends_test.py specification:
    PC_i = b0 + b_UK·D_UK + b_AU·D_AU + Σγ_y·Year_y
               + Σ(δ_UK_y · D_UK · Year_y) + Σ(δ_AU_y · D_AU · Year_y) + ε

    Includes year main effects (dummies) to capture common time trends.
    Uses Wild Cluster Bootstrap covariance for Wald test, consistent with
    the main DID regression inference.
    """
    # Filter to study period
    mask = (df['year'] >= min_year) & (df['year'] <= max_year)
    df_f = df[mask]
    pc_f = pc_scores[mask.values]
    n = len(df_f)
    clusters = df_f['doc_id'].values

    D_UK = (df_f['country'] == 'UK').astype(float).values
    D_AU = (df_f['country'] == 'AU').astype(float).values

    study_years = sorted([y for y in df_f['year'].unique() if y != base_year])

    features = [np.ones(n), D_UK, D_AU]
    names = ['intercept', 'UK', 'AU']

    # Year main effects (captures common time trends for reference group US)
    for y in study_years:
        features.append((df_f['year'] == y).astype(float).values)
        names.append(f'year_{y}')

    # Country × Year interactions (event study coefficients)
    uk_year_indices = []
    au_year_indices = []

    for y in study_years:
        year_dummy = (df_f['year'] == y).astype(float).values
        features.append(D_UK * year_dummy)
        names.append(f'UK_x_{y}')
        uk_year_indices.append(len(names) - 1)

        features.append(D_AU * year_dummy)
        names.append(f'AU_x_{y}')
        au_year_indices.append(len(names) - 1)

    X = np.column_stack(features)

    # Pre-AUKUS year indices (years < 2021)
    pre_years = [y for y in study_years if y < base_year]
    uk_pre_idx = [uk_year_indices[study_years.index(y)] for y in pre_years]
    au_pre_idx = [au_year_indices[study_years.index(y)] for y in pre_years]

    results = {'n_obs': n, 'study_years': study_years, 'base_year': base_year}
    for pc_i in range(3):
        pc_name = f'PC{pc_i + 1}'
        y = pc_f[:, pc_i]

        uk_wald = cluster_robust_wald_test(X, y, clusters, uk_pre_idx, n_bootstrap, seed)
        au_wald = cluster_robust_wald_test(X, y, clusters, au_pre_idx, n_bootstrap, seed)

        # Also get individual coefficients for the event study plot
        XtX_inv = np.linalg.pinv(X.T @ X)  # pinv for numerical stability with many dummies
        beta_hat = XtX_inv @ (X.T @ y)
        residuals = y - X @ beta_hat
        s2 = np.sum(residuals ** 2) / (n - X.shape[1])
        se = np.sqrt(np.diag(XtX_inv * s2))

        coefs = {}
        for idx, name in enumerate(names):
            coefs[name] = {
                'coef': float(beta_hat[idx]),
                'se': float(se[idx]),
                'p': float(2 * (1 - stats.t.cdf(abs(beta_hat[idx] / se[idx]), df=n - X.shape[1])))
            }

        results[pc_name] = {
            'coefficients': coefs,
            'UK_parallel_trends': {
                'cluster_robust_wald': uk_wald['wald_cluster_robust'],
                'cluster_robust_p': uk_wald['p_cluster_robust'],
                'ols_wald': uk_wald['wald_ols'],
                'ols_p': uk_wald['p_ols'],
                'df': uk_wald['df'],
                'n_clusters': uk_wald['n_clusters'],
                'pre_years': pre_years,
                'conclusion': 'satisfied' if uk_wald['p_cluster_robust'] > 0.05 else 'violated'
            },
            'AU_parallel_trends': {
                'cluster_robust_wald': au_wald['wald_cluster_robust'],
                'cluster_robust_p': au_wald['p_cluster_robust'],
                'ols_wald': au_wald['wald_ols'],
                'ols_p': au_wald['p_ols'],
                'df': au_wald['df'],
                'n_clusters': au_wald['n_clusters'],
                'pre_years': pre_years,
                'conclusion': 'satisfied' if au_wald['p_cluster_robust'] > 0.05 else 'violated'
            }
        }

    return results


# ============================================================
# Run Model with Bootstrap
# ============================================================

def run_model_bootstrap(
    X: np.ndarray,
    pc_scores: np.ndarray,
    clusters: np.ndarray,
    feature_names: List[str],
    model_name: str,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict:
    """Run PC regression with Wild Cluster Bootstrap for all 3 PCs."""
    results = {'model_name': model_name, 'feature_names': feature_names}
    t_start = time.time()

    for pc_i in range(3):
        pc_name = f'PC{pc_i + 1}'
        y = pc_scores[:, pc_i]

        logger.info(f"  {model_name} - {pc_name}: bootstrapping ({n_bootstrap} iter)...")
        t0 = time.time()
        boot = wild_cluster_bootstrap(X, y, clusters, n_bootstrap, seed + pc_i)
        elapsed = time.time() - t0
        logger.info(f"    Done in {elapsed:.1f}s")

        pc_result = {}
        for i, name in enumerate(feature_names):
            pc_result[name] = {
                'coef': float(boot['ols_coef'][i]),
                'ols_se': float(boot['ols_se'][i]),
                'ols_p': float(boot['ols_p'][i]),
                'boot_se': float(boot['boot_se'][i]),
                'boot_p': float(boot['boot_p'][i]),
                'se_ratio': float(boot['se_ratio'][i])
            }

        pc_result['_meta'] = {
            'n_obs': int(boot['n_obs']),
            'n_clusters': int(boot['n_clusters']),
            'r_squared': float(boot['r_squared'])
        }
        results[pc_name] = pc_result

    total = time.time() - t_start
    logger.info(f"  {model_name} total: {total:.1f}s")
    return results


# ============================================================
# Significance star helper
# ============================================================

def sig_star(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '.'
    return ''


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("Comprehensive DID Robustness Analysis")
    logger.info("=" * 60)

    # Load data & PCA
    df = load_data()
    pc_scores, pca = prepare_pca(df)
    clusters = df['doc_id'].values

    all_results = {
        'description': 'Comprehensive DID Robustness Analysis',
        'date': '2026-02-06',
        'n_total': len(df),
        'n_clusters': int(df['doc_id'].nunique()),
        'pca_variance': {
            f'PC{i+1}': float(pca.explained_variance_ratio_[i])
            for i in range(3)
        }
    }

    N_BOOT = 1000
    SEED = 42

    # ========== Model 1: Main DID (baseline) ==========
    logger.info("\n" + "=" * 60)
    logger.info("Model 1: Main DID (standard specification, no year controls)")
    logger.info("=" * 60)

    X1, n1 = build_main_model(df)
    r1 = run_model_bootstrap(X1, pc_scores, clusters, n1, "Main_DID", N_BOOT, SEED)
    all_results['model_1_main'] = r1

    # ========== Model 2: Year FE ==========
    logger.info("\n" + "=" * 60)
    logger.info("Model 2: DID with Year Fixed Effects (TWFE)")
    logger.info("=" * 60)

    X2, n2 = build_year_fe_model(df)
    r2 = run_model_bootstrap(X2, pc_scores, clusters, n2, "Year_FE_DID", N_BOOT, SEED)
    all_results['model_2_year_fe'] = r2

    # ========== Model 3: Linear Time Trend ==========
    logger.info("\n" + "=" * 60)
    logger.info("Model 3: DID with Linear Time Trend")
    logger.info("=" * 60)

    X3, n3 = build_time_trend_model(df)
    r3 = run_model_bootstrap(X3, pc_scores, clusters, n3, "Time_Trend_DID", N_BOOT, SEED)
    all_results['model_3_time_trend'] = r3

    # ========== Model 4a: Placebo (fake treatment at 2020) ==========
    logger.info("\n" + "=" * 60)
    logger.info("Model 4a: Placebo Test (fake treatment: year >= 2020)")
    logger.info("=" * 60)

    pre_mask = (df['post_aukus'] == 0).values
    pc_pre = pc_scores[pre_mask]

    X4a, n4a, pre_df_a = build_placebo_model(df, 2020)
    cl4a = pre_df_a['doc_id'].values
    r4a = run_model_bootstrap(X4a, pc_pre, cl4a, n4a, "Placebo_2020", N_BOOT, SEED)
    all_results['model_4a_placebo_2020'] = r4a

    # ========== Model 4b: Placebo (fake treatment at 2021) ==========
    logger.info("\n" + "=" * 60)
    logger.info("Model 4b: Placebo Test (fake treatment: year >= 2021, pre-AUKUS only)")
    logger.info("=" * 60)

    X4b, n4b, pre_df_b = build_placebo_model(df, 2021)
    cl4b = pre_df_b['doc_id'].values
    r4b = run_model_bootstrap(X4b, pc_pre, cl4b, n4b, "Placebo_2021", N_BOOT, SEED)
    all_results['model_4b_placebo_2021'] = r4b

    # ========== Model 5: Restricted Sample (2017+) ==========
    logger.info("\n" + "=" * 60)
    logger.info("Model 5: Restricted Sample (2017+, all countries present)")
    logger.info("=" * 60)

    mask_2017 = (df['year'] >= 2017).values
    df_r = df[mask_2017]
    pc_r = pc_scores[mask_2017]
    X5, n5 = build_main_model(df_r)
    cl5 = df_r['doc_id'].values
    r5 = run_model_bootstrap(X5, pc_r, cl5, n5, "Restricted_2017", N_BOOT, SEED)
    all_results['model_5_restricted'] = r5

    # ========== Model 6: Proper Wald Test (Parallel Trends) ==========
    logger.info("\n" + "=" * 60)
    logger.info("Model 6: Event Study with Proper Wald Test")
    logger.info("=" * 60)

    wald_results = run_event_study_with_proper_wald(df, pc_scores)
    all_results['model_6_proper_wald'] = wald_results

    # ============================================================
    # Summary
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY: Key DID Coefficients Across Models")
    logger.info("=" * 70)

    # Compare UK_x_post and AU_x_post across models 1,2,3,5
    models_compare = [
        ('Main DID',       r1, 'UK_x_post', 'AU_x_post'),
        ('Year FE',        r2, 'UK_x_post', 'AU_x_post'),
        ('Time Trend',     r3, 'UK_x_post', 'AU_x_post'),
        ('Restricted 2017', r5, 'UK_x_post', 'AU_x_post'),
    ]

    for pc in ['PC1', 'PC2', 'PC3']:
        logger.info(f"\n--- {pc} ---")
        logger.info(f"{'Model':<18} {'UK×Post coef':>13} {'boot_p':>8} {'AU×Post coef':>13} {'boot_p':>8}")
        for label, res, uk_var, au_var in models_compare:
            uk_c = res[pc][uk_var]['coef']
            uk_p = res[pc][uk_var]['boot_p']
            au_c = res[pc][au_var]['coef']
            au_p = res[pc][au_var]['boot_p']
            logger.info(
                f"{label:<18} {uk_c:>+13.6f} {uk_p:>7.4f}{sig_star(uk_p):<3} "
                f"{au_c:>+13.6f} {au_p:>7.4f}{sig_star(au_p):<3}"
            )

    # Placebo results
    logger.info(f"\n--- Placebo Tests ---")
    logger.info(f"{'Model':<18} {'UK×FakePost':>13} {'boot_p':>8} {'AU×FakePost':>13} {'boot_p':>8}")
    for pc in ['PC1', 'PC2', 'PC3']:
        for label, res in [('Placebo 2020', r4a), ('Placebo 2021', r4b)]:
            uk_c = res[pc]['UK_x_fake_post']['coef']
            uk_p = res[pc]['UK_x_fake_post']['boot_p']
            au_c = res[pc]['AU_x_fake_post']['coef']
            au_p = res[pc]['AU_x_fake_post']['boot_p']
            logger.info(
                f"  {pc} {label:<14} {uk_c:>+13.6f} {uk_p:>7.4f}{sig_star(uk_p):<3} "
                f"{au_c:>+13.6f} {au_p:>7.4f}{sig_star(au_p):<3}"
            )

    # Cluster-Robust Wald
    logger.info(f"\n--- Parallel Trends: Cluster-Robust vs OLS Wald ---")
    logger.info(f"{'PC':<5} {'Country':<5} {'ClusterW':>10} {'p':>8} {'OLS W':>10} {'p':>8} {'Conclusion':<10}")
    for pc in ['PC1', 'PC2', 'PC3']:
        for country in ['UK', 'AU']:
            key = f'{country}_parallel_trends'
            d = wald_results[pc][key]
            logger.info(
                f"{pc:<5} {country:<5} {d['cluster_robust_wald']:>10.2f} {d['cluster_robust_p']:>8.4f} "
                f"{d['ols_wald']:>10.2f} {d['ols_p']:>8.4f} {d['conclusion']:<10}"
            )

    # Build summary dict
    summary = {'comparison': {}, 'placebo': {}, 'wald': {}}
    for pc in ['PC1', 'PC2', 'PC3']:
        summary['comparison'][pc] = {}
        for var in ['UK_x_post', 'AU_x_post']:
            summary['comparison'][pc][var] = {
                'main': {'coef': r1[pc][var]['coef'], 'boot_p': r1[pc][var]['boot_p']},
                'year_fe': {'coef': r2[pc][var]['coef'], 'boot_p': r2[pc][var]['boot_p']},
                'time_trend': {'coef': r3[pc][var]['coef'], 'boot_p': r3[pc][var]['boot_p']},
                'restricted': {'coef': r5[pc][var]['coef'], 'boot_p': r5[pc][var]['boot_p']},
            }
        summary['placebo'][pc] = {
            'placebo_2020': {
                'UK_coef': r4a[pc]['UK_x_fake_post']['coef'],
                'UK_p': r4a[pc]['UK_x_fake_post']['boot_p'],
                'AU_coef': r4a[pc]['AU_x_fake_post']['coef'],
                'AU_p': r4a[pc]['AU_x_fake_post']['boot_p'],
            },
            'placebo_2021': {
                'UK_coef': r4b[pc]['UK_x_fake_post']['coef'],
                'UK_p': r4b[pc]['UK_x_fake_post']['boot_p'],
                'AU_coef': r4b[pc]['AU_x_fake_post']['coef'],
                'AU_p': r4b[pc]['AU_x_fake_post']['boot_p'],
            }
        }
        summary['wald'][pc] = {}
        for country in ['UK', 'AU']:
            key = f'{country}_parallel_trends'
            summary['wald'][pc][country] = {
                'cluster_robust_wald': wald_results[pc][key]['cluster_robust_wald'],
                'cluster_robust_p': wald_results[pc][key]['cluster_robust_p'],
                'ols_wald': wald_results[pc][key]['ols_wald'],
                'ols_p': wald_results[pc][key]['ols_p'],
                'conclusion': wald_results[pc][key]['conclusion']
            }

    all_results['summary'] = summary

    # ============================================================
    # Save
    # ============================================================
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    output_path = OUTPUT_DIR / 'did_robustness_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    logger.info(f"\nAll results saved to {output_path}")

    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
