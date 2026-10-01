"""The analysis chain of the paper on one matrix of semantic vectors Y (N x 768, float64).

Every function takes Y and the occurrence metadata (country, year, month, post_aukus, doc_id, in row order) and returns
a JSON-ready dictionary:

* H1: MANOVA (Wilks' lambda, Rao's F approximation; Rencher and Christensen, eq. 6.10) on principal-component scores,
  with the period split and the robustness settings;
* H2: the subgroup difference-in-differences with a linear time trend, inference by the wild cluster bootstrap
  (Cameron, Gelbach and Miller 2008; Rademacher weights by document; unrestricted residuals; centred tail p value;
  one generator per component, seeds from :mod:`replication.seeds`); the 95% interval
  ``b +/- quantile(|b* - b|, 0.95, method='linear')`` (not multiplicity-adjusted);
* robustness models (year fixed effects with and without the Post main effect, a linear time trend, 2017+) and the
  placebo models with fake cut-off years on the pre-signing rows;
* the parallel-trends event study (US and UK, 2014-2024, base year 2021) with the joint Wald test of the pre-signing
  UK x year terms under the full bootstrap covariance;
* H3: nearest GPT-2 vocabulary words of each country's mean vector, before and after the signing;
* Euclidean distances between the country mean vectors; the orientation of an own PC1 against the reference PC1.

The bootstrap loops over clusters are vectorised (``weights[cluster_index]``); the random draws, their order and the
arithmetic follow the original implementation. Test outcomes are reported as rejection or non-rejection at the stated
level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import seeds
from .params import params as setting
from .pca import fit_pca, k_for_variance

COUNTRIES = ('US', 'UK', 'AU')


# =========================================================================== H1 MANOVA
def compute_wilks_lambda_f(Y, labels):
    """Wilks' lambda with Rao's F approximation (Rencher & Christensen eq 6.10)."""
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
    try:
        sign_W, logdet_W = np.linalg.slogdet(W)
        sign_T, logdet_T = np.linalg.slogdet(W + B)
        if sign_W <= 0 or sign_T <= 0:
            return np.nan, np.nan, {}
        lambda_wilks = np.exp(logdet_W - logdet_T)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, {}
    p, k, n = n_vars, n_groups, n_total
    if (p ** 2 + (k - 1) ** 2 - 5) > 0:
        t = np.sqrt((p ** 2 * (k - 1) ** 2 - 4) / (p ** 2 + (k - 1) ** 2 - 5))
    else:
        t = 1
    df1 = p * (k - 1)
    df2 = t * (n - 1 - (p + k) / 2) - (p * (k - 1) - 2) / 2
    if t > 0 and df2 > 0:
        lambda_t = lambda_wilks ** (1 / t)
        F = ((1 - lambda_t) / lambda_t) * (df2 / df1)
        p_value = stats.f.sf(F, df1, df2)
    else:
        F, p_value = np.nan, np.nan
    return F, p_value, {'wilks_lambda': float(lambda_wilks), 'df1': float(df1), 'df2': float(df2),
                        'n_groups': int(n_groups), 'n_samples': int(n_total), 'n_vars': int(n_vars),
                        'partial_eta_sq': float(1 - lambda_wilks)}


def residualize_year_fe(Y, years):
    """Residuals of Y on year fixed effects (intercept plus one indicator per year after the first)."""
    unique_years = sorted(np.unique(years))
    n = len(years)
    X = np.ones((n, 1))
    for y in unique_years[1:]:
        X = np.column_stack([X, (years == y).astype(float)])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)
    return Y - X @ beta


MANOVA_NULL = '各组均值相等的联合零假设（经典 MANOVA，假设观测独立）'


def _finite(p) -> bool:
    return p is not None and bool(np.isfinite(p))


def h1_outcome(p, alpha: float = 0.05) -> dict:
    """Label of a MANOVA outcome at the levels .001 and .05. Wilks / Rao F assumes independent
    observations, which repeated occurrences within documents and years do not satisfy, so a rejection is not evidence
    robust to that dependence."""
    if not _finite(p):
        return {'h1_joint_null_rejected': None, 'label': f'p 不可用：未检验{MANOVA_NULL}'}
    if p < alpha:
        return {'h1_joint_null_rejected': True, 'label': f'拒绝{MANOVA_NULL}，{"p < .001" if p < 0.001 else "p < .05"}'}
    return {'h1_joint_null_rejected': False, 'label': f'未拒绝{MANOVA_NULL}，p ≥ .05'}


def manova_period_split(Y, meta: pd.DataFrame, n_pc: int | None = None, pca=None) -> dict:
    """The period-split MANOVA: tests 1-6 with the labels and structure of the original analysis."""
    n_pc = setting('manova')['n_pc'] if n_pc is None else n_pc
    country = meta['country'].to_numpy()
    years = meta['year'].to_numpy()
    post = meta['post_aukus'].to_numpy().astype(bool)
    pca = pca or fit_pca(Y, n_pc)
    S = pca.transform(Y)
    results = []
    F, p, det = compute_wilks_lambda_f(S, country)
    results.append({'test': '1. 全样本（原始）', 'n': int(len(S)), 'F': float(F), 'p': float(p),
                    **h1_outcome(p), 'wilks_lambda': det.get('wilks_lambda')})
    F_r, p_r, det_r = compute_wilks_lambda_f(residualize_year_fe(S, years), country)
    results.append({'test': '1b. 全样本（时间残差）', 'n': int(len(S)), 'F': float(F_r), 'p': float(p_r),
                    **h1_outcome(p_r), 'wilks_lambda': det_r.get('wilks_lambda')})

    def split(mask, label, countries=('US', 'UK', 'AU'), residual=True, residual_label=None):
        Ys, cs, ys = S[mask], country[mask], years[mask]
        F_, p_, det_ = compute_wilks_lambda_f(Ys, cs)
        results.append({'test': label, 'n': int(len(Ys)), 'F': float(F_), 'p': float(p_),
                        'n_per_country': {c: int((cs == c).sum()) for c in countries},
                        **h1_outcome(p_), 'wilks_lambda': det_.get('wilks_lambda'),
                        'df1': det_.get('df1'), 'df2': det_.get('df2')})
        if residual:
            F2, p2, det2 = compute_wilks_lambda_f(residualize_year_fe(Ys, ys), cs)
            results.append({'test': residual_label, 'n': int(len(Ys)), 'F': float(F2), 'p': float(p2),
                            **h1_outcome(p2), 'wilks_lambda': det2.get('wilks_lambda')})

    split(~post, '2. Pre-AUKUS（原始）', residual_label='2b. Pre-AUKUS（时间残差）')
    split(post, '3. Post-AUKUS（原始）', residual_label='3b. Post-AUKUS（时间残差）')
    split((~post) & (years >= 2017), '4. Pre-AUKUS 2017+（原始）', residual=False)
    usuk = (country == 'US') | (country == 'UK')
    split((~post) & usuk, '5. Pre-AUKUS US vs UK', countries=('US', 'UK'), residual=False)
    t6 = setting('manova')['test6']
    split((~post) & usuk & (years >= t6['min_year']), '6. Pre-AUKUS US vs UK（2014+重叠）', countries=('US', 'UK'),
          residual=False)
    return {'method': 'MANOVA Period Split Analysis',
            'description': '分期MANOVA：检验Pre/Post-AUKUS差异是否均显著，以及F值变化',
            'outcome_rule': ('h1_joint_null_rejected = p < .05（经典 Wilks/Rao F，假设观测独立；同一文档与年份内的重复'
                             '出现点不满足该假设，拒绝不代表对这种依赖稳健）'),
            'n_pc': int(n_pc), 'aukus_date': '2021-09 (month-level coding)', 'results': results,
            'interpretation': {
                'pre_aukus_significant': results[2]['p'] < 0.05, 'post_aukus_significant': results[4]['p'] < 0.05,
                'f_ratio_post_over_pre': float(results[4]['F'] / results[2]['F']) if results[2]['F'] > 0 else None,
                'f_ratio_post_over_pre_residualized': (float(results[5]['F'] / results[3]['F'])
                                                       if results[3]['F'] > 0 else None)},
            '_details': {'pca_solver': pca.svd_solver, 'cumulative_variance': float(pca.explained_variance_ratio_.sum())}}


def h1_summary(Y, meta, pca88=None) -> dict:
    """Test 6 at k=88 and at the k99 rule (k from a PCA fitted to all components needed)."""
    pca88 = pca88 or fit_pca(Y, 88)
    base = manova_period_split(Y, meta, 88, pca88)
    full = fit_pca(Y, min(Y.shape))
    k99 = k_for_variance(full.explained_variance_ratio_)
    k99_result = manova_period_split(Y, meta, k99, fit_pca(Y, k99)) if k99 != 88 else base
    t6 = base['results'][8]
    return {'k88': base, 'k99': k99, 'test6_k88': t6, 'test6_k99': k99_result['results'][8],
            'cumulative_variance_88': float(pca88.explained_variance_ratio_.sum())}


def manova_pc_robustness(Y, meta, n_pcs=(3, 5, 10, 20, 30, 50, 70, 88, 99)) -> dict:
    """Three-country MANOVA on PCA-k fitted on all rows, k in ``n_pcs``."""
    country = meta['country'].to_numpy()
    out = []
    for k in n_pcs:
        pca = fit_pca(Y, k)
        F, p, _ = compute_wilks_lambda_f(pca.transform(Y), country)
        out.append({'n_pc': int(k), 'var_explained': float(pca.explained_variance_ratio_.sum() * 100),
                    'F': float(F), 'p': float(p)})
    return {'method': 'MANOVA Robustness Analysis', 'results': out}


def h1_test6_pc_robustness(Y, meta, n_pcs=(3, 5, 10, 20, 30, 50, 70, 88, 99)) -> dict:
    """Principal-component robustness of H1 on its own sample: test 6 (pre-signing US / UK rows from 2014) on the
    scores of a PCA-k fitted on all rows (as in the main MANOVA), Wilks' lambda with Rao's F, k in ``n_pcs``.
    (:func:`manova_pc_robustness` is the three-country full-sample MANOVA over the same grid.)"""
    t6 = setting('manova')['test6']
    country = meta['country'].to_numpy()
    mask = ((~meta['post_aukus'].to_numpy().astype(bool)) & np.isin(country, t6['countries'])
            & (meta['year'].to_numpy() >= t6['min_year']))
    results = []
    for k in n_pcs:
        pca = fit_pca(Y, k)
        F, p, det = compute_wilks_lambda_f(pca.transform(Y)[mask], country[mask])
        results.append({'n_pc': int(k), 'var_explained': float(pca.explained_variance_ratio_.sum() * 100),
                        'n': int(mask.sum()), 'F': float(F), 'p': float(p), 'wilks_lambda': det.get('wilks_lambda'),
                        'df1': det.get('df1'), 'df2': det.get('df2')})
    return {'sample': 'pre-signing US / UK rows from 2014 (H1 test 6)', 'results': results,
            'F_min': min(r['F'] for r in results), 'F_max': max(r['F'] for r in results),
            'p_max': max(r['p'] for r in results)}


def manova_time_robustness(Y, meta, pca88=None) -> dict:
    """Three-country MANOVA on PCA-88: full sample and 2017+, raw and year-residualised."""
    pca88 = pca88 or fit_pca(Y, 88)
    S = pca88.transform(Y)
    country, years = meta['country'].to_numpy(), meta['year'].to_numpy()
    m17 = years >= 2017
    rows = []
    for label, Ys, cs, ys, resid in (('1. 全样本（基线）', S, country, years, False),
                                     ('2. 受限样本（2017+）', S[m17], country[m17], years[m17], False),
                                     ('3. 时间残差（全样本）', S, country, years, True),
                                     ('4. 时间残差（2017+）', S[m17], country[m17], years[m17], True)):
        F, p, _ = compute_wilks_lambda_f(residualize_year_fe(Ys, ys) if resid else Ys, cs)
        rows.append({'test': label, 'n': int(len(Ys)), 'F': float(F), 'p': float(p)})
    return {'method': 'MANOVA Time Robustness Check', 'n_pc': 88, 'results': rows}


# =========================================================================== bootstrap core
def cluster_index(clusters):
    """Sorted unique clusters (numpy.unique) and each row's position among them."""
    unique, inverse = np.unique(clusters, return_inverse=True)
    return unique, inverse.astype(np.int64)


def wcb_interval(beta, bootstrap_betas, level=None, method=None):
    """b +/- Q_level(|b* - b|) with numpy.quantile(method='linear'); not multiplicity-adjusted."""
    params = setting('wcb')
    level = params['interval_level'] if level is None else level
    method = params['quantile_method'] if method is None else method
    half = np.quantile(np.abs(bootstrap_betas - beta[np.newaxis, :]), level, axis=0, method=method)
    return beta - half, beta + half, half


def wcb_v1_main_bootstrap(X, y, clusters, n_bootstrap=seeds.WCB_DRAWS, seed=seeds.WCB_SEED_BASE) -> dict:
    """Wild cluster bootstrap of the main model (Rademacher weights per document, unrestricted residuals; the
    ``RandomState`` draws of the original implementation)."""
    rs = np.random.RandomState(seed)  # == np.random.seed(seed) + np.random.choice (global RNG of the original)
    n_samples, n_params = X.shape
    unique, inverse = cluster_index(clusters)
    n_clusters = len(unique)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    residuals = y - X @ beta
    sigma2 = np.sum(residuals ** 2) / (n_samples - n_params)
    V_ols = sigma2 * XtX_inv
    se_ols = np.sqrt(np.diag(V_ols))
    t_stat = beta / se_ols
    p_ols = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_samples - n_params))
    proj = XtX_inv @ X.T
    fitted = X @ beta
    boot = np.zeros((n_bootstrap, n_params))
    for b in range(n_bootstrap):
        weights = rs.choice([-1, 1], size=n_clusters)
        sample_weights = np.empty(n_samples)
        sample_weights[:] = weights[inverse]
        boot[b] = proj @ (fitted + residuals * sample_weights)
    se_boot = np.std(boot, axis=0, ddof=1)
    p_boot = np.mean(np.abs(boot - beta[np.newaxis, :]) >= np.abs(beta[np.newaxis, :]), axis=0)
    lo, hi, half = wcb_interval(beta, boot)
    return {'beta': beta, 'se_ols': se_ols, 't_stat_ols': t_stat, 'p_val_ols': p_ols, 'se_bootstrap': se_boot,
            'p_val_bootstrap': p_boot, 'n_bootstrap': n_bootstrap, 'n_clusters': n_clusters, 'n_samples': n_samples,
            'n_params': n_params, 'bootstrap_betas': boot, 'ci_low': lo, 'ci_high': hi, 'ci_halfwidth': half}


def wcb_robustness_bootstrap(X, y, clusters, n_bootstrap=seeds.WCB_DRAWS, seed=seeds.WCB_SEED_BASE) -> dict:
    """Wild cluster bootstrap of the robustness models (normal equations with ``inv``; ``RandomState`` draws)."""
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ (X.T @ y)
    residuals = y - X @ beta_hat
    s2 = np.sum(residuals ** 2) / (n - k)
    ols_se = np.sqrt(np.diag(XtX_inv) * s2)
    ols_t = beta_hat / ols_se
    ols_p = 2 * (1 - stats.t.cdf(np.abs(ols_t), df=n - k))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    unique, obs_cluster_idx = cluster_index(clusters)
    n_clusters = len(unique)
    XtX_inv_Xt = XtX_inv @ X.T
    fitted = X @ beta_hat
    rng = np.random.RandomState(seed)
    boot = np.zeros((n_bootstrap, k))
    for b in range(n_bootstrap):
        cluster_weights = rng.choice([-1.0, 1.0], size=n_clusters)
        obs_weights = cluster_weights[obs_cluster_idx]
        boot[b] = XtX_inv_Xt @ (fitted + obs_weights * residuals)
    boot_se = np.std(boot, axis=0, ddof=1)
    boot_p = np.mean(np.abs(boot - beta_hat) >= np.abs(beta_hat), axis=0)
    se_ratio = boot_se / np.where(ols_se > 0, ols_se, 1e-20)
    lo, hi, half = wcb_interval(beta_hat, boot)
    return {'ols_coef': beta_hat, 'ols_se': ols_se, 'ols_p': ols_p, 'boot_se': boot_se, 'boot_p': boot_p,
            'se_ratio': se_ratio, 'r_squared': r_squared, 'n_obs': n, 'n_clusters': n_clusters,
            'bootstrap_betas': boot, 'ci_low': lo, 'ci_high': hi}


# =========================================================================== H2 main (WCB)
def wcb_features(meta: pd.DataFrame):
    """The main H2 design: intercept, UK, AU, linear time trend, Post, UK x Post, AU x Post."""
    uk = (meta['country'] == 'UK').astype(float).to_numpy()
    au = (meta['country'] == 'AU').astype(float).to_numpy()
    time_var = (meta['year'] - setting('wcb')['trend_origin_year']).astype(float).to_numpy()
    post = meta['post_aukus'].astype(float).to_numpy()
    X = np.column_stack([uk, au, time_var, post, uk * post, au * post])
    names = ['UK', 'AU', 'time', 'post_aukus', 'UK_x_post', 'AU_x_post']
    return np.column_stack([np.ones(len(meta)), X]), ['intercept'] + names


def h2_wcb_main(pc_scores3, meta, explained_variance3) -> dict:
    """The main H2 model with the wild cluster bootstrap, on precomputed PCA-3 scores."""
    params = setting('wcb')
    X, names = wcb_features(meta)
    clusters = meta['doc_id'].to_numpy()
    all_results = {}
    for pc_idx in range(3):
        all_results[f'PC{pc_idx + 1}'] = wcb_v1_main_bootstrap(X, pc_scores3[:, pc_idx], clusters,
                                                               params['n_bootstrap'], params['seed_base'] + pc_idx)
    n_samples = X.shape[0]
    output = {'method': 'Wild Cluster Bootstrap', 'description': 'OLS vs Wild Cluster Bootstrap标准误和p值对比',
              'n_bootstrap': params['n_bootstrap'], 'seed_base': params['seed_base'], 'n_samples': int(n_samples),
              'n_clusters': int(all_results['PC1']['n_clusters']),
              'avg_cluster_size': float(n_samples / all_results['PC1']['n_clusters']),
              'pca_explained_variance': {f'PC{i + 1}': float(v) for i, v in enumerate(explained_variance3)},
              'comparison': {}}
    intervals = {}
    for pc, r in all_results.items():
        output['comparison'][pc] = {name: {
            'coef': float(r['beta'][i]), 'se_ols': float(r['se_ols'][i]), 'se_bootstrap': float(r['se_bootstrap'][i]),
            'se_ratio': float(r['se_bootstrap'][i] / r['se_ols'][i]) if r['se_ols'][i] > 0 else None,
            'p_ols': float(r['p_val_ols'][i]), 'p_bootstrap': float(r['p_val_bootstrap'][i]),
            't_stat_ols': float(r['t_stat_ols'][i])} for i, name in enumerate(names)}
        intervals[pc] = {name: {'coef': float(r['beta'][i]), 'ci95_low': float(r['ci_low'][i]),
                                'ci95_high': float(r['ci_high'][i]), 'halfwidth': float(r['ci_halfwidth'][i])}
                         for i, name in enumerate(names)}
    ratios, changes = [], []
    for pc, r in all_results.items():
        for i, name in enumerate(names):
            if name == 'intercept':
                continue
            ratios.append(r['se_bootstrap'][i] / r['se_ols'][i])
            ols_sig, boot_sig = r['p_val_ols'][i] < 0.05, r['p_val_bootstrap'][i] < 0.05
            if ols_sig != boot_sig:
                changes.append({'variable': name, 'pc': pc, 'ols_p': float(r['p_val_ols'][i]),
                                'bootstrap_p': float(r['p_val_bootstrap'][i]), 'ols_significant': bool(ols_sig),
                                'bootstrap_significant': bool(boot_sig)})
    output['summary'] = {'se_ratio_mean': float(np.mean(ratios)), 'se_ratio_median': float(np.median(ratios)),
                         'se_ratio_min': float(np.min(ratios)), 'se_ratio_max': float(np.max(ratios)),
                         'n_conclusion_changes': len(changes), 'conclusion_changes': changes,
                         'overall_assessment': ('全部非截距系数的 OLS p 与 bootstrap p 在 .05 水平上的拒绝/未拒绝相同'
                                                if not changes else
                                                f'{len(changes)} 个非截距系数的 OLS p 与 bootstrap p 在 .05 水平上的拒绝/未拒绝不同')}
    output['_details'] = {'interval': params['interval'], 'intervals_95': intervals,
                     'seeds': [params['seed_base'] + i for i in range(3)],
                     'cluster_order': 'numpy.unique(doc_id)'}
    return output


# =========================================================================== descriptive re-run (alternative encoders)
def wcb_uk_post_rerun(pc1_scores, meta, n_bootstrap: int | None = None, seed: int | None = None) -> dict:
    """The descriptive re-run of the PC1 UK x Post WCB (the main specification of :func:`h2_wcb_main`, same
    features, clusters and centred tail rule) with its own seed and B; p = (k + 1) / (B + 1), k = #{|b* - b| >= |b|}.
    Descriptive only: it never replaces the primary p (B = 1000, seeds 42/43/44)."""
    params = setting('cross_encoder_reporting')
    B = params['rerun_B'] if n_bootstrap is None else n_bootstrap
    seed = params['rerun_seed'] if seed is None else seed
    X, names = wcb_features(meta)
    r = wcb_v1_main_bootstrap(X, np.asarray(pc1_scores, dtype=np.float64), meta['doc_id'].to_numpy(), B, seed)
    j = names.index('UK_x_post')
    k = int(np.sum(np.abs(r['bootstrap_betas'][:, j] - r['beta'][j]) >= np.abs(r['beta'][j])))
    return {'coef': float(r['beta'][j]), 'k': k, 'B': int(B), 'seed': int(seed), 'p': (k + 1) / (B + 1),
            'se_bootstrap': float(r['se_bootstrap'][j]), 'rule': '(k + 1) / (B + 1); descriptive, never the primary p'}


def pre_signing_sd(pc1_scores, meta) -> dict:
    """The SD (ddof = 1) of the PC1 scores of the pre-signing rows (post_aukus = 0)."""
    pre = (meta['post_aukus'].to_numpy().astype(int) == 0)
    values = np.asarray(pc1_scores, dtype=np.float64)[pre]
    return {'sd': float(np.std(values, ddof=1)), 'rows': int(pre.sum()), 'ddof': 1}


def rerun_descriptive(own_pc1, same_axis_pc1, meta) -> dict:
    """Both family members of one analysis (own-PCA PC1 and reference-axis PC1): the descriptive re-run and the
    pre-signing SD of the scores. Computed on every E1-E4 analysis, unconditionally, alongside the primary WCB."""
    return {'rule': 'descriptive re-run (every alternative encoder, unconditional) and the pre-signing PC1 SD '
                    '(ddof = 1) on the same axis; neither changes the primary readings',
            'rerun': {'same_procedure_PC1': wcb_uk_post_rerun(own_pc1, meta),
                      'same_axis_PC1': wcb_uk_post_rerun(same_axis_pc1, meta)},
            'pre_signing_sd': {'same_procedure_PC1': pre_signing_sd(own_pc1, meta),
                               'same_axis_PC1': pre_signing_sd(same_axis_pc1, meta)}}


# =========================================================================== outcome labels
def base_year_post_rows(df: pd.DataFrame, base_year: int) -> dict:
    """Rows of the base year already coded post-AUKUS (month-level Post), per country of the estimation sample."""
    m = (df['year'] == base_year) & (df['post_aukus'].astype(int) == 1)
    return {str(c): int(n) for c, n in df.loc[m, 'country'].value_counts().sort_index().items()}


def pretrend_outcome(p, base_year: int, post_rows: dict, alpha: float = 0.05) -> dict:
    """Label of the joint pre-period Wald test: rejection iff p <= .05. The base year is a whole
    calendar year, so it holds post-signing months when ``post_rows`` is non-empty; not rejecting is no proof of
    parallel trends (low power; Roth 2022)."""
    total = sum(post_rows.values())
    base = (f'基准年 {base_year} 含签署后月份（' + '、'.join(f'{c} {n}' for c, n in post_rows.items()) + ' 行）'
            if total else f'基准年 {base_year} 不含签署后月份')
    if not _finite(p):
        return {'pretrend_joint_null_rejected': None, 'label': f'p 不可用；{base}', 'base_year_post_rows': post_rows}
    if p > alpha:
        return {'pretrend_joint_null_rejected': False, 'label': f'未拒绝签署前联合零假设；{base}；不证明平行趋势',
                'base_year_post_rows': post_rows}
    return {'pretrend_joint_null_rejected': True, 'label': f'拒绝签署前联合零假设；{base}',
            'base_year_post_rows': post_rows}


def placebo_outcome(p, alpha: float = 0.05) -> dict:
    """Label of a placebo UK/AU x fake-Post test: rejection iff bootstrap p <= .05."""
    if not _finite(p):
        return {'null_rejected': None, 'label': '安慰剂 p 不可用'}
    return ({'null_rejected': False, 'label': '安慰剂未拒绝（p > .05）'} if p > alpha
            else {'null_rejected': True, 'label': '安慰剂拒绝（p ≤ .05）'})


# =========================================================================== did_robustness
def build_main_model(df):
    D_UK = (df['country'] == 'UK').astype(float).to_numpy()
    D_AU = (df['country'] == 'AU').astype(float).to_numpy()
    post = df['post_aukus'].astype(float).to_numpy()
    n = len(df)
    X = np.column_stack([np.ones(n), D_UK, D_AU, post, D_UK * post, D_AU * post])
    return X, ['intercept', 'UK', 'AU', 'post_aukus', 'UK_x_post', 'AU_x_post']


def build_year_fe_model(df, with_post: bool = False):
    """Year fixed-effects model: ``with_post=False`` as in the original specification (Post main effect omitted);
    ``with_post=True`` with the Post main effect."""
    D_UK = (df['country'] == 'UK').astype(float).to_numpy()
    D_AU = (df['country'] == 'AU').astype(float).to_numpy()
    post = df['post_aukus'].astype(float).to_numpy()
    n = len(df)
    ref_year = setting('did_robustness')['ref_year_fe']
    year_dummies, year_names = [], []
    for y in sorted(df['year'].unique()):
        if y != ref_year:
            year_dummies.append((df['year'] == y).astype(float).to_numpy())
            year_names.append(f'year_{y}')
    extra, extra_names = ([post], ['post_aukus']) if with_post else ([], [])
    X = np.column_stack([np.ones(n), D_UK, D_AU] + year_dummies + extra + [D_UK * post, D_AU * post])
    return X, ['intercept', 'UK', 'AU'] + year_names + extra_names + ['UK_x_post', 'AU_x_post']


def build_time_trend_model(df):
    D_UK = (df['country'] == 'UK').astype(float).to_numpy()
    D_AU = (df['country'] == 'AU').astype(float).to_numpy()
    post = df['post_aukus'].astype(float).to_numpy()
    time_var = (df['year'] - setting('wcb')['trend_origin_year']).astype(float).to_numpy()
    n = len(df)
    X = np.column_stack([np.ones(n), D_UK, D_AU, time_var, post, D_UK * post, D_AU * post])
    return X, ['intercept', 'UK', 'AU', 'time', 'post_aukus', 'UK_x_post', 'AU_x_post']


def build_placebo_model(df, fake_cutoff_year):
    """Placebo design: pre-AUKUS rows only, fake_post = year >= cutoff."""
    pre_df = df[df['post_aukus'] == 0].copy()
    pre_df['fake_post'] = (pre_df['year'] >= fake_cutoff_year).astype(float)
    D_UK = (pre_df['country'] == 'UK').astype(float).to_numpy()
    D_AU = (pre_df['country'] == 'AU').astype(float).to_numpy()
    fake_post = pre_df['fake_post'].to_numpy()
    n = len(pre_df)
    X = np.column_stack([np.ones(n), D_UK, D_AU, fake_post, D_UK * fake_post, D_AU * fake_post])
    return X, ['intercept', 'UK', 'AU', 'fake_post', 'UK_x_fake_post', 'AU_x_fake_post'], pre_df


def run_model_bootstrap(X, pc_scores, clusters, feature_names, model_name, n_bootstrap=seeds.WCB_DRAWS,
                        seed=seeds.WCB_SEED_BASE,
                        seed_step: int = 1) -> dict:
    """Wild cluster bootstrap of one model on each PC (seed ``seed + pc`` at ``seed_step=1``) and the 95% intervals
    under ``intervals_95``."""
    results = {'model_name': model_name, 'feature_names': feature_names}
    intervals = {}
    for pc_i in range(3):
        pc = f'PC{pc_i + 1}'
        boot = wcb_robustness_bootstrap(X, pc_scores[:, pc_i], clusters, n_bootstrap, seed + seed_step * pc_i)
        pc_result = {name: {'coef': float(boot['ols_coef'][i]), 'ols_se': float(boot['ols_se'][i]),
                            'ols_p': float(boot['ols_p'][i]), 'boot_se': float(boot['boot_se'][i]),
                            'boot_p': float(boot['boot_p'][i]), 'se_ratio': float(boot['se_ratio'][i])}
                     for i, name in enumerate(feature_names)}
        pc_result['_meta'] = {'n_obs': int(boot['n_obs']), 'n_clusters': int(boot['n_clusters']),
                              'r_squared': float(boot['r_squared'])}
        results[pc] = pc_result
        intervals[pc] = {name: [float(boot['ci_low'][i]), float(boot['ci_high'][i])]
                         for i, name in enumerate(feature_names) if 'x' in name}
    results['intervals_95'] = intervals
    return results


def cluster_robust_wald_test(X, y, clusters, test_indices, n_bootstrap=seeds.WCB_DRAWS,
                             seed=seeds.EVENT_STUDY_WALD_SEED) -> dict:
    """Joint Wald test of the coefficients ``test_indices``: bootstrap covariance (Rademacher weights per cluster) and
    OLS covariance, chi-square reference."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)
    residuals = y - X @ beta_hat
    fitted = X @ beta_hat
    unique, obs_cluster_idx = cluster_index(clusters)
    XtX_inv_Xt = XtX_inv @ X.T
    rng = np.random.RandomState(seed)
    boot = np.zeros((n_bootstrap, k))
    for b in range(n_bootstrap):
        cluster_weights = rng.choice([-1.0, 1.0], size=len(unique))
        boot[b] = XtX_inv_Xt @ (fitted + cluster_weights[obs_cluster_idx] * residuals)
    V_boot_full = np.cov(boot.T)
    beta_test = beta_hat[test_indices]
    V_boot_test = V_boot_full[np.ix_(test_indices, test_indices)]
    wald_cluster = float(beta_test @ np.linalg.inv(V_boot_test) @ beta_test)
    s2 = np.sum(residuals ** 2) / (n - k)
    V_ols_test = (XtX_inv * s2)[np.ix_(test_indices, test_indices)]
    wald_ols = float(beta_test @ np.linalg.inv(V_ols_test) @ beta_test)
    df_test = len(test_indices)
    return {'wald_cluster_robust': wald_cluster, 'p_cluster_robust': float(1 - stats.chi2.cdf(wald_cluster, df_test)),
            'wald_ols': wald_ols, 'p_ols': float(1 - stats.chi2.cdf(wald_ols, df_test)), 'df': df_test,
            'n_clusters': len(unique)}


def run_event_study_with_proper_wald(df, pc_scores, base_year=2021, min_year=2017, max_year=2024,
                                     n_bootstrap=seeds.WCB_DRAWS, seed=seeds.EVENT_STUDY_WALD_SEED) -> dict:
    """Event study of the robustness section with the joint Wald test (three countries, 2017-2024; one seed for all
    PCs)."""
    mask = (df['year'] >= min_year) & (df['year'] <= max_year)
    df_f = df[mask]
    pc_f = pc_scores[mask.to_numpy()]
    n = len(df_f)
    clusters = df_f['doc_id'].to_numpy()
    D_UK = (df_f['country'] == 'UK').astype(float).to_numpy()
    D_AU = (df_f['country'] == 'AU').astype(float).to_numpy()
    study_years = sorted([y for y in df_f['year'].unique() if y != base_year])
    features, names = [np.ones(n), D_UK, D_AU], ['intercept', 'UK', 'AU']
    for y in study_years:
        features.append((df_f['year'] == y).astype(float).to_numpy())
        names.append(f'year_{y}')
    uk_idx, au_idx = [], []
    for y in study_years:
        dummy = (df_f['year'] == y).astype(float).to_numpy()
        features.append(D_UK * dummy)
        names.append(f'UK_x_{y}')
        uk_idx.append(len(names) - 1)
        features.append(D_AU * dummy)
        names.append(f'AU_x_{y}')
        au_idx.append(len(names) - 1)
    X = np.column_stack(features)
    pre_years = [y for y in study_years if y < base_year]
    uk_pre = [uk_idx[study_years.index(y)] for y in pre_years]
    au_pre = [au_idx[study_years.index(y)] for y in pre_years]
    post_rows = base_year_post_rows(df_f, base_year)
    results = {'n_obs': n, 'study_years': [int(y) for y in study_years], 'base_year': base_year}
    for pc_i in range(3):
        y = pc_f[:, pc_i]
        uk = cluster_robust_wald_test(X, y, clusters, uk_pre, n_bootstrap, seed)
        au = cluster_robust_wald_test(X, y, clusters, au_pre, n_bootstrap, seed)
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta_hat = XtX_inv @ (X.T @ y)
        residuals = y - X @ beta_hat
        s2 = np.sum(residuals ** 2) / (n - X.shape[1])
        se = np.sqrt(np.diag(XtX_inv * s2))
        coefs = {name: {'coef': float(beta_hat[i]), 'se': float(se[i]),
                        'p': float(2 * (1 - stats.t.cdf(abs(beta_hat[i] / se[i]), df=n - X.shape[1])))}
                 for i, name in enumerate(names)}

        def block(w):
            return {'cluster_robust_wald': w['wald_cluster_robust'], 'cluster_robust_p': w['p_cluster_robust'],
                    'ols_wald': w['wald_ols'], 'ols_p': w['p_ols'], 'df': w['df'], 'n_clusters': w['n_clusters'],
                    'pre_years': [int(v) for v in pre_years],
                    **pretrend_outcome(w['p_cluster_robust'], base_year, post_rows)}
        results[f'PC{pc_i + 1}'] = {'coefficients': coefs, 'UK_parallel_trends': block(uk),
                                    'AU_parallel_trends': block(au)}
    return results


def did_robustness(pc_scores3, meta, explained_variance3) -> dict:
    """Robustness models, including year fixed effects with and without the Post main effect.
    Bootstrap seeds are 42, 43 and 44 for PC1, PC2 and PC3."""
    params = setting('did_robustness')
    N_BOOT, SEED = setting('wcb')['n_bootstrap'], setting('wcb')['seed_base']
    ES_SEED = setting('did_robustness')['event_study']['seed']
    step = 1
    df = meta.reset_index(drop=True)
    clusters = df['doc_id'].to_numpy()
    out = {'description': 'Comprehensive DID Robustness Analysis', 'n_total': len(df),
           'n_clusters': int(df['doc_id'].nunique()),
           'pca_variance': {f'PC{i + 1}': float(explained_variance3[i]) for i in range(3)}}
    X1, n1 = build_main_model(df)
    r1 = run_model_bootstrap(X1, pc_scores3, clusters, n1, 'Main_DID', N_BOOT, SEED, step)
    X2, n2 = build_year_fe_model(df, with_post=False)
    r2 = run_model_bootstrap(X2, pc_scores3, clusters, n2, 'Year_FE_DID', N_BOOT, SEED, step)
    X2b, n2b = build_year_fe_model(df, with_post=True)
    r2b = run_model_bootstrap(X2b, pc_scores3, clusters, n2b, 'Year_FE_DID_with_post', N_BOOT, SEED, step)
    X3, n3 = build_time_trend_model(df)
    r3 = run_model_bootstrap(X3, pc_scores3, clusters, n3, 'Time_Trend_DID', N_BOOT, SEED, step)
    pre_mask = (df['post_aukus'] == 0).to_numpy()
    pc_pre = pc_scores3[pre_mask]
    placebo = {}
    for year, key, name in ((params['placebo_years'][0], 'model_4a_placebo_2020', 'Placebo_2020'),
                            (params['placebo_years'][1], 'model_4b_placebo_2021', 'Placebo_2021')):
        Xp, npl, pre_df = build_placebo_model(df, year)
        placebo[key] = run_model_bootstrap(Xp, pc_pre, pre_df['doc_id'].to_numpy(), npl, name, N_BOOT, SEED, step)
    m17 = (df['year'] >= params['restricted_min_year']).to_numpy()
    df_r = df[m17]
    X5, n5 = build_main_model(df_r)
    r5 = run_model_bootstrap(X5, pc_scores3[m17], df_r['doc_id'].to_numpy(), n5, 'Restricted_2017', N_BOOT, SEED,
                             step)
    es = params['event_study']
    wald = run_event_study_with_proper_wald(df, pc_scores3, es['base_year'], es['min_year'], es['max_year'],
                                            N_BOOT, ES_SEED)
    out.update({'model_1_main': r1, 'model_2_year_fe': r2, 'model_3_time_trend': r3, **placebo,
                'model_5_restricted': r5, 'model_6_proper_wald': wald,
                'model_2b_year_fe_with_post': r2b})
    summary = {'comparison': {}, 'placebo': {}, 'wald': {}}
    r4a, r4b = placebo['model_4a_placebo_2020'], placebo['model_4b_placebo_2021']
    for pc in ('PC1', 'PC2', 'PC3'):
        summary['comparison'][pc] = {var: {
            'main': {'coef': r1[pc][var]['coef'], 'boot_p': r1[pc][var]['boot_p']},
            'year_fe': {'coef': r2[pc][var]['coef'], 'boot_p': r2[pc][var]['boot_p']},
            'time_trend': {'coef': r3[pc][var]['coef'], 'boot_p': r3[pc][var]['boot_p']},
            'restricted': {'coef': r5[pc][var]['coef'], 'boot_p': r5[pc][var]['boot_p']}}
            for var in ('UK_x_post', 'AU_x_post')}
        summary['placebo'][pc] = {label: {'UK_coef': r[pc]['UK_x_fake_post']['coef'],
                                          'UK_p': r[pc]['UK_x_fake_post']['boot_p'],
                                          'AU_coef': r[pc]['AU_x_fake_post']['coef'],
                                          'AU_p': r[pc]['AU_x_fake_post']['boot_p'],
                                          'UK_label': placebo_outcome(r[pc]['UK_x_fake_post']['boot_p'])['label'],
                                          'AU_label': placebo_outcome(r[pc]['AU_x_fake_post']['boot_p'])['label']}
                                  for label, r in (('placebo_2020', r4a), ('placebo_2021', r4b))}
        summary['wald'][pc] = {c: {k: wald[pc][f'{c}_parallel_trends'][k] for k in
                                   ('cluster_robust_wald', 'cluster_robust_p', 'ols_wald', 'ols_p',
                                    'pretrend_joint_null_rejected', 'label')}
                               for c in ('UK', 'AU')}
    summary['placebo_specification'] = ('真实签署前样本（post_aukus = 0），fake_post = year >= 截断年；与主模型不同，未含年趋势项；'
                                        '未拒绝不证明识别成立')
    out['summary'] = summary
    out['_details'] = {'seeds_per_pc': [SEED + step * i for i in range(3)],
                  'model_ids': {
                      'year_fe_without_post': 'model_2_year_fe (original specification, Post main effect omitted)',
                      'year_fe_with_post': 'model_2b_year_fe_with_post (Post main effect added)',
                      'restricted_2017_onward': 'model_5_restricted (v1 main specification without trend, year >= 2017)'}}
    return out


def placebo_with_trend(pc_scores3, meta) -> dict:
    """The placebo as the main model with a fake cut: pre-signing rows only (all countries, all years), fake_post =
    year >= cut for the placebo years, regressors constant, D_UK, D_AU, t = year - trend origin, fake_post, D_UK x
    fake_post, D_AU x fake_post; the wild cluster bootstrap of the robustness models (seed 42 + PC index). Table 6
    reports the placebo without the trend (:func:`build_placebo_model`)."""
    N_BOOT, SEED = setting('wcb')['n_bootstrap'], setting('wcb')['seed_base']
    df = meta.reset_index(drop=True)
    rows = (df['post_aukus'] == 0).to_numpy()
    rename = {'post_aukus': 'fake_post', 'UK_x_post': 'UK_x_fake_post', 'AU_x_post': 'AU_x_fake_post'}
    out = {}
    for cut in setting('did_robustness')['placebo_years']:
        pre = df[rows].copy()
        pre['post_aukus'] = (pre['year'] >= cut).astype(float)
        X, names = build_time_trend_model(pre)
        out[str(cut)] = run_model_bootstrap(X, pc_scores3[rows], pre['doc_id'].to_numpy(),
                                            [rename.get(n, n) for n in names], f'Placebo_{cut}_trend', N_BOOT, SEED, 1)
    return out


# =========================================================================== parallel trends (paper)
def parallel_trends_paper(pc_scores3, meta, explained_variance3) -> dict:
    """The parallel-trends event study of the paper (US and UK; years and base year from the parameters) with the
    joint Wald test of the pre-signing UK x year terms under the full bootstrap covariance."""
    params = setting('parallel_trends')
    base_year, min_year, max_year = params['base_year'], params['min_year'], params['max_year']
    mask = ((meta['year'] >= min_year) & (meta['year'] <= max_year) & (meta['country'].isin(['US', 'UK']))).to_numpy()
    df_f = meta[mask]
    scores = pc_scores3[mask]
    years = sorted([y for y in df_f['year'].unique() if y != base_year])
    df_uk = (df_f['country'] == 'UK').astype(float).to_numpy()
    features, names = [df_uk], ['UK']
    for year in years:
        features.append((df_f['year'] == year).astype(float).to_numpy())
        names.append(f'year_{year}')
    for year in years:
        features.append(df_uk * (df_f['year'] == year).astype(float).to_numpy())
        names.append(f'UK_x_{year}')
    X = np.column_stack([np.ones(len(df_f)), np.column_stack(features)])
    feature_names = ['intercept'] + names
    clusters = df_f['doc_id'].to_numpy()
    N_BOOT, SEED = setting('wcb')['n_bootstrap'], setting('wcb')['seed_base']
    unique, inverse = cluster_index(clusters)
    post_rows = base_year_post_rows(df_f, base_year)
    out = {'method': 'Event Study for Parallel Trends Test', 'base_year': base_year, 'n_samples': int(len(df_f)),
           'n_clusters': int(len(unique)), 'pca_explained_variance': [float(v) for v in explained_variance3],
           'outcome_rule': ('pretrend_joint_null_rejected = Wald p <= .05（签署前 UK x 年份系数的联合零假设，完整 bootstrap '
                            '协方差）；n_significant = 单个签署前系数 bootstrap p < .05 的个数（未经多重调整）；未拒绝不证明'
                            '平行趋势'),
           'parallel_trends_test': {}, 'coefficients': {}}
    for pc_idx in range(3):
        pc = f'PC{pc_idx + 1}'
        y = scores[:, pc_idx]
        rng = np.random.RandomState(SEED + pc_idx)
        n_samples, n_params = X.shape
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta
        sigma2 = np.sum(residuals ** 2) / (n_samples - n_params)
        se_ols = np.sqrt(np.diag(sigma2 * XtX_inv))
        proj = XtX_inv @ X.T
        fitted = X @ beta
        boot = np.zeros((N_BOOT, n_params))
        for b in range(N_BOOT):
            weights = rng.choice([-1, 1], size=len(unique))
            sample_weights = np.empty(n_samples)
            sample_weights[:] = weights[inverse]
            boot[b] = proj @ (fitted + residuals * sample_weights)
        se_boot = np.std(boot, axis=0, ddof=1)
        p_boot = np.mean(np.abs(boot - beta[np.newaxis, :]) >= np.abs(beta[np.newaxis, :]), axis=0)
        coefs = {}
        for i, name in enumerate(feature_names):
            coefs[name] = {'coef': float(beta[i]), 'se': float(se_boot[i]),
                           't': float(beta[i] / se_boot[i]) if se_boot[i] > 0 else 0.0, 'p': float(p_boot[i]),
                           'se_ols': float(se_ols[i])}
        out['coefficients'][pc] = coefs
        pre_years = sorted(y_ for y_ in years if y_ < base_year)
        pre_idx = [feature_names.index(f'UK_x_{yr}') for yr in pre_years]
        coefs_pre = beta[pre_idx]
        cov = np.cov(boot[:, pre_idx], rowvar=False, ddof=1)
        wald = float(coefs_pre @ np.linalg.inv(cov) @ coefs_pre)
        p_value = float(1 - stats.chi2.cdf(wald, len(pre_idx)))
        significant = sum(1 for i in pre_idx if p_boot[i] < 0.05)
        out['parallel_trends_test'][pc] = {'UK': {
            'n_years': len(pre_idx), 'pre_years': [int(v) for v in pre_years], 'wald_stat': wald, 'df': len(pre_idx),
            'p_value': p_value, 'n_significant': int(significant),
            **pretrend_outcome(p_value, base_year, post_rows)}}
    return out


# =========================================================================== H3
def filter_vocabulary(vocab: dict):
    """Vocabulary filter of the nearest-neighbour analysis (tokens sorted by id: only exact ties can move)."""
    stats_ = {'total_vocab': len(vocab), 'special_tokens_removed': 0, 'non_G_prefix_removed': 0,
              'non_alpha_removed': 0, 'single_char_removed': 0, 'kept': 0}
    filtered = []
    for token_str, token_id in sorted(vocab.items(), key=lambda kv: kv[1]):
        if token_str == '<|endoftext|>':
            stats_['special_tokens_removed'] += 1
            continue
        if not token_str.startswith('Ġ'):
            stats_['non_G_prefix_removed'] += 1
            continue
        clean = token_str[1:]
        if not clean.isalpha():
            stats_['non_alpha_removed'] += 1
            continue
        if len(clean) <= 1:
            stats_['single_char_removed'] += 1
            continue
        filtered.append((clean, token_id, clean.lower()))
        stats_['kept'] += 1
    return filtered, stats_


def country_means(Y, meta, mask):
    means, sizes = {}, {}
    for c in COUNTRIES:
        m = mask & (meta['country'] == c).to_numpy()
        if m.sum() == 0:
            continue
        means[c] = np.stack(list(Y[m])).mean(axis=0)
        sizes[c] = int(m.sum())
    return means, sizes


def find_nearest_neighbors(country_mean, embeddings, filtered_tokens, top_k=30):
    """The top-k cosine neighbours of a country mean vector (stable sort; ties reported)."""
    from sklearn.metrics.pairwise import cosine_similarity
    token_ids = [t[1] for t in filtered_tokens]
    sims = cosine_similarity(country_mean.reshape(1, -1), embeddings[token_ids])[0]
    order = np.argsort(-sims, kind='stable')
    results, seen = [], set()
    for idx in order:
        clean, token_id, lower = filtered_tokens[idx]
        if lower in seen:
            continue
        seen.add(lower)
        results.append({'word': lower, 'similarity': round(float(sims[idx]), 6), 'original_form': clean})
        if len(results) >= top_k:
            break
    top = sims[order[:top_k * 4]]
    ties = int((np.diff(top) == 0).sum())
    return results, ties


def analyze_common_unique(results_by_country, top_n=15):
    """Words in the top n of all three countries and words in the top n of one country only."""
    word_sets = {c: set(r['word'] for r in res[:top_n]) for c, res in results_by_country.items()}
    all_c = list(word_sets)
    if len(all_c) < 3:
        return [], {c: list(word_sets.get(c, set())) for c in COUNTRIES}
    common = word_sets[all_c[0]]
    for c in all_c[1:]:
        common = common & word_sets[c]
    unique = {}
    for c in all_c:
        others = set()
        for c2 in all_c:
            if c2 != c:
                others |= word_sets[c2]
        unique[c] = sorted(word_sets[c] - others)
    return sorted(common), unique


def h3_nearest_neighbors(Y, meta, embeddings, vocab) -> dict:
    """H3: the nearest GPT-2 vocabulary words of each country's mean vector, before and after the signing; no PCA."""
    params = setting('h3')
    filtered, fstats = filter_vocabulary(vocab)
    post = meta['post_aukus'].to_numpy().astype(bool)
    all_rows = np.ones(len(meta), dtype=bool)
    pre_means, pre_sizes = country_means(Y, meta, ~post)
    post_means, post_sizes = country_means(Y, meta, post)
    all_means, all_sizes = country_means(Y, meta, all_rows)
    results, ties = {}, {}
    for period, means in (('pre_aukus', pre_means), ('post_aukus', post_means), ('full_period', all_means)):
        results[period], ties[period] = {}, {}
        for c, vec in means.items():
            results[period][c], ties[period][c] = find_nearest_neighbors(vec, embeddings, filtered, params['top_k'])
    analyses = {p: dict(zip(('common', 'unique'), analyze_common_unique(results[p], params['top_n_compare'])))
                for p in results}
    changes = {}
    for c in COUNTRIES:
        pre_w = set(r['word'] for r in results['pre_aukus'].get(c, [])[:params['top_n_compare']])
        post_w = set(r['word'] for r in results['post_aukus'].get(c, [])[:params['top_n_compare']])
        changes[c] = {'entered': sorted(post_w - pre_w), 'exited': sorted(pre_w - post_w)}
    return {'analysis_type': 'nearest_neighbor_words', 'llm_model': 'GPT-2',
            'vocabulary_filter': {'total_vocab': fstats['total_vocab'],
                                  'special_tokens_removed': fstats['special_tokens_removed'],
                                  'non_G_prefix_removed': fstats['non_G_prefix_removed'],
                                  'non_alpha_removed': fstats['non_alpha_removed'],
                                  'single_char_removed': fstats['single_char_removed'],
                                  'effective_tokens_kept': fstats['kept']},
            'sample_sizes': {'total': all_sizes, 'pre_aukus': pre_sizes, 'post_aukus': post_sizes},
            'post_aukus_analysis': {'nearest_words': results['post_aukus'],
                                    'common_words_top15': analyses['post_aukus']['common'],
                                    'unique_words_top15': analyses['post_aukus']['unique']},
            'pre_aukus_analysis': {'nearest_words': results['pre_aukus'],
                                   'common_words_top15': analyses['pre_aukus']['common'],
                                   'unique_words_top15': analyses['pre_aukus']['unique']},
            'full_period_analysis': {'nearest_words': results['full_period'],
                                     'common_words_top15': analyses['full_period']['common'],
                                     'unique_words_top15': analyses['full_period']['unique']},
            'pre_post_comparison': changes, '_details': {'exact_similarity_ties_near_top': ties,
                                                     'token_order': 'sorted by token id; stable sort'}}


def did_h3_distances(Y, meta) -> dict:
    """Euclidean distances between the country mean vectors, before and after the signing."""
    post = meta['post_aukus'].to_numpy().astype(bool)
    out = {}
    for label, mask in (('pre', ~post), ('post', post)):
        means, _ = country_means(Y, meta, mask)
        out[label] = {}
        for a, b in (('US', 'UK'), ('US', 'AU'), ('UK', 'AU')):
            va, vb = means[a], means[b]
            out[label][f'{a}_{b}'] = float(np.linalg.norm(va - vb))
    pairs = {k: {'pre': out['pre'][k], 'post': out['post'][k],
                 'change_percent': (out['post'][k] - out['pre'][k]) / out['pre'][k] * 100} for k in out['pre']}
    pre_avg = float(np.mean([v['pre'] for v in pairs.values()]))
    post_avg = float(np.mean([v['post'] for v in pairs.values()]))
    return {'average_distances': {'pre_aukus_avg_distance': pre_avg, 'post_aukus_avg_distance': post_avg,
                                  'percent_change': (post_avg - pre_avg) / pre_avg * 100},
            'pairwise_distances': pairs}


# =========================================================================== same-axis / orientation
def orientation(own_components, reference_components) -> dict:
    """Own PC1 loading vs reconstructed reference PC1 loading: cosine, sign, thresholds."""
    params = setting('orientation')
    own = np.asarray(own_components[0], dtype=np.float64)
    ref = np.asarray(reference_components[0], dtype=np.float64)
    cos = float(own @ ref / (np.linalg.norm(own) * np.linalg.norm(ref)))
    return {'cos_pc1': cos, 'abs_cos_pc1': abs(cos), 'sign': 1 if cos >= 0 else -1,
            'comparable_at_0.30': abs(cos) >= params['primary_threshold'],
            'descriptive': {str(t): abs(cos) >= t for t in params['descriptive_thresholds']},
            'rule': 'orient own PC1 by the sign of its inner product with the reference PC1 loading; never by '
                    'regression coefficients; |cos| < 0.30 -> "direction not comparable" (p and Holm membership kept)'}


def h2_models(Y, meta, pca3=None) -> dict:
    """PCA-3 (all rows, 'full') + WCB main + did robustness + paper parallel trends."""
    pca3 = pca3 or fit_pca(Y, 3)
    S3 = pca3.transform(Y)
    ev = pca3.explained_variance_ratio_
    return {'pca3': pca3, 'scores3': S3, 'wcb': h2_wcb_main(S3, meta, ev), 'did': did_robustness(S3, meta, ev),
            'parallel_trends': parallel_trends_paper(S3, meta, ev)}
