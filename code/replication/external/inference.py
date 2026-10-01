"""The circular month-block bootstrap, p values and intervals, the MDE tiers, the fake-cutoff
diagnostic, the v1 document-cluster WCB (historical reference) and the Holm adjustment.

Implementation details of the least-squares, resampling and inference algorithms:

* least squares: minimum-norm weighted least squares (:func:`lstsq_fit`); estimability c'P = c' with c_j = 0 on zero
  columns (:func:`estimable_direct`); per-unit sufficient statistics (:class:`UnitStats`); batched Jacobi-scaled
  eigendecomposition with rank ratio 1e-11 and an ambiguity band (:func:`solve_gram`); chunked refits of every draw,
  ambiguous draws refitted by SVD on the materialised rows (:func:`fit_draws`); the original-sample fit
  (:func:`point_fit`);
* resampling: :class:`BlockDraws` (``numpy.random.Generator(PCG64(seed))``, pre starts then post starts, a block of L
  consecutive units wrapping within its own layer, multiplicities by ``np.add.at``); the layers are the calendar
  months of each window (2021-09 wholly post), and the replacement draws continue the same generator after the B
  main draws;
* inference: :func:`r_max` / :func:`order_stat_quantile` / :func:`draw_inference` (p = (1 + #{|t*-t| >= |t|}) /
  (B_e + 1); half-width = the order statistic a_(B_e - r), r = ceil(alpha (B_e + 1)) - 2, so "interval excludes 0"
  <=> "p < alpha") and :func:`holm`; the MDE is the observed-precision MDE (c + z_.80) x SE of the selected test
  (:func:`replication.external.selection.observed_mde`).

Design failures are judged by the estimability of the target contrast in a draw; a failed draw is replaced by
the next replacement draw; more than 1% failures (of B) makes the item "不可用", which is never counted as "not
rejected" (it enters the Holm family with p = 1).

Monthly-score HAC kernel: :func:`bartlett` / :func:`kernel_matrix` give the Bartlett weights k_b(h) = (1 - |h| / b)_+
over true month distances (no wrap) used by the iid-exact corrected sensitivity estimator of
:mod:`replication.external.escov`. On the saturated event study the uncorrected monthly-score HAC is biased towards
zero, so the event-study coefficients are descriptive. Primary M1 and pre-trend slope inference use the methods
selected by :mod:`replication.external.selection`; month-block bootstrap results are also retained for descriptive
comparisons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .params import external_params

RANK_RTOL = 1e-11          # eigenvalue ratio at or below which a direction is dropped
GAP_DROPPED_MAX = 1e-13    # a dropped ratio above this, or a kept ratio below GAP_KEPT_MIN, is ambiguous
GAP_KEPT_MIN = 1e-9
ESTIMABLE_TOL = 1e-8       # relative residual of c'P = c' accepted as estimable
UNAVAILABLE = '不可用'


# =========================================================================== least squares
def lstsq_fit(X, Y, w=None):
    """Minimum-norm weighted least squares via rank-revealing SVD."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if w is not None:
        r = np.sqrt(np.asarray(w, dtype=float))
        X, Y = X * r[:, None], (Y * r[:, None] if Y.ndim == 2 else Y * r)
    beta, _, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    return beta, int(rank), sv


def estimable_direct(X, c, w=None, tol=ESTIMABLE_TOL):
    """Estimability c'P = c' on a materialised (weighted) design, via SVD."""
    X = np.asarray(X, dtype=float)
    if w is not None:
        X = X[np.asarray(w) > 0]
    if X.size == 0:
        return False, float('inf')
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    keep = s > max(X.shape) * np.finfo(float).eps * (s[0] if s.size else 0)
    V = vt[keep].T
    residual = np.linalg.norm(c - V @ (V.T @ c)) / max(np.linalg.norm(c), 1e-300)
    return bool(residual <= tol), float(residual)


@dataclass
class UnitStats:
    """Per-unit sufficient statistics G_u = X_u' W X_u, H_u = X_u' W Y_u."""
    G: np.ndarray
    H: np.ndarray
    rows: np.ndarray
    n_units: int

    @classmethod
    def build(cls, X, Y, units, n_units, w=None):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        units = np.asarray(units, dtype=np.int64)
        w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
        p, q = X.shape[1], Y.shape[1]
        G = np.zeros((n_units, p, p))
        H = np.zeros((n_units, p, q))
        rows = np.zeros(n_units)
        order = np.argsort(units, kind='stable')
        bounds = np.searchsorted(units[order], np.arange(n_units + 1))
        for u in range(n_units):
            idx = order[bounds[u]:bounds[u + 1]]
            if idx.size == 0:
                continue
            Xw = X[idx] * w[idx, None]
            G[u] = X[idx].T @ Xw
            H[u] = Xw.T @ Y[idx]
            rows[u] = w[idx].sum()
        return cls(G=G, H=H, rows=rows, n_units=n_units)

    def aggregate(self, M):
        M = np.atleast_2d(np.asarray(M, dtype=float))
        return np.tensordot(M, self.G, axes=(1, 0)), np.tensordot(M, self.H, axes=(1, 0))


def solve_gram(G, H, contrasts: dict, rank_rtol=RANK_RTOL):
    """Batched minimum-norm solutions and contrast estimability."""
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    diag = np.einsum('bii->bi', G)
    zero = diag <= 0
    s = np.where(zero, 0.0, 1.0 / np.sqrt(np.where(zero, 1.0, diag)))
    Gs = G * s[:, :, None] * s[:, None, :]
    lam, V = np.linalg.eigh(Gs)
    lam_max = lam[:, -1:]
    ratio = lam / np.where(lam_max > 0, lam_max, 1.0)
    keep = ratio > rank_rtol
    kept_min = np.where(keep, ratio, np.inf).min(axis=1)
    dropped_max = np.where(~keep, ratio, -np.inf).max(axis=1)
    ambiguous = (dropped_max > GAP_DROPPED_MAX) | (kept_min < GAP_KEPT_MIN)
    inv = np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)
    Hs = H * s[:, :, None]
    beta_s = np.einsum('bpk,bkq->bpq', V, np.einsum('bpk,bpq->bkq', V, Hs) * inv[:, :, None])
    beta = beta_s * s[:, :, None]
    Vk = V * keep[:, None, :]
    estimates, estimable, residuals = {}, {}, {}
    for name, c in contrasts.items():
        c = np.asarray(c, dtype=float)
        cs = c[None, :] * s
        proj = np.einsum('bpk,bk->bp', Vk, np.einsum('bpk,bp->bk', Vk, cs))
        denom = np.maximum(np.linalg.norm(cs, axis=1), 1e-300)
        res = np.linalg.norm(cs - proj, axis=1) / denom
        on_zero = np.abs(c[None, :] * zero).sum(axis=1) > 0
        estimates[name] = np.einsum('p,bpq->bq', c, beta)
        estimable[name] = (res <= ESTIMABLE_TOL) & ~on_zero
        residuals[name] = np.where(on_zero, np.inf, res)
    return {'estimates': estimates, 'estimable': estimable, 'residual': residuals, 'rank': keep.sum(axis=1),
            'kept_min': kept_min, 'dropped_max': dropped_max, 'ambiguous': ambiguous, 'beta': beta}


def fit_draws(unit_stats: UnitStats, M, contrasts: dict, X=None, Y=None, units=None, w=None, chunk=512) -> dict:
    """Refit for every multiplicity row of M; ambiguous draws fall back to lstsq."""
    M = np.atleast_2d(M)
    out = None
    for start in range(0, len(M), chunk):
        G, H = unit_stats.aggregate(M[start:start + chunk])
        part = solve_gram(G, H, contrasts)
        part.pop('beta')
        if out is None:
            out = {k: ({n: [] for n in v} if isinstance(v, dict) else []) for k, v in part.items()}
        for key, value in part.items():
            if isinstance(value, dict):
                for name, arr in value.items():
                    out[key][name].append(arr)
            else:
                out[key].append(value)
    result = {k: ({n: np.concatenate(a) for n, a in v.items()} if isinstance(v, dict) else np.concatenate(v))
              for k, v in out.items()}
    fallback = np.flatnonzero(result['ambiguous'])
    result['fallback_draws'] = fallback.tolist()
    if fallback.size:
        if X is None:
            raise RuntimeError(f'{fallback.size} draws have an ambiguous rank gap and no fallback data')
        Y2 = Y if np.ndim(Y) == 2 else np.asarray(Y)[:, None]
        base_w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
        for b in fallback:
            weights = M[b][units] * base_w
            keep = weights > 0
            beta, _, _ = lstsq_fit(X[keep], Y2[keep], weights[keep])
            for name, c in contrasts.items():
                ok, res = estimable_direct(X[keep], np.asarray(c, dtype=float))
                result['estimates'][name][b] = np.asarray(c) @ beta
                result['estimable'][name][b] = ok
                result['residual'][name][b] = res
    return result


def point_fit(X, Y, contrasts: dict, w=None) -> dict:
    """Original-sample estimates via SVD lstsq plus the estimability of each contrast."""
    Y2 = Y if np.ndim(Y) == 2 else np.asarray(Y)[:, None]
    beta, rank, _ = lstsq_fit(X, Y2, w)
    out = {'rank': rank, 'p': int(X.shape[1]), 'estimates': {}, 'estimable': {}, 'residual': {}}
    Xw = X if w is None else X * np.sqrt(np.asarray(w))[:, None]
    for name, c in contrasts.items():
        ok, res = estimable_direct(Xw, np.asarray(c, dtype=float))
        out['estimates'][name] = np.asarray(c) @ beta
        out['estimable'][name] = ok
        out['residual'][name] = res
    out['beta'] = beta
    return out


# =========================================================================== monthly-score HAC kernel and normal inference
def bartlett(h, b: float) -> np.ndarray:
    """k_b(h) = (1 - |h| / b)_+: lags 0..b-1 carry weight, |h| >= b none."""
    return np.clip(1.0 - np.abs(np.asarray(h, dtype=float)) / float(b), 0.0, None)


def kernel_matrix(months, b: float) -> np.ndarray:
    """K[t, u] = k_b(m_t - m_u) over the observed calendar months (true month distances: a month without rows is a
    gap, never a neighbour; no wrap from the last month to the first). Positive definite (the triangle is a
    positive-definite function)."""
    m = np.asarray(months, dtype=float)
    return bartlett(m[:, None] - m[None, :], b)


def psd_record(V: np.ndarray) -> dict:
    """The smallest eigenvalue of a covariance relative to its largest (a PSD check; >= -1e-10 expected)."""
    eig = np.linalg.eigvalsh((np.asarray(V, float) + np.asarray(V, float).T) / 2)
    top = float(max(abs(eig.max()), 1e-300))
    return {'min_eigenvalue': float(eig.min()), 'max_eigenvalue': float(eig.max()),
            'min_over_max': float(eig.min() / top), 'psd': bool(eig.min() >= -1e-10 * top)}


SINGULAR_REL_TOL = 1e-12


def normal_inference(estimate: float, var: float, scale: float | None = None, df: float | None = None) -> dict:
    """Normal (z; ``df`` None) or Student-t (``df``) inference of one contrast with its variance: two-sided p, 95% and
    90% intervals. A non-finite variance, a variance <= 0, or one <= 1e-12 x ``scale`` (c' Q^-1 c Var_w(y);
    numerically zero scores, e.g. an exact fit) is a singular covariance ("不可用"); so is a non-positive df."""
    if not np.isfinite(var) or var <= 0.0 or (scale is not None and var <= SINGULAR_REL_TOL * scale):
        return {'status': UNAVAILABLE, 'estimate': float(estimate),
                'reason': f'singular covariance (variance {var!r}; scale {scale!r})'}
    if df is not None and not (np.isfinite(df) and df > 0):
        return {'status': UNAVAILABLE, 'estimate': float(estimate), 'reason': f'degrees of freedom {df!r}'}
    se = math.sqrt(var)
    z = estimate / se
    dist = stats.norm if df is None else stats.t(df)
    c95, c90 = float(dist.ppf(0.975)), float(dist.ppf(0.95))
    return {'status': 'ok', 'estimate': float(estimate), 'se': se, 'z' if df is None else 't': float(z),
            'df': None if df is None else float(df), 'p': float(2.0 * dist.sf(abs(z))),
            'ci95': [estimate - c95 * se, estimate + c95 * se], 'ci90': [estimate - c90 * se, estimate + c90 * se]}


# =========================================================================== month layers and block draws
@dataclass(frozen=True)
class Layers:
    """Units 0..n_pre-1 (pre layer, calendar order) and n_pre..n_pre+n_post-1 (post layer); ``labels`` in order."""
    labels: tuple
    n_pre: int
    n_post: int

    @property
    def n_units(self) -> int:
        return self.n_pre + self.n_post

    def record(self) -> dict:
        return {'units': self.n_units, 'pre_units': self.n_pre, 'post_units': self.n_post,
                'first': self.labels[0] if self.labels else None, 'last': self.labels[-1] if self.labels else None,
                'first_post': self.labels[self.n_pre] if self.n_post else None}


def month_layers(start: str, end: str, first_post: str) -> Layers:
    """The calendar months of [start, end]; months >= first_post form the post layer (2021-09 whole in post)."""
    from .common import month_index, months
    labels = months(start, end)
    n_pre = sum(1 for m in labels if month_index(m) < month_index(first_post))
    return Layers(tuple(labels), n_pre, len(labels) - n_pre)


def day_layers(start: str, end: str, cut_day: str) -> Layers:
    """Day-level cut: the cut month is split into '<YYYY-MM>a' (pre) and '<YYYY-MM>b' (post)."""
    from .common import month_index, months
    cut_month = cut_day[:7]
    labels = []
    for m in months(start, end):
        labels += [f'{m}a', f'{m}b'] if m == cut_month else [m]
    n_pre = sum(1 for m in labels if month_index(m[:7]) < month_index(cut_month) or m == f'{cut_month}a')
    return Layers(tuple(labels), n_pre, len(labels) - n_pre)


@dataclass(frozen=True)
class BlockDraws:
    """Circular block draws on one pair of layers; ``pre_starts`` / ``post_starts`` hold the main draws followed by
    the replacement draws (drawn after the main draws from the same generator)."""
    block_length: int
    B: int
    R: int
    seed: int
    layers: Layers
    pre_starts: np.ndarray
    post_starts: np.ndarray

    @classmethod
    def generate(cls, layers: Layers, block_length: int | None = None, B: int | None = None, seed: int | None = None,
                 replacements: int | None = None) -> 'BlockDraws':
        bp = external_params('bootstrap')
        L = bp['block_length'] if block_length is None else int(block_length)
        B = bp['B'] if B is None else int(B)
        seed = bp['seed'] if seed is None else int(seed)
        R = 2 * failure_cap(B) + 1 if replacements is None else int(replacements)
        if layers.n_pre < 1:
            raise ValueError('the pre layer needs at least one unit')
        rng = np.random.Generator(np.random.PCG64(seed))
        n_pb, n_qb = math.ceil(layers.n_pre / L), math.ceil(layers.n_post / L)
        if layers.n_post == 0:          # pre-trend slope diagnostic: one layer (pre-signing months only)
            pre = rng.integers(0, layers.n_pre, size=(B, n_pb))
            pre_r = rng.integers(0, layers.n_pre, size=(R, n_pb))
            empty = np.zeros((B + R, 0), dtype=np.int64)
            return cls(L, B, R, seed, layers, np.vstack([pre, pre_r]), empty)
        pre = rng.integers(0, layers.n_pre, size=(B, n_pb))
        post = rng.integers(0, layers.n_post, size=(B, n_qb))
        pre_r = rng.integers(0, layers.n_pre, size=(R, n_pb))
        post_r = rng.integers(0, layers.n_post, size=(R, n_qb))
        return cls(L, B, R, seed, layers, np.vstack([pre, pre_r]), np.vstack([post, post_r]))

    def multiplicities(self) -> np.ndarray:
        """(B + R, units): how often each unit enters each draw (circular blocks within the layer, truncated to the
        layer length)."""
        L, n_pre, n_post = self.block_length, self.layers.n_pre, self.layers.n_post
        offsets = np.arange(L)
        pre = ((self.pre_starts[:, :, None] + offsets) % n_pre).reshape(len(self.pre_starts), -1)[:, :n_pre]
        M = np.zeros((len(pre), self.layers.n_units), dtype=np.int64)
        rows = np.arange(len(pre))[:, None]
        np.add.at(M, (rows, pre), 1)
        if n_post:
            post = ((self.post_starts[:, :, None] + offsets) % n_post).reshape(len(self.post_starts), -1)[:, :n_post]
            np.add.at(M, (rows, n_pre + post), 1)
        return M

    def record(self) -> dict:
        M = self.multiplicities()[:self.B]
        n_pre = self.layers.n_pre
        return {'scheme': external_params('bootstrap')['scheme'], 'block_length': self.block_length, 'B': self.B,
                'replacement_draws_available': self.R, 'seed': self.seed, 'layers': self.layers.record(),
                'pre_blocks': int(self.pre_starts.shape[1]), 'post_blocks': int(self.post_starts.shape[1]),
                'pre_slots_per_draw': sorted(set(M[:, :n_pre].sum(1).tolist())),
                'post_slots_per_draw': sorted(set(M[:, n_pre:].sum(1).tolist())),
                'unit_inclusion_rate_min_max': [float((M > 0).mean(0).min()), float((M > 0).mean(0).max())]}


def failure_cap(B: int) -> int:
    """At most 1% of B sampling failures."""
    return int(math.floor(external_params('bootstrap')['max_failure_share'] * B))


def effective(valid: np.ndarray, B: int) -> dict:
    """The first B valid draws of the main + replacement sequence; failures counted before B are reached."""
    valid = np.asarray(valid, bool)
    order = np.flatnonzero(valid)
    cap = failure_cap(B)
    if len(order) < B:
        failures = int((~valid).sum())
        return {'index': None, 'failures': failures, 'available': False,
                'reason': f'only {len(order)} estimable draws of {len(valid)} (need {B}; failures {failures} > {cap})'}
    last = order[B - 1]
    failures = int((~valid[:last + 1]).sum())
    if failures > cap:
        return {'index': None, 'failures': failures, 'available': False,
                'reason': f'{failures} sampling failures > 1% of B ({cap})'}
    return {'index': order[:B], 'failures': failures, 'available': True, 'replacements_used': int(max(0, last + 1 - B))}


# =========================================================================== p values and intervals
def r_max(B_e: int, alpha: float) -> int:
    return int(math.ceil(alpha * (B_e + 1)) - 2)


def order_stat_quantile(abs_dev, alpha: float):
    abs_dev = np.sort(np.asarray(abs_dev, dtype=float))
    B_e = abs_dev.size
    r = r_max(B_e, alpha)
    if r < 0 or B_e == 0:
        return None
    return float(abs_dev[B_e - r - 1])


def draw_inference(estimate: float, draws, alpha: float | None = None) -> dict:
    """p value (+1, centred two-sided tail) and the symmetric absolute-deviation intervals (95%, 90%), with the Monte
    Carlo errors; ``draws`` are the B_e effective draws."""
    alpha = external_params('bootstrap')['alpha'] if alpha is None else alpha
    d = np.asarray(draws, dtype=float)
    B_e = int(d.size)
    out = {'estimate': float(estimate), 'B_e': B_e}
    if B_e == 0:
        return out | {'p': None, 'status': UNAVAILABLE}
    dev = np.abs(d - estimate)
    exceed = int((dev >= abs(estimate)).sum())
    q95, q90 = order_stat_quantile(dev, alpha), order_stat_quantile(dev, 0.10)
    p = (1 + exceed) / (B_e + 1)
    se = float(np.std(d, ddof=1)) if B_e > 1 else None
    out.update(p=p, exceed=exceed, q95=q95, q90=q90,
               ci95=None if q95 is None else [estimate - q95, estimate + q95],
               ci90=None if q90 is None else [estimate - q90, estimate + q90],
               se=se, draw_mean=float(d.mean()),
               mc_se_p=float(math.sqrt(p * (1 - p) / B_e)),
               mc_se_se=None if se is None else float(se / math.sqrt(2 * (B_e - 1))),
               interval='estimate +/- order statistic a_(B_e - r) of |t* - t|, r = ceil(alpha (B_e + 1)) - 2 '
                        '(symmetric absolute deviation; not a percentile-t interval)')
    return out


def holm(pvalues: dict) -> dict:
    """Holm adjustment over a fixed family: every member stays in the family; a member without a usable p enters
    with p = 1."""
    filled = {k: (1.0 if v is None else float(v)) for k, v in pvalues.items()}
    items = sorted(((p, k) for k, p in filled.items()), key=lambda t: (t[0], t[1]))
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return {k: adjusted[k] for k in pvalues}


# =========================================================================== MDE
def tier(mde_sd: float | None) -> dict:
    """Design tiers."""
    lo, hi = external_params('mde')['tiers_sd']
    if mde_sd is None or not np.isfinite(mde_sd):
        return {'tier': UNAVAILABLE, 'inference': False, 'text': 'MDE 不可用：只作描述'}
    if mde_sd <= lo:
        return {'tier': 'A', 'inference': True, 'text': f'MDE ≤ {lo} SD：可检出约 0.1 SD 的标准化效应（原稿估计的量级；近似精度推算）'}
    if mde_sd <= hi:
        return {'tier': 'B', 'inference': True,
                'text': f'{lo} < MDE ≤ {hi} SD：可作推断；不显著结果对约 0.1 SD 量级的效应信息有限'}
    return {'tier': 'C', 'inference': False,
            'text': f'MDE > {hi} SD：只作描述（功效不足；在 Holm 族中按 p = 1 计入）'}


# =========================================================================== v1 document-cluster WCB
def wcb_v1(X, y, clusters, contrasts: dict, pc_index: int, B: int | None = None) -> dict:
    """The document-cluster wild cluster bootstrap of the main analysis (:func:`replication.analysis.wcb_v1_main_bootstrap`)
    on this design: Rademacher weights per sorted document, B = 1000, seed 42 + the zero-based PC index;
    unrestricted residuals. Historical reference only."""
    from ..analysis import wcb_v1_main_bootstrap
    import pandas as pd
    w = external_params('wcb_v1')
    B = w['B'] if B is None else B
    codes = pd.factorize(pd.Series(np.asarray(clusters)), sort=True)[0]
    res = wcb_v1_main_bootstrap(np.asarray(X, float), np.asarray(y, float), codes, n_bootstrap=B,
                                seed=w['seed_base'] + pc_index)
    out = {}
    for name, c in contrasts.items():
        c = np.asarray(c, dtype=float)
        estimate = float(c @ res['beta'])
        draws = res['bootstrap_betas'] @ c
        exceed = int((np.abs(draws - estimate) >= abs(estimate)).sum())
        out[name] = {'estimate': estimate, 'se': float(np.std(draws, ddof=1)), 'p_v1': exceed / B,
                     'p_plus_one': (1 + exceed) / (B + 1)}
    return {'contrasts': out, 'B': B, 'seed': w['seed_base'] + pc_index, 'n_clusters': int(res['n_clusters']),
            'weights': 'Rademacher per document (numpy.unique order), v1 wild_cluster_bootstrap_pc'}


# =========================================================================== fake cutoffs
NEEDS = {'theta': ('US', 'UK'), 'delta_UK': ('UK',), 'delta_US': ('US',), 'delta_AU': ('AU',)}


def fake_cutoff_diagnostic(panel, base, responses: list, cache, estimands=('theta', 'delta_UK')) -> dict:
    """Design (descriptive, fixed in advance): pre-period data only; every month 2016-01..2019-12 with at least
    18 months after it is a candidate fake cut. Per window, a cut counts for an estimand only if the estimand is
    estimable there: theta needs US, UK and the control observed on both sides, delta_UK needs UK and the control
    (AU missing on one side only drops the redundant AU interaction). Month-block inference on the cut's own
    layers (same seed). The effective denominator, the unavailable cuts and their reasons, and the empirical 5%
    rejection rate are reported; a rate above 0.10 labels the estimand's month-block inference "校准不佳"
    (descriptive conclusions). A rate <= 0.10 does not prove valid coverage and does not remove the pre-test
    limitation (Roth 2022); the rate also mixes in any existing trend."""
    from dataclasses import replace
    from . import estimate as est
    from .common import month_index, months
    fc = external_params('fake_cutoffs')
    first_post = month_index(external_params('extract')['post_first_month'])
    cuts = [m for m in months(fc['first'], fc['last']) if first_post - month_index(m) >= fc['min_months_after']]
    pre_panel = panel.loc[~panel['post'].to_numpy(bool)]
    per = {name: {r: [] for r in responses} for name in estimands}
    detail = {}
    for cut in cuts:
        spec = replace(base, name=f'{base.name}_fake_{cut}', fake_cut=cut, model='M1')
        sub = est.select(pre_panel, spec)
        before = sub['ym'].to_numpy() < month_index(cut)
        seen = {c: (bool(((sub['country'] == c).to_numpy() & before).any()),
                    bool(((sub['country'] == c).to_numpy() & ~before).any())) for c in set(sub['country'])}
        controls = spec.controls
        observed = {}
        for name in estimands:
            need = NEEDS[name] + tuple(controls)
            missing = [c for c in need if not all(seen.get(c, (False, False)))]
            observed[name] = missing
        if not any(not v for v in observed.values()):
            detail[cut] = {'status': 'not estimable', 'missing_one_side': observed}
            for name in estimands:
                for r in responses:
                    per[name][r].append({'cut': cut, 'status': UNAVAILABLE, 'reason': f'no observation on one side: '
                                                                                   f'{observed[name]}'})
            continue
        fit = est.fit_spec(pre_panel, spec, responses, cache)
        detail[cut] = {'n': fit['n'], 'layers': fit['layers'], 'dropped_columns': fit['design']['structural']['dropped']}
        for name in estimands:
            for r in responses:
                if observed[name]:
                    per[name][r].append({'cut': cut, 'status': UNAVAILABLE,
                                         'reason': f'no observation on one side: {observed[name]}'})
                    continue
                e = fit['estimands'][name][r]
                if e.get('status') != 'ok':
                    per[name][r].append({'cut': cut, 'status': UNAVAILABLE, 'reason': e.get('reason')})
                    continue
                per[name][r].append({'cut': cut, 'status': 'ok', 'p': e['p'], 'estimate': e['estimate'],
                                     'rejected': bool(e['p'] < fc['alpha'])})
    summary = {}
    for name in estimands:
        summary[name] = {}
        for r in responses:
            rows = per[name][r]
            ok = [x for x in rows if x['status'] == 'ok']
            rate = (sum(x['rejected'] for x in ok) / len(ok)) if ok else None
            summary[name][r] = {'candidates': len(rows), 'effective_denominator': len(ok),
                                'unavailable': len(rows) - len(ok),
                                'unavailable_reasons': sorted({str(x.get('reason')) for x in rows if x['status'] != 'ok'}),
                                'rejections': sum(x['rejected'] for x in ok), 'empirical_rejection_rate': rate,
                                'flag': (UNAVAILABLE if rate is None else
                                         '校准不佳' if rate > fc['flag_above'] else 'ok'),
                                'cuts': rows}
    return {'window': [base.start, base.end], 'sample': base.sample, 'candidates': cuts, 'summary': summary,
            'cut_detail': detail, 'rule': fc,
            'note': '≤ 0.10 不证明覆盖有效，也不消除预检验局限（Roth 2022）；经验拒绝率同时混有既有趋势'}
