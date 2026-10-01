"""Distance change between the US and UK conceptualisations (Table 7; a post-hoc description in no test family).

Sample: US and UK occurrences of 2014-01 .. 2025-11 (month code 100 * year + month); pre-signing = month code < 202109,
post = >= 202109 (the rows' ``post_aukus`` must agree).

Spaces (baseline encoder): (a) main specification, the scores on the first 88 components of the encoder's own PCA,
fitted on all rows and then cut to the sample; (b) the reference PC1; (c) the reference PC1-PC3; (d) the first 88
reference axes; (e) the full 768-dimensional space Y; (f) the full space on the original manuscript's own sample (all
rows, three pairs of countries; US-UK with inference, US-AU and UK-AU description only). Alternative encoders: (d).

Statistic per space and period t: m_ct the occurrence mean of country c; V_ct = G/(G-1) (1/n^2) sum_d s_d s_d' the
document-clustered covariance of the mean (s_d = sum_{i in d} (y_i - m_ct), G documents, n occurrences); the
bias-corrected squared distance D2_t = |m_UK,t - m_US,t|^2 - tr V_UK,t - tr V_US,t (approximately bias-correcting;
with unequal document sizes a small residual bias remains); the primary statistic delta_sq = D2_post - D2_pre.
Described only: D_t = sign(D2_t) sqrt(|D2_t|), D_post - D_pre, the uncorrected |m_UK,t - m_US,t|.

Inference: documents are resampled with replacement within each of the four country x period cells (G per cell),
B = 2,000, ``numpy.random.default_rng(seeds.DISTANCE_SEED)``, fixed draw order (b = 0..B-1; cells US_pre, US_post,
UK_pre, UK_post; ``integers(0, G, size=G)`` over the cell's documents in ``numpy.unique`` order; the sha256 of all draws
is written). V is recomputed in every draw. The bootstrap population is the documents' empirical distribution, whose
value of the statistic is the sample's uncorrected one, so the errors are e*_b = delta_sq*_b - delta_sq_uncorrected and
the 95% pivot interval (Hall 1992) is [delta_sq - q_.975(e*), delta_sq - q_.025(e*)] (``numpy.quantile``, linear);
p = (1 + #{|e*_b| >= |delta_sq|}) / (B + 1); se_MC = sqrt(p (1 - p) / B). Pre-stated reading by the sign of delta_sq
and its interval: > 0 with the interval excluding 0 -> widened; the interval containing 0 -> no change detected;
< 0 with the interval excluding 0 -> narrowed; an interval excluding 0 on the other side -> not applicable.

Hard checks (a failure stops the script): the own PCA-88 PC1 gives the main-model UK x Post coefficient and the
cumulative variance of the H1 analysis within 1e-10; the reference PC1 gives the same-axis UK x Post coefficient within
1e-10; (f) the uncorrected distances of the three pairs, both periods, equal those of the neighbour-word analysis within
1e-10 and the original manuscript's within 1e-4.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import analysis, pca, seeds

CELLS = ('US_pre', 'US_post', 'UK_pre', 'UK_post')
V1_CELLS = CELLS + ('AU_pre', 'AU_post')
V1_PAIRS = (('US_UK', 'US', 'UK'), ('US_AU', 'US', 'AU'), ('UK_AU', 'UK', 'AU'))
PARAMS = {'countries': ['US', 'UK'], 'month_code': '100 * year + month', 'first_month': 201401,
          'last_month': 202511, 'cutoff_month': 202109, 'cells': list(CELLS),
          'cluster': 'doc_id', 'k': 88, 'B': seeds.DISTANCE_DRAWS, 'seed': seeds.DISTANCE_SEED,
          'generator': 'numpy.random.default_rng (PCG64)', 'interval_quantiles': [0.025, 0.975],
          'quantile_method': 'linear', 'hard_check_tol': 1e-10,
          'statistic': 'delta_sq = D2_post - D2_pre (bias-corrected squared distances)',
          'interval': ('pivot centred on the bootstrap-population value: e*_b = delta_sq*_b - delta_sq_uncorrected; '
                       '[delta_sq - q_.975(e*), delta_sq - q_.025(e*)]'),
          'p_rule': '(1 + #{|e*_b| >= |delta_sq|}) / (B + 1)',
          'v1_sample': {'rows': 'all rows of the baseline analysis', 'countries': ['US', 'UK', 'AU'],
                        'first_months': {'UK': 200411, 'US': 201407, 'AU': 201709}, 'cutoff_month': 202109,
                        'pairs': ['US_UK', 'US_AU', 'UK_AU'], 'inference_pairs': ['US_UK'],
                        'v1_tol': 1e-4, 'h3_distances_tol': 1e-10}}
Y_DIM = 768
CHUNK = 100                                # draws per matrix product (the draw order does not depend on it)
SPACE_NAMES = {'own_pca88': 'E0 自身 PCA 前 88 个主成分（H1 所用空间；主规格）',
               'ref_pc1': '参考轴 PC1（单维）',
               'ref_pc1_3': '参考轴 PC1–PC3',
               'ref_88': '参考轴前 88 维',
               'y768': 'E0 全空间：Y 768 维（原稿口径）',
               'y768_v1_sample': 'E0 全空间 768 维，原稿样本口径（全样本）'}
E0_SPACES = ('own_pca88', 'ref_pc1', 'ref_pc1_3', 'ref_88', 'y768')
E1E4_SPACES = ('ref_88',)
READINGS = {'widened': '签署后美英在该空间中的距离扩大（偏差校正）', 'not_detected': '未检出距离变化',
            'narrowed': '距离缩小', 'not_applicable': '判读不适用'}
BIAS_CORRECTION = '近似偏差校正（CR1 迹校正；文档规模不等时存在小的残余偏差）'
V1_DESCRIPTIVE_NOTE = '只作描述（澳大利亚签署前样本不足），不作推断与判读'
CENTRING_NOTE = ('诊断（不改变读法）：自助总体为各格的文档经验分布，其美英均值距离即未校正距离；每次抽样重算 V̂ 的近似偏差校正'
                 '统计量以该总体值为中心，故 D*、Δ* 的自助分布约以未校正值为中心，而非以 D̂、Δ̂ 为中心。每期 D* 约上移 '
                 'tr V̂/(2D)（tr V̂ 为两国之和；tr V̂ 远小于 D² 时）；Δ* 约移动两期上移之差，朝 tr V̂/D 较大的时期。'
                 '枢轴区间以该总体值（未校正平方距离之差）为中心')


class HardCheckError(RuntimeError):
    pass


# =========================================================================== pure computation
def signed_root(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def doc_sums(Z, doc_ids) -> tuple:
    """(documents in ``numpy.unique`` order, rows per document, column sums of ``Z`` per document)."""
    docs, inverse = np.unique(np.asarray(doc_ids), return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    counts = np.bincount(inverse, minlength=len(docs))
    order = np.argsort(inverse, kind='stable')
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    sums = np.add.reduceat(np.asarray(Z, dtype=np.float64)[order], starts, axis=0)
    return docs, counts.astype(np.float64), sums


def cell_moments(Z, doc_ids) -> dict:
    """One country x period cell: n, G, the occurrence mean, the per-document sums s_d of (y_i - mean), and the
    diagonal of the document-clustered covariance of the mean, V = G/(G-1) (1/n^2) sum_d s_d s_d'."""
    Z = np.asarray(Z, dtype=np.float64)
    n = len(Z)
    mean = Z.mean(axis=0)
    docs, counts, sums = doc_sums(Z - mean, doc_ids)
    G = len(docs)
    if G < 2:
        raise ValueError(f'{G} document(s) in a cell: the clustered covariance needs at least 2')
    return {'n': int(n), 'G': int(G), 'mean': mean, 'counts': counts, 'sums': sums,
            'var_diag': G / (G - 1) * (sums ** 2).sum(axis=0) / n ** 2}


def cell_moments_from_doc_sums(counts, raw_sums) -> dict:
    """The same moments from per-document row counts and raw column sums (used by the synthetic calibration)."""
    counts = np.asarray(counts, dtype=np.float64)
    raw_sums = np.asarray(raw_sums, dtype=np.float64)
    n, G = float(counts.sum()), len(counts)
    if G < 2:
        raise ValueError(f'{G} document(s) in a cell: the clustered covariance needs at least 2')
    mean = raw_sums.sum(axis=0) / n
    sums = raw_sums - counts[:, None] * mean
    return {'n': int(round(n)), 'G': int(G), 'mean': mean, 'counts': counts, 'sums': sums,
            'var_diag': G / (G - 1) * (sums ** 2).sum(axis=0) / n ** 2}


def distance(uk: dict, us: dict, cols) -> dict:
    """Bias-corrected distance of two cell means on the columns ``cols`` (a slice)."""
    gap = uk['mean'][cols] - us['mean'][cols]
    raw2 = float(gap @ gap)
    tr_uk, tr_us = float(uk['var_diag'][cols].sum()), float(us['var_diag'][cols].sum())
    d2 = raw2 - tr_uk - tr_us
    return {'D': float(signed_root(d2)), 'D2': d2, 'uncorrected': float(np.sqrt(raw2)), 'uncorrected_sq': raw2,
            'trace_V_UK': tr_uk, 'trace_V_US': tr_us, 'gap': gap}


def draws(G_by_cell: dict, B: int, seed: int):
    """(b, cell, document indices) in the fixed order: b = 0..B-1, then the cells in :data:`CELLS` order; each draw
    ``rng.integers(0, G, size=G)`` of ``numpy.random.default_rng(seed)``."""
    rng = np.random.default_rng(seed)
    for b in range(B):
        for cell in CELLS:
            G = G_by_cell[cell]
            yield b, cell, rng.integers(0, G, size=G)


def replicate_moments(mom: dict, W) -> tuple:
    """Cell means and clustered-variance diagonals of the resampled cells given by the rows of ``W`` (how often each
    document was drawn; every row sums to G). A document drawn twice counts as two clusters."""
    W = np.asarray(W, dtype=np.float64)
    G, n_d, T = mom['G'], mom['counts'], mom['sums']
    k = T.shape[1]
    P = W @ np.hstack([T ** 2, n_d[:, None] * T, T])
    n_star = W @ n_d
    c = P[:, 2 * k:] / n_star[:, None]                    # resampled mean - cell mean (the sums are centred)
    Q = P[:, :k] - 2 * c * P[:, k:2 * k] + c ** 2 * (W @ n_d ** 2)[:, None]
    return mom['mean'][None, :] + c, G / (G - 1) * Q / n_star[:, None] ** 2


def draw_weights(G_by_cell: dict, B: int, seed: int, digest=None) -> dict:
    """cell -> (B, G) matrix of how often each document was drawn in each resample (the order of :func:`draws`)."""
    W = {c: np.zeros((B, G_by_cell[c])) for c in CELLS}
    for b, cell, idx in draws(G_by_cell, B, seed):
        if len(idx) != G_by_cell[cell]:
            raise AssertionError('a draw changed the number of documents of its cell')
        W[cell][b] = np.bincount(idx, minlength=G_by_cell[cell])
        if digest is not None:
            digest.update(np.ascontiguousarray(idx, dtype='<i8').tobytes())
    return W


def bootstrap(moments: dict, B: int, seed: int) -> dict:
    """Means and variance diagonals of every cell in B resamples (documents with replacement within each cell)."""
    G = {c: moments[c]['G'] for c in CELLS}
    k = len(moments[CELLS[0]]['mean'])
    means = {c: np.empty((B, k)) for c in CELLS}
    var = {c: np.empty((B, k)) for c in CELLS}
    digest = hashlib.sha256()
    sequence = draws(G, B, seed)
    for start in range(0, B, CHUNK):
        stop = min(start + CHUNK, B)
        W = {c: np.zeros((stop - start, G[c])) for c in CELLS}
        for _ in range((stop - start) * len(CELLS)):
            b, cell, idx = next(sequence)
            if len(idx) != G[cell]:
                raise AssertionError('a draw changed the number of documents of its cell')
            W[cell][b - start] = np.bincount(idx, minlength=G[cell])
            digest.update(np.ascontiguousarray(idx, dtype='<i8').tobytes())
        for c in CELLS:
            means[c][start:stop], var[c][start:stop] = replicate_moments(moments[c], W[c])
    return {'means': means, 'var_diag': var, 'draws_sha256': digest.hexdigest(), 'documents_per_draw': G}


def bootstrap_from_weights(moments: dict, W: dict) -> dict:
    means, var = {}, {}
    for c in CELLS:
        means[c], var[c] = replicate_moments(moments[c], W[c])
    return {'means': means, 'var_diag': var}


def replicate_squared(boot: dict, cols) -> dict:
    """Bias-corrected squared distances D2*_t of every resample (V recomputed in every draw) and delta_sq*."""
    out = {}
    for t in ('pre', 'post'):
        gap = boot['means'][f'UK_{t}'][:, cols] - boot['means'][f'US_{t}'][:, cols]
        out[t] = (gap ** 2).sum(axis=1) - boot['var_diag'][f'UK_{t}'][:, cols].sum(axis=1) \
            - boot['var_diag'][f'US_{t}'][:, cols].sum(axis=1)
    out['delta_sq'] = out['post'] - out['pre']
    return out


def replicate_distances(boot: dict, cols) -> dict:
    sq = replicate_squared(boot, cols)
    out = {t: signed_root(sq[t]) for t in ('pre', 'post')}
    out['delta'] = out['post'] - out['pre']
    return out


def quantiles(values) -> tuple:
    lo, hi = np.quantile(np.asarray(values, dtype=np.float64), PARAMS['interval_quantiles'],
                         method=PARAMS['quantile_method'])
    return float(lo), float(hi)


def pivot(estimate: float, population: float, replicates) -> dict:
    """The bootstrap errors e*_b = replicate_b - population and the pivot interval
    [estimate - q_.975(e*), estimate - q_.025(e*)]."""
    errors = np.asarray(replicates, dtype=np.float64) - population
    q_lo, q_hi = quantiles(errors)
    return {'interval_95': [estimate - q_hi, estimate - q_lo], 'errors': errors}


def pivot_p(estimate: float, errors) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    B = len(errors)
    k = int(np.sum(np.abs(errors) >= abs(estimate)))
    p = (1 + k) / (B + 1)
    return {'p': p, 'tail_count': k, 'B': B, 'se_mc': float(np.sqrt(p * (1 - p) / B)),
            'rule': '(1 + #{|e*_b| >= |delta_sq|}) / (B + 1), e*_b = delta_sq*_b - delta_sq_uncorrected; '
                    'se_MC = sqrt(p (1 - p) / B)'}


def reading(estimate: float, interval) -> dict:
    """The pre-stated mechanical reading by the sign of delta_sq and its interval (the p value does not enter)."""
    lo, hi = interval
    if lo <= 0 <= hi:
        key = 'not_detected'
    elif estimate > 0 and lo > 0:
        key = 'widened'
    elif estimate < 0 and hi < 0:
        key = 'narrowed'
    else:
        key = 'not_applicable'
    return {'key': key, 'text': READINGS[key]}


def centring(point: dict, sqrt_reps: dict, sq_reps: dict, delta_sq: float, delta_sq_population: float) -> dict:
    """Where the bootstrap distributions sit (a diagnostic; no reading)."""
    out = {'note': CENTRING_NOTE}
    for t in ('pre', 'post'):
        out[t] = {'mean_D_star': float(np.mean(sqrt_reps[t])), 'median_D_star': float(np.median(sqrt_reps[t])),
                  'D_estimate': point[t]['D'], 'D_bootstrap_population': point[t]['uncorrected'],
                  'trace_V_sum': point[t]['trace_V_UK'] + point[t]['trace_V_US']}
    delta = point['post']['D'] - point['pre']['D']
    population = point['post']['uncorrected'] - point['pre']['uncorrected']
    out['delta'] = {'mean_star': float(np.mean(sqrt_reps['delta'])), 'median_star': float(np.median(sqrt_reps['delta'])),
                    'estimate': delta, 'bootstrap_population': population,
                    'mean_star_minus_estimate': float(np.mean(sqrt_reps['delta'])) - delta}
    mean_sq = float(np.mean(sq_reps['delta_sq']))
    out['delta_sq'] = {'mean_star': mean_sq, 'median_star': float(np.median(sq_reps['delta_sq'])),
                       'estimate': delta_sq, 'bootstrap_population': delta_sq_population,
                       'mean_star_minus_estimate': mean_sq - delta_sq,
                       'mean_star_minus_population': mean_sq - delta_sq_population}
    return out


def space_result(space: str, moments: dict, boot: dict, cols) -> dict:
    """One space: delta_sq with its pivot interval, p and reading; the per-period D2 with the same pivot
    construction; the square-root and uncorrected forms as description."""
    width = len(range(*cols.indices(len(moments[CELLS[0]]['mean']))))
    point = {t: distance(moments[f'UK_{t}'], moments[f'US_{t}'], cols) for t in ('pre', 'post')}
    sq = replicate_squared(boot, cols)
    estimate = point['post']['D2'] - point['pre']['D2']
    population = point['post']['uncorrected_sq'] - point['pre']['uncorrected_sq']
    piv = pivot(estimate, population, sq['delta_sq'])
    out = {'name': SPACE_NAMES[space], 'dim': width}
    for t in ('pre', 'post'):
        out[t] = {**{k: v for k, v in point[t].items() if k != 'gap'},
                  'D2_interval_95': pivot(point[t]['D2'], point[t]['uncorrected_sq'], sq[t])['interval_95']}
    out['delta_sq'] = {'estimate': estimate, 'interval_95': piv['interval_95'], 'uncorrected': population,
                       **pivot_p(estimate, piv['errors']),
                       'interval_rule': 'pivot: [delta_sq - q_.975(e*), delta_sq - q_.025(e*)], '
                                        'e*_b = delta_sq*_b - delta_sq_uncorrected'}
    out['descriptive'] = {'D_pre': point['pre']['D'], 'D_post': point['post']['D'],
                          'delta_D': point['post']['D'] - point['pre']['D'],
                          'note': 'square-root forms (D = sign(D2) sqrt|D2|): description only'}
    out['uncorrected'] = {'pre': point['pre']['uncorrected'], 'post': point['post']['uncorrected'],
                          'change': point['post']['uncorrected'] - point['pre']['uncorrected']}
    out['reading'] = reading(estimate, piv['interval_95'])
    out['bootstrap_centring'] = centring(point, replicate_distances(boot, cols), sq, estimate, population)
    if width == 1:
        g = {t: float(point[t]['gap'][0]) for t in ('pre', 'post')}
        out['signed_gap'] = {'pre': g['pre'], 'post': g['post'], 'abs_change': abs(g['post']) - abs(g['pre']),
                             'definition': 'g_t = m_UK,t - m_US,t on the reference PC1; |g_post| - |g_pre|'}
    return out


def analyse(Z, cells, doc_ids, spaces: dict, B: int, seed: int) -> dict:
    cells, doc_ids = np.asarray(cells), np.asarray(doc_ids)
    moments = {c: cell_moments(Z[cells == c], doc_ids[cells == c]) for c in CELLS}
    boot = bootstrap(moments, B, seed)
    return {'cells': {c: {'n': moments[c]['n'], 'documents': moments[c]['G']} for c in CELLS},
            'bootstrap': bootstrap_record(boot, B, seed),
            'spaces': {s: space_result(s, moments, boot, cols) for s, cols in spaces.items()}}


def bootstrap_record(boot: dict, B: int, seed: int) -> dict:
    return {'B': int(B), 'seed': int(seed), 'generator': PARAMS['generator'], 'numpy_version': np.__version__,
            'draws_sha256': boot['draws_sha256'], 'documents_per_draw': boot['documents_per_draw'],
            'order': ('for b = 0..B-1, for cell in (US_pre, US_post, UK_pre, UK_post): '
                      'rng.integers(0, G_cell, size=G_cell) over the documents of the cell in numpy.unique(doc_id) '
                      'order; one rng = numpy.random.default_rng(seed); the clustered covariance is recomputed in '
                      'every draw (a document drawn twice is two clusters)')}


def pair_description(first: dict, second: dict) -> dict:
    """A descriptive pair of (f): per period D2, D and the uncorrected distance, and their changes."""
    point = {t: distance(second[t], first[t], slice(None)) for t in ('pre', 'post')}
    out = {t: {k: v for k, v in point[t].items() if k != 'gap'} for t in ('pre', 'post')}
    out['delta_sq'] = point['post']['D2'] - point['pre']['D2']
    out['delta_D'] = point['post']['D'] - point['pre']['D']
    out['uncorrected'] = {'pre': point['pre']['uncorrected'], 'post': point['post']['uncorrected'],
                          'change': point['post']['uncorrected'] - point['pre']['uncorrected'],
                          'change_percent': (point['post']['uncorrected'] - point['pre']['uncorrected'])
                          / point['pre']['uncorrected'] * 100}
    return out


def check_item(value: float, reference, tol: float) -> dict:
    diff = None if reference is None else abs(float(value) - float(reference))
    return {'value': float(value), 'reference': None if reference is None else float(reference),
            'abs_diff': diff, 'tol': tol, 'pass': diff is not None and diff <= tol}


def v1_block(Y, meta, did_h3: dict, v1_distances: dict, B: int, seed: int) -> dict:
    """(f): the full 768-dimensional space on the original manuscript's own sample (all rows) and its three pairs.
    US-UK: statistic, pivot bootstrap and reading; US-AU, UK-AU: description only. Hard checks: the uncorrected
    distances equal the neighbour-word analysis's (``did_h3``, the same reconstruction on the same Y) within 1e-10 and
    the original manuscript's (``v1_distances``) within 1e-4, all pairs and periods; first months as stated; every row
    in the sample."""
    spec = PARAMS['v1_sample']
    smp = v1_sample(meta)
    if smp['problems']:
        raise HardCheckError('(f) sample: ' + '; '.join(smp['problems']))
    Z = np.asarray(Y, dtype=np.float64)
    mask = smp['mask']
    cells, docs, Zs = smp['cells'][mask], smp['docs'][mask], Z[mask]
    moments = {c: cell_moments(Zs[cells == c], docs[cells == c]) for c in V1_CELLS}
    by_country = {c: {t: moments[f'{c}_{t}'] for t in ('pre', 'post')} for c in spec['countries']}
    pairs = {name: pair_description(by_country[a], by_country[b]) for name, a, b in V1_PAIRS}
    reference = did_h3.get('pairwise_distances') or {}
    reproduction, checks = {}, {}
    for name, _, _ in V1_PAIRS:
        reproduction[name] = {}
        for t in ('pre', 'post'):
            mine = pairs[name]['uncorrected'][t]
            ref = (reference.get(name) or {}).get(t)
            v1 = v1_distances[name][t]
            reproduction[name][t] = {'value': mine, 'v1': v1, 'diff_v1': mine - v1, 'h3_distances': ref,
                                     'diff_h3_distances': None if ref is None else mine - ref}
            checks[f'{name}_{t}_uncorrected_equals_h3_distances_json'] = check_item(
                mine, ref, spec['h3_distances_tol'])
            checks[f'{name}_{t}_uncorrected_reproduces_v1'] = check_item(mine, v1, spec['v1_tol'])
        v1_change = (v1_distances[name]['post'] - v1_distances[name]['pre']) / v1_distances[name]['pre'] * 100
        reproduction[name]['change_percent'] = {'value': pairs[name]['uncorrected']['change_percent'], 'v1': v1_change}
    checks['first_months_as_stated'] = {'value': smp['counts']['first_month'], 'reference': spec['first_months'],
                                        'pass': smp['counts']['first_month'] == spec['first_months']}
    checks['all_rows_in_sample'] = {'value': smp['counts']['rows_in_sample'], 'reference': smp['counts']['rows_total'],
                                    'pass': smp['counts']['rows_in_sample'] == smp['counts']['rows_total']}
    failed = [k for k, v in checks.items() if not v['pass']]
    if failed:
        raise HardCheckError(f'hard checks failed: {failed}')
    usuk = {c: moments[c] for c in CELLS}
    boot = bootstrap(usuk, B, seed)
    return {'name': SPACE_NAMES['y768_v1_sample'],
            'definition': ('all rows of the baseline analysis (UK from 2004-11, US from 2014-07, AU from 2017-09); '
                           'post = month code >= 202109; the baseline Y, 768 dimensions; country means as in the '
                           'original manuscript'),
            'sample': smp['counts'], 'cells': {c: {'n': moments[c]['n'], 'documents': moments[c]['G']} for c in V1_CELLS},
            'bootstrap': bootstrap_record(boot, B, seed),
            'US_UK': space_result('y768_v1_sample', usuk, boot, slice(0, Z.shape[1])),
            'pairs': {name: {**pairs[name], 'note': None if name in spec['inference_pairs'] else V1_DESCRIPTIVE_NOTE}
                      for name, _, _ in V1_PAIRS},
            'v1_reproduction': {'source': 'original_outputs/did_h3_verification.json (computed_results.pairwise_distances)',
                                'pairs': reproduction}, 'checks': checks}


def define_cells(meta, countries, first=None, last=None) -> dict:
    """Rows of ``countries`` with month code in [first, last] (None: unbounded), their country x period cell and the
    structural problems."""
    import pandas as pd
    country = meta['country'].to_numpy().astype(str)
    year, month = meta['year'].to_numpy().astype(np.int64), meta['month'].to_numpy().astype(np.int64)
    code = year * 100 + month
    problems = []
    if not ((month >= 1) & (month <= 12)).all():
        problems.append(f'{int(((month < 1) | (month > 12)).sum())} rows with a month outside 1..12')
    chosen = np.isin(country, list(countries))
    mask = chosen & (code >= (first if first is not None else code.min())) \
        & (code <= (last if last is not None else code.max()))
    cut = PARAMS['cutoff_month']
    post = code >= cut
    flag = meta['post_aukus'].to_numpy().astype(bool)
    disagree = int((mask & (post != flag)).sum())
    if disagree:
        problems.append(f'{disagree} sample rows whose post_aukus differs from month code >= {cut}')
    cells = np.char.add(country.astype(str), np.where(post, '_post', '_pre'))
    cells = np.where(mask, cells, '').astype(str)
    docs = np.asarray(meta['doc_id'].to_numpy()).astype(str)
    nested = pd.DataFrame({'d': docs[mask], 'c': cells[mask]}).drop_duplicates().groupby('d').size()
    if (nested > 1).any():
        problems.append(f'{int((nested > 1).sum())} documents in more than one country x period cell')
    return {'mask': mask, 'cells': cells, 'docs': docs, 'problems': problems, 'code': code, 'country': country,
            'chosen': chosen}


def _cell_sizes(base: dict, cells) -> list:
    out = []
    for c in cells:
        G = len(np.unique(base['docs'][base['cells'] == c]))
        if G < 2:
            out.append(f'cell {c}: {G} document(s); at least 2 are needed')
    return out


def sample(meta) -> dict:
    first, last = PARAMS['first_month'], PARAMS['last_month']
    base = define_cells(meta, PARAMS['countries'], first, last)
    pair, code = base['chosen'], base['code']
    return {'mask': base['mask'], 'cells': base['cells'], 'docs': base['docs'],
            'problems': base['problems'] + _cell_sizes(base, CELLS),
            'counts': {'rows_total': int(len(meta)), 'rows_in_sample': int(base['mask'].sum()),
                       'excluded_other_countries': int((~pair).sum()),
                       'excluded_before_first_month': int((pair & (code < first)).sum()),
                       'excluded_after_last_month': int((pair & (code > last)).sum())}}


def v1_sample(meta) -> dict:
    spec = PARAMS['v1_sample']
    base = define_cells(meta, spec['countries'])
    firsts = {c: int(base['code'][base['country'] == c].min()) for c in spec['countries']
              if (base['country'] == c).any()}
    lasts = {c: int(base['code'][base['country'] == c].max()) for c in spec['countries']
             if (base['country'] == c).any()}
    return {'mask': base['mask'], 'cells': base['cells'], 'docs': base['docs'],
            'problems': base['problems'] + _cell_sizes(base, V1_CELLS),
            'counts': {'rows_total': int(len(meta)), 'rows_in_sample': int(base['mask'].sum()),
                       'first_month': firsts, 'last_month': lasts}}


def uk_x_post(pc1_scores, meta) -> float:
    """The main-model UK x Post OLS coefficient on one score."""
    X, names = analysis.wcb_features(meta)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ np.asarray(pc1_scores, dtype=np.float64)
    return float(beta[names.index('UK_x_post')])


def space_matrix(baseline: bool, Y, axes: dict) -> tuple:
    """(Z of all rows, space -> column slice, own PCA or None): baseline [own PCA-88 | reference 88 | Y (768)],
    alternative encoders [reference 88]."""
    k = PARAMS['k']
    ref = pca.scores(Y, axes['mean'], axes['components'][:k])
    if not baseline:
        return ref, {'ref_88': slice(0, k)}, None
    model = pca.fit_pca(Y, k)
    own = model.transform(Y)
    Y = np.asarray(Y, dtype=np.float64)
    return (np.hstack([own, ref, Y]),
            {'own_pca88': slice(0, k), 'ref_pc1': slice(k, k + 1), 'ref_pc1_3': slice(k, k + 3),
             'ref_88': slice(k, 2 * k), 'y768': slice(2 * k, 2 * k + Y.shape[1])}, model)


def hard_checks(baseline: bool, Z, spaces: dict, model, meta, reference: dict) -> dict:
    """Checks against the H1 / H2 / same-axis results of the same Y (a failure stops the script)."""
    tol = PARAMS['hard_check_tol']
    n = len(meta)
    want = {'wcb n_samples': reference['wcb'].get('n_samples'), 'rows expected': reference['rows']}
    out = {'rows_equal_analysis': {'value': n, 'reference': want,
                                              'pass': all(v == n for v in want.values())}}
    ref_pc1 = Z[:, spaces['ref_88'].start]
    out['reference_pc1_uk_x_post_equals_same_axis_json'] = check_item(
        uk_x_post(ref_pc1, meta), ((reference['same_axis'].get('UK_x_post') or {}).get('PC1') or {}).get('coef'), tol)
    if baseline:
        out['own_pca88_pc1_uk_x_post_equals_wcb_json'] = check_item(
            uk_x_post(Z[:, 0], meta),
            ((reference['wcb'].get('comparison') or {}).get('PC1') or {}).get('UK_x_post', {}).get('coef'), tol)
        out['own_pca88_cumulative_variance_equals_h1_json'] = check_item(
            float(model.explained_variance_ratio_.sum()), reference['h1'].get('cumulative_variance_88'), tol)
        width = spaces['y768'].stop - spaces['y768'].start
        out['y768_has_768_dimensions'] = {'value': int(width), 'reference': Y_DIM, 'pass': int(width) == Y_DIM}
    return out


def compute(baseline: bool, Y, meta, axes: dict, reference: dict, v1_distances: dict | None,
            B: int = PARAMS['B'], seed: int = PARAMS['seed']) -> dict:
    """The full result of one encoder (stops on a failed hard check or a structural problem)."""
    smp = sample(meta)
    if smp['problems']:
        raise HardCheckError('; '.join(smp['problems']))
    Z, spaces, model = space_matrix(baseline, Y, axes)
    checks = hard_checks(baseline, Z, spaces, model, meta, reference)
    failed = [name for name, item in checks.items() if not item['pass']]
    if failed:
        raise HardCheckError(f'hard checks failed: {failed}')
    mask = smp['mask']
    out = analyse(Z[mask], smp['cells'][mask], smp['docs'][mask], spaces, B, seed)
    out['sample'] = smp['counts']
    out['checks'] = checks
    if baseline:
        out['v1_sample'] = v1_block(Y, meta, reference['did_h3_distances'], v1_distances, B, seed)
    if model is not None:
        out['own_pca'] = {'function': 'replication.pca.fit_pca', 'n_components': int(model.n_components_),
                          'svd_solver': model.svd_solver, 'n_samples': int(model.n_samples_),
                          'cumulative_variance': float(model.explained_variance_ratio_.sum())}
    return out


def summary_rows(results: list) -> list:
    """One row per encoder and space (the combined table)."""
    return [{'encoder': r['encoder'], 'code': r['code'], 'space': space, 'name': s['name'], 'cells': r['cells'],
             'D2_pre': s['pre']['D2'], 'D2_pre_interval_95': s['pre']['D2_interval_95'],
             'D2_post': s['post']['D2'], 'D2_post_interval_95': s['post']['D2_interval_95'],
             'delta_sq': s['delta_sq']['estimate'], 'delta_sq_interval_95': s['delta_sq']['interval_95'],
             'p': s['delta_sq']['p'], 'se_mc': s['delta_sq']['se_mc'], 'reading': s['reading'],
             'descriptive': s['descriptive'], 'uncorrected': s['uncorrected']}
            for r in results for space, s in r['spaces'].items()]
