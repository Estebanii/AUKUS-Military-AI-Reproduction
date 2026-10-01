#!/usr/bin/env python3
"""Supplementary (optional; not needed for any printed number): synthetic calibration of the distance-change interval
and reading of Table 7.

Only synthetic data. The one real input is the cell structure of the main distance sample: the documents of the four
cells US/UK x pre/post (2014-01 to 2025-11) and their occurrence counts, from the occurrence metadata (no vector is
read). Every simulated data set has exactly these documents and counts and is analysed by the code path of Table 7
(:func:`replication.distance.space_result` on :func:`replication.distance.cell_moments_from_doc_sums`) with the same
B = 2,000 bootstrap draws (seed ``seeds.DISTANCE_SEED``; the draw matrices depend only on the cell sizes and the seed).

Model per cell (country c, period t): y_i = mu_ct + u_d + e_i with a document effect u_d ~ N(0, rho Lambda) and an
occurrence term e_i ~ N(0, (1 - rho) Lambda), tr Lambda = 1; the document sums are drawn directly:
T_d = n_d mu_ct + sqrt(n_d^2 rho + n_d (1 - rho)) Lambda^(1/2) z_d. US means are 0; the UK mean is a_t v (v a fixed
random unit vector), so the true squared distance is D2_t = a_t^2 and the true delta_sq = D2_post - D2_pre.
ratio = tr V_post / D2 (both countries), the noise-to-signal scale of a setting. Baseline: rho = 0.3 and a decaying
spectrum lambda_j ~ 1/(j + 5); sensitivity: rho = 0.7, and an isotropic spectrum.

Settings: true delta_sq = 0 at ratios 10, 3, 1, 0.3, 0.1, 0.03, 0.01 (baseline) and at 3 and 0.3 (each sensitivity);
then delta_sq = +/- 1, 2, 3 x SD0 at ratios 3, 1 and 0.1 (baseline; SD0 = the SD of the estimate in the delta_sq = 0
setting of that ratio). Reported per setting: the reading rates ("扩大", "缩小", "未检出", "判读不适用"), 95% coverage
of the true delta_sq, power, the p <= .05 rate, bias and SD of the estimate, with Monte Carlo SEs. Stop rule: any
delta_sq = 0 setting whose misreading rate ("扩大" + "缩小") exceeds 0.10.

Variants: ``--k 88`` (default; the space of the main specification), ``--k 1`` / ``--k 3`` (single axis, PC1-PC3),
``--variant e`` (768 dimensions on the main-sample cells), ``--variant f`` (768 dimensions on the full-sample US/UK
cells of the original manuscript), ``--stress`` (true D = 0 exactly and the near-zero ratios 30 and 100; k = 1, 3, 88).
Data seeds: ``numpy.random.SeedSequence([seeds.CALIBRATION_DATA_SEED, setting index])``. Runtime with 4 workers:
k = 88 about 12 minutes; k = 1 or 3 and --stress about 1-2 minutes; e / f about 70-80 minutes.
Writes results/supplementary/distance_calibration{_k1,_k3,_e768,_f768,_stress}.{json,md}.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import _common  # noqa: E402,F401

import numpy as np  # noqa: E402

from replication import data, output, seeds  # noqa: E402
from replication import distance as pdist  # noqa: E402

K = 88
DATA_SEED = seeds.CALIBRATION_DATA_SEED       # data-generating seeds: SeedSequence([DATA_SEED, setting index])
ZERO_RATIOS = (10, 3, 1, 0.3, 0.1, 0.03, 0.01)
SENSITIVITY_RATIOS = (3, 0.3)
NONZERO_RATIOS = (3, 1, 0.1)
MULTIPLES = (-3, -2, -1, 1, 2, 3)
STOP_RATE = 0.10
STRESS_DIMS = (1, 3, 88)
STRESS_RATIOS = (None, 30, 100)          # None: true D = 0 exactly


# =========================================================================== the real cell structure (metadata)
def cell_structure(variant: str = 'main') -> dict:
    """cell -> occurrence counts of its documents (numpy.unique(doc_id) order), from the occurrence metadata;
    ``variant`` f: the original manuscript's full sample (US/UK)."""
    meta = data.analysis_meta()
    smp = pdist.v1_sample(meta) if variant == 'f' else pdist.sample(meta)
    if smp['problems']:
        raise SystemExit('; '.join(smp['problems']))
    out = {}
    for c in pdist.CELLS:
        docs = smp['docs'][smp['cells'] == c]
        _, counts = np.unique(docs, return_counts=True)
        out[c] = counts.astype(np.float64)
    return {'counts': out, 'source': {'file': 'corpus/occurrences.parquet (data bundle)',
                                      'columns_read': ['doc_id', 'country', 'year', 'month', 'post_aukus'],
                                      'sample_counts': smp['counts']}}


def describe(counts: dict) -> dict:
    out = {}
    for c, n_d in counts.items():
        q = np.quantile(n_d, [0.5, 0.9, 0.99])
        out[c] = {'occurrences': int(n_d.sum()), 'documents': int(len(n_d)), 'mean_per_document': float(n_d.mean()),
                  'median': float(q[0]), 'p90': float(q[1]), 'p99': float(q[2]), 'max': int(n_d.max()),
                  'share_single': float(np.mean(n_d == 1)), 'sum_n2': float((n_d ** 2).sum())}
    return out


# =========================================================================== model
def spectrum(kind: str, k: int | None = None) -> np.ndarray:
    k = K if k is None else k
    if kind == 'isotropic':
        return np.ones(k) / k
    head = 1.0 / (np.arange(1, min(k, 88) + 1) + 5.0)
    if k <= 88:
        return head / head.sum()
    return np.concatenate([0.99 * head / head.sum(), np.full(k - 88, 0.01 / (k - 88))])


def expected_trace(counts: dict, rho: float) -> dict:
    """The model's true variance trace of each cell mean (tr Lambda = 1), summed over the two countries; not the
    finite-sample E[V-hat] (the CR1 trace correction is only approximately bias-correcting)."""
    tr = {c: float(((n ** 2) * rho + n * (1 - rho)).sum() / n.sum() ** 2) for c, n in counts.items()}
    return {t: tr[f'US_{t}'] + tr[f'UK_{t}'] for t in ('pre', 'post')}


def direction(k: int | None = None) -> np.ndarray:
    """A fixed random unit vector in the first min(k, 88) dimensions."""
    k = K if k is None else k
    v = np.zeros(k)
    v[:min(k, 88)] = np.random.default_rng([DATA_SEED, seeds.CALIBRATION_DIRECTION_KEY]).normal(size=min(k, 88))
    return v / np.linalg.norm(v)


def simulate(rng, counts: dict, rho: float, lam: np.ndarray, a: dict, v: np.ndarray) -> dict:
    moments = {}
    root = np.sqrt(lam)
    k = len(lam)
    for c, n_d in counts.items():
        country, t = c.split('_')
        mu = a[t] * v if country == 'UK' else np.zeros(k)
        scale = np.sqrt(n_d ** 2 * rho + n_d * (1 - rho))
        sums = n_d[:, None] * mu + scale[:, None] * (rng.normal(size=(len(n_d), k)) * root)
        moments[c] = pdist.cell_moments_from_doc_sums(n_d, sums)
    return moments


# =========================================================================== one setting
_W = None


def _init(W):
    global _W
    _W = W


def run_setting(task: dict) -> dict:
    counts, reps = task['counts'], task['reps']
    k = task.get('k', K)
    lam, v = spectrum(task['spectrum'], k), direction(k)
    a = {'pre': np.sqrt(task['D2_pre']), 'post': np.sqrt(task['D2_post'])}
    truth = task['D2_post'] - task['D2_pre']
    rng = np.random.default_rng([DATA_SEED, task['index']])
    est, lo, hi, keys, ps = [], [], [], [], []
    for _ in range(reps):
        moments = simulate(rng, counts, task['rho'], lam, a, v)
        r = pdist.space_result('own_pca88', moments, pdist.bootstrap_from_weights(moments, _W), slice(0, k))
        assert r['dim'] == k
        d = r['delta_sq']
        est.append(d['estimate'])
        lo.append(d['interval_95'][0])
        hi.append(d['interval_95'][1])
        keys.append(r['reading']['key'])
        ps.append(d['p'])
    est, lo, hi, ps = map(np.asarray, (est, lo, hi, ps))
    keys = np.asarray(keys)

    def rate(mask):
        p = float(np.mean(mask))
        return {'rate': p, 'mc_se': float(np.sqrt(p * (1 - p) / reps))}
    out = {**{key: val for key, val in task.items() if key != 'counts'}, 'truth_delta_sq': truth,
           'widened': rate(keys == 'widened'), 'narrowed': rate(keys == 'narrowed'),
           'not_detected': rate(keys == 'not_detected'), 'not_applicable': rate(keys == 'not_applicable'),
           'coverage_95': rate((lo <= truth) & (truth <= hi)), 'p_le_05': rate(ps <= 0.05),
           'bias': float(est.mean() - truth), 'sd_estimate': float(est.std(ddof=1)),
           'mean_interval_width': float(np.mean(hi - lo))}
    out['misread'] = rate(np.isin(keys, ['widened', 'narrowed'])) if truth == 0 else \
        rate(keys == ('narrowed' if truth > 0 else 'widened'))
    out['power'] = None if truth == 0 else rate(keys == ('widened' if truth > 0 else 'narrowed'))
    return out


def tasks_zero(counts, reps) -> list:
    out = []
    for rho, spec, ratios, tag in ((0.3, 'decaying', ZERO_RATIOS, 'baseline'),
                                   (0.7, 'decaying', SENSITIVITY_RATIOS, 'rho=0.7'),
                                   (0.3, 'isotropic', SENSITIVITY_RATIOS, 'isotropic')):
        tr = expected_trace(counts, rho)
        for ratio in ratios:
            D2 = tr['post'] / ratio
            out.append({'group': tag, 'rho': rho, 'spectrum': spec, 'ratio': ratio, 'ratio_post': tr['post'] / D2,
                        'ratio_pre': tr['pre'] / D2, 'D2_pre': D2, 'D2_post': D2, 'multiple_of_sd0': 0,
                        'reps': reps, 'counts': counts})
    return out


def tasks_nonzero(counts, reps, zero_results) -> list:
    out = []
    tr = expected_trace(counts, 0.3)
    sd0 = {z['ratio']: z['sd_estimate'] for z in zero_results if z['group'] == 'baseline'}
    for ratio in NONZERO_RATIOS:
        D2 = tr['post'] / ratio
        for m in MULTIPLES:
            delta = m * sd0[ratio]
            pre, post = (D2, D2 + delta) if delta > 0 else (D2 - delta, D2)
            out.append({'group': 'baseline', 'rho': 0.3, 'spectrum': 'decaying', 'ratio': ratio,
                        'ratio_post': tr['post'] / post, 'ratio_pre': tr['pre'] / pre, 'D2_pre': pre, 'D2_post': post,
                        'multiple_of_sd0': m, 'sd0': sd0[ratio], 'reps': reps, 'counts': counts})
    return out


def tasks_stress(counts, reps) -> list:
    """Stress settings (true delta_sq = 0 throughout): D = 0 exactly and ratios 30, 100 (baseline model)."""
    tr = expected_trace(counts, 0.3)
    out = []
    for k in STRESS_DIMS:
        for ratio in STRESS_RATIOS:
            D2 = 0.0 if ratio is None else tr['post'] / ratio
            out.append({'group': f'stress k={k}', 'k': k, 'rho': 0.3, 'spectrum': 'decaying',
                        'ratio': 'D=0' if ratio is None else ratio,
                        'ratio_post': float('inf') if ratio is None else tr['post'] / D2,
                        'ratio_pre': float('inf') if ratio is None else tr['pre'] / D2,
                        'D2_pre': D2, 'D2_post': D2, 'multiple_of_sd0': 0, 'reps': reps, 'counts': counts})
    return out


def run_all(tasks, W, workers) -> list:
    if workers <= 1:
        _init(W)
        return [run_setting(t) for t in tasks]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, initializer=_init, initargs=(W,)) as pool:
        return pool.map(run_setting, tasks)


# =========================================================================== report
def table(results) -> list:
    lines = ['| 组 | 比值 | 实际 tr V/D² 前 / 后 | 真实 Δ_sq（×SD0） | 扩大 | 缩小 | 未检出 | 判读不适用 | 95% 覆盖率 | 功效 | '
             'p≤.05 | 偏差/SD |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        def pct(key):
            return f'{r[key]["rate"]:.3f}'
        power = '–' if r['power'] is None else f'{r["power"]["rate"]:.3f}'
        lines.append(f'| {r["group"]} | {r["ratio"]:g} | {r["ratio_pre"]:.3g} / {r["ratio_post"]:.3g} | '
                     f'{r["multiple_of_sd0"]:+d} | '
                     f'{pct("widened")} | {pct("narrowed")} | {pct("not_detected")} | {pct("not_applicable")} | '
                     f'{pct("coverage_95")} | {power} | {pct("p_le_05")} | {r["bias"] / r["sd_estimate"]:+.3f} |')
    return lines


def stress_table(results) -> list:
    lines = ['| 维数 k | 设定 | 实际 tr V/D² 前 / 后 | 扩大 | 缩小 | 未检出 | 判读不适用 | 误读合计（扩大 + 缩小） | 95% 覆盖率 | p≤.05 |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        setting = '真实 D = 0' if r['ratio'] == 'D=0' else f'比值 {r["ratio"]:g}'
        ratios = '∞ / ∞' if r['ratio'] == 'D=0' else f'{r["ratio_pre"]:.3g} / {r["ratio_post"]:.3g}'
        lines.append(f'| {r["k"]} | {setting} | {ratios} | {r["widened"]["rate"]:.3f} | {r["narrowed"]["rate"]:.3f} | '
                     f'{r["not_detected"]["rate"]:.3f} | {r["not_applicable"]["rate"]:.3f} | {r["misread"]["rate"]:.3f} | '
                     f'{r["coverage_95"]["rate"]:.3f} | {r["p_le_05"]["rate"]:.3f} |')
    return lines


def weights(counts: dict) -> tuple:
    G = {c: len(n) for c, n in counts.items()}
    digest = hashlib.sha256()
    W = pdist.draw_weights(G, pdist.PARAMS['B'], pdist.PARAMS['seed'], digest)
    return W, digest.hexdigest()


def write(stem: str, record: dict, text: str) -> None:
    output.write_json(f'supplementary/{stem}.json', record)
    output.write_text(f'supplementary/{stem}.md', text)
    print(f'[calibration] -> results/supplementary/{stem}.json, .md', flush=True)


def run_stress(args) -> int:
    started = time.time()
    structure = cell_structure('main')
    counts = structure['counts']
    W, draws_sha = weights(counts)
    results = run_all([{**t, 'index': seeds.CALIBRATION_STRESS_INDEX_OFFSET + i}
                       for i, t in enumerate(tasks_stress(counts, args.reps))], W,
                      args.workers)
    worst = max(results, key=lambda r: r['misread']['rate'])
    stop = worst['misread']['rate'] > STOP_RATE
    record = {'purpose': 'stress calibration of the distance reading (descriptive; no method or reading change)',
              'dims': list(STRESS_DIMS), 'settings': ['D=0', 30, 100], 'truth_delta_sq': 0.0,
              'B': pdist.PARAMS['B'], 'bootstrap_seed': pdist.PARAMS['seed'], 'data_seed': DATA_SEED,
              'draws_sha256': draws_sha, 'reps_per_setting': args.reps, 'model': 'baseline (rho 0.3, decaying)',
              'cell_structure': describe(counts), 'cell_source': structure['source'],
              'expected_trace_note': 'model true variance trace of the cell means, not the finite-sample E[V-hat]',
              'stop_rule': f'any D = 0 / near-zero setting with misreading rate > {STOP_RATE} -> STOP',
              'stop': bool(stop), 'worst': {k: worst[k] for k in ('group', 'ratio', 'misread')},
              'results': results, 'numpy_version': np.__version__, 'runtime_seconds': round(time.time() - started, 1)}
    lines = ['# 距离读法的压力校准（合成数据；描述性，不改方法与读法）', '',
             '- 真实 Δ_sq = 0；设定为真实 D = 0（两期美英均值相同）与趋零的比值 30、100（tr V_post/D²）；k = 1、3、88；'
             f'主规格样本的真实格规模；每设定 {args.reps} 次；B = {pdist.PARAMS["B"]}，自助种子 {pdist.PARAMS["seed"]}（与表 7 相同）；'
             '基线模型（ρ = 0.3，衰减谱）。',
             '- 比值中的 tr V 是模型下均值的真实方差迹，不是有限样本下的 E[V̂]（CR1 迹校正只是近似偏差校正）。',
             f'- 停止规则：任一设定的误读率（扩大 + 缩小）> {STOP_RATE} 即停止。结果：' + ('**停止**' if stop else '未触发')
             + f'（最大误读率 {worst["misread"]["rate"]:.3f}，{worst["group"]}，设定 {worst["ratio"]}）', '']
    lines += stress_table(results) + ['']
    text = '\n'.join(lines) + '\n'
    print(text)
    write('distance_calibration_stress', record, text)
    return 3 if stop else 0


def main(argv=None) -> int:
    global K, NONZERO_RATIOS
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--reps', type=int, default=seeds.CALIBRATION_REPS,
                    help=f'replications per setting (default and reference: {seeds.CALIBRATION_REPS})')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 3))
    ap.add_argument('--k', type=int, default=88, help='dimensions (88: the main specification; 1 or 3: supplement)')
    ap.add_argument('--variant', choices=['main', 'e', 'f'], default='main',
                    help='e = 768 dimensions on the main-sample cells, f = 768 dimensions on the full-sample cells')
    ap.add_argument('--nonzero-ratios', type=float, nargs='*', default=None,
                    help='ratios of the non-zero settings (default 3 1 0.1)')
    ap.add_argument('--stress', action='store_true', help='stress settings: D = 0, ratios 30 and 100; k 1/3/88')
    args = ap.parse_args(argv)
    if args.reps < 1:
        ap.error('--reps must be >= 1')
    if args.stress:
        return run_stress(args)
    K = 768 if args.variant in ('e', 'f') else args.k
    if args.nonzero_ratios is not None:
        NONZERO_RATIOS = tuple(args.nonzero_ratios)
    suffix = {'e': '_e768', 'f': '_f768'}.get(args.variant, '' if K == 88 else f'_k{K}')
    if args.reps != seeds.CALIBRATION_REPS:
        suffix += f'_reps{args.reps}'
        print(f'[calibration] note: {args.reps} replications (the reference calibration uses '
              f'{seeds.CALIBRATION_REPS})', flush=True)
    started = time.time()
    structure = cell_structure(args.variant)
    counts = structure['counts']
    W, draws_sha = weights(counts)
    zero = run_all([{**t, 'index': i} for i, t in enumerate(tasks_zero(counts, args.reps))], W, args.workers)
    nonzero_tasks = tasks_nonzero(counts, args.reps, zero)
    nonzero = run_all([{**t, 'index': seeds.CALIBRATION_NONZERO_INDEX_OFFSET + i} for i, t in enumerate(nonzero_tasks)],
                      W, args.workers)
    worst = max(zero, key=lambda r: r['misread']['rate'])
    stop = worst['misread']['rate'] > STOP_RATE
    record = {'purpose': 'synthetic calibration of the distance-change interval and reading',
              'variant': args.variant, 'k': K, 'nonzero_ratios': list(NONZERO_RATIOS),
              'B': pdist.PARAMS['B'], 'bootstrap_seed': pdist.PARAMS['seed'], 'data_seed': DATA_SEED,
              'draws_sha256': draws_sha, 'reps_per_setting': args.reps,
              'cell_structure': describe(counts), 'cell_source': structure['source'],
              'expected_trace': {'rho=0.3': expected_trace(counts, 0.3), 'rho=0.7': expected_trace(counts, 0.7)},
              'stop_rule': f'any delta_sq = 0 setting with misreading rate > {STOP_RATE} -> STOP',
              'stop': bool(stop), 'worst_zero_setting': {k: worst[k] for k in ('group', 'ratio', 'misread')},
              'results_zero': zero, 'results_nonzero': nonzero, 'numpy_version': np.__version__,
              'runtime_seconds': round(time.time() - started, 1)}
    title = {'e': '# 距离区间的合成校准 (e)：768 维，主规格样本的格',
             'f': '# 距离区间的合成校准 (f)：768 维，原稿全样本的美英格'}.get(args.variant) or (
        '# 距离区间的合成校准（88 维，主规格）' if K == 88 else f'# 距离区间的合成校准：补充 k = {K}')
    lines = [title, '',
             '- 格结构（只读元数据列）：' + '；'.join(f'{c} {d["documents"]} 篇 / {d["occurrences"]} 个出现点'
                                         for c, d in record['cell_structure'].items()),
             f'- 维数 {K}；每设定 {args.reps} 次；B = {pdist.PARAMS["B"]}，自助种子 {pdist.PARAMS["seed"]}（与表 7 相同的抽样）',
             f'- 停止规则：任一 Δ_sq = 0 设定的误读率（扩大 + 缩小）> {STOP_RATE} 即停止。结果：'
             + ('**停止**' if stop else '未触发') + f'（最大误读率 {worst["misread"]["rate"]:.3f}，'
               f'{worst["group"]}，比值 {worst["ratio"]:g}）', '',
             '## Δ_sq = 0', ''] + table(zero) + ['', '## Δ_sq ≠ 0（基线模型）', ''] + table(nonzero) + ['']
    text = '\n'.join(lines) + '\n'
    print(text)
    write(f'distance_calibration{suffix}', record, text)
    return 3 if stop else 0


if __name__ == '__main__':
    raise SystemExit(main())
