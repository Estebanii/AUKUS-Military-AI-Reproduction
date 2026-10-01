"""The event study (period coding 2014..2020, 2021a, 2021b, 2022..2025; base 2020), its sensitivity
tests (joint test, TOST, HonestDiD) and the placebos.

The event study is saturated in country x event period, so each coefficient's monthly scores sum to zero within its
period and the monthly-score HAC is biased towards zero (balanced iid: E(V)/V = 1 - T_tau^-2 sum_{t,u in tau}
k_b(t - u); b = 6: about 0.58 / 0.49 / 0.30 for 12 / 8 / 4 months). Hence:

* the event-study coefficients are descriptive point estimates;
* the Wald (joint), TOST and HonestDiD on the saturated event study are not valid inference; they are reported only
  as sensitivity under the estimators of :mod:`replication.external.escov` (``external_params('event_study')
  ['sensitivity_estimators']``): (a) the design-exact iid-corrected Bartlett HAC (iid-exact marginal variances) and
  (b) month-clustered CR2 with Bell-McCaffrey / Imbens-Kolesar degrees of freedom and the HTZ joint test
  (cross-month independence), each labelled with its assumption and the pre-specified simulation judgements;
* the low-dimensional pre-trend diagnostic (:mod:`replication.external.pretrend`) is the inference on pre-trends (linear only);
* no sampling-failure gate; structurally non-estimable or singular -> "不可用"; the month-block absence shares of the
  event periods are kept as the result-independent record of why the draws are not used here.

Implementation:

* the event design: event-period FE with 2020 as the reference, D_g x period, theta@t = UK - US;
* TOST by the item-wise interval rule;
* HonestDiD: the coefficients and the full covariance re-expressed relative to the last pre period 2021a
  (b' = L b, Sigma' = L Sigma L', :func:`renormalise`), the relative-magnitudes sensitivity through
  ``honest_rm.R`` (:func:`run_honestdid`, with a summary of its grid warnings), and a closed-form identified set as
  a cross-check (:func:`rm_identified_set`).

Design rules: the frozen pre sets and HonestDiD time grid per window (S_SG: 2014-2019 and 2021a; S_CA 2017+:
2017-2019 and 2021a); AU has coefficients only where it has observations. TOST: every pre coefficient's 90%
interval inside +/-0.10 SD (the locked SD). HonestDiD: M-bar in {0, 0.25, ..., 2}, l_vec = equal weights over
the post periods, alpha 0.05; any numerical failure is "不可用". Reported whatever they show; never used to change a
window or model.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import escov
from . import estimate as est
from . import inference as inf
from .params import MEMBERS, external_params

R_SCRIPT = Path(__file__).with_name('honest_rm.R')
FAMILIES = ('theta', 'delta_UK', 'delta_US')


def rscript() -> str:
    return shutil.which('Rscript') or '/usr/local/bin/Rscript'


def event_period(year, month) -> np.ndarray:
    year, month = np.asarray(year, int), np.asarray(month, int)
    out = year.astype(str).astype(object)
    in2021 = year == 2021
    out[in2021 & (month <= 8)] = '2021a'
    out[in2021 & (month >= 9)] = '2021b'
    return out.astype(str)


def period_order(periods) -> list:
    def key(p):
        return (int(p[:4]), p[4:])
    return sorted(set(periods), key=key)


def build_event_design(sub: pd.DataFrame, spec: est.Spec) -> est.Design:
    """intercept; D_US, D_UK, D_AU; period FE (reference 2020); D_g x period (every period except 2020)."""
    reference = external_params('event_study')['reference']
    n = len(sub)
    country = sub['country'].to_numpy()
    periods = event_period(sub['year'], sub['month'])
    present = period_order(periods)
    if reference not in present:
        raise ValueError('the event study needs the reference year 2020 in the sample')
    cols, names, groups = [np.ones(n)], ['intercept'], ['constant']
    for g in MEMBERS:
        cols.append((country == g).astype(float))
        names.append(f'D_{g}')
        groups.append('country levels')
    for p in present:
        if p == reference:
            continue
        cols.append((periods == p).astype(float))
        names.append(f'ev_{p}')
        groups.append('time terms')
    for g in MEMBERS:
        for p in present:
            if p == reference:
                continue
            cols.append((country == g).astype(float) * (periods == p))
            names.append(f'{g}_x_{p}')
            groups.append('event-period terms')
    X = np.column_stack(cols)

    def e(name):
        v = np.zeros(len(names))
        if name in names:
            v[names.index(name)] = 1.0
        return v
    contrasts = {}
    for p in present:
        if p == reference:
            continue
        contrasts[f'theta@{p}'] = e(f'UK_x_{p}') - e(f'US_x_{p}')
        for g in MEMBERS:
            contrasts[f'delta_{g}@{p}'] = e(f'{g}_x_{p}')
    return est.Design(X=X, names=names, groups=groups, contrasts=contrasts,
                      report={'model': 'event study', 'reference': reference, 'periods': present, 'n': n,
                              'base': '+'.join(spec.controls), 'p_full': X.shape[1]})


def frozen_sets(spec: est.Spec) -> tuple:
    """(sample, frozen pre periods, post periods) of the window."""
    es = external_params('event_study')
    sample = 'CA' if spec.sample == 'CA' else 'SG'
    pre_set = [p for p in es['pre_sets'][sample] if p >= spec.start[:4] or p == '2021a']
    post_set = [p for p in es['post'] if int(p[:4]) <= int(spec.end[:4])]
    return sample, pre_set, post_set


def event_design(panel: pd.DataFrame, spec: est.Spec) -> dict:
    """The rows, weights, full and structurally reduced event design of ``spec`` (no outcome is read)."""
    sub = est.select(panel, spec)
    post = est.post_indicator(sub, spec)
    w, w_record = est.row_weights(sub, spec, post)
    design = build_event_design(sub, spec)
    struct = est.structural(design.X, design.names, design.groups, w)
    contrasts, status = est.reduce_contrasts(design, struct, w)
    return {'sub': sub, 'w': w, 'w_record': w_record, 'design': design, 'struct': struct, 'contrasts': contrasts,
            'status': status, 'X': design.X[:, struct['kept']]}


def period_absence(sub: pd.DataFrame, spec: est.Spec, cache: est.DrawCache | None) -> dict | None:
    """Result-independent record of why the event study left the month-block draws: the share of the B main
    draws that hold no month of an event period."""
    if cache is None:
        return None
    units, layers = est.unit_index(sub, spec)
    draws, M = cache.get(layers, spec.block_length)
    periods_row = event_period(sub['year'], sub['month'])
    main = M[:draws.B]
    shares = {str(p): float((main[:, np.unique(units[periods_row == p])].sum(axis=1) == 0).mean())
              for p in period_order(periods_row)}
    return {'shares': shares, 'B': int(draws.B), 'block_length': int(draws.block_length),
            'note': '月块抽样不用于事件研究的起因（与结果无关，只用日历与块）：月块抽样中不含某事件期任何月份的比例。'}


def sensitivity_labels(simulation: dict | None, estimator: str) -> dict:
    """The pre-specified simulation judgements of one sensitivity estimator (per coefficient and per family)."""
    proc = ((simulation or {}).get('procedures') or {}).get(f'es:{estimator}') or {}
    return {'coefficients': proc.get('by_item') or {}, 'families': proc.get('by_joint') or {}}


def event_study(panel: pd.DataFrame, spec: est.Spec, responses: list, cache: est.DrawCache | None, sd: dict,
                simulation: dict | None = None) -> dict:
    """Per response column: the event coefficients (descriptive point estimates) and, per
    sensitivity estimator of ``external_params('event_study')['sensitivity_estimators']`` (:mod:`replication.external.escov`),
    the coefficient intervals, the joint test, TOST and HonestDiD on the frozen sets, each labelled with its
    assumption and the pre-specified simulation judgements (``simulation``: the sample's block of the coverage /
    size simulation). None of the sensitivity results is valid inference."""
    es = external_params('event_study')
    ed = event_design(panel, spec)
    sub, design, struct, contrasts, status = ed['sub'], ed['design'], ed['struct'], ed['contrasts'], ed['status']
    Y = sub[responses].to_numpy(float)
    sample, pre_set, post_set = frozen_sets(spec)
    out = {'spec': spec.record(), 'n': int(len(sub)), 'periods': design.report['periods'],
           'inference': es['inference'], 'sensitivity_estimators': es['sensitivity_estimators'],
           'pre_set_frozen': pre_set, 'post_set': post_set,
           'design': {**design.report, 'structural': struct, 'contrast_status': {k: v for k, v in status.items()}},
           'weights': ed['w_record'], 'block_draw_period_absence': period_absence(sub, spec, cache),
           'simulation_supplied': simulation is not None, 'PCs': {}}
    names = [n for n in design.contrasts if n in contrasts]
    w = ed['w']
    month = sub['ym'].to_numpy(np.int64)
    if not names:
        for r in responses:
            out['PCs'][r] = {'status': inf.UNAVAILABLE, 'reason': 'no event coefficient is estimable',
                             'coefficients': {n: {'status': inf.UNAVAILABLE, 'reason': status[n]['reason']}
                                              for n in design.contrasts}}
        return out
    X = ed['X']
    beta, rank, _ = inf.lstsq_fit(X, Y, w)
    C = np.stack([contrasts[n] for n in names])
    estimates = C @ beta                                               # (k, q)
    E = Y - X @ beta
    wv = np.ones(len(X)) if w is None else np.asarray(w, float)
    Q_inv = np.linalg.pinv(X.T @ (X * wv[:, None]))
    iid = np.einsum('kp,pq,kq->k', C, Q_inv, C)
    y_scale = ((wv[:, None] * (Y - (wv[:, None] * Y).sum(0) / wv.sum()) ** 2).sum(0) / wv.sum())
    estimators = []
    for spec_e in es['sensitivity_estimators']:
        try:
            obj = escov.prepare(spec_e, X, month, w, C)
            estimators.append((spec_e, obj, obj.contrast_cov(E), None))
        except (ValueError, np.linalg.LinAlgError) as error:
            estimators.append((spec_e, None, None, f'{type(error).__name__}: {error}'))
    jobs = []
    for j, r in enumerate(responses):
        coefs = {}
        for name in design.contrasts:
            if name not in contrasts:
                coefs[name] = {'status': inf.UNAVAILABLE, 'reason': status[name]['reason']}
            else:
                coefs[name] = {'status': 'ok', 'estimate': float(estimates[names.index(name), j]),
                               'reading': '描述性'}
        scale = iid * y_scale[j]
        sens = {}
        for spec_e, obj, covs, error in estimators:
            key = spec_e['name']
            if obj is None:
                sens[key] = {'status': inf.UNAVAILABLE, 'reason': error, 'valid_inference': False}
                continue
            labels = sensitivity_labels(simulation, key)
            V = covs[j]
            block = {'label': obj.label, 'assumption': obj.assumption, 'valid_inference': False,
                     'estimator': obj.record(), 'coefficients': {}, 'families': {}, 'psd': inf.psd_record(V)}
            for name in names:
                k = names.index(name)
                df = None if obj.df is None else float(obj.df[k])
                entry = inf.normal_inference(float(estimates[k, j]), float(V[k, k]), float(scale[k]), df)
                entry.pop('estimate', None)
                entry['simulation'] = labels['coefficients'].get(name)
                block['coefficients'][name] = entry
            eps = es['tost_sd'] * sd[r]
            for fam in FAMILIES:
                pre_names = [f'{fam}@{p}' for p in pre_set]
                post_names = [f'{fam}@{p}' for p in post_set]
                res, job = family_tests(pre_names, post_names, names, estimates[:, j], V, eps, block['coefficients'],
                                        obj, fam, r, scale)
                res['simulation'] = labels['families'].get(fam)
                block['families'][fam] = res
                if job is not None:
                    jobs.append((res, job))
            sens[key] = block
        out['PCs'][r] = {'coefficients': coefs, 'sensitivity': sens, 'tost_epsilon': es['tost_sd'] * sd[r],
                         'locked_sd': sd[r]}
    run_honest_jobs(jobs)
    return out


def family_tests(pre_names, post_names, names, estimates, cov, eps, intervals, obj, fam_name='', response='',
                 scale=None) -> tuple:
    """(result, HonestDiD job or None) of one family under one sensitivity estimator: the joint test of the frozen
    pre coefficients (``obj.joint``), TOST with that estimator's 90% intervals, and the HonestDiD inputs (pre + post
    coefficients and their full covariance). Sensitivity only."""
    out = {'pre': pre_names, 'post': post_names, 'valid_inference': False}
    missing = [n for n in pre_names if n not in names]
    if missing:
        reason = f'frozen pre coefficients not estimable (structural): {missing}'
        out['joint'] = {'status': inf.UNAVAILABLE, 'reason': reason}
        out['tost'] = {'status': inf.UNAVAILABLE, 'reason': reason}
        out['honestdid'] = {'status': inf.UNAVAILABLE, 'reason': 'pre coefficients missing'}
        return out, None
    idx = [names.index(n) for n in pre_names]
    bvec = np.asarray([float(estimates[i]) for i in idx])
    out['joint'] = obj.joint(idx, bvec, cov[np.ix_(idx, idx)], None if scale is None else float(np.max(scale[idx])))
    items = {}
    for n in pre_names:
        c = intervals.get(n) or {}
        ci = c.get('ci90')
        items[n] = {'estimate': float(estimates[names.index(n)]), 'ci90': ci,
                    'within': None if ci is None else bool(-eps <= ci[0] and ci[1] <= eps)}
    flags = [v['within'] for v in items.values()]
    out['tost'] = {'epsilon': eps, 'items': items, 'pass': None if any(f is None for f in flags) else bool(all(flags)),
                   'rule': "every frozen pre coefficient: its 90% interval (this estimator's reference distribution) "
                           'inside [-0.10 SD, +0.10 SD] (item-wise)'}
    if fam_name not in external_params('honestdid')['estimands'] or response not in external_params('honestdid')['responses']:
        out['honestdid'] = {'status': 'not run', 'reason': 'HonestDiD runs for theta and delta_UK on PC1'}
        return out, None
    post_ok = [n for n in post_names if n in names]
    if len(post_ok) != len(post_names) or '2021a' not in {n.split('@')[1] for n in pre_names}:
        out['honestdid'] = {'status': inf.UNAVAILABLE, 'reason': 'post coefficients or 2021a not estimable'}
        return out, None
    all_names = pre_names + post_names
    idx = [names.index(n) for n in all_names]
    betahat = np.asarray([float(estimates[i]) for i in idx])
    sigma = cov[np.ix_(idx, idx)]
    top = float(np.linalg.eigvalsh((sigma + sigma.T) / 2).max())
    if scale is not None and top <= inf.SINGULAR_REL_TOL * float(np.max(scale[idx])):
        out['honestdid'] = {'status': inf.UNAVAILABLE, 'reason': f'singular covariance (max eigenvalue {top:.3e})'}
        return out, None
    out['honestdid'] = {'status': 'pending'}
    return out, (all_names, betahat, sigma)


def run_honest_jobs(jobs: list) -> None:
    """Run the HonestDiD jobs (independent R processes, 45-120 s each) concurrently and fill their results in place;
    each result depends only on its own inputs."""
    if not jobs:
        return
    workers = max(1, min(len(jobs), os.cpu_count() or 1, 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: honest_family(*job[1]), jobs))
    for (res, _), result in zip(jobs, results):
        res['honestdid'] = result


# =========================================================================== HonestDiD
def renormalise(names: list, betahat, sigma) -> dict:
    """Coefficients relative to 2020 -> HonestDiD order relative to 2021a with the full
    covariance (b' = L b, Sigma' = L Sigma L'); pre = the frozen pre periods except 2021a, then 2020 (-b_2021a);
    post = the post periods."""
    es = external_params('event_study')
    reference = es['reference']
    periods = [n.split('@')[1] for n in names]
    if '2021a' not in periods:
        raise ValueError('2021a coefficient required for the HonestDiD re-normalisation')
    pre = [p for p in periods if p not in es['post'] and p != '2021a'] + [reference]
    post = [p for p in periods if p in es['post']]
    order = pre + post
    idx = {p: i for i, p in enumerate(periods)}
    a = idx['2021a']
    L = np.zeros((len(order), len(periods)))
    for r, p in enumerate(order):
        if p != reference:
            L[r, idx[p]] = 1.0
        L[r, a] -= 1.0
    b = L @ np.asarray(betahat, dtype=float)
    S = L @ np.asarray(sigma, dtype=float) @ L.T
    return {'order': order, 'numPrePeriods': len(pre), 'numPostPeriods': len(post), 'betahat': b.tolist(),
            'sigma': S.tolist(), 'normalised_to': '2021a', 'L': L.tolist()}


def run_honestdid(betahat, sigma, num_pre: int, num_post: int, l_vec, mbar_grid: list | None = None) -> dict:
    """HonestDiD relative-magnitudes sensitivity with the prespecified grid (``mbar_grid``: a single M-bar for the
    in-restriction simulation check; the robust interval at an M-bar does not depend on the other grid values); any failure
    raises (the caller writes "不可用")."""
    h = external_params('honestdid')
    payload = {'betahat': list(map(float, betahat)), 'sigma': np.asarray(sigma, float).ravel().tolist(),
               'numPrePeriods': num_pre, 'numPostPeriods': num_post, 'l_vec': list(map(float, l_vec)),
               'alpha': h['alpha'], 'Mbar_grid': h['Mbar'] if mbar_grid is None else list(mbar_grid),
               'gridPoints': h['grid_points'],
               'bisection_tol': h['bisection_tol'], 'max_doublings': h['max_doublings']}
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = Path(tmp) / 'in.json', Path(tmp) / 'out.json'
        inp.write_text(json.dumps(payload))
        proc = subprocess.run([rscript(), str(R_SCRIPT), str(inp), str(outp)], capture_output=True, text=True,
                              timeout=1800)
        if proc.returncode != 0 or not outp.is_file():
            raise RuntimeError(f'HonestDiD failed: {proc.stderr[-1500:]}')
        result = json.loads(outp.read_text())
    result['input'] = {k: payload[k] for k in ('numPrePeriods', 'numPostPeriods', 'l_vec', 'alpha', 'Mbar_grid',
                                               'gridPoints', 'max_doublings')}
    result['implementation'] = 'R HonestDiD createSensitivityResults_relativeMagnitudes (C-LF)'
    grid = result.get('grid') or []
    result['warnings_summary'] = {'default_grid_open_at_Mbar': [g['Mbar'] for g in grid if g.get('default_grid_open')],
                                  'still_truncated_at_Mbar': [g['Mbar'] for g in grid if g.get('truncated')]}
    return result


def rm_identified_set(betahat, num_pre: int, num_post: int, l_vec, Mbar: float) -> list:
    """Closed-form identified set of l' tau_post under Delta^RM (cross-check only)."""
    b = np.asarray(betahat, dtype=float)
    pre = np.concatenate([b[:num_pre], [0.0]])
    D = float(np.max(np.abs(np.diff(pre)))) if num_pre >= 1 else 0.0
    l = np.asarray(l_vec, dtype=float)
    tails = np.cumsum(l[::-1])[::-1]
    width = Mbar * D * float(np.abs(tails).sum())
    centre = float(l @ b[num_pre:num_pre + num_post])
    return [centre - width, centre + width]


def honest_family(names: list, betahat, sigma) -> dict:
    """Design for one family: rebase to 2021a (full covariance), l_vec = equal weights over the post periods,
    the frozen M-bar grid; the robust intervals and the breakdown M-bar*; any failure -> "不可用"."""
    try:
        ren = renormalise(names, betahat, sigma)
        eig = np.linalg.eigvalsh(np.asarray(ren['sigma']))
        if eig.min() <= 0:
            return {'status': inf.UNAVAILABLE, 'reason': f'rebased covariance not positive definite ({eig.min():.3e})',
                    'renormalised': ren}
        T = ren['numPostPeriods']
        l_vec = np.full(T, 1.0 / T)
        result = run_honestdid(ren['betahat'], ren['sigma'], ren['numPrePeriods'], T, l_vec)
    except Exception as error:          # noqa: BLE001 - the design failure rule
        return {'status': inf.UNAVAILABLE, 'reason': f'{type(error).__name__}: {error}'[:600]}
    result['status'] = 'ok'
    result['renormalised'] = ren
    result['l_vec'] = l_vec.tolist()
    result['identified_set_python'] = {str(m): rm_identified_set(ren['betahat'], ren['numPrePeriods'], T, l_vec, m)
                                       for m in (0.5, 1.0, 2.0)}
    result['target'] = 'equal-weight average of the post-period event coefficients (not the M1 coefficient)'
    return result


def honest_interval_at(names: list, betahat, sigma, Mbar: float) -> dict:
    """The HonestDiD robust interval of the target at one M-bar (the frozen settings otherwise): rebased to
    2021a with the full covariance; {'lb', 'ub', 'status'} (in-restriction simulation check)."""
    try:
        ren = renormalise(names, betahat, sigma)
        T = ren['numPostPeriods']
        res = run_honestdid(ren['betahat'], ren['sigma'], ren['numPrePeriods'], T, np.full(T, 1.0 / T), [Mbar])
        g = (res.get('grid') or [{}])[0]
        return {'status': 'ok', 'lb': g.get('lb'), 'ub': g.get('ub'), 'truncated': g.get('truncated')}
    except Exception as error:          # noqa: BLE001 - recorded, never raised
        return {'status': inf.UNAVAILABLE, 'reason': f'{type(error).__name__}: {error}'[:300]}


# =========================================================================== placebos
def placebos(panel: pd.DataFrame, base: est.Spec, responses: list, cache: est.DrawCache) -> dict:
    """Pre-period data only; fake cuts 2019-09 and 2020-09, and v1's year starts 2020 and 2021 (fake Post = year >=
    2020 / 2021); M1 with the fake Post; month-block inference on each placebo's own layers (the pre-period months
    split at the fake cut); sample sizes and layer lengths recorded."""
    from dataclasses import replace
    pl = external_params('placebo')
    cuts = list(pl['month_cuts']) + [f'{y}-01' for y in pl['v1_year_starts']]
    out = {}
    for cut in cuts:
        spec = replace(base, name=f'{base.name}_placebo_{cut}', fake_cut=cut, model='M1')
        out[cut] = est.fit_spec(panel, spec, responses, cache)
        out[cut]['origin'] = 'the design month cut' if cut in pl['month_cuts'] else "v1's year-start cut"
    return out
