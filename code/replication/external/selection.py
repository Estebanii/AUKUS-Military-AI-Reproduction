"""The inference selection: the result-blind selection of the M1 inference and of the pre-trend slope inference.

Design only (no outcome is read): the frozen five error scenarios on the real rows of each window (S_SG, S_CA), Y = e
(every estimand is 0). Candidates, frozen before any result (``external_params('inference_selection')``):

* C1 / C2: the month-block bootstrap of :func:`replication.external.estimate.fit_spec` with L = 12 / 24 (the real draws, B,
  seed, replacement draws and the <= 1% target-estimability rule; the +1 p value and the symmetric order-statistic
  interval);
* C3: the studentised (percentile-t) month-block bootstrap, L = 12: in every draw the month-clustered CR0 standard error
  is re-estimated on the resampled data (every drawn copy of a month is a cluster; unit scores at the draw's
  coefficients); T = t / SE, T* = (t* - t) / SE*; p = (1 + #{|T*| >= |T|}) / (B_e + 1); half-width SE x the order
  statistic of |T*| (the same r rule, so "interval excludes 0" <=> "p < alpha");
* C4: month-clustered CR2 (:class:`replication.external.escov.Cr2Month`) with the Satterthwaite (Bell-McCaffrey / Imbens-Kolesar)
  t for single contrasts (HTZ for joint restrictions; the estimands in scope are single contrasts);
* C5-C8: C1-C4 with the worst-case calibrated critical value: S = |estimate| / nominal 95% half-width; c*_a = the
  maximum over the five development scenarios of the (1 - a) quantile of S; test S > c*_a; interval estimate +/-
  c*_a x the nominal 95% half-width; p = the maximum over the scenarios of the development tail probability of S.

Development (seed 20260928, 2,000 replications per scenario): a candidate is admitted when every scenario x estimand
in scope has coverage95, coverage90 and size05 all 通过 (the thresholds of the simulation, one-sided 95% Clopper-Pearson bounds).
Selection: the smallest worst-case size05 upper bound; ties -> the wider mean standardised 95% interval; then the
candidate number. Verification: the winner with seed 20260929 (calibrated candidates keep the development critical
values); anything other than 通过 everywhere -> the fallback: descriptive (it enters the Holm family with p = 1), no further
switching.
Reported: every candidate x scenario x estimand x metric with the Monte Carlo counts and bounds, the selection trace,
the verification, the simulated sizes at alpha / j (j = 1..10) of the winner and its simulated-power MDE per unit
error SD in each scenario (descriptive only).

MDE (the frozen form MDE = (c + z_.80) x SE with c the critical value
of the test actually used): :func:`observed_mde`. SE is the selected candidate's standard error on the real data and
c its actual critical value on the SE scale (C8: the calibrated c*_a x the Satterthwaite t_.975 of the contrast) at the
Holm first-step level alpha / m, with the unadjusted alpha alongside; MDE_sd = MDE / the locked pre-period SD, tiers
0.10 / 0.25. Only the SE enters from the outcome (through the residuals); no estimate or p is returned. The
five scenario MDEs never set a tier.

Development admission of C5-C8 is in-sample (their critical values are calibrated on the same replications), which
makes it nearly automatic; the independent-seed verification is the out-of-sample check.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import escov
from . import estimate as est
from . import inference as inf
from .params import CONTROLS, external_params
from .simulation import BlockProcedure, Errors, judge, with_row_ids

METRICS = ('coverage95', 'coverage90', 'size05')

# Scope statement of the selected inference (C8), copied into the summary records
GUARANTEE_SCOPE = ('主推断为冻结五种误差情形校准的 CR2 统计量（C8）；开发模拟用于选择与校准，独立种子模拟支持预定的模拟验收标准'
                   '（95% 覆盖 ≥ .93，5% 尺寸 ≤ .06）。该保障仅限于冻结的情形集合，不外推至任意相关结构；不保证精确 5% 尺寸'
                   '或 Holm 族错误率控制。')


def _spec(which: str, sample: str, block_length: int | None = None) -> est.Spec:
    from dataclasses import replace
    from .pretrend import slope_spec
    base = est.main_spec(sample) if which == 'M1' else slope_spec(sample)
    return base if block_length is None else replace(base, block_length=int(block_length),
                                                     name=f'{base.name}_L{block_length}')


# =========================================================================== base candidates (C1-C4)
class BlockCandidate:
    """C1 / C2: the month-block bootstrap with the candidate's block length."""

    def __init__(self, name, panel_r, which, sample, position, cache, targets, levels, block_length):
        self.name, self.levels = name, list(levels)
        self.proc = BlockProcedure(name, panel_r, _spec(which, sample, block_length), position, cache, targets)
        self.items = self.proc.items
        self.record = {'kind': 'block', 'block_length': int(block_length), **{k: v for k, v in self.proc.record.items()
                                                                               if k in ('B', 'n', 'unavailable')}}

    def stats(self, E_base):
        out = self.proc.stats(E_base, self.levels)
        return {nm: {'estimate': v['estimate'], 'p': v['p'], 'h': v['h']} for nm, v in out.items()}

    def precision(self, E_base):
        """Per estimand: the SE (the SD of the effective bootstrap draws t*) and the half-widths; no estimate."""
        pr = self.proc
        Hstar = np.tensordot(pr.M, pr.unit_scores(E_base), axes=(1, 0))
        st = pr.stats(E_base, self.levels)
        return {nm: {'se': np.einsum('bp,bpr->br', pr.R[nm], Hstar[pr.index[nm]]).std(axis=0, ddof=1),
                     'h': st[nm]['h'], 'se_kind': 'SD of the month-block bootstrap draws'} for nm in self.items}


class StudentizedCandidate(BlockCandidate):
    """C3: percentile-t month-block bootstrap with a month-clustered CR0 SE re-estimated in every draw."""

    chunk = 1024

    def __init__(self, name, panel_r, which, sample, position, cache, targets, levels, block_length):
        super().__init__(name, panel_r, which, sample, position, cache, targets, levels, block_length)
        self.record['kind'] = 'block_studentized'
        pr = self.proc
        self.RG = {nm: np.einsum('bp,upq->buq', pr.R[nm], pr.G_units) for nm in self.items}   # (B_e, U, p)
        self.RG_full = {nm: np.einsum('p,upq->uq', pr.R_full[nm], pr.G_units) for nm in self.items}

    def stats(self, E_base):
        pr = self.proc
        Hu = pr.unit_scores(E_base)                                      # (U, p, r)
        U, p, r = Hu.shape
        Hfull = Hu.sum(axis=0)
        beta = pr.P_full @ Hfull                                          # (p, r)
        Hflat = Hu.transpose(1, 0, 2).reshape(p, U * r)
        out = {}
        for nm in self.items:
            idx = pr.index[nm]
            t = pr.R_full[nm] @ Hfull
            a0 = (pr.R_full[nm] @ Hflat).reshape(U, r) - self.RG_full[nm] @ beta
            se = np.sqrt((a0 ** 2).sum(axis=0))
            T = np.where(se > 0, t / np.where(se > 0, se, 1.0), np.inf)
            dev = np.empty((len(idx), r))
            for s in range(0, len(idx), self.chunk):
                ix = idx[s:s + self.chunk]
                Mi = pr.M[ix].astype(float)                               # (c, U)
                Hstar = np.tensordot(Mi, Hu, axes=(1, 0))                 # (c, p, r)
                bstar = np.matmul(pr.P[ix], Hstar)                        # (c, p, r)
                Rb = pr.R[nm][s:s + self.chunk]                           # (c, p)
                tstar = np.einsum('cp,cpr->cr', Rb, Hstar)
                a = (Rb @ Hflat).reshape(len(ix), U, r) - np.matmul(self.RG[nm][s:s + self.chunk], bstar)
                vstar = np.einsum('cu,cur->cr', Mi, a * a)
                ok = vstar > 0
                dev[s:s + len(ix)] = np.where(ok, np.abs(tstar - t) / np.sqrt(np.where(ok, vstar, 1.0)), np.inf)
            B_e = dev.shape[0]
            pos = {a_: B_e - inf.r_max(B_e, a_) - 1 for a_ in set(self.levels) | {0.05, 0.10}}
            valid = sorted({k for k in pos.values() if 0 <= k < B_e})
            part = np.partition(dev, valid, axis=0) if valid else dev
            exceed = (dev >= np.abs(T)).sum(axis=0)
            out[nm] = {'estimate': t, 'p': (1 + exceed) / (B_e + 1), 'se': se,
                       'h': {a_: (se * part[k] if 0 <= k < B_e else np.full(r, np.inf)) for a_, k in pos.items()}}
        return out

    def precision(self, E_base):
        """Per estimand: the month-clustered CR0 SE and the percentile-t half-widths; no estimate is returned."""
        return {nm: {'se': v['se'], 'h': v['h'], 'se_kind': 'month-clustered CR0'} for nm, v in self.stats(E_base).items()}


class Cr2Candidate:
    """C4: month-clustered CR2 with Satterthwaite t (single contrasts)."""

    def __init__(self, name, panel_r, which, sample, position, targets, levels):
        self.name, self.levels = name, list(levels)
        spec = _spec(which, sample)
        sub = est.select(panel_r, spec)
        self.rows = np.array([position[i] for i in sub['_row'].to_numpy()], dtype=np.int64)
        post = est.post_indicator(sub, spec)
        w, _ = est.row_weights(sub, spec, post)
        design = est.build_design(sub, spec, post)
        struct = est.structural(design.X, design.names, design.groups, w)
        contrasts, status = est.reduce_contrasts(design, struct, w)
        self.items = [k for k in targets if k in contrasts]
        X = design.X[:, struct['kept']]
        wv = np.ones(len(X)) if w is None else np.asarray(w, float)
        self.X = X
        self.A = np.linalg.pinv(X.T @ (X * wv[:, None])) @ (X * wv[:, None]).T
        self.C = np.stack([contrasts[k] for k in self.items])
        self.obj = escov.Cr2Month(X, sub['ym'].to_numpy(np.int64), w, self.C)
        self.record = {'kind': 'cr2', 'n': int(len(X)), **self.obj.record(),
                       'unavailable': {k: v['reason'] for k, v in status.items() if k in targets and k not in contrasts}}

    def stats(self, E_base):
        E = E_base[self.rows]
        B = self.A @ E
        V = self.obj.contrast_cov(E - self.X @ B)
        est_ = self.C @ B
        out = {}
        for k, nm in enumerate(self.items):
            se = np.sqrt(np.clip(V[:, k, k], 0, None))
            df = float(self.obj.df[k])
            tval = np.where(se > 0, np.abs(est_[k]) / np.where(se > 0, se, 1.0), np.inf)
            out[nm] = {'estimate': est_[k], 'p': 2 * stats.t.sf(tval, df),
                       'h': {a: se * float(stats.t.ppf(1 - a / 2, df)) for a in set(self.levels) | {0.05, 0.10}}}
        return out

    def precision(self, E_base):
        """Per estimand: the CR2 SE and the Satterthwaite half-widths. The coefficients enter only through the
        residuals; no contrast estimate is formed."""
        E = E_base[self.rows]
        V = self.obj.contrast_cov(E - self.X @ (self.A @ E))
        out = {}
        for k, nm in enumerate(self.items):
            se = np.sqrt(np.clip(V[:, k, k], 0, None))
            df = float(self.obj.df[k])
            out[nm] = {'se': se, 'df': df, 'se_kind': 'month-clustered CR2',
                       'h': {a: se * float(stats.t.ppf(1 - a / 2, df)) for a in set(self.levels) | {0.05, 0.10}}}
        return out


def build_candidates(panel_r, which, sample, position, cache, cfg) -> dict:
    sel = cfg
    targets = sel['procedures'][which]['estimands']
    levels = sel['levels']
    out = {}
    for name, c in sel['candidates'].items():
        if c['kind'] == 'block':
            out[name] = BlockCandidate(name, panel_r, which, sample, position, cache, targets, levels, c['block_length'])
        elif c['kind'] == 'block_studentized':
            out[name] = StudentizedCandidate(name, panel_r, which, sample, position, cache, targets, levels,
                                             c['block_length'])
        elif c['kind'] == 'cr2':
            out[name] = Cr2Candidate(name, panel_r, which, sample, position, targets, levels)
    return out


# =========================================================================== simulation of the base candidates
def simulate(panel: pd.DataFrame, sample: str, which: str, seed: int, cfg: dict | None = None,
             names: list | None = None, cache: est.DrawCache | None = None, progress=None) -> dict:
    """Per base candidate (C1-C4, or ``names``), scenario and estimand: the per-replication estimate, p value and
    half-widths at the frozen levels, from the frozen scenarios with ``seed`` (design only)."""
    cfg = external_params('inference_selection') if cfg is None else cfg
    sim = external_params('simulation')
    cache = est.DrawCache() if cache is None else cache
    panel_r = with_row_ids(panel)
    spec_main = est.main_spec(sample)
    base = est.select(panel_r, spec_main)
    position = {int(i): k for k, i in enumerate(base['_row'].to_numpy())}
    errors = Errors(base, spec_main)
    cands = build_candidates(panel_r, which, sample, position, cache, cfg)
    if names is not None:
        cands = {k: v for k, v in cands.items() if k in names}
    R, batch = int(cfg['replications']), int(sim['batch'])
    seeds = np.random.SeedSequence([int(seed), CONTROLS.index(sample)]).spawn(len(sim['scenarios']))
    out = {name: {} for name in cands}
    for (scen, sc), ss in zip(sim['scenarios'].items(), seeds):
        if progress:
            progress(f'inference selection {which} {sample} seed {seed} {scen}')
        rng = np.random.Generator(np.random.PCG64(ss))
        parts = {name: [] for name in cands}
        done = 0
        while done < R:
            r = min(batch, R - done)
            E = errors.draw(rng, sc, r)
            for name, cand in cands.items():
                parts[name].append(cand.stats(E))
            done += r
        for name, cand in cands.items():
            out[name][scen] = {nm: {'estimate': np.concatenate([p_[nm]['estimate'] for p_ in parts[name]]),
                                    'p': np.concatenate([p_[nm]['p'] for p_ in parts[name]]),
                                    'h': {a: np.concatenate([p_[nm]['h'][a] for p_ in parts[name]])
                                          for a in parts[name][0][nm]['h']}}
                               for nm in cand.items}
    records = {name: cand.record for name, cand in cands.items()}
    return {'draws': out, 'records': records, 'seed': int(seed), 'replications': R}


# =========================================================================== evaluation, calibration, selection
def calibration(dev: dict, levels) -> dict:
    """c*_a per estimand: the maximum over scenarios of the (1 - a) quantile of S = |estimate| / h_0.05 (development)."""
    out = {}
    for nm in next(iter(dev.values())):
        S = {scen: np.abs(v[nm]['estimate']) / v[nm]['h'][0.05] for scen, v in dev.items()}
        out[nm] = {'critical': {a: float(max(np.quantile(s, 1 - a, method='higher') for s in S.values()))
                                for a in set(levels) | {0.05, 0.10}},
                   'null_S': {scen: np.sort(s) for scen, s in S.items()}}
    return out


def calibrated_p(S_obs: np.ndarray, null_S: dict) -> np.ndarray:
    """max over scenarios of (1 + #{S_s >= S_obs}) / (R + 1)."""
    S_obs = np.atleast_1d(np.asarray(S_obs, float))
    best = np.zeros(len(S_obs))
    for s in null_S.values():
        ge = len(s) - np.searchsorted(s, S_obs, side='left')
        best = np.maximum(best, (1 + ge) / (len(s) + 1))
    return best


def decisions(block: dict, cal: dict | None, alpha: float) -> tuple:
    """(reject, half-width) arrays at level ``alpha`` for one scenario x estimand block (uncalibrated or calibrated)."""
    if cal is None:
        h = block['h'][alpha]
        return block['p'] < alpha, h
    c = cal['critical'][alpha]
    S = np.abs(block['estimate']) / block['h'][0.05]
    return S > c, c * block['h'][0.05]


def evaluate(draws: dict, cal: dict | None, thr: dict) -> dict:
    """Per scenario x estimand: the three judgements, the standardised width; and the worst-case size upper bound."""
    per, worst, widths = {}, 0.0, []
    for scen, block in draws.items():
        per[scen] = {}
        for nm, v in block.items():
            n = len(v['estimate'])
            rej95, h95 = decisions(v, None if cal is None else cal[nm], 0.05)
            rej90, _ = decisions(v, None if cal is None else cal[nm], 0.10)
            sd = float(np.std(v['estimate'], ddof=1))
            width = float(np.mean(h95)) / sd if sd > 0 else float('nan')
            j = {'coverage95': judge(int((~rej95).sum()), n, thr['coverage95'], 'coverage'),
                 'coverage90': judge(int((~rej90).sum()), n, thr['coverage90'], 'coverage'),
                 'size05': judge(int(rej95.sum()), n, thr['size05'], 'size'), 'standardised_width95': width}
            per[scen][nm] = j
            worst = max(worst, j['size05']['upper'])
            widths.append(width)
    admitted = all(v[m]['judgement'] == '通过' for b in per.values() for v in b.values() for m in METRICS)
    return {'by_scenario': per, 'worst_size_upper': worst, 'mean_standardised_width95': float(np.nanmean(widths)),
            'admitted': admitted}


def candidate_views(dev_draws: dict, cfg: dict) -> dict:
    """(draws, calibration or None) per candidate C1-C8 from the base-candidate development draws."""
    views = {}
    for name, c in cfg['candidates'].items():
        if c['kind'] == 'calibrated':
            if c['base'] in dev_draws:
                views[name] = (c['base'], calibration(dev_draws[c['base']], cfg['levels']))
        elif name in dev_draws:
            views[name] = (name, None)
    return views


def select(dev: dict, cfg: dict, thr: dict) -> dict:
    """Development evaluation of C1-C8, admission and the selection trace."""
    views = candidate_views(dev['draws'], cfg)
    evals = {name: evaluate(dev['draws'][base], cal, thr) for name, (base, cal) in views.items()}
    admitted = [n for n, e in evals.items() if e['admitted']]
    order = sorted(admitted, key=lambda n: (evals[n]['worst_size_upper'], -evals[n]['mean_standardised_width95'],
                                            int(n[1:])))
    trace = [{'candidate': n, 'worst_size_upper': evals[n]['worst_size_upper'],
              'mean_standardised_width95': evals[n]['mean_standardised_width95']} for n in order]
    return {'evaluations': evals, 'admitted': admitted, 'ranking': trace, 'winner': order[0] if order else None,
            'calibration': {n: ({nm: {'critical': v['critical']} for nm, v in cal.items()} if cal else None)
                            for n, (base, cal) in views.items()}, 'views': views}


def holm_sizes(draws: dict, cal: dict | None) -> dict:
    """Simulated sizes at alpha / j (j = 1..10) per scenario x estimand (p values; calibrated p for C5-C8)."""
    out = {}
    for scen, block in draws.items():
        out[scen] = {}
        for nm, v in block.items():
            if cal is None:
                pv = v['p']
            else:
                pv = calibrated_p(np.abs(v['estimate']) / v['h'][0.05], cal[nm]['null_S'])
            n = len(pv)
            out[scen][nm] = {f'alpha/{j}': {'level': 0.05 / j, **_rate(int((pv < 0.05 / j).sum()), n)}
                             for j in range(1, 11)}
    return out


def _rate(x, n):
    from .simulation import cp_bounds
    lo, hi = cp_bounds(x, n)
    return {'rate': x / n, 'x': x, 'n': n, 'lower': lo, 'upper': hi}


def mde_unit(draws: dict, cal: dict | None, alpha: float, power: float = 0.80) -> dict:
    """Per scenario x estimand: the smallest delta >= 0 with simulated power >= ``power`` at level ``alpha`` (the
    estimate shifts by delta, half-widths do not; unit-variance errors)."""
    out = {}
    for scen, block in draws.items():
        out[scen] = {}
        for nm, v in block.items():
            t = v['estimate']
            if cal is None:
                crit = v['h'][alpha] if alpha in v['h'] else None
            else:
                c = cal[nm]['critical'].get(alpha)
                crit = None if c is None else c * v['h'][0.05]
            if crit is None:
                out[scen][nm] = None
                continue

            def pw(d):
                return float(np.mean(np.abs(t + d) > crit))
            lo, hi = 0.0, float(np.max(crit) * 4 + np.max(np.abs(t)) + 1e-12)
            if pw(hi) < power:
                out[scen][nm] = float('inf')
                continue
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if pw(mid) < power else (lo, mid)
            out[scen][nm] = hi
    return out


# =========================================================================== driver (both windows, M1 and slope)
def run(panel: pd.DataFrame, progress=None, cache: est.DrawCache | None = None) -> dict:
    """The inference selection record: per procedure (M1, slope) and window, the development evaluation of C1-C8,
    the selection trace, the verification of the winner (or the fallback), the alpha / j sizes and the MDE_unit."""
    cfg = external_params('inference_selection')
    thr = external_params('simulation')['thresholds']
    cache = est.DrawCache() if cache is None else cache
    out = {'decision': cfg['decision'], 'rule': cfg, 'thresholds': thr, 'outcome_read': False, 'procedures': {}}
    for which in cfg['procedures']:
        out['procedures'][which] = {}
        for sample in CONTROLS:
            dev = simulate(panel, sample, which, cfg['development_seed'], cfg, cache=cache, progress=progress)
            sel = select(dev, cfg, thr)
            block = {'records': dev['records'], 'development': _public(sel), 'winner': sel['winner']}
            if sel['winner'] is None:
                block.update(status='descriptive', reason='no candidate admitted (selection fallback)')
            else:
                base, cal = sel['views'][sel['winner']]
                ver = simulate(panel, sample, which, cfg['verification_seed'], cfg, names=[base], cache=cache,
                               progress=progress)
                ev = evaluate(ver['draws'][base], cal, thr)
                block['verification'] = {'seed': cfg['verification_seed'], 'by_scenario': ev['by_scenario'],
                                         'worst_size_upper': ev['worst_size_upper'], 'passed': ev['admitted']}
                block['holm_sizes'] = holm_sizes(ver['draws'][base], cal)
                if which == 'M1':
                    levels = sorted(set(cfg['levels']))
                    block['mde_unit'] = {str(a): mde_unit(ver['draws'][base], cal, a, cfg['mde']['power'])
                                         for a in levels}
                block['status'] = 'selected' if ev['admitted'] else 'descriptive'
                if not ev['admitted']:
                    block['reason'] = 'verification not 通过 everywhere (selection fallback)'
                block['selected_calibration'] = ({nm: {'critical': v['critical'],
                                                       'null_S': {s: x.tolist() for s, x in v['null_S'].items()}}
                                                  for nm, v in cal.items()} if cal else None)
            out['procedures'][which][sample] = block
    return out


def _public(sel: dict) -> dict:
    return {k: v for k, v in sel.items() if k != 'views'}


# =========================================================================== application to the real panel (estimation)
def _selected(panel: pd.DataFrame, blk: dict, which: str, sample: str, responses: list, cfg: dict,
              cache: est.DrawCache | None):
    """(winner, its base candidate built on the real rows, the real responses, the calibration or None)."""
    winner = blk['winner']
    spec = cfg['candidates'][winner]
    base_name = spec['base'] if spec['kind'] == 'calibrated' else winner
    cache = est.DrawCache() if cache is None else cache
    panel_r = with_row_ids(panel)
    base = est.select(panel_r, est.main_spec(sample))
    position = {int(i): k for k, i in enumerate(base['_row'].to_numpy())}
    cand = build_candidates(panel_r, which, sample, position, cache,
                            {**cfg, 'candidates': {base_name: cfg['candidates'][base_name]}})[base_name]
    cal = blk.get('selected_calibration') if spec['kind'] == 'calibrated' else None
    return winner, cand, base[responses].to_numpy(float), cal


def _lvl(d: dict, a: float):
    """A dict keyed by level (float or its JSON string)."""
    if a in d:
        return d[a]
    for k, v in d.items():
        if abs(float(k) - a) < 1e-12:
            return v
    raise KeyError(a)


def apply(panel: pd.DataFrame, record: dict, which: str, sample: str, responses: list,
          cache: est.DrawCache | None = None) -> dict:
    """The selected inference of ``which`` (M1 or slope) in ``sample`` on the real responses: per estimand in
    scope and response, the estimate, p (calibrated p for C5-C8) and the 95% / 90% intervals of the selected candidate;
    'descriptive' when the window fell back (it then enters the Holm family with p = 1)."""
    cfg = external_params('inference_selection')
    blk = record['procedures'][which][sample]
    targets = cfg['procedures'][which]['estimands']
    if blk.get('status') != 'selected':
        return {'status': 'descriptive', 'reason': blk.get('reason'), 'winner': blk.get('winner'),
                'estimands': {nm: {r: {'status': 'descriptive (inference selection)'} for r in responses} for nm in targets}}
    winner, cand, Y, cal = _selected(panel, blk, which, sample, responses, cfg, cache)
    st = cand.stats(Y)
    out = {}
    for nm in targets:
        out[nm] = {}
        for j, r in enumerate(responses):
            if nm not in st:
                out[nm][r] = {'status': inf.UNAVAILABLE, 'reason': 'not estimable (structural or sampling rule)'}
                continue
            t = float(st[nm]['estimate'][j])
            h95, h90 = float(_lvl(st[nm]['h'], 0.05)[j]), float(_lvl(st[nm]['h'], 0.10)[j])
            if cal is None:
                p = float(st[nm]['p'][j])
                w95, w90 = h95, h90
            else:
                c = cal[nm]['critical']
                S = abs(t) / h95 if h95 > 0 else float('inf')
                null = {s: np.asarray(v, float) for s, v in cal[nm]['null_S'].items()}
                p = float(calibrated_p(np.array([S]), null)[0])
                w95, w90 = float(_lvl(c, 0.05)) * h95, float(_lvl(c, 0.10)) * h95
            out[nm][r] = {'status': 'ok', 'estimate': t, 'p': p, 'ci95': [t - w95, t + w95], 'ci90': [t - w90, t + w90],
                          'candidate': winner}
    return {'status': 'selected', 'winner': winner, 'candidate': cfg['candidates'][winner], 'estimands': out}


# =========================================================================== MDE (inference selection)
def residual_sd(panel: pd.DataFrame, sample: str, responses: list) -> dict:
    """The occurrence-weighted residual SD of the M1 fit per response (the scale of the unit-variance simulation
    errors); no coefficient is returned."""
    spec = est.main_spec(sample)
    sub = est.select(panel, spec)
    post = est.post_indicator(sub, spec)
    w, _ = est.row_weights(sub, spec, post)
    design = est.build_design(sub, spec, post)
    struct = est.structural(design.X, design.names, design.groups, w)
    X = design.X[:, struct['kept']]
    Y = sub[responses].to_numpy(float)
    beta, _, _ = inf.lstsq_fit(X, Y, w)
    E = Y - X @ beta
    wv = np.ones(len(X)) if w is None else np.asarray(w, float)
    return {r: float(np.sqrt((wv * E[:, j] ** 2).sum() / wv.sum())) for j, r in enumerate(responses)}


def observed_mde(panel: pd.DataFrame, record: dict, sample: str, responses: list, sd: dict,
                 cache: est.DrawCache | None = None) -> dict:
    """The MDE (the frozen form MDE = (c + z_.80) x SE, c the critical value of the
    test actually used) for the M1 inference selected in ``sample``: per estimand in scope and response, the selected
    candidate's SE on the real data, its critical value on the SE scale at every frozen level a -- calibrated
    candidates (C5-C8): c_a = c*_a x nominal 95% half-width / SE (C8: c*_a x the Satterthwaite t_.975 of the contrast);
    uncalibrated: half-width at a / SE -- MDE_a = (c_a + z_.80) x SE and MDE_sd_a = MDE_a / the locked SD. The outcome
    enters only through the residuals of the SE; no estimate or p is computed for, or returned by, this record.
    A window without a verified candidate (selection fallback) has no MDE ('descriptive')."""
    cfg = external_params('inference_selection')
    blk = record['procedures']['M1'][sample]
    targets = cfg['procedures']['M1']['estimands']
    if blk.get('status') != 'selected':
        return {'status': 'descriptive', 'winner': blk.get('winner'),
                'reason': blk.get('reason') or 'no verified candidate: descriptive (inference selection)'}
    z = float(external_params('mde')['z_power'])
    levels = sorted(set(cfg['levels']) | {0.05})
    winner, cand, Y, cal = _selected(panel, blk, 'M1', sample, responses, cfg, cache)
    prec = cand.precision(Y)
    out = {}
    for nm in targets:
        out[nm] = {}
        for j, r in enumerate(responses):
            if nm not in prec:
                out[nm][r] = {'status': inf.UNAVAILABLE, 'reason': 'not estimable (structural or sampling rule)'}
                continue
            se = float(prec[nm]['se'][j])
            h95 = float(_lvl(prec[nm]['h'], 0.05)[j])
            ok = np.isfinite(se) and se > 0
            by = {}
            for a in levels:
                c_star = None if cal is None else float(_lvl(cal[nm]['critical'], a))
                crit = c_star * h95 if cal is not None else float(_lvl(prec[nm]['h'], a)[j])
                c = crit / se if ok and np.isfinite(crit) else None
                mde = None if c is None else (c + z) * se
                by[str(a)] = {'c_calibrated': c_star, 'c': c, 'mde': mde,
                              'mde_sd': None if mde is None or not sd.get(r) else mde / sd[r]}
            out[nm][r] = {'status': 'ok', 'se': se if ok else None, 'se_kind': prec[nm].get('se_kind'),
                          'df': prec[nm].get('df'), 'by_level': by}
    return {'status': 'selected', 'winner': winner, 'candidate': cfg['candidates'][winner], 'z_power': z,
            'locked_sd': sd, 'form': 'MDE = (c + z_.80) x SE (observed precision)', 'estimands': out}


def mde_report(panel: pd.DataFrame, record: dict, sample: str, responses: list, sd: dict) -> dict:
    """Descriptive only (never sets a tier): per level, estimand in scope and scenario, the
    simulated-power MDE of the verified winner per unit error SD (``mde_unit``), and per response its conversion
    MDE_unit x the M1 residual SD / the locked SD."""
    blk = record['procedures']['M1'][sample]
    if blk.get('mde_unit') is None:
        return {'status': blk.get('status'), 'reason': 'no verified candidate: descriptive (inference selection)'}
    rsd = residual_sd(panel, sample, responses)
    out = {'status': blk.get('status'), 'winner': blk.get('winner'), 'descriptive_only': True, 'residual_sd': rsd,
           'locked_sd': sd, 'mde_unit_by_level': {}, 'by_level': {}}
    for lvl, per in blk['mde_unit'].items():
        names = next(iter(per.values()))
        out['mde_unit_by_level'][str(lvl)] = {nm: {scen: v[nm] for scen, v in per.items()} for nm in names}
        out['by_level'][str(lvl)] = {nm: {r: {scen: (None if v[nm] is None else v[nm] * rsd[r] / sd[r])
                                              for scen, v in per.items()} for r in responses}
                                     for nm in names}
    return out
