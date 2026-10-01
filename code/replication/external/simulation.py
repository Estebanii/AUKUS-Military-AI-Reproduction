"""The coverage / size simulation of the procedures actually used (fixed together with the MDE, before any
estimate).

Generic over scenarios and procedures (``external_params('simulation')``). For every sample (SG, CA) the real design is used --
the rows of the sample's main M1 specification with their countries, calendar months, documents and weights; no
outcome column is ever read -- and Y = e (every coefficient, slope and event contrast is 0), e drawn per frozen
scenario (idiosyncratic, document, country-month AR(1), common-month AR(1) components). One draw of e per batch is
shared by every procedure:

* ``M1`` and ``slope`` / ``slope_L3`` / ``slope_L12`` (bootstrap procedures): the full month-block procedure of
  :func:`replication.external.estimate.fit_spec` with the real draws (B = 9,999, seed 20260928, the same block calendar,
  replacement draws and the <= 1% target-estimability rule): the centred two-sided +1 p value and the symmetric
  order-statistic intervals; per estimand coverage of the 95% and 90% intervals and the size of the 5% test. The
  per-draw Gram solutions depend on the design only and are computed once (draws with an ambiguous rank gap use the
  pseudo-inverse of the draw's Gram matrix; the real procedure refits them on the materialised rows);
* ``es:<estimator>`` (fixed-design procedures; :mod:`replication.external.escov`): the event-study sensitivity estimators; per
  event coefficient coverage of the 95% / 90% intervals (normal or Student-t by the estimator) and the size of each
  family's pre-period joint test.

Judgement (prespecified simulation acceptance criteria, one-sided 95% Clopper-Pearson bounds): coverage 95% >= .93 and 90% >= .88,
size <= .06; clearly outside = 失败, straddling = 未定, clearly inside = 通过. Every scenario is reported; 通过 is not a
proof of validity. Descriptive only (a zero-effect simulation cannot validate them): the TOST pass rate with one pre
coefficient on the equivalence boundary, and the HonestDiD robust-interval coverage of the target under a deviation
path on the Delta^RM(M-bar) boundary.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from . import escov
from . import estimate as est
from . import eventstudy
from . import inference as inf
from .common import month_index
from .params import CONTROLS, external_params

FAMILIES = eventstudy.FAMILIES


# =========================================================================== judgement
def cp_bounds(x: int, n: int, level: float = 0.95) -> tuple:
    """One-sided ``level`` Clopper-Pearson (lower, upper) bounds of a binomial proportion x / n."""
    a = 1.0 - level
    lower = 0.0 if x <= 0 else float(stats.beta.ppf(a, x, n - x + 1))
    upper = 1.0 if x >= n else float(stats.beta.ppf(1.0 - a, x + 1, n - x))
    return lower, upper


def judge(x: int, n: int, threshold: float, kind: str) -> dict:
    """``kind`` 'coverage' (at least ``threshold``) or 'size' (at most ``threshold``)."""
    lower, upper = cp_bounds(int(x), int(n))
    if kind == 'coverage':
        verdict = '失败' if upper < threshold else '通过' if lower >= threshold else '未定'
    else:
        verdict = '失败' if lower > threshold else '通过' if upper <= threshold else '未定'
    return {'rate': (x / n) if n else None, 'x': int(x), 'n': int(n), 'lower': lower, 'upper': upper,
            'threshold': threshold, 'judgement': verdict}


def rate(x: int, n: int) -> dict:
    lower, upper = cp_bounds(int(x), int(n))
    return {'rate': (x / n) if n else None, 'x': int(x), 'n': int(n), 'lower': lower, 'upper': upper}


# =========================================================================== errors
class Errors:
    """The frozen error scenarios on the base rows (the sample's main M1 rows)."""

    def __init__(self, base: pd.DataFrame, spec: est.Spec):
        self.n = len(base)
        self.doc = pd.factorize(base['doc_id'].astype(str))[0]
        countries = sorted(set(base['country']))
        self.country = base['country'].map({c: i for i, c in enumerate(countries)}).to_numpy()
        self.n_countries = len(countries)
        m0 = month_index(spec.start)
        self.cal = base['ym'].to_numpy(np.int64) - m0
        self.T = month_index(spec.end) - m0 + 1

    @staticmethod
    def _ar1(rng, shape: tuple, rho: float) -> np.ndarray:
        eps = rng.standard_normal(shape)
        out = np.empty(shape)
        out[:, 0] = eps[:, 0]
        scale = math.sqrt(1.0 - rho * rho)
        for t in range(1, shape[1]):
            out[:, t] = rho * out[:, t - 1] + scale * eps[:, t]
        return out

    def draw(self, rng, sc: dict, r: int) -> np.ndarray:
        E = math.sqrt(sc.get('idiosyncratic', 0.0)) * rng.standard_normal((self.n, r))
        if sc.get('document'):
            E += math.sqrt(sc['document']) * rng.standard_normal((self.doc.max() + 1, r))[self.doc]
        if sc.get('country_month'):
            E += math.sqrt(sc['country_month']) * self._ar1(rng, (self.n_countries, self.T, r), sc['rho'])[
                self.country, self.cal]
        if sc.get('common_month'):
            E += math.sqrt(sc['common_month']) * self._ar1(rng, (1, self.T, r), sc['rho'])[0][self.cal]
        return E


def with_row_ids(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out['_row'] = np.arange(len(out), dtype=np.int64)
    return out


# =========================================================================== bootstrap procedures (M1, slope)
class BlockProcedure:
    """A month-block procedure of :func:`replication.external.estimate.fit_spec` on fixed draws (design-only precomputation)."""

    kind = 'bootstrap'

    def __init__(self, name: str, panel_r: pd.DataFrame, spec: est.Spec, position: dict, cache: est.DrawCache,
                 targets: list | None = None):
        self.name, self.spec = name, spec
        sub = est.select(panel_r, spec)
        self.rows = np.array([position[i] for i in sub['_row'].to_numpy()], dtype=np.int64)
        post = est.post_indicator(sub, spec)
        w, _ = est.row_weights(sub, spec, post)
        design = est.build_design(sub, spec, post)
        struct = est.structural(design.X, design.names, design.groups, w)
        contrasts, status = est.reduce_contrasts(design, struct, w)
        if targets is not None:
            contrasts = {k: v for k, v in contrasts.items() if k in targets}
            status = {k: v for k, v in status.items() if k in targets}
        X = design.X[:, struct['kept']]
        n, p = X.shape
        wv = np.ones(n) if w is None else np.asarray(w, float)
        units, layers = est.unit_index(sub, spec)
        draws, self.M = cache.get(layers, spec.block_length)
        self.X, self.w, self.units, self.layers, self.contrasts = X, wv, units, layers, contrasts
        self.B = draws.B
        self.unavailable = {k: v['reason'] for k, v in status.items() if k not in contrasts}
        G_units = inf.UnitStats.build(X, np.zeros((n, 1)), units, layers.n_units, wv).G
        Gstar = np.tensordot(self.M, G_units, axes=(1, 0))
        eye = np.broadcast_to(np.eye(p), Gstar.shape)
        sol = inf.solve_gram(Gstar, eye, contrasts)
        R, valid = sol['estimates'], sol['estimable']
        self.P = sol['beta']                                              # (B + R, p, p): the draw's G*^+
        for b in np.flatnonzero(sol['ambiguous']):
            pinv = np.linalg.pinv(Gstar[b])
            self.P[b] = pinv
            for nm, c in contrasts.items():
                R[nm][b] = np.asarray(c) @ pinv
                valid[nm][b] = inf.estimable_direct(Gstar[b], np.asarray(c))[0]
        full = inf.solve_gram(G_units.sum(axis=0)[None], np.eye(p)[None], contrasts)
        self.G_units, self.P_full = G_units, full['beta'][0]
        self.items, self.R, self.index, self.R_full = [], {}, {}, {}
        for nm in contrasts:
            eff = inf.effective(valid[nm], self.B)
            if not eff['available'] or not full['estimable'][nm][0]:
                self.unavailable[nm] = eff.get('reason') or 'not estimable on the full sample'
                continue
            self.items.append(nm)
            self.index[nm] = eff['index']
            self.R[nm] = R[nm][eff['index']]
            self.R_full[nm] = full['estimates'][nm][0]
        order = np.argsort(units, kind='stable')
        self.unit_bounds = np.searchsorted(units[order], np.arange(layers.n_units + 1))
        self.order = order
        self.Xw = (X * wv[:, None])[order]
        self.n_units, self.p = layers.n_units, p
        self.record = {'spec': spec.record(), 'n': int(n), 'layers': layers.record(), 'B': int(self.B),
                       'block_length': int(spec.block_length), 'items': self.items, 'unavailable': self.unavailable,
                       'ambiguous_draws_pseudo_inverse': int(sol['ambiguous'].sum())}

    def new_counts(self) -> dict:
        return {nm: {'cov95': 0, 'cov90': 0, 'rej05': 0} for nm in self.items}

    def unit_scores(self, E_base: np.ndarray) -> np.ndarray:
        """H_u = X_u' W e_u per resampling unit: (units, p, replications)."""
        E = E_base[self.rows][self.order]
        r = E.shape[1]
        Hu = np.zeros((self.n_units, self.p, r))
        for u in range(self.n_units):
            a, b = self.unit_bounds[u], self.unit_bounds[u + 1]
            if b > a:
                Hu[u] = self.Xw[a:b].T @ E[a:b]
        return Hu

    def stats(self, E_base: np.ndarray, levels=(0.05, 0.10)) -> dict:
        """Per estimand and replication (column of ``E_base``): the estimate, the +1 p value and the half-widths at
        ``levels`` (order statistics of |t* - t|; 'q95' / 'q90' for 0.05 / 0.10), exactly as
        :func:`replication.external.inference.draw_inference`."""
        Hu = self.unit_scores(E_base)
        Hfull = Hu.sum(axis=0)
        Hstar = np.tensordot(self.M, Hu, axes=(1, 0))                 # (B + R, p, r)
        out = {}
        for nm in self.items:
            t = self.R_full[nm] @ Hfull
            star = np.einsum('bp,bpr->br', self.R[nm], Hstar[self.index[nm]])
            dev = np.abs(star - t)
            B_e = dev.shape[0]
            pos = {a: B_e - inf.r_max(B_e, a) - 1 for a in set(levels) | {0.05, 0.10}}
            valid = sorted({k for k in pos.values() if 0 <= k < B_e})
            part = np.partition(dev, valid, axis=0) if valid else dev
            exceed = (dev >= np.abs(t)).sum(axis=0)
            h = {a: (part[k] if 0 <= k < B_e else np.full(dev.shape[1], np.inf)) for a, k in pos.items()}  # r < 0: none
            out[nm] = {'estimate': t, 'p': (1 + exceed) / (B_e + 1), 'q95': h[0.05], 'q90': h[0.10], 'h': h, 'B_e': B_e}
        return out

    def accumulate(self, E_base: np.ndarray, counts: dict) -> None:
        for nm, s in self.stats(E_base).items():
            counts[nm]['cov95'] += int((np.abs(s['estimate']) <= s['q95']).sum())
            counts[nm]['cov90'] += int((np.abs(s['estimate']) <= s['q90']).sum())
            counts[nm]['rej05'] += int((s['p'] < 0.05).sum())


# =========================================================================== fixed-design procedures (event study)
class EventProcedures:
    """The event-study sensitivity estimators (:mod:`replication.external.escov`) on one shared fixed design."""

    kind = 'fixed'

    def __init__(self, panel_r: pd.DataFrame, spec: est.Spec, position: dict, estimator_specs: list, cfg: dict):
        ed = eventstudy.event_design(panel_r, spec)
        sub, design, contrasts = ed['sub'], ed['design'], ed['contrasts']
        self.rows = np.array([position[i] for i in sub['_row'].to_numpy()], dtype=np.int64)
        X = ed['X']
        w = ed['w']
        wv = np.ones(len(X)) if w is None else np.asarray(w, float)
        self.X = X
        self.A = np.linalg.pinv(X.T @ (X * wv[:, None])) @ (X * wv[:, None]).T
        self.names = [nm for nm in design.contrasts if nm in contrasts]
        self.C = np.stack([contrasts[nm] for nm in self.names])
        month = sub['ym'].to_numpy(np.int64)
        self.estimators = {}
        for spec_e in estimator_specs:
            obj = escov.prepare(spec_e, X, month, w, self.C)
            self.estimators[spec_e['name']] = {'obj': obj, 'crit95': obj.critical(0.95), 'crit90': obj.critical(0.90)}
        sample, pre_set, post_set = eventstudy.frozen_sets(spec)
        self.sample, self.pre_set, self.post_set = sample, pre_set, post_set
        self.pre_idx = {}
        for fam in FAMILIES:
            pre = [f'{fam}@{q}' for q in pre_set]
            self.pre_idx[fam] = [self.names.index(x) for x in pre] if all(x in self.names for x in pre) else None
        periods = eventstudy.event_period(sub['year'], sub['month'])
        uk = sub['country'].to_numpy() == 'UK'
        tb = cfg['tost_boundary']
        self.eps = float(tb['epsilon_sd'])                                # Var(e) = 1 in every scenario
        self.tost_families = [fam for fam in tb['families'] if self.pre_idx.get(fam)]
        shift = self.eps * (uk & (periods == '2021a'))
        self.delta_tost = self.C @ (self.A @ shift)
        hd = cfg['honestdid_in_restriction']
        d, mbar = float(hd['d_sd']), float(hd['Mbar'])
        path = {'2021a': d, **{q: d * (1 + (k + 1) * mbar) for k, q in enumerate(post_set)}}
        shift_hd = np.zeros(len(sub))
        for q, v in path.items():
            shift_hd[uk & (periods == q)] = v
        self.delta_hd = self.C @ (self.A @ shift_hd)
        self.hd_reps = int(hd['replications'])
        fam = hd['family']
        names_hd = [f'{fam}@{q}' for q in pre_set + post_set]
        self.hd_names = names_hd if all(x in self.names for x in names_hd) else None
        self.hd_idx = [self.names.index(x) for x in names_hd] if self.hd_names else None
        self.hd_path = path
        self.record = {'spec': spec.record(), 'n': int(len(sub)), 'coefficients': self.names,
                       'estimators': {k: v['obj'].record() for k, v in self.estimators.items()},
                       'tost_boundary_shift': {'rows': 'UK x 2021a', 'epsilon': self.eps},
                       'honestdid_path_uk': path}

    def new_counts(self) -> dict:
        k = len(self.names)
        return {name: {'cov95': np.zeros(k, int), 'cov90': np.zeros(k, int), 'rej05': np.zeros(k, int),
                       'singular': np.zeros(k, int),
                       'joint': {fam: {'rej05': 0, 'n': 0} for fam in FAMILIES if self.pre_idx[fam]},
                       'tost': {fam: 0 for fam in self.tost_families}} for name in self.estimators}

    def stats(self, E_base: np.ndarray) -> tuple:
        """(event estimates (k, r), {estimator: covariance (r, k, k)}) of the replications in ``E_base``."""
        E = E_base[self.rows]
        Bh = self.A @ E
        resid = E - self.X @ Bh
        return self.C @ Bh, {name: e['obj'].contrast_cov(resid) for name, e in self.estimators.items()}

    def accumulate(self, E_base: np.ndarray, counts: dict, keep_hd: list | None = None) -> None:
        est_, covs = self.stats(E_base)
        for name, e in self.estimators.items():
            V = covs[name]                                                 # (r, k, k)
            var = np.einsum('rkk->kr', V)
            ok = var > 0
            se = np.sqrt(np.where(ok, var, 1.0))
            c = counts[name]
            c['singular'] += (~ok).sum(axis=1)
            within95 = ok & (np.abs(est_) <= e['crit95'][:, None] * se)
            c['cov95'] += within95.sum(axis=1)
            c['cov90'] += (ok & (np.abs(est_) <= e['crit90'][:, None] * se)).sum(axis=1)
            c['rej05'] += (ok & ~within95).sum(axis=1)
            for fam, ix in self.pre_idx.items():
                if not ix:
                    continue
                Vs = V[:, ix][:, :, ix]
                b = est_[ix].T
                try:
                    sol = np.linalg.solve(Vs, b[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    sol = np.stack([np.linalg.pinv(m) @ v for m, v in zip(Vs, b)])
                T2 = np.einsum('ra,ra->r', b, sol)
                pv = escov.joint_p_vector(e['obj'], ix, T2)
                c['joint'][fam]['rej05'] += int(np.nansum(pv < 0.05))
                c['joint'][fam]['n'] += int(np.isfinite(pv).sum())
            eb = est_ + self.delta_tost[:, None]
            half = e['crit90'][:, None] * se
            for fam in c['tost']:
                ix = self.pre_idx[fam]
                inside = ok[ix] & (eb[ix] - half[ix] >= -self.eps) & (eb[ix] + half[ix] <= self.eps)
                c['tost'][fam] += int(inside.all(axis=0).sum())
            if keep_hd is not None and self.hd_idx is not None:
                have = sum(1 for x in keep_hd if x[0] == name)
                ix = self.hd_idx
                for r in range(min(est_.shape[1], max(0, self.hd_reps - have))):
                    keep_hd.append((name, (est_[ix, r] + self.delta_hd[ix]).copy(), V[r][np.ix_(ix, ix)].copy()))


# =========================================================================== driver
def build(panel_r: pd.DataFrame, sample: str, cfg: dict, cache: est.DrawCache) -> tuple:
    """(base rows, errors, procedures) of one sample."""
    spec = est.main_spec(sample)
    base = est.select(panel_r, spec)
    position = {int(i): k for k, i in enumerate(base['_row'].to_numpy())}
    from .pretrend import slope_spec
    procs = {}
    es_specs = []
    for name in cfg['procedures']:
        if name == 'M1':
            procs[name] = BlockProcedure(name, panel_r, spec, position, cache)
        elif name == 'slope':
            procs[name] = BlockProcedure(name, panel_r, slope_spec(sample), position, cache)
        elif name.startswith('slope_L'):
            L = int(name[len('slope_L'):])
            procs[name] = BlockProcedure(name, panel_r, slope_spec(sample, L), position, cache)
        elif name.startswith('es:'):
            wanted = name[3:]
            es_specs += [s for s in external_params('event_study')['sensitivity_estimators'] if s['name'] == wanted]
        else:
            raise ValueError(f'unknown simulation procedure {name!r}')
    events = EventProcedures(panel_r, spec, position, es_specs, cfg) if es_specs else None
    return base, Errors(base, spec), procs, events


def summarise_block(proc: BlockProcedure, counts: dict, n: int, thr: dict) -> dict:
    return {'items': {nm: {'coverage95': judge(c['cov95'], n, thr['coverage95'], 'coverage'),
                           'coverage90': judge(c['cov90'], n, thr['coverage90'], 'coverage'),
                           'size05': judge(c['rej05'], n, thr['size05'], 'size')} for nm, c in counts.items()}}


def summarise_event(events: EventProcedures, name: str, counts: dict, n: int, thr: dict) -> dict:
    c = counts[name]
    items = {}
    for k, nm in enumerate(events.names):
        m = n - int(c['singular'][k])
        items[nm] = {'coverage95': judge(int(c['cov95'][k]), n, thr['coverage95'], 'coverage'),
                     'coverage90': judge(int(c['cov90'][k]), n, thr['coverage90'], 'coverage'),
                     'size05': judge(int(c['rej05'][k]), max(m, 1), thr['size05'], 'size'),
                     'singular_replications': int(c['singular'][k])}
    joint = {fam: {'size05': judge(v['rej05'], max(v['n'], 1), thr['size05'], 'size'), 'n_finite': v['n']}
             for fam, v in c['joint'].items()}
    tost = {fam: rate(x, n) for fam, x in c['tost'].items()}
    return {'items': items, 'joint': joint, 'tost_boundary': tost}


def compact(proc_scen: dict) -> tuple:
    """(by_item, by_joint): scenario -> judgement strings per estimand (the labels attached to the outputs)."""
    by_item, by_joint = {}, {}
    for scen, block in proc_scen.items():
        for nm, v in block['items'].items():
            by_item.setdefault(nm, {})[scen] = {k: v[k]['judgement'] for k in ('coverage95', 'coverage90', 'size05')}
        for fam, v in (block.get('joint') or {}).items():
            by_joint.setdefault(fam, {})[scen] = v['size05']['judgement']
    return by_item, by_joint


def verdict_counts(proc_scen: dict) -> dict:
    out = {'通过': 0, '未定': 0, '失败': 0}
    for block in proc_scen.values():
        for v in block['items'].values():
            for k in ('coverage95', 'coverage90', 'size05'):
                out[v[k]['judgement']] += 1
        for v in (block.get('joint') or {}).values():
            out[v['size05']['judgement']] += 1
    return out


def run_honestdid_check(events: EventProcedures, kept: list, cfg: dict) -> dict:
    """Descriptive: the HonestDiD robust interval at M-bar under the in-restriction deviation path; the rate at which
    it contains the true target 0 (treatment effect 0), per estimator."""
    hd = cfg['honestdid_in_restriction']
    if events.hd_names is None:
        return {'status': inf.UNAVAILABLE, 'reason': 'the family is not estimable in this window'}
    workers = max(1, min(len(kept), 8))

    def one(job):
        return eventstudy.honest_interval_at(events.hd_names, job[1], job[2], float(hd['Mbar']))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, kept))
    out = {}
    for name in events.estimators:
        rows = [res for job, res in zip(kept, results) if job[0] == name]
        ok = [x for x in rows if x.get('status') == 'ok' and x.get('lb') is not None and x.get('ub') is not None]
        covered = sum(1 for x in ok if x['lb'] <= 0.0 <= x['ub'])
        out[name] = {'coverage_of_target_0': rate(covered, len(ok)) if ok else None, 'runs': len(rows),
                     'unavailable': len(rows) - len(ok),
                     'truncated': sum(1 for x in ok if x.get('truncated'))}
    return {'rule': hd, 'path_uk': events.hd_path, 'by_estimator': out,
            'reading': '描述性：零效应覆盖验证不了 HonestDiD；此处只描述在 Delta^RM 边界上的偏离路径下稳健区间覆盖真实目标 0 的比例'}


def simulate_sample(panel: pd.DataFrame, sample: str, cfg: dict | None = None, cache: est.DrawCache | None = None,
                    progress=None) -> dict:
    cfg = external_params('simulation') if cfg is None else cfg
    cache = est.DrawCache() if cache is None else cache
    panel_r = with_row_ids(panel)
    base, errors, procs, events = build(panel_r, sample, cfg, cache)
    R, batch = int(cfg['replications']), int(cfg['batch'])
    thr = cfg['thresholds']
    hd = cfg['honestdid_in_restriction']
    kept_hd = [] if (events is not None and sample == hd['sample']) else None
    seeds = np.random.SeedSequence([int(cfg['seed']), CONTROLS.index(sample)]).spawn(len(cfg['scenarios']))
    results = {name: {} for name in procs}
    ev_results = {name: {} for name in (events.estimators if events else {})}
    for (scen, sc), ss in zip(cfg['scenarios'].items(), seeds):
        if progress:
            progress(f'simulation {sample} {scen}')
        rng = np.random.Generator(np.random.PCG64(ss))
        counts = {name: proc.new_counts() for name, proc in procs.items()}
        ev_counts = events.new_counts() if events else None
        done = 0
        while done < R:
            r = min(batch, R - done)
            E = errors.draw(rng, sc, r)
            for name, proc in procs.items():
                proc.accumulate(E, counts[name])
            if events:
                events.accumulate(E, ev_counts, kept_hd if (kept_hd is not None and scen == hd['scenario']) else None)
            done += r
        for name, proc in procs.items():
            results[name][scen] = summarise_block(proc, counts[name], R, thr)
        for name in ev_results:
            ev_results[name][scen] = summarise_event(events, name, ev_counts, R, thr)
    out = {'sample': sample, 'n_rows': int(len(base)), 'replications': R, 'procedures': {}}
    for name, proc in procs.items():
        by_item, by_joint = compact(results[name])
        out['procedures'][name] = {'kind': proc.kind, 'design': proc.record, 'scenarios': results[name],
                                   'by_item': by_item, 'by_joint': by_joint, 'verdicts': verdict_counts(results[name])}
    for name, res in ev_results.items():
        by_item, by_joint = compact(res)
        obj = events.estimators[name]['obj']
        out['procedures'][f'es:{name}'] = {'kind': events.kind, 'label': obj.label, 'assumption': obj.assumption,
                                           'design': events.record, 'scenarios': res, 'by_item': by_item,
                                           'by_joint': by_joint, 'verdicts': verdict_counts(res)}
    descriptive = {'tost_boundary': {name: {scen: blk['tost_boundary'] for scen, blk in res.items()}
                                     for name, res in ev_results.items()},
                   'tost_reading': cfg['tost_boundary']['reading']}
    if kept_hd is not None:
        if progress:
            progress(f'simulation {sample} HonestDiD in-restriction ({len(kept_hd)} R runs)')
        descriptive['honestdid_in_restriction'] = run_honestdid_check(events, kept_hd, cfg)
    out['descriptive'] = descriptive
    return out


def run(panel: pd.DataFrame, progress=None) -> dict:
    """The simulation record of both samples (design only; no outcome is read)."""
    cfg = external_params('simulation')
    cache = est.DrawCache()
    return {'decision': cfg['decision'], 'rule': cfg, 'outcome_read': False,
            'samples': {sample: simulate_sample(panel, sample, cfg, cache, progress) for sample in CONTROLS}}
