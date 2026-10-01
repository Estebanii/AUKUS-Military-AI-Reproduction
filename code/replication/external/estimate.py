"""Models M0-M3 and the pre-trend slope model of the external-control analysis, their estimands, the structural
column handling (fixed-priority pivoted QR) and the month-block bootstrap fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import inference as inf
from .common import month_index
from .params import MEMBERS, external_params

GROUPS = ('constant', 'country levels', 'time terms', 'Post main effect', 'interactions', 'event-period terms',
          'term FE')


@dataclass(frozen=True)
class Spec:
    name: str
    sample: str = 'SG'                  # 'SG', 'CA' or 'SG+CA' (pooled control)
    start: str = '2014-01'
    end: str = '2025-11'
    model: str = 'M1'                   # M0 | M1 | M2 | M3
    weights: str = 'occurrence'         # occurrence | document | genre_fixed
    cut: str = 'month'                  # month (2021-09 whole in post) | day (2021-09-16)
    block_length: int = 6
    drop_months: tuple = ()             # donut
    genres: tuple = ()                  # () = all genres
    drop_countries: tuple = ()
    drop_terms: tuple = ()
    drop_docs: tuple = ()
    fake_cut: str | None = None         # placebo / fake cutoff (pre-period data only; 'YYYY-MM')
    description: str = ''
    pre_only: bool = False              # pre-trend slope diagnostic: pre-signing rows, one pre-month layer

    @property
    def controls(self) -> tuple:
        return tuple(self.sample.split('+'))

    def record(self) -> dict:
        return asdict(self)


def main_spec(sample: str, model: str = 'M1', **kw) -> Spec:
    w = external_params('windows')[sample if sample in ('SG', 'CA') else 'pooled']
    return Spec(name=f'{model}_{sample}', sample=sample, start=w['start'], end=w['end'], model=model, **kw)


# =========================================================================== samples
def select(panel: pd.DataFrame, spec: Spec, members_only: bool = False) -> pd.DataFrame:
    countries = set(MEMBERS) | (set() if (members_only or spec.model == 'M0') else set(spec.controls))
    countries -= set(spec.drop_countries)
    ym = panel['ym'].to_numpy()
    mask = panel['country'].isin(countries).to_numpy() & (ym >= month_index(spec.start)) & (ym <= month_index(spec.end))
    if spec.drop_months:
        lo, hi = (month_index(m) for m in spec.drop_months)
        mask &= ~((ym >= lo) & (ym <= hi))
    if spec.genres:
        mask &= panel['genre'].isin(spec.genres).to_numpy()
    if spec.drop_terms:
        mask &= ~panel['term'].str.lower().isin([t.lower() for t in spec.drop_terms]).to_numpy()
    if spec.drop_docs:
        mask &= ~panel['doc_id'].isin(spec.drop_docs).to_numpy()
    if spec.fake_cut is not None or spec.pre_only:
        mask &= ~panel['post'].to_numpy(bool)
    return panel.loc[mask].reset_index(drop=True)


def post_indicator(sub: pd.DataFrame, spec: Spec) -> np.ndarray:
    if spec.fake_cut is not None:
        return sub['ym'].to_numpy() >= month_index(spec.fake_cut)
    if spec.cut == 'day':
        return (sub['publish_date'].astype(str) >= external_params('sensitivity')['day_cut']).to_numpy()
    return sub['post'].to_numpy(bool)


def row_weights(sub: pd.DataFrame, spec: Spec, post: np.ndarray) -> tuple:
    n = len(sub)
    if spec.weights == 'occurrence':
        return None, {'weights': 'occurrence equal weight'}
    if spec.weights == 'document':
        counts = sub.groupby('doc_id')['doc_id'].transform('size').to_numpy(float)
        return 1.0 / counts, {'weights': 'document equal weight (1 / occurrences of the document)',
                              'documents': int(sub['doc_id'].nunique())}
    if spec.weights == 'genre_fixed':
        # needs every genre of a country in both periods; otherwise 不可用
        w = np.ones(n)
        rec, missing = {}, {}
        for country, rows in sub.groupby('country').groups.items():
            idx = np.asarray(list(rows))
            pre, post_ = idx[~post[idx]], idx[post[idx]]
            s_pre = sub['genre'].iloc[pre].value_counts(normalize=True)
            s_post = sub['genre'].iloc[post_].value_counts(normalize=True)
            pre_only = sorted(set(s_pre.index) - set(s_post.index))
            post_only = sorted(set(s_post.index) - set(s_pre.index))
            if pre_only or post_only:
                missing[country] = {'pre_only': pre_only, 'post_only': post_only}
            elif len(s_post):
                g = sub['genre'].iloc[post_].to_numpy()
                w[post_] = (s_pre / s_post).reindex(g).to_numpy()
            rec[country] = {'pre_shares': s_pre.round(6).to_dict(), 'post_shares': s_post.round(6).to_dict()}
        record = {'weights': 'fixed genre composition (pre-period shares per country)', 'countries': rec}
        if missing:
            record['unavailable'] = f'no common genre support in both periods: {missing}'
            return None, record
        return w, record
    raise ValueError(f'unknown weights {spec.weights!r}')


# =========================================================================== design matrices
@dataclass
class Design:
    X: np.ndarray
    names: list
    groups: list
    contrasts: dict
    report: dict = field(default_factory=dict)


def _dummies(values: np.ndarray, prefix: str, base=None):
    levels = sorted(set(values.tolist()), key=str)
    base = levels[0] if base is None else base
    cols, names = [], []
    for level in levels:
        if level == base:
            continue
        cols.append((values == level).astype(float))
        names.append(f'{prefix}{level}')
    return cols, names, base


def build_design(sub: pd.DataFrame, spec: Spec, post: np.ndarray) -> Design:
    n = len(sub)
    country = sub['country'].to_numpy()
    cols, names, groups = [np.ones(n)], ['intercept'], ['constant']
    if spec.model == 'slope':
        return slope_design(sub, spec)
    post_f = post.astype(float)
    if spec.model == 'M0':
        levels, interacted = ('UK', 'AU'), ('UK', 'AU')
    else:
        levels, interacted = MEMBERS, MEMBERS
    for g in levels:
        cols.append((country == g).astype(float))
        names.append(f'D_{g}')
        groups.append('country levels')
    if spec.model == 'M2':
        c, nm, base = _dummies(sub['year'].to_numpy(), 'year_')
        cols += c
        names += nm
        groups += ['time terms'] * len(nm)
    else:
        cols.append((sub['year'].to_numpy(float) - external_params('trend_origin_year')))
        names.append('time')
        groups.append('time terms')
    cols.append(post_f)
    names.append('post')
    groups.append('Post main effect')
    for g in interacted:
        cols.append((country == g).astype(float) * post_f)
        names.append(f'{g}_x_post')
        groups.append('interactions')
    if spec.model == 'M3':
        c, nm, base = _dummies(sub['term'].str.lower().to_numpy(), 'term_')
        cols += c
        names += nm
        groups += ['term FE'] * len(nm)
    X = np.column_stack(cols)
    contrasts = contrasts_for(names, spec.model)
    return Design(X=X, names=names, groups=groups, contrasts=contrasts,
                  report={'model': spec.model, 'base': 'US' if spec.model == 'M0' else '+'.join(spec.controls),
                          'n': n, 'p_full': X.shape[1]})


def slope_design(sub: pd.DataFrame, spec: Spec) -> Design:
    """intercept; D_US, D_UK, D_AU; t (linear month trend, per year); D_g x t."""
    n = len(sub)
    country = sub['country'].to_numpy()
    t = (sub['ym'].to_numpy(float) - month_index(f'{external_params("trend_origin_year")}-01')) / 12.0
    cols, names, groups = [np.ones(n)], ['intercept'], ['constant']
    for g in MEMBERS:
        cols.append((country == g).astype(float))
        names.append(f'D_{g}')
        groups.append('country levels')
    cols.append(t)
    names.append('month_trend')
    groups.append('time terms')
    for g in MEMBERS:
        cols.append((country == g).astype(float) * t)
        names.append(f'{g}_x_trend')
        groups.append('interactions')
    X = np.column_stack(cols)
    return Design(X=X, names=names, groups=groups, contrasts=contrasts_for(names, 'slope'),
                  report={'model': 'slope (pre-signing linear month trends)', 'base': '+'.join(spec.controls), 'n': n,
                          'p_full': X.shape[1], 'trend_unit': 'per year of a linear calendar-month trend'})


def contrasts_for(names: list, model: str) -> dict:
    def e(name):
        v = np.zeros(len(names))
        if name in names:
            v[names.index(name)] = 1.0
        return v
    if model == 'M0':
        return {'theta': e('UK_x_post')}
    if model == 'slope':
        return {'theta_slope': e('UK_x_trend') - e('US_x_trend'), 'slope_UK': e('UK_x_trend'),
                'slope_US': e('US_x_trend'), 'slope_AU': e('AU_x_trend')}
    return {'theta': e('UK_x_post') - e('US_x_post'), 'delta_UK': e('UK_x_post'), 'delta_US': e('US_x_post'),
            'delta_AU': e('AU_x_post')}


def structural(X: np.ndarray, names: list, groups: list, w=None, tol: float | None = None) -> dict:
    """Fixed-priority pivoted QR (modified Gram-Schmidt twice on the weighted columns)."""
    tol = external_params('qr_rel_tol') if tol is None else tol
    Xw = X if w is None else X * np.sqrt(np.asarray(w, float))[:, None]
    rank_of = {g: i for i, g in enumerate(GROUPS)}
    order = sorted(range(X.shape[1]), key=lambda j: (rank_of[groups[j]], j))
    basis, kept, dropped = [], [], []
    for j in order:
        x = Xw[:, j]
        norm = float(np.linalg.norm(x))
        if norm == 0.0:
            dropped.append({'column': names[j], 'group': groups[j], 'reason': 'all-zero column (no observation)'})
            continue
        r = x.copy()
        for _ in range(2):
            for q in basis:
                r -= (q @ r) * q
        rel = float(np.linalg.norm(r)) / norm
        if rel <= tol:
            dropped.append({'column': names[j], 'group': groups[j], 'reason': 'linearly dependent on higher-priority '
                            'columns', 'relative_residual': rel})
            continue
        basis.append(r / np.linalg.norm(r))
        kept.append(j)
    kept_sorted = sorted(kept)
    return {'kept': kept_sorted, 'kept_names': [names[j] for j in kept_sorted], 'dropped': dropped,
            'rank': len(kept), 'priority_order': [names[j] for j in order],
            'priority_groups': list(GROUPS), 'tolerance': tol,
            'rule': 'only all-zero columns and columns linearly dependent on higher-priority columns are '
                    'dropped (column space kept); priority: constant, country levels, time terms, Post main effect, '
                    'interactions, event-period terms, term FE'}


def reduce_contrasts(design: Design, struct: dict, w=None) -> tuple:
    Xw = design.X if w is None else design.X * np.sqrt(np.asarray(w, float))[:, None]
    reduced, status = {}, {}
    for name, c in design.contrasts.items():
        if not np.any(c):
            status[name] = {'estimable': False, 'reason': 'the contrast has no column in this model'}
            continue
        ok, residual = inf.estimable_direct(Xw, c)
        status[name] = {'estimable': bool(ok), 'residual': residual,
                        'reason': None if ok else 'not in the row space of the design (structural; 不可用)'}
        if ok:
            reduced[name] = c[struct['kept']]
    return reduced, status


# =========================================================================== fitting
def unit_index(sub: pd.DataFrame, spec: Spec) -> tuple:
    from .common import month_label
    first_post = external_params('extract')['post_first_month']
    if spec.fake_cut is not None:
        end = month_label(month_index(first_post) - 1)
        layers = inf.month_layers(spec.start, min(spec.end, end, key=month_index), spec.fake_cut)
        labels = sub['ym'].map(month_label).to_numpy()
    elif spec.pre_only:  # slope diagnostic: one layer of pre-signing months
        end = month_label(month_index(first_post) - 1)
        layers = inf.month_layers(spec.start, min(spec.end, end, key=month_index), first_post)
        labels = sub['ym'].map(month_label).to_numpy()
    elif spec.cut == 'day':
        layers = inf.day_layers(spec.start, spec.end, external_params('sensitivity')['day_cut'])
        cut_month = external_params('sensitivity')['day_cut'][:7]
        ym = sub['ym'].map(month_label).to_numpy()
        post_day = (sub['publish_date'].astype(str) >= external_params('sensitivity')['day_cut']).to_numpy()
        labels = np.where(ym == cut_month, np.where(post_day, f'{cut_month}b', f'{cut_month}a'), ym)
    else:
        layers = inf.month_layers(spec.start, spec.end, first_post)
        labels = sub['ym'].map(month_label).to_numpy()
    if spec.drop_months:  # donut: the removed months leave the calendar
        lo, hi = (month_index(m) for m in spec.drop_months)
        gone = {lab for lab in layers.labels if lo <= month_index(lab[:7]) <= hi}
        kept = [lab for lab in layers.labels if lab not in gone]
        n_pre = sum(1 for lab in layers.labels[:layers.n_pre] if lab not in gone)
        layers = inf.Layers(tuple(kept), n_pre, len(kept) - n_pre)
    index = {lab: i for i, lab in enumerate(layers.labels)}
    units = np.array([index[x] for x in labels], dtype=np.int64)
    return units, layers


class DrawCache:
    def __init__(self, B: int | None = None, seed: int | None = None):
        self.B, self.seed, self.cache = B, seed, {}

    def get(self, layers, block_length: int):
        key = (layers.labels, layers.n_pre, block_length)
        if key not in self.cache:
            draws = inf.BlockDraws.generate(layers, block_length, self.B, self.seed)
            self.cache[key] = (draws, draws.multiplicities())
        return self.cache[key]


def fit_spec(panel: pd.DataFrame, spec: Spec, responses: list, cache: DrawCache, *, point: bool = True,
             keep_draws: bool = False) -> dict:
    sub = select(panel, spec)
    post = post_indicator(sub, spec)
    w, w_record = row_weights(sub, spec, post)
    if w_record.get('unavailable'):
        return {'spec': spec.record(), 'n': int(len(sub)), 'weights': w_record,
                'estimands': {name: {r: {'status': inf.UNAVAILABLE, 'reason': w_record['unavailable']}
                                     for r in responses} for name in contrasts_for([], spec.model)}}
    design = build_design(sub, spec, post)
    struct = structural(design.X, design.names, design.groups, w)
    contrasts, status = reduce_contrasts(design, struct, w)
    X_r = design.X[:, struct['kept']]
    Y = sub[responses].to_numpy(float)
    units, layers = unit_index(sub, spec)
    draws, M = cache.get(layers, spec.block_length)
    out = {'spec': spec.record(), 'n': int(len(sub)), 'design': {**design.report, 'structural': struct,
                                                                 'contrast_status': status},
           'weights': w_record, 'layers': layers.record(), 'draws': draws.record(),
           'rows_by_country_period': rows_by_country_period(sub, post),
           'estimands': {}}
    if not contrasts:
        for name in design.contrasts:
            out['estimands'][name] = {r: {'status': inf.UNAVAILABLE, 'reason': status[name]['reason']} for r in responses}
        return out
    stats_ = inf.UnitStats.build(X_r, Y, units, layers.n_units, w)
    res = inf.fit_draws(stats_, M, contrasts, X=X_r, Y=Y, units=units, w=w)
    pt = inf.point_fit(X_r, Y, contrasts, w) if point else None
    out['fallback_draws'] = len(res['fallback_draws'])
    for name in design.contrasts:
        if name not in contrasts:
            out['estimands'][name] = {r: {'status': inf.UNAVAILABLE, 'reason': status[name]['reason']} for r in responses}
            continue
        eff = inf.effective(res['estimable'][name], draws.B)
        per = {}
        for j, r in enumerate(responses):
            if not eff['available']:
                per[r] = {'status': inf.UNAVAILABLE, 'reason': eff['reason'], 'failures': eff['failures']}
                continue
            d = res['estimates'][name][eff['index'], j]
            entry = {'status': 'ok', 'failures': eff['failures'], 'replacements_used': eff.get('replacements_used', 0)}
            if point:
                entry.update(inf.draw_inference(float(pt['estimates'][name][j]), d))
            else:
                entry.update({'B_e': int(d.size), 'se': float(np.std(d, ddof=1))})
            if keep_draws:
                entry['_draws'] = d
            per[r] = entry
        out['estimands'][name] = per
    if keep_draws:
        out['_valid'] = {k: v for k, v in res['estimable'].items()}
        out['_all_estimates'] = res['estimates']
    return out


def rows_by_country_period(sub: pd.DataFrame, post: np.ndarray) -> dict:
    table = pd.crosstab(sub['country'].to_numpy(), np.where(post, 'post', 'pre'))
    return {f'{c}_{p}': int(table.loc[c, p]) for c in table.index for p in table.columns}


def locked_sd(panel: pd.DataFrame, spec: Spec, responses: list) -> dict:
    sub = select(panel, spec)
    pre = ~sub['post'].to_numpy(bool)
    return {r: float(np.std(sub.loc[pre, r].to_numpy(float), ddof=1)) for r in responses}


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj
