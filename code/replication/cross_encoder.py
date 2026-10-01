"""The comparison across encoders (Table 8): four alternative encoders (ModernBERT-large, RoBERTa-large,
Ettin-encoder-1b, DeBERTa-v2-xlarge) on the 37,808 common-support occurrences.

* Same procedure: each encoder's own PCA; its PC1 is oriented by the sign of the inner product of its loading with
  the reference PC1 loading (never by a coefficient); |cos| < 0.30 means "direction not comparable" (p and family
  membership kept).
* Same axis: the encoder's Y projected on the reference axes (the PCA-88 of the original submission's vectors,
  ``data.original_vectors``; not of the re-encoded baseline).
* Two Holm families over the four encoders (PC1 UK x Post, wild cluster bootstrap p, B = 1,000); read separately.
* Reporting additions (descriptive, never changing a reading): the Monte Carlo SE of each bootstrap p and a boundary
  flag when p +/- 2 se_MC changes the Holm reading; a B = 9,999 re-run of every member; whether each 95% interval
  contains the original submission's PC1 UK x Post coefficient; the standardized effect; the baseline encoder on the
  same rows; linear CKA (Kornblith et al. 2019) between the encoders' Y.
"""
from __future__ import annotations

import numpy as np

from . import analysis, pca
from .amatrix import fit_a, transform
from .data import ALTERNATIVES
from .params import params

FAMILY_NAMES = {'same_procedure': 'same_procedure (PC1, Holm over 4)', 'same_axis': 'same_axis (PC1, Holm over 4)'}
FAMILY_KEYS = {'same_procedure': 'same_procedure_PC1', 'same_axis': 'same_axis_PC1'}


def linear_cka(X, Y) -> float:
    """Linear CKA (Kornblith et al. 2019) of two row-aligned matrices (feature-space form)."""
    X = np.asarray(X, np.float64) - np.asarray(X, np.float64).mean(axis=0)
    Y = np.asarray(Y, np.float64) - np.asarray(Y, np.float64).mean(axis=0)
    xy = np.linalg.norm(X.T @ Y) ** 2
    return float(xy / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p values (family = the keys)."""
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m, running, out = len(items), 0.0, {}
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * float(p)))
        out[name] = running
    return out


def family(raw: dict) -> dict:
    """Raw p of the four members and their Holm-adjusted p (the family size is 4)."""
    members = {e: float(raw[e]) for e in ALTERNATIVES}
    return {'raw': members, 'holm': holm(members), 'size': len(members)}


def orient(coef: float, interval: tuple, cos_pc1: float) -> dict:
    o = params('orientation')
    sign = 1 if cos_pc1 >= 0 else -1
    lo, hi = sorted((sign * interval[0], sign * interval[1]))
    return {'cos_pc1': cos_pc1, 'abs_cos_pc1': abs(cos_pc1), 'sign': sign, 'oriented_coef': sign * coef,
            'oriented_interval_95': [lo, hi], 'comparable_at_0.30': abs(cos_pc1) >= o['primary_threshold'],
            'descriptive': {str(t): abs(cos_pc1) >= t for t in o['descriptive_thresholds']},
            'direction': ('可比（|cos| ≥ 0.30，按载荷内积符号定向；不证明轴语义一致）' if abs(cos_pc1) >= o['primary_threshold']
                          else '方向不可比（|cos| < 0.30；p 与族成员资格保留）')}


def _sign_text(same: bool) -> str:
    return '同号' if same else '不同号'


def label_text(key: str) -> str:
    return params('cross_encoder_reporting')['labels'][key]


def _holm_text(family_name: str, rejected: bool, alpha: float) -> str:
    return (f'{family_name} Holm 校正 p < {alpha:g}（拒绝零假设）' if rejected
            else f'{family_name} Holm 校正 p ≥ {alpha:g}（{label_text("holm_not_rejected")}）')


def readings(rows: dict, f1: dict, f2: dict, v1_sign: int, alpha: float = 0.05, mc: dict | None = None,
             effects: dict | None = None) -> dict:
    """The pre-stated conditions and their labels. Same procedure: direction comparable, oriented PC1
    UK x Post of the original sign, Holm p < alpha. Same axis: reference-axis PC1 UK x Post of the original sign,
    Holm p < alpha. The two families are read separately."""
    out = {}
    for e in ALTERNATIVES:
        r = rows[e]
        o = r['orientation']
        comparable = bool(o['comparable_at_0.30'])
        sp_sign, sp_rejected = bool(np.sign(o['oriented_coef']) == v1_sign), bool(f1['holm'][e] < alpha)
        sa_sign, sa_rejected = bool(np.sign(r['same_axis']['coef']) == v1_sign), bool(f2['holm'][e] < alpha)
        sp = (f'定向后与原稿{_sign_text(sp_sign)}；{_holm_text("同程序族", sp_rejected, alpha)}' if comparable
              else '方向不可比（|cos| < 0.30）；p 与族成员资格保留')
        sa = f'参考轴 PC1 与原稿{_sign_text(sa_sign)}；{_holm_text("同轴族", sa_rejected, alpha)}'
        texts = {'same_procedure': sp, 'same_axis': sa}
        for fam in texts:
            effect = ((effects or {}).get(fam) or {}).get(e) or {}
            if effect.get('reading') and (fam == 'same_axis' or comparable):
                texts[fam] += f'；{effect["reading"]}'
            flag = (((mc or {}).get(fam) or {}).get(e) or {}).get('flag')
            if flag:
                texts[fam] += f'；{flag}'
        out[e] = {**texts, 'conditions': {
            'same_procedure': {'direction_comparable_at_0.30': comparable, 'same_sign_as_v1': sp_sign,
                               'holm_p_below_alpha': sp_rejected},
            'same_axis': {'same_sign_as_v1': sa_sign, 'holm_p_below_alpha': sa_rejected}}}
    return out


def mc_se(p: float, B: int) -> float:
    """Binomial Monte Carlo standard error of a bootstrap p from B draws."""
    return float(np.sqrt(p * (1 - p) / B))


def p_display(p: float, B: int) -> str:
    """A centred-tail p of 0 is shown as "< 0.001" (< 1/B at B = 1000), never as an exact 0."""
    return params('cross_encoder_reporting')['p_zero_display'] if p == 0 else f'{p:.3f}'


def mc_precision(raw: dict, alpha: float = 0.05, B: int | None = None, k: float | None = None) -> dict:
    """Per member: se_MC = sqrt(p(1 - p)/B), the tail count, and the Holm reading recomputed with its own p replaced by
    p - k*se_MC and p + k*se_MC (clipped to [0, 1]); a change of reading adds the boundary flag."""
    p_ = params('cross_encoder_reporting')
    B = p_['mc_B'] if B is None else B
    k = p_['se_multiplier'] if k is None else k
    fam = family(raw)
    out = {}
    for e in ALTERNATIVES:
        p = fam['raw'][e]
        se = mc_se(p, B)
        lo, hi = max(0.0, p - k * se), min(1.0, p + k * se)
        rejected = fam['holm'][e] < alpha
        at_lo = holm({**fam['raw'], e: lo})[e] < alpha
        at_hi = holm({**fam['raw'], e: hi})[e] < alpha
        boundary = bool(at_lo != rejected or at_hi != rejected)
        out[e] = {'p': p, 'p_display': p_display(p, B), 'tail_count': int(round(p * B)), 'B': B, 'se_mc': se,
                  'p_minus': lo, 'p_plus': hi, 'holm_p': fam['holm'][e], 'holm_rejected': bool(rejected),
                  'holm_rejected_at_p_minus': bool(at_lo), 'holm_rejected_at_p_plus': bool(at_hi),
                  'boundary': boundary, 'flag': p_['boundary_flag'] if boundary else None}
    return out


def effect_consistency(coef: float, interval, pre_sd: float | None, v1_coef: float,
                       holm_rejected: bool | None = None) -> dict:
    """Whether the 95% interval contains the original PC1 UK x Post coefficient and excludes 0, and the standardized
    effect coef / SD(pre-signing PC1 scores on the same axis, ddof = 1)."""
    lo, hi = sorted(float(v) for v in interval)
    contains = bool(lo <= v1_coef <= hi)
    reading = label_text('consistent_but_imprecise') if contains and holm_rejected is False else None
    return {'coef': float(coef), 'interval_95': [lo, hi], 'v1_coef': float(v1_coef), 'contains_v1_coef': contains,
            'excludes_0': bool(not lo <= 0 <= hi), 'pre_signing_sd': pre_sd,
            'standardized_effect': (float(coef) / pre_sd) if pre_sd else None, 'holm_rejected': holm_rejected,
            'reading': reading}


def reporting_section(rows: dict, f1: dict, f2: dict, alpha: float, v1_coef: float, baseline: dict) -> dict:
    """Monte Carlo precision of every raw p, the descriptive re-run of all 8 members and the effect consistency, with
    the baseline encoder on the same rows beside them."""
    mc = {'same_procedure': mc_precision({e: f1['raw'][e] for e in rows}, alpha),
          'same_axis': mc_precision({e: f2['raw'][e] for e in rows}, alpha)}
    rerun, effects = {'same_procedure': {}, 'same_axis': {}}, {'same_procedure': {}, 'same_axis': {}}
    for e, row in rows.items():
        desc = row['rerun']
        o = row['orientation']
        primary = {'same_procedure': row['own_pc1']['coef'], 'same_axis': row['same_axis']['coef']}
        for fam, key in FAMILY_KEYS.items():
            r = desc['rerun'][key]
            rerun[fam][e] = {**r, 'coef_equals_primary': r['coef'] == primary[fam]}
        effects['same_procedure'][e] = {
            **effect_consistency(o['oriented_coef'], o['oriented_interval_95'],
                                 desc['pre_signing_sd']['same_procedure_PC1']['sd'], v1_coef, f1['holm'][e] < alpha),
            'direction_comparable_at_0.30': o['comparable_at_0.30'], 'oriented': True}
        effects['same_axis'][e] = effect_consistency(
            row['same_axis']['coef'], row['same_axis']['interval_95'], desc['pre_signing_sd']['same_axis_PC1']['sd'],
            v1_coef, f2['holm'][e] < alpha)
    return {'rule': params('cross_encoder_reporting'), 'mc_precision': mc, 'descriptive_rerun': rerun,
            'effect_consistency': effects, 'members_rerun': sum(len(v) for v in rerun.values()),
            'e0_common_support_baseline': baseline}


def baseline_on_common_rows(U_t, U_a, anchor_words, labels: dict, rows, axes: dict, meta, v1_coef: float) -> dict:
    """The baseline encoder computed the way the members are (reported only; not a family member): its A on all
    anchors, Y of the common-support target rows, own PCA-3 of those rows (oriented by the reference PC1 loading) and
    the reference axes; for PC1 UK x Post of each, the primary WCB (B = 1,000, seed 42) coefficient, p and 95%
    interval, the descriptive re-run, the pre-signing SD and the effect consistency."""
    fit = fit_a(np.asarray(U_a), anchor_words, labels)
    rows = np.asarray(rows, np.int64)
    Y = transform(np.asarray(U_t)[rows], fit.A, fit.U_mean)
    sub = meta.iloc[rows].reset_index(drop=True)
    pca3 = pca.fit_pca(Y, 3)
    ori = analysis.orientation(pca3.components_, axes['components'])
    scores = {'same_procedure_PC1': pca3.transform(Y)[:, 0],
              'same_axis_PC1': pca.scores(Y, axes['mean'], axes['components'][:3])[:, 0]}
    X, names = analysis.wcb_features(sub)
    j = names.index('UK_x_post')
    wcb = params('wcb')
    out = {'rows': int(len(rows)), 'orientation': {k: ori[k] for k in ('cos_pc1', 'sign', 'comparable_at_0.30')},
           'rule': 'reported only; the baseline encoder is not a family member'}
    for key, pc1 in scores.items():
        r = analysis.wcb_v1_main_bootstrap(X, pc1, sub['doc_id'].to_numpy(), wcb['n_bootstrap'], wcb['seed_base'])
        sign = ori['sign'] if key == 'same_procedure_PC1' else 1
        coef, lo, hi = sign * r['beta'][j], sign * r['ci_low'][j], sign * r['ci_high'][j]
        out[key] = {'primary': {'coef_raw': float(r['beta'][j]), 'p': float(r['p_val_bootstrap'][j]),
                                'interval_95_raw': [float(r['ci_low'][j]), float(r['ci_high'][j])]},
                    'rerun': analysis.wcb_uk_post_rerun(pc1, sub),
                    'pre_signing_sd': analysis.pre_signing_sd(pc1, sub)}
        out[key]['effect_consistency'] = effect_consistency(coef, (lo, hi), out[key]['pre_signing_sd']['sd'], v1_coef)
    return out
