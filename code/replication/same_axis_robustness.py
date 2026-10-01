"""Same-axis robustness of the alternative encoders (cross-encoder test, Table 8; a post-hoc description in no test family).

For each alternative encoder (and for the baseline encoder cut to the 37,808 common-support rows) the robustness
models of :mod:`replication.analysis` run unchanged on the reference axes PC1-PC3 and stand beside the same models on
the encoder's own axes:

* placebos on the real pre-signing sample with fake cut-offs year >= 2020 and year >= 2021;
* the parallel-trends event study (US-UK 2014-2024, base year 2021) with the joint Wald test;
* the year fixed-effects model with the Post main effect, and the 2017+ sample;
* the main model (for context).

Wild cluster bootstrap B = 1,000, seeds 42/43/44; se_MC = sqrt(p (1 - p) / B) beside every bootstrap p; the own PC1 is
oriented by the sign of its loading's inner product with the reference PC1 loading, and marked "direction not
comparable" when |cos| < 0.30 (p kept). Writes results/cross_encoder/same_axis_robustness/<encoder>.json, the summary
and a CSV table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis, data, output, pca, pipeline
from .cross_encoder import mc_se, p_display
from .params import params

LABEL = '事后描述性分析，不计入 Holm 检验族'
PCS = ('PC1', 'PC2', 'PC3')
MODELS = (('placebo_2020', 'model_4a_placebo_2020', ('UK_x_fake_post', 'AU_x_fake_post'), 'placebo'),
          ('placebo_2021', 'model_4b_placebo_2021', ('UK_x_fake_post', 'AU_x_fake_post'), 'placebo'),
          ('year_fe_with_post', 'model_2b_year_fe_with_post', ('UK_x_post',), 'effect'),
          ('restricted_2017', 'model_5_restricted', ('UK_x_post',), 'effect'))
NOT_COMPARABLE = '方向不可比（|cos|<0.30）'
SUMMARY_ROWS = ('placebo_2020.UK_x_fake_post', 'placebo_2021.UK_x_fake_post', 'pretrend_wald',
                'year_fe_with_post.UK_x_post', 'restricted_2017.UK_x_post')


def variance_shares(Y, S) -> list:
    """Share of the total variance of Y (ddof = 1) on each axis."""
    total = float(np.var(np.asarray(Y, dtype=np.float64), axis=0, ddof=1).sum())
    return [float(np.var(S[:, k], ddof=1) / total) for k in range(S.shape[1])]


def robustness(S3, meta, variance3) -> dict:
    return {'did': analysis.did_robustness(S3, meta, variance3),
            'parallel_trends': analysis.parallel_trends_paper(S3, meta, variance3),
            'main_wcb': analysis.h2_wcb_main(S3, meta, variance3)}


def same_axis(Y, meta, axes) -> dict:
    S3 = pca.scores(Y, axes['mean'], axes['components'][:3])
    shares = variance_shares(Y, S3)
    return {'scores3': S3, 'variance_share_of_Y': shares,
            'reference_variance_ratio_on_v1': [float(v) for v in axes['explained_variance_ratio'][:3]],
            **robustness(S3, meta, shares)}


def own_axis_computed(Y, meta, axes) -> dict:
    """The baseline encoder on the common rows: own PCA-3 fitted on these rows (full solver), oriented by the reference
    PC1 loading."""
    pca3 = pca.fit_pca(Y, 3)
    S3 = pca3.transform(Y)
    return {'scores3': S3, 'orientation': analysis.orientation(pca3.components_, axes['components']),
            'explained_variance_ratio': [float(v) for v in pca3.explained_variance_ratio_],
            **robustness(S3, meta, pca3.explained_variance_ratio_)}


def _cell(p, B: int, kind: str) -> dict:
    out = {'p': float(p), 'p_display': p_display(float(p), B), 'se_mc': mc_se(float(p), B),
           'tail_count': int(round(float(p) * B)), 'B': B}
    if kind == 'placebo':
        out['label'] = analysis.placebo_outcome(p)['label']
    else:
        out['label'] = '拒绝系数为零的零假设（p ≤ .05）' if p <= 0.05 else '未拒绝系数为零的零假设（p > .05）'
    return out


def model_row(model: dict, pc: str, var: str, sign: int, kind: str, B: int) -> dict:
    r = model[pc][var]
    iv = ((model.get('intervals_95') or {}).get(pc) or {}).get(var)
    interval = sorted((sign * iv[0], sign * iv[1])) if iv else None
    return {'coef': sign * float(r['coef']), 'interval_95': interval, **_cell(r['boot_p'], B, kind)}


def main_row(wcb: dict, pc: str, sign: int, B: int) -> dict:
    r = wcb['comparison'][pc]['UK_x_post']
    iv = wcb['_details']['intervals_95'][pc]['UK_x_post']
    return {'coef': sign * float(r['coef']), 'interval_95': sorted((sign * iv['ci95_low'], sign * iv['ci95_high'])),
            **_cell(r['p_bootstrap'], B, 'effect')}


def pretrend_row(pt: dict, pc: str) -> dict:
    r = pt['parallel_trends_test'][pc]['UK']
    label = r.get('label') or ('拒绝签署前联合零假设' if r['p_value'] <= 0.05 else '未拒绝签署前联合零假设；不证明平行趋势')
    return {'wald': float(r['wald_stat']), 'df': int(r['df']), 'p': float(r['p_value']),
            'n_significant': r.get('n_significant'), 'label': label, 'se_mc': None,
            'note': 'χ² p（完整 bootstrap 协方差）；se_MC 不适用'}


def table(did: dict, pt: dict, wcb: dict, signs: dict) -> dict:
    B = params('wcb')['n_bootstrap']
    out = {}
    for pc in PCS:
        sign = signs.get(pc, 1)
        rows = {f'{key}.{var}': model_row(did[model], pc, var, sign, kind, B)
                for key, model, variables, kind in MODELS for var in variables}
        rows['pretrend_wald'] = pretrend_row(pt, pc)
        rows['main.UK_x_post'] = main_row(wcb, pc, sign, B)
        out[pc] = rows
    return out


def direction(orientation: dict) -> dict:
    comparable = bool(orientation['comparable_at_0.30'])
    return {'cos_pc1': float(orientation['cos_pc1']), 'abs_cos_pc1': abs(float(orientation['cos_pc1'])),
            'sign': int(orientation['sign']), 'comparable_at_0.30': comparable,
            'note': None if comparable else NOT_COMPARABLE}


def build(slot: str, baseline_e0: bool, Y, meta, axes, own: dict) -> dict:
    same = same_axis(Y, meta, axes)
    sign = int(own['orientation']['sign'])
    dirn = direction(own['orientation'])
    result = {'label': LABEL, 'slot': slot, 'baseline_e0': baseline_e0, 'rows': int(len(Y)),
              'rule': 'post-hoc description; the robustness models with identical parameters on the reference axes '
                      'PC1-PC3; WCB B = 1000, seeds 42/43/44; in no family',
              'same_axis': {k: v for k, v in same.items() if k != 'scores3'},
              'own_axis': {'source': own['source'], 'orientation': own['orientation'], 'direction_pc1': dirn},
              'table': {'same_axis': table(same['did'], same['parallel_trends'], same['main_wcb'], {}),
                        'own_axis': table(own['did'], own['parallel_trends'], own['wcb'], {'PC1': sign})}}
    for row in result['table']['own_axis']['PC1'].values():
        row.update(direction_comparable=dirn['comparable_at_0.30'], direction_note=dirn['note'])
    return result, same


def run_all(axes: dict, analyses: dict, baseline_meta) -> dict:
    """Every encoder; ``analyses``: the alternative encoders' analysis results (plain dicts, as written)."""
    results = {}
    rows = pipeline.common_rows('targets')
    Y = pipeline.load_fit(data.BASELINE)['Y'][rows]
    meta = baseline_meta.iloc[rows].reset_index(drop=True)
    computed = own_axis_computed(Y, meta, axes)
    own = {'source': '在共同支持上按同一程序计算（基线编码器全样本分析的 Y 取共同支持行）',
           'orientation': computed['orientation'], 'did': computed['did'],
           'parallel_trends': computed['parallel_trends'], 'wcb': computed['main_wcb']}
    result, _ = build(data.BASELINE, True, Y, meta, axes, output.plain(own))
    result['checks'] = {}
    results[f'{data.BASELINE}（共同支持基线）'] = result
    for e in data.ALTERNATIVES:
        a = analyses[e]
        Ye = pipeline.load_fit(e)['Y']
        own = {'source': '该编码器分析的 did 与 parallel_trends 结果', 'orientation': a['orientation'], 'did': a['did'],
               'parallel_trends': a['parallel_trends'], 'wcb': a['wcb']}
        result, same = build(e, False, Ye, pipeline.encoder_meta(e), axes, own)
        mine = same['main_wcb']['comparison']
        reported = a['same_axis']['UK_x_post']
        result['checks'] = {'same_axis_main_equals_analysis': all(
            mine[pc]['UK_x_post'][k] == reported[pc][k] for pc in PCS for k in ('coef', 'p_bootstrap'))}
        if not result['checks']['same_axis_main_equals_analysis']:
            raise RuntimeError(f'{e}: the same-axis main WCB differs from the analysis (other axes or Y)')
        results[e] = result
    for key, result in results.items():
        slot = result['slot']
        output.write_json(f'cross_encoder/same_axis_robustness/{slot}.json', result)
    summary = {'label': LABEL, 'encoders': {k: {'status': 'computed', **output.plain(v)} for k, v in results.items()}}
    output.write_json('cross_encoder/same_axis_robustness/summary.json', summary)
    table_rows = []
    for key, result in results.items():
        s, o = result['table']['same_axis']['PC1'], result['table']['own_axis']['PC1']
        d = result['own_axis']['direction_pc1']
        for row in SUMMARY_ROWS:
            table_rows.append({'encoder': key, 'abs_cos_pc1': d['abs_cos_pc1'], 'direction_comparable':
                               d['comparable_at_0.30'], 'analysis': row,
                               'reference_axis_coef': s[row].get('coef'), 'reference_axis_p': s[row]['p'],
                               'own_axis_coef': o[row].get('coef'), 'own_axis_p': o[row]['p']})
    output.write_text('tables/table8_same_axis_robustness.csv', pd.DataFrame(table_rows).to_csv(index=False))
    return results
