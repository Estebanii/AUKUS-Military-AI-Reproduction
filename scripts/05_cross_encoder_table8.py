#!/usr/bin/env python3
"""Table 8: the four alternative encoders on the 37,808 common-support occurrences.

For each alternative encoder the full analysis chain (as for the baseline, plus the reference-axis model, the
orientation of its own PC1 and the descriptive B = 9,999 re-run); then the two Holm families, the readings,
the Monte Carlo precision, the effect consistency, the baseline encoder on the same rows and the linear CKA; then the
same-axis robustness (cross-encoder test, Table 8): placebos and the pre-signing Wald on the reference axes and on each
encoder's own axes. Writes results/alternatives/<encoder>/*.json, results/cross_encoder/*.json and
results/tables/table8_*.csv.
"""
import _common  # noqa: F401

import numpy as np
import pandas as pd

from replication import cross_encoder as X
from replication import data, output, pipeline, same_axis_robustness
from replication.params import params

NAMES = ('manova', 'h1', 'manova_pc_robustness', 'manova_time_robustness', 'wcb', 'did', 'parallel_trends', 'h3',
         'did_h3_distances', 'same_axis', 'orientation', 'meta', 'rerun_descriptive')


def main() -> int:
    timer = output.Timer('05_cross_encoder_table8')
    bundle = pipeline.gpt2_bundle()
    axes = pipeline.load_axes()
    v1_coef = data.archived_json('original_outputs/wild_cluster_bootstrap_results.json')[
        'comparison']['PC1']['UK_x_post']['coef']
    v1_sign = 1 if v1_coef >= 0 else -1
    rows, raw1, raw2, Ys, analyses = {}, {}, {}, {}, {}
    for e in data.ALTERNATIVES:
        print(f'[05] analysis chain: {e}', flush=True)
        Y = pipeline.load_fit(e)['Y']
        meta = pipeline.encoder_meta(e)
        r = pipeline.run_analyses(Y, meta, bundle, axes, rerun=True)
        for name in NAMES:
            output.write_json(f'alternatives/{e}/{name}.json', r[name])
        r = output.plain(r)
        analyses[e] = r
        wcb, same, ori = r['wcb'], r['same_axis'], r['orientation']
        own = wcb['comparison']['PC1']['UK_x_post']
        ci = wcb['_details']['intervals_95']['PC1']['UK_x_post']
        o = X.orient(own['coef'], (ci['ci95_low'], ci['ci95_high']), ori['cos_pc1'])
        rows[e] = {'orientation': o, 'own_pc1': {'coef': own['coef'], 'p': own['p_bootstrap'],
                                                 'interval_95': [ci['ci95_low'], ci['ci95_high']]},
                   'same_axis': {'coef': same['UK_x_post']['PC1']['coef'], 'p': same['UK_x_post']['PC1']['p_bootstrap'],
                                 'interval_95': [same['intervals_95']['PC1']['ci95_low'],
                                                 same['intervals_95']['PC1']['ci95_high']]},
                   'pc1_pc2_eigengap': ori.get('pc1_pc2_eigengap'), 'rerun': r['rerun_descriptive']}
        raw1[e], raw2[e] = own['p_bootstrap'], same['UK_x_post']['PC1']['p_bootstrap']
        Ys[e] = Y
    f1, f2 = X.family(raw1), X.family(raw2)
    alpha = params('families')['alpha']
    print('[05] baseline encoder on the common-support rows', flush=True)
    U_t, _ = data.encoding(data.BASELINE, 'targets')
    U_a, _ = data.encoding(data.BASELINE, 'anchors')
    baseline = X.baseline_on_common_rows(U_t, U_a, bundle['anchor_words'], bundle['labels'],
                                         pipeline.common_rows('targets'), axes, data.analysis_meta(), v1_coef)
    reporting = X.reporting_section(rows, f1, f2, alpha, v1_coef, baseline)
    read = X.readings(rows, f1, f2, v1_sign, alpha, mc=reporting['mc_precision'],
                      effects=reporting['effect_consistency'])
    original = data.original_vectors()[pipeline.common_rows('targets')]
    allY = {**Ys, 'original vectors (common support)': original}
    names = list(allY)
    cka = {a: {b: X.linear_cka(allY[a], allY[b]) for b in names} for a in names}
    result = {'families': {X.FAMILY_NAMES['same_procedure']: f1, X.FAMILY_NAMES['same_axis']: f2},
              'rows': rows, 'readings': read, 'v1_reference_pc1_uk_x_post': v1_coef, 'cka_linear_Y': cka,
              'rule': params('families'), 'orientation_rule': params('orientation'), 'reporting': reporting}
    output.write_json('cross_encoder/cross_encoder.json', result)
    table = []
    for e in data.ALTERNATIVES:
        o, s = rows[e]['orientation'], rows[e]['same_axis']
        table.append({'encoder': data.DISPLAY[e], 'abs_cos_pc1': o['abs_cos_pc1'],
                      'direction_comparable': o['comparable_at_0.30'],
                      'same_procedure_coef_oriented': o['oriented_coef'],
                      'same_procedure_ci95_low': o['oriented_interval_95'][0],
                      'same_procedure_ci95_high': o['oriented_interval_95'][1],
                      'same_procedure_p': f1['raw'][e], 'same_procedure_holm_p': f1['holm'][e],
                      'same_axis_coef': s['coef'], 'same_axis_ci95_low': s['interval_95'][0],
                      'same_axis_ci95_high': s['interval_95'][1], 'same_axis_p': f2['raw'][e],
                      'same_axis_holm_p': f2['holm'][e], 'h1_test6_F': analyses[e]['h1']['test6_k88']['F'],
                      'h1_test6_p': analyses[e]['h1']['test6_k88']['p']})
    output.write_text('tables/table8_cross_encoder.csv', pd.DataFrame(table).to_csv(index=False))
    print('[05] same-axis robustness (cross-encoder test, Table 8)', flush=True)
    same_axis_robustness.run_all(axes, analyses, baseline_meta=data.analysis_meta())
    timer.done()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
