#!/usr/bin/env python3
"""H2 with the baseline encoder: Table 4 (subgroup DiD with a linear time trend, wild cluster bootstrap), Table 5
(parallel-trends event study, joint Wald), Table 6 (placebos with fake cut-offs 2020 and 2021), the placebos with the
time trend (footnote 80), the robustness models, the same-axis model on the reference axes, the orientation of the
own PC1, and Figure 3 (event-study coefficients). Writes results/h2/{wcb,did,parallel_trends,placebo_with_trend,
same_axis,orientation,meta}.json and results/figures/figure3_event_study.*
"""
import _common  # noqa: F401

from replication import analysis, data, figures, output, pca, pipeline


def main() -> int:
    timer = output.Timer('03_h2_did_tables4_6_fig3')
    encoder = data.BASELINE
    Y = pipeline.load_fit(encoder)['Y']
    meta = pipeline.encoder_meta(encoder)
    axes = pipeline.load_axes()
    pca88 = pca.fit_pca(Y, 88)
    h2 = analysis.h2_models(Y, meta)
    output.write_json('h2/wcb.json', h2['wcb'])
    output.write_json('h2/did.json', h2['did'])
    output.write_json('h2/parallel_trends.json', h2['parallel_trends'])
    output.write_json('h2/placebo_with_trend.json', analysis.placebo_with_trend(h2['scores3'], meta))
    scores = pca.scores(Y, axes['mean'], axes['components'][:3])
    same_axis = analysis.h2_wcb_main(scores, meta, axes['explained_variance_ratio'][:3])
    output.write_json('h2/same_axis.json', {
        'wcb_on_reference_axes': same_axis,
        'UK_x_post': {pc: same_axis['comparison'][pc]['UK_x_post'] for pc in ('PC1', 'PC2', 'PC3')},
        'intervals_95': {pc: same_axis['_details']['intervals_95'][pc]['UK_x_post'] for pc in ('PC1', 'PC2', 'PC3')}})
    eigengap = float(h2['pca3'].explained_variance_[0] - h2['pca3'].explained_variance_[1])
    output.write_json('h2/orientation.json', {**analysis.orientation(h2['pca3'].components_, axes['components']),
                                              'pc1_pc2_eigengap': eigengap})
    output.write_json('h2/meta.json', {
        'n_pc': int(pca88.n_components_), 'pca_solver': pca88.svd_solver, 'pca3_solver': h2['pca3'].svd_solver,
        'pca88_n_samples': int(pca88.n_samples_), 'pca3_n_samples': int(h2['pca3'].n_samples_),
        'wcb_seeds': h2['wcb']['_details']['seeds'],
        'pca3_explained_variance': h2['pca3'].explained_variance_ratio_.tolist(), 'pc1_pc2_eigengap': eigengap})
    figures.figure3_event_study(h2['parallel_trends'], 'figures/figure3_event_study')
    timer.done()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
