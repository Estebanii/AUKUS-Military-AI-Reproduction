#!/usr/bin/env python3
"""H1 (Table 3, footnotes 72 and 78, and the alternative-encoder F values of Section V.2).

Baseline encoder: MANOVA on PCA-88 scores (period split, test 6 = pre-signing US vs UK from 2014), the 99%-variance
rule, the PCA-k robustness of test 6 (k = 3 ... 99; footnote 72), the PCA-k robustness of the three-country
full-sample MANOVA and the time robustness. Alternative encoders:
test 6 on their own PCA-88 of the common-support rows. Writes results/h1/<encoder>/{manova,h1,...}.json.
"""
import _common  # noqa: F401

from replication import analysis, data, output, pca, pipeline


def main() -> int:
    timer = output.Timer('02_h1_manova_table3')
    for encoder in data.ENCODERS:
        print(f'[02] {encoder}', flush=True)
        Y = pipeline.load_fit(encoder)['Y']
        meta = pipeline.encoder_meta(encoder)
        pca88 = pca.fit_pca(Y, 88)
        h1 = analysis.h1_summary(Y, meta, pca88)
        base = f'h1/{encoder}'
        output.write_json(f'{base}/manova.json', h1['k88'])
        output.write_json(f'{base}/h1.json', {k: v for k, v in h1.items() if k != 'k88'})
        if encoder == data.BASELINE:
            output.write_json(f'{base}/h1_test6_pc_robustness.json', analysis.h1_test6_pc_robustness(Y, meta))
            output.write_json(f'{base}/manova_pc_robustness.json', analysis.manova_pc_robustness(Y, meta))
            output.write_json(f'{base}/manova_time_robustness.json', analysis.manova_time_robustness(Y, meta, pca88))
    timer.done()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
