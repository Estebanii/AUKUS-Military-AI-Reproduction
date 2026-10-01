#!/usr/bin/env python3
"""Table 7 and footnote 81: the change of the US-UK distance from before to after the signing.

Baseline encoder: the main specification (own PCA-88, 2014-2025/11) and the reference PC1, PC1-3, first 88 axes, the
full 768-dimensional space and (f) the full space on the original manuscript's own sample; alternative encoders: the
first 88 reference axes. Pivot bootstrap over documents, B = 2,000, seed from replication.seeds. Hard checks against
the H1 / H2 / neighbour-word results (scripts 02-05), which therefore run first.
Writes results/distance/<encoder>.json, results/distance/summary.json and results/tables/table7_distance_change.csv.
"""
import _common  # noqa: F401

import pandas as pd

from replication import data, distance, output, pipeline

ROLE = {True: '主规格（E0 自身 88 主成分；另报参考轴空间以便并列）', False: '描述性稳健性（参考轴前 88 维）'}
CODE = {'deberta-v3-base': 'E0', 'ModernBERT-large': 'E1', 'roberta-large': 'E2', 'ettin-encoder-1b': 'E3',
        'deberta-v2-xlarge': 'E4'}


def reference_inputs(encoder: str) -> dict:
    if encoder == data.BASELINE:
        return {'wcb': output.read_json('h2/wcb.json'), 'same_axis': output.read_json('h2/same_axis.json'),
                'h1': output.read_json(f'h1/{encoder}/h1.json'),
                'did_h3_distances': output.read_json('h3/distances.json'), 'rows': 37866}
    return {'wcb': output.read_json(f'alternatives/{encoder}/wcb.json'),
            'same_axis': output.read_json(f'alternatives/{encoder}/same_axis.json'),
            'rows': int(len(pipeline.common_rows('targets')))}


def main() -> int:
    timer = output.Timer('06_distance_table7')
    axes = pipeline.load_axes()
    v1 = data.archived_json('original_outputs/did_h3_verification.json')['computed_results']['pairwise_distances']
    results = []
    for encoder in data.ENCODERS:
        print(f'[06] {encoder}', flush=True)
        baseline = encoder == data.BASELINE
        Y = pipeline.load_fit(encoder)['Y']
        meta = pipeline.encoder_meta(encoder)
        body = distance.compute(baseline, Y, meta, axes, reference_inputs(encoder), v1 if baseline else None)
        result = {'label': '事后描述性分析，不计入 Holm 检验族', 'encoder': encoder, 'code': CODE[encoder],
                  'role': ROLE[baseline], 'bias_correction': distance.BIAS_CORRECTION, **body,
                  'params': {**distance.PARAMS,
                             'spaces': list(distance.E0_SPACES if baseline else distance.E1E4_SPACES)}}
        output.write_json(f'distance/{encoder}.json', result)
        results.append(output.plain(result))
    summary = {'label': results[0]['label'], 'bias_correction': distance.BIAS_CORRECTION,
               'rows': distance.summary_rows(results), 'v1_sample': results[0]['v1_sample']}
    output.write_json('distance/summary.json', summary)
    table = []
    for row in summary['rows']:
        table.append({'encoder': data.DISPLAY[row['encoder']], 'space': row['space'],
                      'D2_pre': row['D2_pre'], 'D2_pre_ci95_low': row['D2_pre_interval_95'][0],
                      'D2_pre_ci95_high': row['D2_pre_interval_95'][1], 'D2_post': row['D2_post'],
                      'D2_post_ci95_low': row['D2_post_interval_95'][0], 'D2_post_ci95_high': row['D2_post_interval_95'][1],
                      'delta_sq': row['delta_sq'], 'delta_sq_ci95_low': row['delta_sq_interval_95'][0],
                      'delta_sq_ci95_high': row['delta_sq_interval_95'][1], 'p': row['p'], 'se_mc': row['se_mc'],
                      'reading': row['reading']['text']})
    f = summary['v1_sample']['US_UK']
    table.append({'encoder': 'DeBERTa-v3-base', 'space': 'y768_v1_sample (full sample)', 'D2_pre': f['pre']['D2'],
                  'D2_pre_ci95_low': f['pre']['D2_interval_95'][0], 'D2_pre_ci95_high': f['pre']['D2_interval_95'][1],
                  'D2_post': f['post']['D2'], 'D2_post_ci95_low': f['post']['D2_interval_95'][0],
                  'D2_post_ci95_high': f['post']['D2_interval_95'][1], 'delta_sq': f['delta_sq']['estimate'],
                  'delta_sq_ci95_low': f['delta_sq']['interval_95'][0], 'delta_sq_ci95_high': f['delta_sq']['interval_95'][1],
                  'p': f['delta_sq']['p'], 'se_mc': f['delta_sq']['se_mc'], 'reading': f['reading']['text']})
    output.write_text('tables/table7_distance_change.csv', pd.DataFrame(table).to_csv(index=False))
    timer.done()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
