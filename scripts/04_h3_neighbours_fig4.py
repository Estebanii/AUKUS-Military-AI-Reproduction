#!/usr/bin/env python3
"""H3 with the baseline encoder: the 30 nearest GPT-2 vocabulary words (cosine) of each country's mean semantic vector
before and after the signing, the words common to / unique among the three countries' top 15, and Figure 4.
Writes results/h3/h3.json, results/h3/distances.json and results/figures/figure4_neighbours.*"""
import _common  # noqa: F401

from replication import analysis, data, figures, output, pipeline


def main() -> int:
    timer = output.Timer('04_h3_neighbours_fig4')
    encoder = data.BASELINE
    Y = pipeline.load_fit(encoder)['Y']
    meta = pipeline.encoder_meta(encoder)
    bundle = pipeline.gpt2_bundle()
    h3 = analysis.h3_nearest_neighbors(Y, meta, bundle['wte'], bundle['tokenizer'].get_vocab())
    output.write_json('h3/h3.json', h3)
    output.write_json('h3/distances.json', analysis.did_h3_distances(Y, meta))
    figures.figure4_neighbours(h3, 'figures/figure4_neighbours')
    timer.done()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
