#!/usr/bin/env python3
"""Figures 1 and 2: PCA and t-SNE scatter plots of the semantic vectors.

Both are drawn from the original semantic vectors, as in the manuscript; --encoder deberta-v3-base draws them from
the baseline re-encoding (needs script 00). Writes results/figures/figure1_pca.*, figure2_tsne.* and
figures_1_2.json.
"""
import _common  # noqa: F401

import argparse

from replication import data, figures, output, pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--encoder', choices=['original', data.BASELINE], default='original',
                    help='vectors to draw (default: the original vectors, as in the manuscript)')
    args = ap.parse_args(argv)
    timer = output.Timer('08_figures_1_2')
    Y = data.original_vectors() if args.encoder == 'original' else pipeline.load_fit(data.BASELINE)['Y']
    meta = data.analysis_meta()
    f1 = figures.figure1_pca(Y, meta, 'figures/figure1_pca')
    f2 = figures.figure2_tsne(Y, meta, 'figures/figure2_tsne')
    output.write_json('figures/figures_1_2.json', {'vectors': args.encoder, 'rows': int(len(Y)), 'figure1': f1,
                                                   'figure2': f2})
    timer.done()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
