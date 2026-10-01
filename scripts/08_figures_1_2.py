#!/usr/bin/env python3
"""Figures 1 and 2: PCA (PC1 x PC2 by country and by period) and t-SNE of the semantic vectors.

As in the manuscript, both figures are drawn from the original semantic vectors (the vectors of the original
submission, in the data bundle); ``--encoder deberta-v3-base`` draws them from the baseline re-encoding instead
(needs script 00). The axis labels of Figure 1 print the PC1 / PC2 variance shares; compare.py checks them against this
figure's own PCA (figures_1_2.json). No full-precision value of the published figure is archived; the baseline encoder's
analysis PCA-3 (results/h2/meta.json) rounds to the same printed shares.
Writes results/figures/figure1_pca.{png,pdf,csv}, figure2_tsne.{png,pdf,csv} and figures_1_2.json.
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
