#!/usr/bin/env python3
"""Step 0: the measurement of every encoder.

* the reference axes: PCA-88 (full SVD) of the original semantic vectors of all 37,866 occurrences;
* the GPT-2 labels of the 100 anchor words;
* for each of the five encoders, the A matrix fitted on the anchor occurrences and the semantic vectors
  Y = (U - U_mean) A^T of the target occurrences (baseline encoder: 37,866 rows; alternative encoders: the 37,808
  common-support rows).

Writes results/intermediate/ (reference_axes.npz, <encoder>_A.npz, <encoder>_Y.npy, <encoder>_A_fit.json).
"""
import _common  # noqa: F401  (paths, BLAS threads)

from replication import data, output, pca, pipeline


def main() -> int:
    timer = output.Timer('00_prepare')
    print('[00] reference axes (PCA-88 of the original vectors)', flush=True)
    pipeline.save_axes(pca.reference_axes(data.original_vectors()))
    print('[00] GPT-2 labels of the anchor words', flush=True)
    bundle = pipeline.gpt2_bundle()
    for encoder in data.ENCODERS:
        print(f'[00] A matrix and Y: {encoder}', flush=True)
        pipeline.save_fit(encoder, pipeline.fit_encoder(encoder, bundle))
    timer.done()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
