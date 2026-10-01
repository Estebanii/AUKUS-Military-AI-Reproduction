#!/usr/bin/env python3
"""Step 0: the reference axes, the GPT-2 anchor labels and, for each encoder, the A matrix and the semantic
vectors Y. Writes results/intermediate/.
"""
import _common  # noqa: F401

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
