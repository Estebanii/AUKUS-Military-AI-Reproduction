"""The measurement pipeline: GPT-2 labels, the A matrix of each encoder, the semantic vectors Y, the reference axes and
the full analysis chain of one Y. Intermediate arrays are cached under ``results/intermediate`` by
``scripts/00_prepare.py`` and read by the other scripts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import analysis, data, labels as labels_mod, pca
from .amatrix import fit_a, transform

INTERMEDIATE = 'intermediate'


# --------------------------------------------------------------------------- GPT-2 labels
def gpt2_bundle() -> dict:
    """GPT-2 tokenizer and input embeddings (pinned files, hash-verified), the labels of the 100 anchor words and the
    anchor word of each of the 49,999 anchor occurrences (row order of every anchor encoding)."""
    source = labels_mod.resolve_gpt2(data.gpt2_dir())
    tokenizer, wte = labels_mod.load_gpt2(source)
    words_all = data.anchor_words()
    labels, token_ids = labels_mod.anchor_labels(words_all, tokenizer, wte)
    return {'gpt2': source, 'labels': labels, 'label_token_ids': token_ids, 'tokenizer': tokenizer, 'wte': wte,
            'anchor_words': data.anchor_occurrences()['anchor_word'].to_numpy()}


# --------------------------------------------------------------------------- common support
def map_rows(cmap: pd.DataFrame, kind: str) -> pd.DataFrame:
    sub = cmap[cmap['kind'] == kind].sort_values('cs_position').reset_index(drop=True)
    if not (sub['cs_position'].to_numpy() == np.arange(len(sub))).all():
        raise ValueError(f'{kind}: the map positions are not 0..n-1')
    return sub


def common_rows(kind: str = 'targets') -> np.ndarray:
    """Original row indices of the common-support rows (ascending)."""
    return map_rows(data.common_support_map(), kind)['orig_row_index'].to_numpy()


# --------------------------------------------------------------------------- A and Y of one encoder
def fit_encoder(encoder: str, bundle: dict) -> dict:
    """A and Y of one encoder. Baseline: all 37,866 target and 49,999 anchor occurrences. Alternative encoders: the
    common-support rows (37,808 targets, 49,999 anchors), strict A fit. Every encoding is checked to follow the row
    order of the metadata (targets) and of the anchor occurrences (anchors)."""
    U_t, rows_t = data.encoding(encoder, 'targets')
    U_a, rows_a = data.encoding(encoder, 'anchors')
    meta = data.analysis_meta()
    anchors = data.anchor_occurrences()
    if encoder == data.BASELINE:
        t_orig = a_orig = None
        expected_t = [f't:{int(o)}' for o in meta['occurrence_id']]
        expected_a = [f'a:{int(r)}' for r in anchors['source_row']]
        words = bundle['anchor_words']
        strict = False
    else:
        cmap = data.common_support_map()
        mt, ma = map_rows(cmap, 'targets'), map_rows(cmap, 'anchors')
        t_orig, a_orig = mt['orig_row_index'].to_numpy(), ma['orig_row_index'].to_numpy()
        if rows_t['row_key'].tolist() != mt['row_key'].tolist() or rows_a['row_key'].tolist() != ma['row_key'].tolist():
            raise ValueError(f'{encoder}: the encoded row keys differ from the common-support map')
        expected_t = [f't:{int(o)}' for o in meta['occurrence_id'].to_numpy()[t_orig]]
        expected_a = [f'a:{int(r)}' for r in anchors['source_row'].to_numpy()[a_orig]]
        words = bundle['anchor_words'][a_orig]
        strict = True
    if rows_t['row_key'].tolist() != expected_t or rows_a['row_key'].tolist() != expected_a:
        raise ValueError(f'{encoder}: the row keys of the encodings do not follow the corpus row order')
    fit = fit_a(U_a, words, bundle['labels'], strict=strict)
    Y = transform(U_t, fit.A, fit.U_mean)
    return {'encoder': encoder, 'fit': fit, 'Y': Y, 'U_t': U_t, 'U_a': U_a, 'rows_t': rows_t, 'rows_a': rows_a,
            't_orig': t_orig, 'a_orig': a_orig, 'words': words}


def fit_record(fit) -> dict:
    return {'n_train': fit.n_train, 'n_matched': fit.n_matched, 'whitened': fit.whitened, 'procrustes': fit.procrustes,
            'train_mse': fit.train_mse, 'precision': fit.precision, 'shape': list(fit.A.shape),
            'diagnostics': fit.diagnostics}


def save_fit(encoder: str, fitted: dict) -> None:
    out = data.results_dir() / INTERMEDIATE
    out.mkdir(parents=True, exist_ok=True)
    fit = fitted['fit']
    np.savez(out / f'{encoder}_A.npz', A=fit.A, U_mean=fit.U_mean, V_mean=fit.V_mean)
    np.save(out / f'{encoder}_Y.npy', fitted['Y'])
    (out / f'{encoder}_A_fit.json').write_text(json.dumps(fit_record(fit), indent=1) + '\n', encoding='utf-8')


def load_fit(encoder: str) -> dict:
    out = data.results_dir() / INTERMEDIATE
    if not (out / f'{encoder}_Y.npy').is_file():
        raise FileNotFoundError(f'{out}/{encoder}_Y.npy: run scripts/00_prepare.py first')
    with np.load(out / f'{encoder}_A.npz') as npz:
        arrays = {k: npz[k] for k in npz.files}
    return {'encoder': encoder, 'Y': np.load(out / f'{encoder}_Y.npy'), **arrays}


def save_axes(axes: dict) -> None:
    out = data.results_dir() / INTERMEDIATE
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'reference_axes.npz', **axes)


def load_axes() -> dict:
    path = data.results_dir() / INTERMEDIATE / 'reference_axes.npz'
    if not path.is_file():
        raise FileNotFoundError(f'{path}: run scripts/00_prepare.py first')
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def encoder_meta(encoder: str) -> pd.DataFrame:
    """The occurrence metadata in the row order of the encoder's Y (baseline: all rows; others: common support)."""
    meta = data.analysis_meta()
    if encoder == data.BASELINE:
        return meta
    return meta.iloc[common_rows('targets')].reset_index(drop=True)


def analysis_table(encoder: str, Y, U_t=None) -> pd.DataFrame:
    """The occurrence table of one encoder's analysis: corpus metadata (encoder row order), Y and optionally U (for the
    external-control panel)."""
    table = data.occurrence_meta()
    if encoder != data.BASELINE:
        table = table.iloc[common_rows('targets')].reset_index(drop=True)
    else:
        table = table.copy()
    if U_t is not None:
        table['U_vector'] = list(np.asarray(U_t, dtype=np.float32))
    table['Y_vector_global'] = list(Y)
    table['Y_norm_global'] = np.linalg.norm(Y, axis=1)
    return table


# --------------------------------------------------------------------------- the analysis chain of one Y
def run_analyses(Y, meta, bundle, axes: dict | None, rerun: bool = False) -> dict:
    """The full chain on one Y: H1 (MANOVA, k = 88 and the 99% rule), PCA-k robustness, H2 (WCB main model, robustness
    models, parallel trends), H3 neighbours, distances, and -- with the reference axes -- the same-axis WCB and the
    orientation of the own PC1. ``rerun`` (alternative encoders): the descriptive B = 9,999 re-run of both PC1 UK x
    Post members and the pre-signing PC1 SDs."""
    pca88 = pca.fit_pca(Y, 88)
    h1 = analysis.h1_summary(Y, meta, pca88)
    h2 = analysis.h2_models(Y, meta)
    h3 = analysis.h3_nearest_neighbors(Y, meta, bundle['wte'], bundle['tokenizer'].get_vocab())
    out = {'manova': h1['k88'], 'h1': {k: v for k, v in h1.items() if k != 'k88'},
           'manova_pc_robustness': analysis.manova_pc_robustness(Y, meta),
           'manova_time_robustness': analysis.manova_time_robustness(Y, meta, pca88),
           'wcb': h2['wcb'], 'did': h2['did'], 'parallel_trends': h2['parallel_trends'], 'h3': h3,
           'did_h3_distances': analysis.did_h3_distances(Y, meta),
           'meta': {'n_pc': int(pca88.n_components_), 'pca_solver': pca88.svd_solver,
                    'pca3_solver': h2['pca3'].svd_solver, 'pca88_n_samples': int(pca88.n_samples_),
                    'pca3_n_samples': int(h2['pca3'].n_samples_),
                    'wcb_seeds': h2['wcb']['_details']['seeds'],
                    'pca3_explained_variance': h2['pca3'].explained_variance_ratio_.tolist(),
                    'pc1_pc2_eigengap': float(h2['pca3'].explained_variance_[0] - h2['pca3'].explained_variance_[1])}}
    if axes is not None:
        scores = pca.scores(Y, axes['mean'], axes['components'][:3])
        same_axis = analysis.h2_wcb_main(scores, meta, axes['explained_variance_ratio'][:3])
        out['same_axis'] = {'wcb_on_reference_axes': same_axis,
                            'UK_x_post': {pc: same_axis['comparison'][pc]['UK_x_post'] for pc in ('PC1', 'PC2', 'PC3')},
                            'intervals_95': {pc: same_axis['_details']['intervals_95'][pc]['UK_x_post']
                                             for pc in ('PC1', 'PC2', 'PC3')}}
        out['orientation'] = {**analysis.orientation(h2['pca3'].components_, axes['components']),
                              'pc1_pc2_eigengap': out['meta']['pc1_pc2_eigengap']}
        if rerun:
            out['rerun_descriptive'] = analysis.rerun_descriptive(h2['scores3'][:, 0], scores[:, 0], meta)
    elif rerun:
        raise ValueError('the descriptive re-run needs the reference axes')
    return out
