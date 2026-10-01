"""The estimation panel of one encoder: the three-country rows of its main analysis and the Singapore / Canada rows
of its external-control encoding, both measured by the three-country A of the main analysis.

* Y of a control row = (U - U_mean_3c) A^T with the main-analysis A and U_mean (never refitted).
* Same-axis scores: the reference axes: (Y - mean) components^T, PC1-3.
* Same-procedure scores: the encoder's own PCA-3 of its three-country Y (all rows of the main-analysis Y table,
  ``svd_solver='full'``, the main-analysis rule), re-fitted from that Y and checked against the explained variance
  ratios of the main analysis; PC_k is oriented by the sign of its loading's inner product with the
  reference PC_k loading (PC1: the design rule, checked against the recorded orientation; PC2/PC3: the same
  rule, descriptive). Controls never enter A or either PCA.
* Rows up to 2025-11; ``post`` = year_month >= 2021-09 (as in the original analysis; 2021-09 wholly post).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..amatrix import transform
from ..pca import fit_pca
from .common import ExternalControlError, month_index, ym_index
from .params import external_params

RESPONSES = ['ref_PC1', 'ref_PC2', 'ref_PC3', 'own_PC1', 'own_PC2', 'own_PC3']


def stack(column) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in column])


def own_pca(Y3: np.ndarray, axes: dict, recorded_meta: dict, recorded_orientation: dict) -> dict:
    pca = fit_pca(Y3, 3)
    ratio = pca.explained_variance_ratio_
    recorded = np.asarray(recorded_meta.get('pca3_explained_variance') or [], float)
    if recorded.size != 3 or not np.allclose(ratio, recorded, rtol=1e-9, atol=1e-12):
        raise ExternalControlError(f'the re-fitted own PCA-3 explained variance {ratio.tolist()} differs from the '
                                   f'main-analysis record {recorded.tolist()}')
    ref = np.asarray(axes['components'][:3], np.float64)
    cos = [float(pca.components_[k] @ ref[k] / (np.linalg.norm(pca.components_[k]) * np.linalg.norm(ref[k])))
           for k in range(3)]
    signs = [1 if c >= 0 else -1 for c in cos]
    if recorded_orientation and signs[0] != recorded_orientation.get('sign'):
        raise ExternalControlError(f'the own PC1 orientation {signs[0]} differs from the main-analysis record '
                                   f'{recorded_orientation.get("sign")}')
    return {'pca': pca, 'cos': cos, 'signs': signs, 'explained_variance_ratio': ratio.tolist()}


def scores(Y: np.ndarray, axes: dict, own: dict) -> np.ndarray:
    ref = (np.asarray(Y, np.float64) - axes['mean']) @ np.asarray(axes['components'][:3]).T
    mine = own['pca'].transform(np.asarray(Y, np.float64)) * np.asarray(own['signs'])[None, :]
    return np.column_stack([ref, mine])


def build(bound: dict, U_targets: np.ndarray, target_rows: pd.DataFrame, target_meta: pd.DataFrame,
          documents: pd.DataFrame) -> tuple:
    """(panel, record). ``target_rows``: the rows table of the encoding (row keys in encoding order); ``target_meta``:
    the metadata of the external-control target occurrences; ``documents``: the external-control document table
    (publication date, genre)."""
    last = month_index(external_params('extract')['cut_last_month'])
    first_post = month_index(external_params('extract')['post_first_month'])
    y = bound['y_table']
    Y3 = stack(y['Y_vector_global'])
    own = own_pca(Y3, bound['axes'], bound['meta'], bound['orientation'])
    doc = documents.set_index('article_id')
    t3 = pd.DataFrame({'row_key': [f'v1:{int(o)}' for o in y['occurrence_id']], 'country': y['country'].to_numpy(),
                       'year': y['year'].astype(int).to_numpy(), 'month': y['month'].astype(int).to_numpy(),
                       'doc_id': y['doc_id'].astype(str).to_numpy(), 'term': y['term'].astype(str).to_numpy(),
                       'post_v1': y['post_aukus'].to_numpy(bool)})
    meta = target_meta.set_index('row_key')
    keys = target_rows['row_key'].astype(str).to_numpy()
    if not set(keys) <= set(meta.index):
        raise ExternalControlError('encoded target rows are not in the target metadata')
    m = meta.loc[keys]
    tc = pd.DataFrame({'row_key': keys, 'country': m['country'].to_numpy(), 'year': m['year'].astype(int).to_numpy(),
                       'month': m['month'].astype(int).to_numpy(), 'doc_id': m['article_id'].astype(str).to_numpy(),
                       'term': m['term'].astype(str).to_numpy(), 'post_v1': m['post_aukus'].to_numpy(bool)})
    Yc = transform(np.asarray(U_targets, np.float64), bound['A'], bound['U_mean'])
    panel = pd.concat([t3, tc], ignore_index=True)
    Y = np.vstack([Y3, Yc])
    panel['ym'] = ym_index(panel['year'], panel['month'])
    panel['post'] = panel['ym'].to_numpy() >= first_post
    if not (panel['post'] == panel['post_v1']).all():
        raise ExternalControlError('post flags differ from year_month >= 2021-09')
    panel['publish_date'] = panel['doc_id'].map(doc['publish_date']).astype(str)
    panel['genre'] = panel['doc_id'].map(doc['genre']).astype(str)
    if panel['publish_date'].isin(['nan', 'None']).any():
        raise ExternalControlError('rows without a document publication date in the document table')
    S = scores(Y, bound['axes'], own)
    for j, name in enumerate(RESPONSES):
        panel[name] = S[:, j]
    keep = panel['ym'].to_numpy() <= last
    record = {'rows_total': int(len(panel)), 'rows_after_cut': int(keep.sum()),
              'rows_by_country': panel.loc[keep, 'country'].value_counts().to_dict(),
              'own_pca': {'explained_variance_ratio': own['explained_variance_ratio'], 'cos_with_reference': own['cos'],
                          'signs': own['signs'], 'fit_rows': int(len(Y3))},
              'y_rule': '(U - U_mean_3c) A^T with the A of the main analysis (controls never refit A or PCA)'}
    panel = panel.loc[keep].reset_index(drop=True)
    return panel.drop(columns=['post_v1']), record, Yc, own
