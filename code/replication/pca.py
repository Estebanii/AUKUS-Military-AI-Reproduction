"""Principal components (scikit-learn PCA, full SVD; deterministic) and the reference axes.

The reference axes are the first 88 principal components of the original vectors of all 37,866 occurrences
(``sklearn.decomposition.PCA(n_components=88, svd_solver='full')``): their mean, loadings and explained-variance
ratios. Scores on an axis set are ``(Y - mean) @ components.T``.
"""
from __future__ import annotations

import numpy as np

from .params import params


def fit_pca(Y, n_components=None, svd_solver=None):
    from sklearn.decomposition import PCA
    p = params('pca')
    pca = PCA(n_components=p['n_components'] if n_components is None else n_components,
              svd_solver=p['svd_solver'] if svd_solver is None else svd_solver)
    pca.fit(Y)
    return pca


def k_for_variance(explained_variance_ratio, threshold=None) -> int | None:
    """Smallest k with cumulative explained variance >= threshold (None if not reached)."""
    threshold = params('pca')['k99_threshold'] if threshold is None else threshold
    cumulative = np.cumsum(explained_variance_ratio)
    hit = np.flatnonzero(cumulative >= threshold)
    return int(hit[0] + 1) if hit.size else None


def reference_axes(Y) -> dict:
    """The reference axes: PCA-88 (full SVD) of all rows of ``Y``."""
    pca = fit_pca(Y)
    return {'mean': pca.mean_, 'components': pca.components_, 'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_, 'singular_values': pca.singular_values_}


def scores(Y, mean, components):
    return (np.asarray(Y, dtype=np.float64) - mean) @ components.T
