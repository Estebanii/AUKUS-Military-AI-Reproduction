"""The A matrix of the embedding regression: a linear map from the encoder's hidden states to the GPT-2 label space.

``A`` is the ridge solution (lambda = 0.1) of the centred anchor labels V on the whitened, centred hidden states U of
the anchor occurrences, shrunk towards a Procrustes prior (Khodak et al. 2018, "A la carte" embeddings; Rodriguez,
Spirling and Stewart 2023). Every step after the float32 input U is float64. The Procrustes prior is the rank-r part
``W_r Z_r^T`` of the SVD of ``M = V_c^T U_w`` (r = #{singular values > 1e-8 x the largest}): M has rank at most (number
of distinct labels - 1) = 99, so only this part is identified. The semantic vector of an occurrence is
``Y = (U - U_mean) @ A.T`` (768 dimensions).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from .labels import label_matrix
from .params import params


@dataclass
class AFit:
    A: np.ndarray
    A0: np.ndarray
    U_mean: np.ndarray
    V_mean: np.ndarray
    lambda_: float
    n_train: int
    n_matched: int
    whitened: bool
    procrustes: bool
    train_mse: float
    precision: str
    diagnostics: dict = field(default_factory=dict)


def compute_whitening_transform(X, eps=1e-6):
    """ZCA whitening of X (eigenvalues clipped at ``eps``); dtype follows X."""
    mean = X.mean(axis=0)
    X_c = X - mean
    cov = X_c.T @ X_c / (X.shape[0] - 1)
    eigenvalues, eigenvectors = linalg.eigh(cov)
    clipped = int((eigenvalues < eps).sum())
    eigenvalues = np.clip(eigenvalues, eps, None)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    W = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    return W, mean, clipped


def compute_procrustes_prior(U, V, rank_rel_tol):
    """The rank-r part ``W_r @ Zt_r`` of the Procrustes solution, r = #{S > tol * S[0]}; also returns S."""
    U_c = U - U.mean(axis=0)
    V_c = V - V.mean(axis=0)
    M = V_c.T @ U_c
    W, S, Zt = linalg.svd(M, full_matrices=False)
    r = int((S > rank_rel_tol * S[0]).sum()) if S.size and S[0] > 0 else 0
    return W[:, :r] @ Zt[:r], S


def fit_a(U_matrix, word_labels, labels: dict, *, strict: bool = False) -> AFit:
    """Fit A on the anchor occurrences (``U_matrix`` rows, their words ``word_labels``, the GPT-2 ``labels``).

    ``strict`` (alternative encoders): a whitening or Procrustes step that cannot run raises instead of silently
    falling back to no whitening / a zero prior. The baseline encoder uses the default."""
    p = params('a_matrix')
    lam, whiten, procrustes, eps = p['lambda'], p['whiten'], p['procrustes_prior'], p['whiten_eps']
    rank_tol = p['a0_rank_rel_tol']
    V_valid, valid = label_matrix(word_labels, labels)
    U_valid = U_matrix[valid].astype(np.float64)
    V_valid = V_valid.astype(np.float64)
    U_mean = U_valid.mean(axis=0)
    V_mean = V_valid.mean(axis=0)
    U_c = U_valid - U_mean
    V_c = V_valid - V_mean
    d_U = U_c.shape[1]
    n = U_c.shape[0]
    diagnostics = {'n_train': int(n), 'd_in': int(d_U), 'd_out': int(V_c.shape[1]),
                   'distinct_labels': int(len(np.unique(np.asarray([str(w).lower() for w in
                                                                     np.asarray(word_labels)[valid]]))))}
    W_u = None
    if strict and (whiten or procrustes) and not n > d_U:
        raise RuntimeError(f'strict A fit: {n} training rows <= d_in {d_U}; whitening/Procrustes would be skipped')
    if whiten and n > d_U:
        try:
            W_u, _, clipped = compute_whitening_transform(U_c, eps)
            U_c = U_c @ W_u
            diagnostics['whitening_eigenvalues_clipped'] = clipped
        except Exception as error:  # noqa: BLE001 - original behaviour: fall back to no whitening
            if strict:
                raise RuntimeError(f'strict A fit: whitening failed ({type(error).__name__}: {error})') from error
            diagnostics['whitening_error'] = f'{type(error).__name__}: {error}'
            W_u = None
    if strict:
        diagnostics['whiten_eps'] = float(eps)
    if procrustes and n > d_U:
        A0, S = compute_procrustes_prior(U_c, V_c, rank_tol)
        rank = int((S > S.max() * rank_tol).sum()) if S.size else 0
        diagnostics['procrustes_M_rank'] = rank
        diagnostics['procrustes_M_singular_values_tail'] = [float(s) for s in S[max(rank - 2, 0):rank + 2]]
    else:
        A0 = np.zeros((V_c.shape[1], U_c.shape[1]))
    UTU = U_c.T @ U_c + lam * np.eye(d_U)
    VTU = V_c.T @ U_c + lam * A0
    A = linalg.solve(UTU.T, VTU.T).T
    V_pred = U_c @ A.T
    mse = float(np.mean((V_c - V_pred) ** 2))
    if W_u is not None:
        A = A @ W_u
    diagnostics['dtypes'] = {'U_mean': str(U_mean.dtype), 'A0': str(A0.dtype), 'A': str(A.dtype)}
    diagnostics['rule'] = p['rule']
    diagnostics['a0_rank_rel_tol'] = rank_tol
    return AFit(A=A, A0=A0, U_mean=U_mean, V_mean=V_mean, lambda_=float(lam), n_train=int(n),
                n_matched=int(len(valid)), whitened=W_u is not None, procrustes=bool(procrustes and n > d_U),
                train_mse=mse, precision='float64', diagnostics=diagnostics)


def transform(U, A, U_mean):
    """``(U - U_mean) @ A.T``."""
    single = U.ndim == 1
    if single:
        U = U.reshape(1, -1)
    Y = (U - U_mean) @ A.T
    return Y.squeeze() if single else Y
