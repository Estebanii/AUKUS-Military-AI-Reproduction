"""Covariance estimators for the sensitivity checks of the event study: an iid-corrected Bartlett HAC on monthly
scores, and month-clustered CR2 with Bell-McCaffrey degrees of freedom and the HTZ joint test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import inference as inf

SINGULAR_REL_TOL = inf.SINGULAR_REL_TOL


def _sorted_months(month):
    month = np.asarray(month, dtype=np.int64)
    order = np.argsort(month, kind='stable')
    months, starts = np.unique(month[order], return_index=True)
    bounds = list(starts) + [len(month)]
    return order, months, bounds


def _inv_sqrt_psd(M: np.ndarray, tol: float = 1e-10) -> tuple:
    vals, vecs = np.linalg.eigh((M + M.T) / 2)
    top = max(float(vals.max()), 1e-300)
    keep = vals > tol * top
    inv = np.where(keep, 1.0 / np.sqrt(np.where(keep, vals, 1.0)), 0.0)
    return (vecs * inv) @ vecs.T, int((~keep).sum())


def chi2_joint(b: np.ndarray, S: np.ndarray, scale: float | None = None) -> dict:
    q = len(b)
    eig = np.linalg.eigvalsh((S + S.T) / 2)
    if eig.min() <= eig.max() * 1e-12 or (scale is not None and eig.max() <= SINGULAR_REL_TOL * scale):
        return {'status': inf.UNAVAILABLE, 'reason': f'singular covariance (eigenvalues {eig.min():.3e} .. '
                                                     f'{eig.max():.3e})'}
    W = float(b @ np.linalg.solve(S, b))
    return {'status': 'ok', 'test': 'Wald chi-square', 'statistic': W, 'df': q, 'p': float(stats.chi2.sf(W, q))}


class HacIidExact:
    """Iid-corrected Bartlett HAC on monthly scores (sensitivity only)."""

    name = 'hac_bartlett_iid_exact'
    label = '设计精确校正的 Bartlett HAC（b = 6）'
    assumption = 'iid 误差下各对比的边际方差无偏；联合矩阵保留 HAC 相关（未校正）；正态参照；非有效推断，仅作敏感性'
    df = None

    def __init__(self, X, month, w, C, bandwidth: int = 6):
        X = np.asarray(X, dtype=float)
        C = np.atleast_2d(np.asarray(C, dtype=float))
        w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
        self.bandwidth = int(bandwidth)
        self.order, self.months, self.bounds = _sorted_months(month)
        Xw = X * w[:, None]
        Q = X.T @ Xw
        self.Q_inv = np.linalg.inv(Q)
        self.Xw_sorted = Xw[self.order]
        self.K = inf.kernel_matrix(self.months, self.bandwidth)
        self.G = C @ self.Q_inv
        R = X.T @ (Xw * w[:, None])                                   # X'W^2 X
        Xs, ws = X[self.order], w[self.order]
        starts = self.bounds[:-1]
        kappa, expected, true = [], [], []
        for c in C:
            g = ws * (Xs @ (self.Q_inv @ c))
            D = np.add.reduceat(g * g, starts)
            F = np.add.reduceat(g[:, None] * Xs, starts, axis=0)
            Fw = np.add.reduceat((g * ws)[:, None] * Xs, starts, axis=0)
            e = float(np.sum(np.diag(self.K) * D) - 2 * np.sum((self.K @ Fw @ self.Q_inv) * F)
                      + np.sum((self.K @ F @ self.Q_inv @ R @ self.Q_inv) * F))
            v = float(c @ self.Q_inv @ R @ self.Q_inv @ c)
            expected.append(e)
            true.append(v)
            kappa.append(v / e if e > 0 else np.nan)
        self.kappa = np.asarray(kappa)
        self.expected_iid, self.true_iid = np.asarray(expected), np.asarray(true)

    def record(self) -> dict:
        return {'estimator': self.name, 'bandwidth': self.bandwidth, 'months': int(len(self.months)),
                'kappa_min_max': [float(np.nanmin(self.kappa)), float(np.nanmax(self.kappa))]}

    def contrast_cov(self, E) -> np.ndarray:
        E = np.asarray(E, dtype=float)
        E = E[:, None] if E.ndim == 1 else E
        Es = E[self.order]
        S = np.stack([self.Xw_sorted[a:b].T @ Es[a:b] for a, b in zip(self.bounds[:-1], self.bounds[1:])])
        U = np.einsum('kp,tpr->tkr', self.G, S)
        KU = np.einsum('tu,ukr->tkr', self.K, U)
        V = np.einsum('tar,tbr->rab', U, KU)
        d = np.sqrt(self.kappa)
        return V * d[None, :, None] * d[None, None, :]

    def critical(self, level: float) -> np.ndarray:
        return np.full(len(self.kappa), float(stats.norm.ppf(0.5 + level / 2)))

    def joint(self, idx, b, S, scale=None) -> dict:
        out = chi2_joint(np.asarray(b, float), np.asarray(S, float), scale)
        out['reference'] = 'chi-square (normal; iid-exact marginal variances, HAC correlations uncorrected)'
        return out


class Cr2Month:
    """Month-clustered CR2 with Bell-McCaffrey degrees of freedom and the HTZ joint test."""

    name = 'cr2_month'
    label = '按月聚类的 CR2（Bell–McCaffrey/Imbens–Kolesár 自由度，HTZ 联合检验）'
    assumption = '跨月独立（允许月内跨国家、文档与出现点相关）；非有效推断，仅作敏感性'

    def __init__(self, X, month, w, C):
        X = np.asarray(X, dtype=float)
        C = np.atleast_2d(np.asarray(C, dtype=float))
        w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float)
        self.sw = np.sqrt(w)
        Xt = X * self.sw[:, None]
        self.Q_inv = np.linalg.inv(Xt.T @ Xt)
        self.order, self.months, self.bounds = _sorted_months(month)
        Xs = Xt[self.order]
        Bc = self.Q_inv @ C.T                                            # (p, k)
        self.Y, F, ZZ = [], [], []
        self.pseudo_clusters = 0
        for a, b in zip(self.bounds[:-1], self.bounds[1:]):
            Xc = Xs[a:b]
            H = Xc @ self.Q_inv @ Xc.T
            A, zero = _inv_sqrt_psd(np.eye(len(Xc)) - H)
            self.pseudo_clusters += int(zero > 0)
            Yc = A @ (Xc @ Bc)                                           # (n_t, k)
            self.Y.append(Yc)
            F.append(Yc.T @ Xc)                                          # (k, p)
            ZZ.append(Yc.T @ Yc)                                         # (k, k)
        self.F = np.stack(F)                                             # (T, k, p)
        self.ZZ = np.stack(ZZ)                                           # (T, k, k)
        k = C.shape[0]
        self.working_expectation = np.array([np.trace(self.gamma(j, j)) for j in range(k)])
        self.working_truth = np.einsum('kp,pq,kq->k', C, self.Q_inv, C)
        dfs = []
        for j in range(k):
            G = self.gamma(j, j)
            den = float(np.sum(G * G))
            dfs.append(float(np.trace(G)) ** 2 / den if den > 0 else np.nan)
        self.df = np.asarray(dfs)
        self._eta = {}

    def gamma(self, a: int, b: int) -> np.ndarray:
        """Gamma_ab = G_a' G_b (T x T)."""
        return np.diag(self.ZZ[:, a, b]) - self.F[:, a, :] @ self.Q_inv @ self.F[:, b, :].T

    def record(self) -> dict:
        ratio = self.working_expectation / self.working_truth
        return {'estimator': self.name, 'clusters': int(len(self.months)), 'pseudo_inverse_clusters': self.pseudo_clusters,
                'df_min_max': [float(np.nanmin(self.df)), float(np.nanmax(self.df))],
                'working_model_unbiasedness_min_max': [float(ratio.min()), float(ratio.max())]}

    def contrast_cov(self, E) -> np.ndarray:
        E = np.asarray(E, dtype=float)
        E = E[:, None] if E.ndim == 1 else E
        Es = (E * self.sw[:, None])[self.order]
        u = np.stack([Yc.T @ Es[a:b] for Yc, a, b in zip(self.Y, self.bounds[:-1], self.bounds[1:])])   # (T, k, r)
        return np.einsum('tar,tbr->rab', u, u)

    def critical(self, level: float) -> np.ndarray:
        return np.array([float(stats.t.ppf(0.5 + level / 2, d)) if np.isfinite(d) and d > 0 else np.nan
                         for d in self.df])

    def eta(self, idx) -> float:
        key = tuple(int(i) for i in idx)
        if key not in self._eta:
            q = len(key)
            Gam = np.stack([np.stack([self.gamma(a, b) for b in key]) for a in key])       # (q, q, T, T)
            Omega = np.einsum('abii->ab', Gam)
            L, _ = _inv_sqrt_psd(Omega)
            Gn = np.einsum('sa,tb,abij->stij', L, L, Gam)
            total = 0.0
            for s in range(q):
                for t in range(q):
                    total += float(np.sum(Gn[s, s] * Gn[t, t]) + np.sum(Gn[s, t] * Gn[s, t].T))
            self._eta[key] = q * (q + 1) / total if total > 0 else float('nan')
        return self._eta[key]

    def joint(self, idx, b, S, scale=None) -> dict:
        b, S = np.asarray(b, float), np.asarray(S, float)
        q = len(b)
        eig = np.linalg.eigvalsh((S + S.T) / 2)
        if eig.min() <= eig.max() * 1e-12 or (scale is not None and eig.max() <= SINGULAR_REL_TOL * scale):
            return {'status': inf.UNAVAILABLE, 'reason': f'singular covariance (eigenvalues {eig.min():.3e} .. '
                                                         f'{eig.max():.3e})'}
        eta = self.eta(idx)
        df2 = eta - q + 1
        if not np.isfinite(eta) or df2 <= 0:
            return {'status': inf.UNAVAILABLE, 'reason': f'HTZ degrees of freedom eta = {eta!r} <= q - 1 = {q - 1}'}
        T2 = float(b @ np.linalg.solve(S, b))
        Fstat = df2 / (eta * q) * T2
        return {'status': 'ok', 'test': 'HTZ (Pustejovsky-Tipton 2018)', 'statistic': Fstat, 'T2': T2, 'df1': q,
                'df2': float(df2), 'eta': float(eta), 'p': float(stats.f.sf(Fstat, q, df2)),
                'reference': 'F(q, eta - q + 1); CR2 clustered by calendar month (cross-month independence)'}

    def joint_p(self, idx, T2: np.ndarray) -> np.ndarray:
        q = len(idx)
        eta = self.eta(idx)
        df2 = eta - q + 1
        if not np.isfinite(eta) or df2 <= 0:
            return np.full(len(T2), np.nan)
        return stats.f.sf(df2 / (eta * q) * np.asarray(T2, float), q, df2)


ESTIMATORS = {HacIidExact.name: HacIidExact, Cr2Month.name: Cr2Month}


def prepare(spec: dict, X, month, w, C):
    params = {k: v for k, v in spec.items() if k != 'name'}
    return ESTIMATORS[spec['name']](X, month, w, C, **params)


def joint_p_vector(est_obj, idx, T2: np.ndarray) -> np.ndarray:
    if hasattr(est_obj, 'joint_p'):
        return est_obj.joint_p(idx, T2)
    return stats.chi2.sf(np.asarray(T2, float), len(idx))
