"""rdrobust's CRV2 and CRV3 cluster-robust variances (rdrobust_vce and
.rdrobust_vce_qq_cluster in R's rdrobust 4.1.1), from per-cluster sums.

Both adjust each cluster's score before the outer products are summed, with
no small-sample factor:

- CRV3 (the jackknife-like correction): score_g = u_g + L_g (G - L_g)^-1 u_g,
  G the window's Gram matrix.
- CRV2 (Bell and McCaffrey's bias-reduced linearization): the same idea with
  (I - H_gg)^(-1/2), from the eigendecomposition of C L_g C', C the upper
  Cholesky factor of G^-1.

L_g and u_g (and, for the robust variance with h != b, P_g, uQ_g and uq_g)
come from one compiled pass over the window (_kernels.cluster_sums*). The
small matrix work is batched over clusters here. Every score is linear in
the residuals, so a fuzzy or covariate-adjusted fit uses the residuals of
the combined response, as rdrobust's sum over its columns does.
"""

from __future__ import annotations

import numpy as np

EPS = np.finfo(float).eps


def _inverse_or(A, fallback):
    """Inverses of a stack of symmetric matrices, as R's solve(); where one
    is numerically singular, its generalized inverse (MASS::ginv) or zeros."""
    lam, V = np.linalg.eigh(A)
    absl = np.abs(lam)
    amax = absl.max(axis=-1, keepdims=True)
    singular = (absl.min(axis=-1, keepdims=True) <= EPS * amax)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(absl > 0, 1.0 / lam, 0.0)
    if fallback == "ginv":
        keep = absl > np.sqrt(EPS) * amax
        inv = np.where(singular & ~keep, 0.0, inv)
    else:                                     # zeros
        inv = np.where(singular, 0.0, inv)
    return (V * inv[..., None, :]) @ np.swapaxes(V, -1, -2)


def _outer_sum(scores):
    return scores.T @ scores


def meat(L, u, seen, Ginv, mode):
    """sum_g score_g score_g' for the conventional variance (and the robust
    one with h = b, and the bandwidth steps): mode 2 (CRV2) or 3 (CRV3)."""
    L, u = L[seen], u[seen]
    if mode == 3:
        try:
            G = np.linalg.inv(Ginv)
        except np.linalg.LinAlgError:
            G = np.linalg.pinv(Ginv)
        Minv = _inverse_or(G[None] - L, "ginv")
        score = u + np.einsum("gij,gj->gi", L, np.einsum("gij,gj->gi", Minv, u))
        return _outer_sum(score)
    try:
        C = np.linalg.cholesky(Ginv).T         # R's chol(): upper, C'C = Ginv
    except np.linalg.LinAlgError:
        return _outer_sum(u)
    F = C[None] @ L @ C.T[None]
    F = 0.5 * (F + np.swapaxes(F, 1, 2))
    lam, V = np.linalg.eigh(F)
    s2 = np.maximum(lam, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        coef = np.where(s2 < 1e-14, 0.0, (1 / np.sqrt(np.maximum(1 - s2, 1e-8)) - 1) / s2)
    Ru = u @ C.T                               # C u_g, per cluster
    adj = np.einsum("gij,gj->gi", V, coef * np.einsum("gji,gj->gi", V, Ru))
    score = u + np.einsum("gij,gj->gi", L, adj @ C)    # L_g C' adj
    return _outer_sum(score)


def meat_qq(L, Pg, uQ, uq, seen, Ginv_q, mode):
    """sum_g score_g score_g' for the robust variance with h != b
    (.rdrobust_vce_qq_cluster): mode 2 (CRV2) or 3 (CRV3)."""
    L, Pg, uQ, uq = L[seen], Pg[seen], uQ[seen], uq[seen]
    if mode == 3:
        try:
            G = np.linalg.inv(Ginv_q)
        except np.linalg.LinAlgError:
            G = np.linalg.pinv(Ginv_q)
        Minv = _inverse_or(G[None] - L, "zeros")
        adj = np.einsum("gij,gj->gi", Minv, uq)
        return _outer_sum(uQ + np.einsum("gij,gj->gi", Pg, adj))
    vals, V = np.linalg.eigh(L)
    sig = np.sqrt(np.maximum(vals, 0.0))
    smax = sig.max(axis=1, keepdims=True)
    keep = sig > smax * 1e-10
    sig0 = np.where(keep, sig, 0.0)
    with np.errstate(divide="ignore"):
        dinv = np.where(keep, 1.0 / np.where(keep, sig, 1.0), 0.0)
    T = sig0[:, :, None] * (np.swapaxes(V, 1, 2) @ Ginv_q[None] @ V) * sig0[:, None, :]
    T = 0.5 * (T + np.swapaxes(T, 1, 2))
    tau, W = np.linalg.eigh(T)
    tau = np.clip(tau, 0.0, 1 - 1e-10)
    gamma = 1 / np.sqrt(1 - tau) - 1
    A = V @ (dinv[:, :, None] * W)
    Lam = (A * gamma[:, None, :]) @ np.swapaxes(A, 1, 2)
    adj = np.einsum("gij,gj->gi", Lam, uq)
    simple = ~keep.any(axis=1)
    score = uQ + np.where(simple[:, None], 0.0, np.einsum("gij,gj->gi", Pg, adj))
    return _outer_sum(score)
