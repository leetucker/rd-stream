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
small matrix work runs here, compiled and in parallel over clusters: each
cluster's symmetric eigendecomposition (2 x 2 to about 6 x 6) by cyclic
Jacobi rotations, which are exact to rounding at these sizes, and each chunk
of clusters adds its scores' outer products into a P x P total of its own.
Every score is linear in the residuals, so a fuzzy or covariate-adjusted fit
uses the residuals of the combined response, as rdrobust's sum over its
columns does.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

EPS = np.finfo(float).eps
_SQRT_EPS = np.sqrt(EPS)


@njit(cache=True)
def _eigh(A):
    """Eigenvalues and eigenvectors (columns) of a small symmetric matrix:
    cyclic Jacobi rotations until the off-diagonal part is negligible."""
    n = A.shape[0]
    a = A.copy()
    V = np.eye(n)
    for _ in range(100):
        off = 0.0
        tot = 0.0
        for p in range(n):
            tot += a[p, p] * a[p, p]
            for q in range(p + 1, n):
                off += a[p, q] * a[p, q]
        tot += 2.0 * off
        if off == 0.0 or off <= 1e-36 * tot:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p, q]
                if apq == 0.0:
                    continue
                theta = (a[q, q] - a[p, p]) / (2.0 * apq)
                t = 1.0 / (abs(theta) + np.sqrt(theta * theta + 1.0))
                if theta < 0.0:
                    t = -t
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp = a[k, p]
                    akq = a[k, q]
                    a[k, p] = c * akp - s * akq
                    a[k, q] = s * akp + c * akq
                for k in range(n):
                    apk = a[p, k]
                    aqk = a[q, k]
                    a[p, k] = c * apk - s * aqk
                    a[q, k] = s * apk + c * aqk
                a[p, q] = 0.0
                a[q, p] = 0.0
                for k in range(n):
                    vkp = V[k, p]
                    vkq = V[k, q]
                    V[k, p] = c * vkp - s * vkq
                    V[k, q] = s * vkp + c * vkq
    lam = np.empty(n)
    for i in range(n):
        lam[i] = a[i, i]
    return lam, V


@njit(cache=True)
def _inverse_or(A, ginv):
    """The inverse of a symmetric matrix, as R's solve(); when numerically
    singular, its generalized inverse (MASS::ginv's tolerance, `ginv`) or
    zeros."""
    n = A.shape[0]
    lam, V = _eigh(A)
    amax = 0.0
    amin = np.inf
    for i in range(n):
        a = abs(lam[i])
        amax = max(amax, a)
        amin = min(amin, a)
    singular = amin <= EPS * amax
    inv = np.zeros(n)
    for i in range(n):
        a = abs(lam[i])
        if a > 0.0:
            inv[i] = 1.0 / lam[i]
        if singular and (not ginv or a <= _SQRT_EPS * amax):
            inv[i] = 0.0
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0.0
            for m in range(n):
                s += V[i, m] * inv[m] * V[j, m]
            out[i, j] = s
    return out


@njit(cache=True, inline="always")
def _mv(A, x, out):
    """out = A x, in loops (BLAS calls cost more than the work at these sizes)."""
    for i in range(A.shape[0]):
        acc = 0.0
        for j in range(A.shape[1]):
            acc += A[i, j] * x[j]
        out[i] = acc


@njit(cache=True, inline="always")
def _mtv(A, x, out):
    """out = A' x."""
    for i in range(A.shape[1]):
        acc = 0.0
        for j in range(A.shape[0]):
            acc += A[j, i] * x[j]
        out[i] = acc


@njit(cache=True, inline="always")
def _sandwich(A, B, out):
    """out = A B A' (B symmetric), symmetrized."""
    n, m = A.shape
    for i in range(n):
        for j in range(i, n):
            acc = 0.0
            for a in range(m):
                ra = 0.0
                for b in range(m):
                    ra += B[a, b] * A[j, b]
                acc += A[i, a] * ra
            out[i, j] = acc
            out[j, i] = acc


@njit(cache=True)
def _chunks(n):
    nchunk = max(1, min(256, n // 256))
    return nchunk, (n + nchunk - 1) // nchunk if n else 0


@njit(cache=True)
def _add_outer(M, s):
    P = s.shape[0]
    for i in range(P):
        for j in range(P):
            M[i, j] += s[i] * s[j]


@njit(parallel=True, cache=True)
def _meat3(L, u, idx, G):
    """CRV3: sum_g score_g score_g', score_g = u_g + L_g (G - L_g)^+ u_g."""
    P = u.shape[1]
    nchunk, step = _chunks(idx.shape[0])
    part = np.zeros((nchunk, P, P))
    for c in prange(nchunk):
        s = np.empty(P)
        mu = np.empty(P)
        for ii in range(c * step, min(idx.shape[0], (c + 1) * step)):
            g = idx[ii]
            _mv(_inverse_or(G - L[g], True), u[g], mu)
            for i in range(P):
                acc = u[g, i]
                for j in range(P):
                    acc += L[g, i, j] * mu[j]
                s[i] = acc
            _add_outer(part[c], s)
    return part.sum(axis=0)


@njit(parallel=True, cache=True)
def _meat2(L, u, idx, C):
    """CRV2: score_g = u_g + L_g C' V (coef * V' C u_g), V and coef from the
    eigendecomposition of C L_g C'."""
    P = u.shape[1]
    nchunk, step = _chunks(idx.shape[0])
    part = np.zeros((nchunk, P, P))
    for c in prange(nchunk):
        s = np.empty(P)
        F = np.empty((P, P))
        coef = np.empty(P)
        a1 = np.empty(P)
        a2 = np.empty(P)
        w = np.empty(P)
        for ii in range(c * step, min(idx.shape[0], (c + 1) * step)):
            g = idx[ii]
            _sandwich(C, L[g], F)
            lam, V = _eigh(F)
            for i in range(P):
                s2 = max(lam[i], 0.0)
                coef[i] = 0.0 if s2 < 1e-14 else (1.0 / np.sqrt(max(1.0 - s2, 1e-8)) - 1.0) / s2
            _mv(C, u[g], a1)                    # C u
            _mtv(V, a1, a2)                     # V' C u
            for i in range(P):
                a2[i] *= coef[i]
            _mv(V, a2, a1)                      # adj
            _mtv(C, a1, w)                      # C' adj
            for i in range(P):
                acc = u[g, i]
                for j in range(P):
                    acc += L[g, i, j] * w[j]
                s[i] = acc
            _add_outer(part[c], s)
    return part.sum(axis=0)


@njit(parallel=True, cache=True)
def _meat_qq3(L, Pg, uQ, uq, idx, G):
    """CRV3 with h != b: score_g = uQ_g + P_g (G - L_g)^-1 uq_g (zero where
    G - L_g is singular)."""
    P = uQ.shape[1]
    nchunk, step = _chunks(idx.shape[0])
    part = np.zeros((nchunk, P, P))
    for c in prange(nchunk):
        s = np.empty(P)
        adj = np.empty(uq.shape[1])
        for ii in range(c * step, min(idx.shape[0], (c + 1) * step)):
            g = idx[ii]
            _mv(_inverse_or(G - L[g], False), uq[g], adj)
            _mv(Pg[g], adj, s)
            for i in range(P):
                s[i] += uQ[g, i]
            _add_outer(part[c], s)
    return part.sum(axis=0)


@njit(parallel=True, cache=True)
def _meat_qq2(L, Pg, uQ, uq, idx, Ginv_q):
    """CRV2 with h != b (.rdrobust_vce_qq_cluster)."""
    P = uQ.shape[1]
    Pq = uq.shape[1]
    nchunk, step = _chunks(idx.shape[0])
    part = np.zeros((nchunk, P, P))
    for c in prange(nchunk):
        s = np.empty(P)
        T = np.empty((Pq, Pq))
        a1 = np.empty(Pq)
        adj = np.empty(Pq)
        for ii in range(c * step, min(idx.shape[0], (c + 1) * step)):
            g = idx[ii]
            vals, V = _eigh(L[g])
            sig = np.empty(Pq)
            smax = 0.0
            for i in range(Pq):
                sig[i] = np.sqrt(max(vals[i], 0.0))
                smax = max(smax, sig[i])
            any_keep = False
            sig0 = np.zeros(Pq)
            dinv = np.zeros(Pq)
            for i in range(Pq):
                if sig[i] > smax * 1e-10:
                    any_keep = True
                    sig0[i] = sig[i]
                    dinv[i] = 1.0 / sig[i]
            if not any_keep:
                for i in range(P):
                    s[i] = uQ[g, i]
                _add_outer(part[c], s)
                continue
            Vt = np.ascontiguousarray(V.T)
            _sandwich(Vt, Ginv_q, T)
            for i in range(Pq):
                for j in range(Pq):
                    T[i, j] *= sig0[i] * sig0[j]
            tau, W = _eigh(T)
            gamma = np.empty(Pq)
            for i in range(Pq):
                tt = min(max(tau[i], 0.0), 1.0 - 1e-10)
                gamma[i] = 1.0 / np.sqrt(1.0 - tt) - 1.0
            A = np.empty((Pq, Pq))
            for i in range(Pq):
                for j in range(Pq):
                    acc = 0.0
                    for m in range(Pq):
                        acc += V[i, m] * dinv[m] * W[m, j]
                    A[i, j] = acc
            _mtv(A, uq[g], a1)
            for i in range(Pq):
                a1[i] *= gamma[i]
            _mv(A, a1, adj)
            _mv(Pg[g], adj, s)
            for i in range(P):
                s[i] += uQ[g, i]
            _add_outer(part[c], s)
    return part.sum(axis=0)


def meat(L, u, seen, Ginv, mode):
    """sum_g score_g score_g' for the conventional variance (and the robust
    one with h = b, and the bandwidth steps): mode 2 (CRV2) or 3 (CRV3)."""
    idx = np.flatnonzero(seen)
    L, u = np.ascontiguousarray(L), np.ascontiguousarray(u)
    if mode == 3:
        try:
            G = np.linalg.inv(Ginv)
        except np.linalg.LinAlgError:
            G = np.linalg.pinv(Ginv)
        return _meat3(L, u, idx, np.ascontiguousarray(G))
    try:
        C = np.linalg.cholesky(Ginv).T         # R's chol(): upper, C'C = Ginv
    except np.linalg.LinAlgError:
        uu = u[idx]
        return uu.T @ uu
    return _meat2(L, u, idx, np.ascontiguousarray(C))


def meat_qq(L, Pg, uQ, uq, seen, Ginv_q, mode):
    """sum_g score_g score_g' for the robust variance with h != b
    (.rdrobust_vce_qq_cluster): mode 2 (CRV2) or 3 (CRV3)."""
    idx = np.flatnonzero(seen)
    args = [np.ascontiguousarray(a) for a in (L, Pg, uQ, uq)]
    if mode == 3:
        try:
            G = np.linalg.inv(Ginv_q)
        except np.linalg.LinAlgError:
            G = np.linalg.pinv(Ginv_q)
        return _meat_qq3(*args, idx, np.ascontiguousarray(G))
    return _meat_qq2(*args, idx, np.ascontiguousarray(Ginv_q))
