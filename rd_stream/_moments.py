"""Window sums for many bandwidths on one side: block power sums.

On one side of the cutoff, a kernel window |x - c| < h is a prefix of the
units (sorted by distance d = |x - c|), and the kernels are polynomials in
d / h on the window. So every sum a fit needs,

    sum_{i < k} poly(d_i) d_i^a v_i,

is a combination of the prefix power sums S_m(k) = sum_{i < k} (d_i / D)^m v_i
(D the side's largest distance). `PowerTable` holds them at the ends of
blocks of BLOCK units (cumulated with compensation) and adds the partial
block at the end directly, so a window's sums cost O(BLOCK) whatever its
size. A window always starts at the cutoff, so no sum is a difference of
two large prefix sums. The kernel polynomial's expansion costs about two
digits for the highest powers, which leaves the sums accurate to about
1e-14 relative.

`GridSide` puts this to work for the fits of a bandwidth grid: tables of
the kernel weights and of the response columns, and, for the
nearest-neighbor variance, of the residuals' cross-products. rdrobust's
nearest-neighbor residuals depend on the window only through its far edge:
a unit's neighbors are the nearest units within the window, so only the
last few runs (tie blocks) of a narrower window differ from the widest
window's. `GridSide` computes every run's residual map once, at the widest
window, and recomputes the runs at a narrower window's edge.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from . import _kernels as K

BLOCK = 256


# ---------------------------------------------------------------------------
# compiled pieces

@njit(parallel=True, cache=True)
def _block_sums(d, V, inv_D, M, nblk):
    nb = V.shape[1]
    T = np.zeros((nblk, M + 1, nb))
    for b in prange(nblk):
        lo = b * BLOCK
        hi = lo + BLOCK
        acc = np.zeros((M + 1, nb))
        for i in range(lo, hi):
            w = d[i] * inv_D
            p = 1.0
            for m in range(M + 1):
                for c in range(nb):
                    acc[m, c] += p * V[i, c]
                p *= w
        T[b] = acc
    return T


@njit(cache=True)
def _cumulate(T):
    """Running sums over blocks, with Neumaier's compensation."""
    nblk, M1, nb = T.shape
    cum = np.zeros((nblk + 1, M1, nb))
    s = np.zeros((M1, nb))
    comp = np.zeros((M1, nb))
    for b in range(nblk):
        for m in range(M1):
            for c in range(nb):
                x = T[b, m, c]
                t = s[m, c] + x
                if abs(s[m, c]) >= abs(x):
                    comp[m, c] += (s[m, c] - t) + x
                else:
                    comp[m, c] += (x - t) + s[m, c]
                s[m, c] = t
                cum[b + 1, m, c] = t + comp[m, c]
    return cum


@njit(cache=True)
def _prefix(d, V, cum, inv_D, M, k):
    b = k // BLOCK
    out = cum[b].copy()
    nb = V.shape[1]
    for i in range(b * BLOCK, k):
        w = d[i] * inv_D
        p = 1.0
        for m in range(M + 1):
            for c in range(nb):
                out[m, c] += p * V[i, c]
            p *= w
    return out


@njit(cache=True)
def _direct(d, v, inv_D, lo, hi, M):
    """sum_{i in [lo, hi)} (d_i / D)^m v_i, m = 0..M."""
    out = np.zeros(M + 1)
    for i in range(lo, hi):
        w = d[i] * inv_D
        p = v[i - lo]
        for m in range(M + 1):
            out[m] += p
            p *= w
    return out


@njit(parallel=True, cache=True)
def run_sums(D, cnt, rows, run_start):
    """Each run's column sums and observation count."""
    nr = run_start.shape[0] - 1
    nc = D.shape[1]
    rsum = np.zeros((nr, nc))
    rcnt = np.zeros(nr, np.int64)
    for r in prange(nr):
        n = 0
        for i in range(run_start[r], run_start[r + 1]):
            for c in range(nc):
                rsum[r, c] += D[i, c]
            n += 1 if rows else cnt[i]
        rcnt[r] = n
    return rsum, rcnt


@njit(parallel=True, cache=True)
def nn_runs(d, rsum, rcnt, run_start, nrun, nobs, matches, r_lo, r_hi):
    """rdrobust's nearest-neighbor residual map of runs [r_lo, r_hi) in the
    window of the first `nrun` runs (`nobs` observations): each unit's
    residual is a D_i - cb (a row) or a dbar_i - cb (a collapsed unit's
    mean), with `hi` the end of the run's neighbor set. The search is
    _kernels.nn_residuals'."""
    n = r_hi - r_lo
    nc = rsum.shape[1]
    a = np.empty(n)
    cb = np.empty((n, nc))
    hi_out = np.empty(n, np.int64)
    cap = matches if matches < nobs - 1 else nobs - 1
    for idx in prange(n):
        r = r_lo + idx
        lo = 0
        lo += r
        hi = lo + 1
        p0 = run_start[r]
        tot = rcnt[r] - 1
        while tot < cap:
            if lo - 1 < 0:
                tot += rcnt[hi]
                hi += 1
            elif hi >= nrun:
                lo -= 1
                tot += rcnt[lo]
            else:
                dl = d[p0] - d[run_start[lo - 1]]
                dr = d[run_start[hi]] - d[p0]
                tol = (dl if dl > dr else dr) * 1.4901161193847656e-08
                if dl - dr > tol:
                    tot += rcnt[hi]
                    hi += 1
                elif dr - dl > tol:
                    lo -= 1
                    tot += rcnt[lo]
                else:
                    tot += rcnt[hi]
                    hi += 1
                    lo -= 1
                    tot += rcnt[lo]
        S = np.zeros(nc)
        nn = 0
        for j in range(lo, hi):
            for c in range(nc):
                S[c] += rsum[j, c]
            nn += rcnt[j]
        J = nn - 1
        f = math.sqrt(J / (J + 1.0))
        a[idx] = f * (J + 1.0) / J
        for c in range(nc):
            cb[idx, c] = f * S[c] / J
        hi_out[idx] = hi
    return a, cb, hi_out


@njit(parallel=True, cache=True)
def residual_products(D, M2, cntf, f2, use_f, rows, run_start, a, cb, r_lo, r_hi, pairs):
    """Each unit's residual cross-products E_jl (the pairs j <= l) for runs
    [r_lo, r_hi), with the residual map a, cb indexed from r_lo: f^2 e_j e_l
    for a row, a^2 M2_jl + n (a dbar_j - cb_j)(a dbar_l - cb_l) for a unit."""
    u0 = run_start[r_lo]
    nu = run_start[r_hi] - u0
    npair = pairs.shape[0]
    E = np.empty((nu, npair))
    for idx in prange(r_hi - r_lo):
        r = r_lo + idx
        ar = a[idx]
        for i in range(run_start[r], run_start[r + 1]):
            for q in range(npair):
                j = pairs[q, 0]
                jj = pairs[q, 1]
                if rows:
                    v = (ar * D[i, j] - cb[idx, j]) * (ar * D[i, jj] - cb[idx, jj])
                    if use_f:
                        v *= f2[i]
                else:
                    n = cntf[i]
                    ej = ar * D[i, j] / n - cb[idx, j]
                    ejj = ar * D[i, jj] / n - cb[idx, jj]
                    v = ar * ar * M2[i, j, jj] + n * ej * ejj
                E[i - u0, q] = v
    return E


@njit(parallel=True, cache=True)
def moments_mixed(t, v, f, use_f, k, h, b, kern, A):
    """sum_{i<k} f_i^2 K_h(t_i) K_b(t_i) t_i^a v_i for a = 0..A (the robust
    variance's cross term)."""
    nchunk = max(1, min(512, k // 4096))
    step = (k + nchunk - 1) // nchunk if k else 0
    part = np.zeros((nchunk, A + 1))
    for c in prange(nchunk):
        lo = c * step
        hi = min(k, lo + step)
        acc = np.zeros(A + 1)
        for i in range(lo, hi):
            w = K.kernel_value(t[i], h, kern) * K.kernel_value(t[i], b, kern)
            if w == 0.0:
                continue
            if use_f:
                w *= f[i] * f[i]
            p = w * v[i]
            ti = t[i]
            for a in range(A + 1):
                acc[a] += p
                p *= ti
        part[c, :] = acc
    out = np.zeros(A + 1)
    for c in range(nchunk):
        out += part[c]
    return out


# ---------------------------------------------------------------------------
# kernels as polynomials in w = d / D

def kernel_poly(kern, h, D):
    """K(d/h)/h on the window as coefficients of powers of w = d / D."""
    r = D / h
    if kern == K.TRIANGULAR:
        return np.array([1.0, -r]) / h
    if kern == K.EPANECHNIKOV:
        return np.array([0.75, 0.0, -0.75 * r * r]) / h
    return np.array([0.5]) / h


class PowerTable:
    """Prefix power sums of the columns of V over the first `extent` units
    of a side (see the module docstring)."""

    def __init__(self, d, V, D, M, extent):
        self.d, self.D, self.M = d, D, M
        self.inv_D = 1.0 / D
        self.V = np.ascontiguousarray(V.reshape(V.shape[0], -1))
        self.extent = extent
        nblk = extent // BLOCK
        self.cum = _cumulate(_block_sums(d, self.V, self.inv_D, M, nblk))

    def prefix(self, k):
        """S[m, c] = sum_{i<k} (d_i / D)^m V[i, c], m = 0..M."""
        if k > self.extent:
            raise ValueError("window beyond the table")
        return _prefix(self.d, self.V, self.cum, self.inv_D, self.M, k)


def _combine(S, coef, D, A, sgn):
    """out[a] = sum_j coef_j sgn^a D^a S[j + a]: the moments in t = x - c."""
    J = coef.shape[0]
    out = np.empty((A + 1,) + S.shape[1:])
    for a in range(A + 1):
        out[a] = np.tensordot(coef, S[a:a + J], axes=(0, 0)) * (sgn * D) ** a
    return out


class GridSide:
    """A side's tables for fits at many bandwidths within `extent` units."""

    def __init__(self, side, extent, *, p, q, vce, matches):
        s = side
        self.s, self.extent, self.vce, self.matches = s, extent, vce, matches
        self.D = float(s.d[extent - 1]) if extent and s.d[extent - 1] > 0 else 1.0
        self.sgn = -1.0 if extent and s.t[0] < 0 else 1.0
        deg = 4                      # Epanechnikov squared
        self.M_w = max(3 * p + 1, 2 * q) + 2
        self.M_E = max(2 * p, p + q, 2 * q) + deg
        if s.rows:
            w = s.f if s.f is not None else np.ones(s.n_units)
            Dw = s.D if s.f is None else s.D * s.f[:, None]
        else:
            w, Dw = s.cntf, s.D
        self.tw = PowerTable(s.d, w, self.D, self.M_w, extent)
        self.tD = PowerTable(s.d, Dw, self.D, q + 2, extent)
        self.nn = None
        if vce == "nn":
            self._nn_tables()

    # ---- the kernel weights and the response columns

    def moments_w(self, k, h, kern, A):
        S = self.tw.prefix(k)[:, 0]
        return _combine(S, kernel_poly(kern, h, self.D), self.D, A, self.sgn)

    def moments_D(self, k, h, kern, A):
        S = self.tD.prefix(k)
        return _combine(S, kernel_poly(kern, h, self.D), self.D, A, self.sgn)

    # ---- nearest-neighbor residuals

    def _nn_tables(self):
        s = self.s
        self.nrun = int(np.searchsorted(s.run_start, self.extent, side="left"))
        rows = s.rows
        cnt = s.cnt
        self.rsum, self.rcnt = run_sums(s.D, cnt, rows, s.run_start[:self.nrun + 1])
        self.nobs = int(self.rcnt.sum())
        self.a, self.cb, self.hi = nn_runs(s.d, self.rsum, self.rcnt, s.run_start, self.nrun,
                                           self.nobs, self.matches, 0, self.nrun)
        nc = s.ncol
        self.pairs = np.array([(j, jj) for j in range(nc) for jj in range(j, nc)], np.int64)
        E = self._products(0, self.nrun, self.a, self.cb)
        self.tE = PowerTable(s.d, E, self.D, self.M_E, self.extent)

    def _products(self, r_lo, r_hi, a, cb):
        s = self.s
        f2 = s.f * s.f if s.f is not None else np.zeros(1)
        M2 = s.M2 if not s.rows else np.zeros((1, 1, 1))
        return residual_products(s.D, M2, s.cntf, f2, s.f is not None, s.rows, s.run_start,
                                 a, cb, r_lo, r_hi, self.pairs)

    def residuals_at(self, k):
        """The residual products at the window of the first k units: the
        edge units whose products differ from the table's, with the
        differences, or None if the window is too small for the tables."""
        s = self.s
        nrun = int(np.searchsorted(s.run_start, k, side="left"))
        nobs = int(self.rcnt[:nrun].sum())
        cap_full = min(self.matches, self.nobs - 1)
        if min(self.matches, nobs - 1) != cap_full or nrun == 0:
            return None
        r_lo = max(0, nrun - self.matches - 2)
        bad = np.nonzero(self.hi[r_lo:nrun] > nrun)[0]
        if bad.size == 0:
            return (k, 0, np.zeros((0, self.pairs.shape[0])))
        r0 = r_lo + int(bad[0])
        a, cb, _ = nn_runs(s.d, self.rsum, self.rcnt, s.run_start, nrun, nobs, self.matches,
                           r0, nrun)
        E_new = self._products(r0, nrun, a, cb)
        E_old = self._products(r0, nrun, self.a[r0:nrun], self.cb[r0:nrun])
        return (k, int(s.run_start[r0]), E_new - E_old)

    def ss_moments(self, edge, svec, kq, coef, A):
        """sum_{i<kq} poly(w_i) t_i^a ss_i(s), ss_i(s) the squared residual of
        the combined response D s (times f^2), poly given by `coef` in w."""
        k, u0, dE = edge
        cw = np.array([svec[j] * svec[jj] * (1.0 if j == jj else 2.0) for j, jj in self.pairs])
        Mneed = coef.shape[0] - 1 + A
        S = self.tE.prefix(kq)[:Mneed + 1] @ cw
        if dE.shape[0] and kq > u0:
            hi = min(kq, k)
            dss = dE[:hi - u0] @ cw
            S = S + _direct(self.s.d, dss, 1.0 / self.D, u0, hi, Mneed)
        return _combine(S, coef, self.D, A, self.sgn)
