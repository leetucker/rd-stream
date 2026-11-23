"""Compiled passes over one side of the cutoff.

A side's data are *units* in order of distance from the cutoff, so every
kernel window |x - c| < h is a prefix of the side. A unit is one observation,
or (when the running variable has mass points and nothing ties observations
together beyond their x) all the observations at one value of x, carried as
their count n, the sum of y and the within-value sum of squares M2. A *run*
is a maximal set of consecutive units at the same distance: rdrobust's tie
block, over which the nearest-neighbor variance matches whole.

Every sum is a kernel-weighted power sum over a window:

    sum_i (f_i K(u_i) / h)^wpow * t_i^a * v_i,   u_i = t_i / h,  t_i = x_i - c,

with f_i the sampling weight. Kernels: 0 triangular, 1 Epanechnikov,
2 uniform, written as rdrobust_kweight writes them (the triangular and
Epanechnikov windows exclude |u| = 1, the uniform one includes it).
"""

import math

import numpy as np
from numba import njit, prange

NN_TOL = 1.4901161193847656e-08          # sqrt(eps): rdrobust 4.1.1's relative tie tolerance
TRIANGULAR, EPANECHNIKOV, UNIFORM = 0, 1, 2
CHUNK = 4096


@njit(cache=True, inline="always")
def kernel_value(t, h, kern):
    """K(t/h)/h as rdrobust_kweight computes it."""
    u = abs(t / h)
    if kern == UNIFORM:
        return 0.5 / h if u <= 1.0 else 0.0
    if u >= 1.0:
        return 0.0
    if kern == EPANECHNIKOV:
        return 0.75 * (1.0 - u * u) / h
    return (1.0 - u) / h


@njit(cache=True)
def window_len(t, h, kern):
    """The number of units with positive kernel weight, for |t| ascending."""
    lo, hi = 0, t.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        u = abs(t[mid] / h)
        inside = u <= 1.0 if kern == UNIFORM else u < 1.0
        if inside:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def runs(d):
    """Starts of the runs of equal d (and len(d) at the end)."""
    n = d.shape[0]
    nr = 0
    for i in range(n):
        if i == 0 or d[i] != d[i - 1]:
            nr += 1
    start = np.empty(nr + 1, np.int64)
    j = 0
    for i in range(n):
        if i == 0 or d[i] != d[i - 1]:
            start[j] = i
            j += 1
    start[nr] = n
    return start


def _chunks(k):
    nchunk = max(1, min(512, k // CHUNK))
    return nchunk, (k + nchunk - 1) // nchunk if k else 0


@njit(parallel=True, cache=True)
def moments(t, v, use_v, f, use_f, k, h, kern, wpow, A):
    """sum_{i<k} (f_i K/h)^wpow t_i^a v_i for a = 0..A."""
    nchunk = max(1, min(512, k // 4096))
    step = (k + nchunk - 1) // nchunk if k else 0
    part = np.zeros((nchunk, A + 1))
    for c in prange(nchunk):
        lo = c * step
        hi = min(k, lo + step)
        acc = np.zeros(A + 1)
        for i in range(lo, hi):
            w = kernel_value(t[i], h, kern)
            if w == 0.0:
                continue
            if use_f:
                w *= f[i]
            if wpow == 2:
                w *= w
            if use_v:
                w *= v[i]
            p = w
            ti = t[i]
            for a in range(A + 1):
                acc[a] += p
                p *= ti
        part[c, :] = acc
    out = np.zeros(A + 1)
    for c in range(nchunk):
        out += part[c]
    return out


@njit(parallel=True, cache=True)
def nn_residuals(d, cnt, sy, m2, y_row, rows, run_start, nrun, matches, signed):
    """rdrobust_res(vce="nn") on the window of the first `nrun` runs.

    Each run is expanded once over whole runs, nearest first, until it holds
    at least `matches` other observations (rdrobust 4.1.1's tolerance for
    equal distances, both directions on a tie). For an observation i with J
    others in its window and window total S, the residual is
    sqrt(J/(J+1)) (y_i - (S - y_i)/J) = a y_i - c. Returns each unit's sum of
    squared residuals a^2 M2 + n (a ybar - c)^2 and, when `signed` (units
    are rows), each row's residual."""
    k = run_start[nrun]
    nobs = 0
    for i in range(k):
        nobs += cnt[i] if not rows else 1
    rsum = np.empty(nrun)
    rcnt = np.empty(nrun, np.int64)
    for r in prange(nrun):
        s = 0.0
        n = 0
        for i in range(run_start[r], run_start[r + 1]):
            s += sy[i]
            n += 1 if rows else cnt[i]
        rsum[r] = s
        rcnt[r] = n
    ss = np.empty(k)
    res = np.empty(k if signed else 0)
    cap = matches if matches < nobs - 1 else nobs - 1
    for r in prange(nrun):
        lo = 0
        lo += r                       # runs [lo, hi); not an alias of r
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
        S = 0.0
        n = 0
        for j in range(lo, hi):
            S += rsum[j]
            n += rcnt[j]
        J = n - 1
        f = math.sqrt(J / (J + 1.0))
        a = f * (J + 1.0) / J
        cb = f * S / J
        for i in range(p0, run_start[r + 1]):
            if rows:
                e = a * y_row[i] - cb
                ss[i] = e * e
                if signed:
                    res[i] = e
            else:
                e = a * (sy[i] / cnt[i]) - cb
                ss[i] = a * a * m2[i] + cnt[i] * e * e
    return ss, res


@njit(parallel=True, cache=True)
def hc_residuals(t, cnt, sy, m2, rows, f, use_f, k, h, kern, beta, Ginv, mode, factor, signed):
    """rdrobust_res for vce hc0..hc3 on the window [0, k): w_adj (y - r'beta),
    with w_adj 1 (hc0), `factor` (hc1: sqrt(n/(n-d)), or 1 with clusters),
    1/sqrt(1-h_ii) (hc2) or 1/(1-h_ii) (hc3), h_ii = w_i r_i' Ginv r_i.
    Returns each unit's sum of squared residuals and, when `signed`, each
    row's residual."""
    P = beta.shape[0]
    ss = np.empty(k)
    res = np.empty(k if signed else 0)
    for i in prange(k):
        ti = t[i]
        m = 0.0
        pw = 1.0
        for j in range(P):
            m += pw * beta[j]
            pw *= ti
        wadj = factor
        if mode >= 2:
            w = kernel_value(ti, h, kern)
            if use_f:
                w *= f[i]
            hii = 0.0
            pj = 1.0
            for j in range(P):
                pl = 1.0
                for jj in range(P):
                    hii += pj * Ginv[j, jj] * pl
                    pl *= ti
                pj *= ti
            hii *= w
            one = 1.0 - hii
            if one < 1e-8:
                one = 1e-8
            wadj = math.sqrt(1.0 / one) if mode == 2 else 1.0 / one
        if rows:
            e = wadj * (sy[i] - m)
            ss[i] = e * e
            if signed:
                res[i] = e
        else:
            ybar = sy[i] / cnt[i]
            e = ybar - m
            ss[i] = wadj * wadj * (m2[i] + cnt[i] * e * e)
    return ss, res


@njit(parallel=True, cache=True)
def poly_scores(t, f, use_f, k, h, kern, P, res):
    """Rows (f K/h) [1, t, .., t^(P-1)] * res_i over the window [0, k)."""
    out = np.empty((k, P))
    for i in prange(k):
        w = kernel_value(t[i], h, kern)
        if use_f:
            w *= f[i]
        w *= res[i]
        for j in range(P):
            out[i, j] = w
            w *= t[i]
    return out


@njit(parallel=True, cache=True)
def final_rows(t, f, use_f, k, h, b, kern, p, q, g, L):
    """The final step's per-unit design over the window [0, k):
    RW_p = w_h [1, .., t^p] and Q = RW_p - h^(p+1) m L', m = w_b sum_l t^l g_l
    (rdrobust's Q_q)."""
    P = p + 1
    RW = np.empty((k, P))
    Q = np.empty((k, P))
    hp1 = h ** (p + 1)
    for i in prange(k):
        ti = t[i]
        wh = kernel_value(ti, h, kern)
        wb = kernel_value(ti, b, kern)
        if use_f:
            wh *= f[i]
            wb *= f[i]
        m = 0.0
        pw = 1.0
        for jj in range(q + 1):
            m += pw * g[jj]
            pw *= ti
        m *= wb
        pw = 1.0
        for j in range(P):
            RW[i, j] = wh * pw
            Q[i, j] = RW[i, j] - hp1 * m * L[j]
            pw *= ti
    return RW, Q


@njit(cache=True)
def cluster_meat(scores, cl, n_clusters):
    """sum over clusters of (sum of the cluster's score rows) outer itself,
    and the number of clusters present."""
    k, P = scores.shape
    sums = np.zeros((n_clusters, P))
    seen = np.zeros(n_clusters, np.bool_)
    for i in range(k):
        g = cl[i]
        seen[g] = True
        for j in range(P):
            sums[g, j] += scores[i, j]
    M = np.zeros((P, P))
    G = 0
    for g in range(n_clusters):
        if seen[g]:
            G += 1
            for j in range(P):
                for jj in range(P):
                    M[j, jj] += sums[g, j] * sums[g, jj]
    return M, G


@njit(parallel=True, cache=True)
def collapse_runs(x, y, start):
    """Count, sum of y and within-value sum of squares of each run of equal x
    (x sorted): the units of the collapsed representation."""
    nr = start.shape[0] - 1
    cnt = np.empty(nr, np.int64)
    sy = np.empty(nr)
    m2 = np.empty(nr)
    for r in prange(nr):
        lo, hi = start[r], start[r + 1]
        s = 0.0
        for i in range(lo, hi):
            s += y[i]
        mean = s / (hi - lo)
        q = 0.0
        for i in range(lo, hi):
            e = y[i] - mean
            q += e * e
        cnt[r] = hi - lo
        sy[r] = s
        m2[r] = q
    return cnt, sy, m2


@njit(parallel=True, cache=True)
def collapse_runs_multi(x, D, start):
    """Count, column sums and within-value cross-products of each run of
    equal x (x sorted), for the response matrix D."""
    nr = start.shape[0] - 1
    nc = D.shape[1]
    cnt = np.empty(nr, np.int64)
    SD = np.empty((nr, nc))
    M2 = np.empty((nr, nc, nc))
    for r in prange(nr):
        lo, hi = start[r], start[r + 1]
        mean = np.zeros(nc)
        for i in range(lo, hi):
            for j in range(nc):
                mean[j] += D[i, j]
        for j in range(nc):
            SD[r, j] = mean[j]
            mean[j] /= hi - lo
        acc = np.zeros((nc, nc))
        for i in range(lo, hi):
            for j in range(nc):
                ej = D[i, j] - mean[j]
                for jj in range(nc):
                    acc[j, jj] += ej * (D[i, jj] - mean[jj])
        M2[r] = acc
        cnt[r] = hi - lo
    return cnt, SD, M2


@njit(parallel=True, cache=True)
def weights(t, f, use_f, k, h, kern):
    """f K(t/h)/h over the window [0, k)."""
    out = np.empty(k)
    for i in prange(k):
        w = kernel_value(t[i], h, kern)
        out[i] = w * f[i] if use_f else w
    return out
