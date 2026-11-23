"""Gate 2 prototype: rdrobust's default fit, compiled and threaded.

Reproduces rdrobust(y, x) (R 4.1.1 / Python 2.1.1) for the default path:
sharp RD at cutoff c, p = 1, q = 2, triangular kernel, bwselect = "mserd",
vce = "nn" with 3 matches, masspoints = "adjust" (bwcheck = 10 when mass
points are detected), bwrestrict, stdvars = True for the bandwidth step,
scaleregul = 1.

Layout: each side of the cutoff is stored in order of distance from the
cutoff (the left side reversed), so every kernel window |x - c| < h is a
prefix of its side. A window's length is a binary search; its neighbor
residuals and kernel-weighted power sums are compiled parallel passes over
the prefix. Nothing is held per observation beyond x, y, the duplicate
counts and one residual vector.

    python prototype.py data.parquet [--variant x|xd] [--compare result.json]
"""

import argparse
import json
import math
import time

import numpy as np
import polars as pl
from numba import get_num_threads, njit, prange

EPS = np.finfo(float).eps
SQRT_EPS = math.sqrt(EPS)
C_TRIANGULAR = 2.576


# ---------------------------------------------------------------------------
# compiled kernels

@njit(cache=True)
def dup_blocks(d):
    """rdrobust's dups (size of each point's tie block) and dupsid (1-based
    position within it), for d sorted ascending, plus the block count."""
    n = d.shape[0]
    dups = np.empty(n, np.int64)
    dupsid = np.empty(n, np.int64)
    i = 0
    nb = 0
    while i < n:
        j = i + 1
        while j < n and d[j] == d[i]:
            j += 1
        for k in range(i, j):
            dups[k] = j - i
            dupsid[k] = k - i + 1
        nb += 1
        i = j
    return dups, dupsid, nb


@njit(cache=True)
def kth_block_distance(d, k):
    """Distance of the k-th distinct value from the cutoff (1-based)."""
    n = d.shape[0]
    seen = 0
    i = 0
    while i < n:
        seen += 1
        if seen == k:
            return d[i]
        j = i + 1
        while j < n and d[j] == d[i]:
            j += 1
        i = j
    return d[n - 1]


@njit(cache=True)
def window_len(t, h):
    """Number of points with |t / h| < 1 (rdrobust's w > 0), for |t| ascending."""
    lo, hi = 0, t.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if abs(t[mid] / h) < 1.0:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(parallel=True, cache=True)
def nn_residuals(d, y, dups, dupsid, k, matches):
    """rdrobust_res(vce="nn") on the window [0, k): sqrt(J/(J+1)) * (y_i -
    mean of the J nearest same-side neighbors, whole tie blocks, both
    directions on a tie within rdrobust 4.1.1's relative tolerance)."""
    res = np.empty(k)
    cap = matches if matches < k - 1 else k - 1
    for pos in prange(k):
        rpos = dups[pos] - dupsid[pos]
        lpos = dupsid[pos] - 1
        while lpos + rpos < cap:
            if pos - lpos - 1 < 0:
                rpos += dups[pos + rpos + 1]
            elif pos + rpos + 1 >= k:
                lpos += dups[pos - lpos - 1]
            else:
                dl = d[pos] - d[pos - lpos - 1]
                dr = d[pos + rpos + 1] - d[pos]
                tol = (dl if dl > dr else dr) * 1.4901161193847656e-08
                if dl - dr > tol:
                    rpos += dups[pos + rpos + 1]
                elif dr - dl > tol:
                    lpos += dups[pos - lpos - 1]
                else:
                    rpos += dups[pos + rpos + 1]
                    lpos += dups[pos - lpos - 1]
        lo = pos - lpos
        hi = pos + rpos + 1
        if hi > k:
            hi = k
        J = (hi - lo) - 1
        s = 0.0
        for j in range(lo, hi):
            s += y[j]
        s -= y[pos]
        res[pos] = math.sqrt(J / (J + 1.0)) * (y[pos] - s / J)
    return res


@njit(cache=True)
def block_starts(dupsid):
    """Start of each tie block (dupsid == 1) and its size."""
    n = dupsid.shape[0]
    nb = 0
    for i in range(n):
        if dupsid[i] == 1:
            nb += 1
    start = np.empty(nb + 1, np.int64)
    j = 0
    for i in range(n):
        if dupsid[i] == 1:
            start[j] = i
            j += 1
    start[nb] = n
    return start


@njit(parallel=True, cache=True)
def nn_residuals_blocks(d, y, start, nbk, matches):
    """rdrobust_res(vce="nn") on the window of the first nbk tie blocks.
    Every row of a block has the same neighbor window (whole blocks), so
    each block is expanded once and the window summed from block totals:
    O(rows + blocks), not O(rows x block size)."""
    k = start[nbk]
    bsum = np.empty(nbk)
    for b in prange(nbk):
        s = 0.0
        for i in range(start[b], start[b + 1]):
            s += y[i]
        bsum[b] = s
    res = np.empty(k)
    cap = matches if matches < k - 1 else k - 1
    for b in prange(nbk):
        blo = 0
        blo += b                     # blocks [blo, bhi); not an alias of b
        bhi = blo + 1
        p0 = start[b]
        tot = start[b + 1] - p0 - 1
        while tot < cap:
            if blo - 1 < 0:
                tot += start[bhi + 1] - start[bhi]
                bhi += 1
            elif bhi >= nbk:
                blo -= 1
                tot += start[blo + 1] - start[blo]
            else:
                dl = d[p0] - d[start[blo - 1]]
                dr = d[start[bhi]] - d[p0]
                tol = (dl if dl > dr else dr) * 1.4901161193847656e-08
                if dl - dr > tol:
                    tot += start[bhi + 1] - start[bhi]
                    bhi += 1
                elif dr - dl > tol:
                    blo -= 1
                    tot += start[blo + 1] - start[blo]
                else:
                    tot += start[bhi + 1] - start[bhi]
                    bhi += 1
                    blo -= 1
                    tot += start[blo + 1] - start[blo]
        S = 0.0
        for j in range(blo, bhi):
            S += bsum[j]
        J = (start[bhi] - start[blo]) - 1
        f = math.sqrt(J / (J + 1.0))
        for i in range(p0, start[b + 1]):
            res[i] = f * (y[i] - (S - y[i]) / J)
    return res


def nn_res(s, k, matches):
    """NN residuals on the window [0, k) of side s (k is block-aligned)."""
    nbk = int(np.searchsorted(s.start, k, side="left"))
    return nn_residuals_blocks(s.d, s.y, s.start, nbk, matches)


@njit(parallel=True, cache=True)
def moments(t, v, k, h, wpow, A, use_v):
    """sum_{i<k} (K(t_i/h)/h)^wpow * t_i^a * v_i for a = 0..A, triangular K."""
    nchunk = max(1, min(256, k // 4096))
    part = np.zeros((nchunk, A + 1))
    step = (k + nchunk - 1) // nchunk
    for c in prange(nchunk):
        lo = c * step
        hi = min(k, lo + step)
        acc = np.zeros(A + 1)
        for i in range(lo, hi):
            u = t[i] / h
            w = (1.0 - abs(u)) / h
            if w <= 0.0:
                continue
            ww = w if wpow == 1 else w * w
            if use_v:
                ww *= v[i]
            p = ww
            for a in range(A + 1):
                acc[a] += p
                p *= t[i]
        part[c, :] = acc
    out = np.zeros(A + 1)
    for c in range(nchunk):
        out += part[c]
    return out


@njit(parallel=True, cache=True)
def final_scores(t, y, res, k, h, b, p, q, g, L):
    """The final step's sums over the window [0, k):
    sum RW_p y, sum Q y, sum RW_p RW_p' res^2, sum Q Q' res^2, where
    RW_p = w_h [1, t, .., t^p], m = w_b sum_l t^l g_l and
    Q = RW_p - h^(p+1) m L'."""
    P = p + 1
    nchunk = max(1, min(256, k // 4096))
    step = (k + nchunk - 1) // nchunk
    part = np.zeros((nchunk, 2 * P + 2 * P * P))
    hp1 = h ** (p + 1)
    for c in prange(nchunk):
        lo = c * step
        hi = min(k, lo + step)
        acc = np.zeros(2 * P + 2 * P * P)
        rw = np.empty(P)
        Q = np.empty(P)
        for i in range(lo, hi):
            ti = t[i]
            uh = abs(ti / h)
            ub = abs(ti / b)
            wh = (1.0 - uh) / h if uh < 1.0 else 0.0
            wb = (1.0 - ub) / b if ub < 1.0 else 0.0
            m = 0.0
            pw = 1.0
            for l in range(q + 1):
                m += pw * g[l]
                pw *= ti
            m *= wb
            pw = 1.0
            for j in range(P):
                rw[j] = wh * pw
                Q[j] = rw[j] - hp1 * m * L[j]
                pw *= ti
            r2 = res[i] * res[i]
            for j in range(P):
                acc[j] += rw[j] * y[i]
                acc[P + j] += Q[j] * y[i]
                for l in range(P):
                    acc[2 * P + j * P + l] += rw[j] * rw[l] * r2
                    acc[2 * P + P * P + j * P + l] += Q[j] * Q[l] * r2
        part[c, :] = acc
    out = np.zeros(2 * P + 2 * P * P)
    for c in range(nchunk):
        out += part[c]
    return out


# ---------------------------------------------------------------------------
# one side of the cutoff

class Side:
    """One side's data, in order of distance from the cutoff."""

    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.t = x - c
        self.d = np.abs(self.t)
        self.dups, self.dupsid, self.M = dup_blocks(self.d)
        self.start = block_starts(self.dupsid)
        self.n = x.shape[0]

    def with_y(self, y):
        s = Side.__new__(Side)
        s.x, s.t, s.d, s.y = self.x, self.t, self.d, y
        s.dups, s.dupsid, s.M, s.n, s.start = self.dups, self.dupsid, self.M, self.n, self.start
        return s

    def scaled(self, x_sd, y_sd, c):
        s = Side.__new__(Side)
        s.x = self.x / x_sd
        s.y = self.y / y_sd
        s.t = s.x - c / x_sd
        s.d = np.abs(s.t)
        s.dups, s.dupsid, s.M, s.n, s.start = self.dups, self.dupsid, self.M, self.n, self.start
        return s


def gram(mom, o):
    return np.array([[mom[j + l] for l in range(o + 1)] for j in range(o + 1)])


def inv_spd(G):
    try:
        L = np.linalg.cholesky(G)
        Li = np.linalg.inv(L)
        return Li.T @ Li
    except np.linalg.LinAlgError:
        return np.linalg.pinv(G)


class BandwidthSide:
    """rdrobust_bw for one side, with the pilot residuals at c_bw shared."""

    def __init__(self, s, c_bw, nnmatch):
        self.s = s
        self.c_bw = c_bw
        self.kV = window_len(s.t, c_bw)
        self.resV2 = nn_res(s, self.kV, nnmatch) ** 2
        self.nnmatch = nnmatch

    def vfit(self, o, nu):
        s, h, k = self.s, self.c_bw, self.kV
        m1 = moments(s.t, s.y, k, h, 1, 2 * o + o + 1, False)
        my = moments(s.t, s.y, k, h, 1, o, True)
        m2 = moments(s.t, self.resV2, k, h, 2, 2 * o, True)
        Gi = inv_spd(gram(m1, o))
        V_V = (Gi @ gram(m2, o) @ Gi)[nu, nu]
        v = np.array([m1[j + o + 1] for j in range(o + 1)]) / h ** (o + 1)
        Hp = np.array([h ** j for j in range(o + 1)])
        BConst = (Hp * (Gi @ v))[nu]
        return V_V, BConst

    def bw(self, o, nu, o_B, h_B, scale):
        s = self.s
        V_V, BConst = self.vfit(o, nu)
        kB = window_len(s.t, h_B)
        m1 = moments(s.t, s.y, kB, h_B, 1, 2 * o_B, False)
        my = moments(s.t, s.y, kB, h_B, 1, o_B, True)
        Gi = inv_spd(gram(m1, o_B))
        beta = Gi @ my
        BWreg = 0.0
        if scale > 0:
            r2 = nn_res(s, kB, self.nnmatch) ** 2
            m2 = moments(s.t, r2, kB, h_B, 2, 2 * o_B, True)
            V_B = (Gi @ gram(m2, o_B) @ Gi)[-1, -1]
            BWreg = 3 * BConst ** 2 * V_B
        B = math.sqrt(2 * (o + 1 - nu)) * BConst * beta[-1]
        V = (2 * nu + 1) * self.c_bw ** (2 * nu + 1) * V_V
        R = scale * (2 * (o + 1 - nu)) * BWreg
        return V, B, R, 1 / (2 * o + 3)


def quantile_type2(xs, prob):
    n = xs.shape[0]
    hq = n * prob
    j = int(math.floor(hq))
    return 0.5 * (xs[j - 1] + xs[j]) if hq == j else xs[j]


# ---------------------------------------------------------------------------
# the fit

def make_sides(x_sorted, y_sorted, c=0.0):
    split = int(np.searchsorted(x_sorted, c, side="left"))       # x < c on the left
    left = Side(np.ascontiguousarray(x_sorted[:split][::-1]),
                np.ascontiguousarray(y_sorted[:split][::-1]), c)
    right = Side(x_sorted[split:], y_sorted[split:], c)
    return left, right, split


def outcome_sides(left, right, split, y_sorted):
    return (left.with_y(np.ascontiguousarray(y_sorted[:split][::-1])),
            right.with_y(y_sorted[split:]))


def rd_fit(x_sorted, y_sorted, c=0.0, p=1, q=2, nnmatch=3, scaleregul=1.0, sides=None):
    """x_sorted ascending (y in the same order). Returns a dict like the
    gate runners' RESULT."""
    timings = {}
    t0 = time.perf_counter()
    left, right = sides if sides is not None else make_sides(x_sorted, y_sorted, c)[:2]
    timings["sides_s"] = time.perf_counter() - t0
    h, b = select_bandwidth(left, right, x_sorted, y_sorted, c, p, q, nnmatch, scaleregul)
    timings["bandwidth_s"] = time.perf_counter() - t0 - timings["sides_s"]
    t0 = time.perf_counter()
    r = estimate(left, right, h, b, p, q, nnmatch)
    timings["estimate_s"] = time.perf_counter() - t0
    r.update(timings)
    return r


def select_bandwidth(left, right, x_sorted, y_sorted, c, p, q, nnmatch, scaleregul):
    x_sd = float(np.std(x_sorted, ddof=1))
    y_sd = float(np.std(y_sorted, ddof=1))
    x_iq = quantile_type2(x_sorted, 0.75) - quantile_type2(x_sorted, 0.25)
    BWp = min(1.0, (x_iq / x_sd) / 1.349)                    # stdvars
    x_min, x_max = float(x_sorted[0]), float(x_sorted[-1])
    range_l = abs(c - x_min) / x_sd
    range_r = abs(c - x_max) / x_sd
    M_l, M_r = left.M, right.M
    bwcheck = None
    if 1 - M_l / left.n >= 0.2 or 1 - M_r / right.n >= 0.2:
        bwcheck = 10
    sl = left.scaled(x_sd, y_sd, c)
    sr = right.scaled(x_sd, y_sd, c)
    cs = c / x_sd
    c_bw = C_TRIANGULAR * BWp * (M_l + M_r) ** (-1 / 5)
    bw_max_l = abs(cs - x_min / x_sd)
    bw_max_r = abs(cs - x_max / x_sd)
    bw_max = max(bw_max_l, bw_max_r)
    c_bw = min(c_bw, bw_max)
    if bwcheck is not None:
        bw_min_l = kth_block_distance(sl.d, min(bwcheck, M_l)) * (1 + SQRT_EPS)
        bw_min_r = kth_block_distance(sr.d, min(bwcheck, M_r)) * (1 + SQRT_EPS)
        c_bw = max(c_bw, bw_min_l, bw_min_r)

    bl = BandwidthSide(sl, c_bw, nnmatch)
    br = BandwidthSide(sr, c_bw, nnmatch)
    C_d_l = bl.bw(q + 1, q + 1, q + 2, range_l * (1 + SQRT_EPS), 0)
    C_d_r = br.bw(q + 1, q + 1, q + 2, range_r * (1 + SQRT_EPS), 0)
    d_bw = ((C_d_l[0] + C_d_r[0]) / (C_d_r[1] - C_d_l[1]) ** 2) ** C_d_l[3]
    d_bw = min(d_bw, bw_max)
    if bwcheck is not None:
        d_bw = max(d_bw, bw_min_l, bw_min_r)
    C_b_l = bl.bw(q, p + 1, q + 1, d_bw, scaleregul)
    C_b_r = br.bw(q, p + 1, q + 1, d_bw, scaleregul)
    b_bw = ((C_b_l[0] + C_b_r[0]) / ((C_b_r[1] - C_b_l[1]) ** 2
                                     + scaleregul * (C_b_r[2] + C_b_l[2]))) ** C_b_l[3]
    b_bw = min(b_bw, bw_max)
    C_h_l = bl.bw(p, 0, q, b_bw, scaleregul)
    C_h_r = br.bw(p, 0, q, b_bw, scaleregul)
    h_bw = ((C_h_l[0] + C_h_r[0]) / ((C_h_r[1] - C_h_l[1]) ** 2
                                     + scaleregul * (C_h_r[2] + C_h_l[2]))) ** C_h_l[3]
    h_bw = min(h_bw, bw_max)
    return x_sd * h_bw, x_sd * b_bw


def estimate(left, right, h, b, p=1, q=2, nnmatch=3):
    """rdrobust's estimates and standard errors at bandwidths h and b."""
    out = {}
    for name, s in (("l", left), ("r", right)):
        kh = window_len(s.t, h)
        kb = window_len(s.t, b)
        k = max(kh, kb)
        res = nn_res(s, k, nnmatch)
        mq = moments(s.t, s.y, kb, b, 1, 2 * q, False)
        Gq_inv = inv_spd(gram(mq, q))
        mp = moments(s.t, s.y, kh, h, 1, 2 * p + p + 1, False)
        Gp_inv = inv_spd(gram(mp, p))
        L = np.array([mp[j + p + 1] for j in range(p + 1)]) / h ** (p + 1)
        P = p + 1
        sc = final_scores(s.t, s.y, res, k, h, b, p, q, Gq_inv[p + 1, :].copy(), L)
        RWy, Qy = sc[:P], sc[P:2 * P]
        Mcl = sc[2 * P:2 * P + P * P].reshape(P, P)
        Mrb = sc[2 * P + P * P:].reshape(P, P)
        out[name] = dict(beta_p=Gp_inv @ RWy, beta_bc=Gp_inv @ Qy,
                         V_cl=Gp_inv @ Mcl @ Gp_inv, V_rb=Gp_inv @ Mrb @ Gp_inv, n_h=kh)
    tau_cl = out["r"]["beta_p"][0] - out["l"]["beta_p"][0]
    tau_bc = out["r"]["beta_bc"][0] - out["l"]["beta_bc"][0]
    se_cl = math.sqrt(out["l"]["V_cl"][0, 0] + out["r"]["V_cl"][0, 0])
    se_rb = math.sqrt(out["l"]["V_rb"][0, 0] + out["r"]["V_rb"][0, 0])
    return {"coef": [tau_cl, tau_bc, tau_bc], "se": [se_cl, se_cl, se_rb],
            "h": [h, h], "b": [b, b], "n_h": [out["l"]["n_h"], out["r"]["n_h"]]}


def load_sorted(path, variant):
    df = pl.read_parquet(path, columns=[variant, "y"])
    df = df.sort(variant, maintain_order=True)
    return np.array(df[variant].to_numpy()), np.array(df["y"].to_numpy())


def warm():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-1, 1, 5000))
    rd_fit(x, x + (x >= 0) + rng.normal(size=5000))
    xd = np.round(x * 20) / 20
    rd_fit(xd, xd + (xd >= 0) + rng.normal(size=5000))


def hwm_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024**2
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--variant", default="x")
    ap.add_argument("--compare")
    a = ap.parse_args()
    warm()
    t0 = time.perf_counter()
    df = pl.read_parquet(a.data, columns=[a.variant, "y"])
    t1 = time.perf_counter()
    df = df.sort(a.variant, maintain_order=True)
    x, y = np.array(df[a.variant].to_numpy()), np.array(df["y"].to_numpy())   # writable: one numba type
    del df
    t2 = time.perf_counter()
    r = rd_fit(x, y)
    t3 = time.perf_counter()
    r.update(read_s=t1 - t0, sort_s=t2 - t1, fit_s=t3 - t1, total_s=t3 - t0,
             peak_gb=hwm_gb(), threads=get_num_threads())
    print("RESULT " + json.dumps(r, default=float))


if __name__ == "__main__":
    main()
