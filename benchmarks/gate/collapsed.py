"""Gate 3: rdrobust's default fit on a running variable with mass points,
computed from one row per distinct value.

A streaming group-by reduces the data to, for each distinct x: the count n,
the mean of y and the within-value sum of squares M2. With rdrobust's tie
handling every observation in a tie block has the same neighbor window, so
its NN residual is a * y_i - c_B, and the block's sum of squared residuals
is a^2 M2 + n (a ybar - c_B)^2: the whole fit runs on the M distinct values.

    python collapsed.py data.parquet [--variant xd]
"""

import argparse
import json
import math
import time

import numpy as np
import polars as pl
from numba import get_num_threads, njit, prange

from prototype import (C_TRIANGULAR, SQRT_EPS, final_scores, gram, hwm_gb, inv_spd,
                       moments, quantile_type2, window_len)


@njit(parallel=True, cache=True)
def block_nn_ss(d, cnt, sy, ybar, m2, k, matches):
    """Each block's sum of squared NN residuals on the window of blocks [0, k)."""
    nobs = 0
    for j in range(k):
        nobs += cnt[j]
    cap = matches if matches < nobs - 1 else nobs - 1
    out = np.empty(k)
    for j in prange(k):
        lo = 0
        lo += j                             # blocks [lo, hi); not an alias of j
        hi = lo + 1
        tot = cnt[j] - 1
        while tot < cap:
            if lo - 1 < 0:
                tot += cnt[hi]
                hi += 1
            elif hi >= k:
                lo -= 1
                tot += cnt[lo]
            else:
                dl = d[j] - d[lo - 1]
                dr = d[hi] - d[j]
                tol = (dl if dl > dr else dr) * 1.4901161193847656e-08
                if dl - dr > tol:
                    tot += cnt[hi]
                    hi += 1
                elif dr - dl > tol:
                    lo -= 1
                    tot += cnt[lo]
                else:
                    tot += cnt[hi]
                    hi += 1
                    lo -= 1
                    tot += cnt[lo]
        S = 0.0
        n = 0
        for i in range(lo, hi):
            S += sy[i]
            n += cnt[i]
        J = n - 1
        f = math.sqrt(J / (J + 1.0))
        a = f * (J + 1.0) / J
        cb = f * S / J
        e = a * ybar[j] - cb
        out[j] = a * a * m2[j] + cnt[j] * e * e
    return out


class Blocks:
    """One side's distinct values in order of distance from the cutoff."""

    def __init__(self, x, cnt, ybar, m2, c):
        self.x, self.cnt, self.ybar, self.m2 = x, cnt, ybar, m2
        self.t = x - c
        self.d = np.abs(self.t)
        self.cntf = cnt.astype(np.float64)
        self.sy = self.cntf * ybar
        self.M = x.shape[0]
        self.n = int(cnt.sum())
        self.cum = np.cumsum(cnt)

    def scaled(self, x_sd, y_sd, c):
        return Blocks(self.x / x_sd, self.cnt, self.ybar / y_sd, self.m2 / y_sd**2, c / x_sd)

    def nobs(self, k):
        return int(self.cum[k - 1]) if k > 0 else 0


class BandwidthBlocks:
    def __init__(self, s, c_bw, nnmatch):
        self.s, self.c_bw, self.nnmatch = s, c_bw, nnmatch
        self.kV = window_len(s.t, c_bw)
        self.ssV = block_nn_ss(s.d, s.cnt, s.sy, s.ybar, s.m2, self.kV, nnmatch)

    def vfit(self, o, nu):
        s, h, k = self.s, self.c_bw, self.kV
        m1 = moments(s.t, s.cntf, k, h, 1, 2 * o + 1, True)
        m2 = moments(s.t, self.ssV, k, h, 2, 2 * o, True)
        Gi = inv_spd(gram(m1, o))
        V_V = (Gi @ gram(m2, o) @ Gi)[nu, nu]
        v = np.array([m1[j + o + 1] for j in range(o + 1)]) / h ** (o + 1)
        Hp = np.array([h ** j for j in range(o + 1)])
        return V_V, (Hp * (Gi @ v))[nu]

    def bw(self, o, nu, o_B, h_B, scale):
        s = self.s
        V_V, BConst = self.vfit(o, nu)
        kB = window_len(s.t, h_B)
        m1 = moments(s.t, s.cntf, kB, h_B, 1, 2 * o_B, True)
        my = moments(s.t, s.sy, kB, h_B, 1, o_B, True)
        Gi = inv_spd(gram(m1, o_B))
        beta = Gi @ my
        BWreg = 0.0
        if scale > 0:
            ss = block_nn_ss(s.d, s.cnt, s.sy, s.ybar, s.m2, kB, self.nnmatch)
            m2 = moments(s.t, ss, kB, h_B, 2, 2 * o_B, True)
            BWreg = 3 * BConst ** 2 * (Gi @ gram(m2, o_B) @ Gi)[-1, -1]
        B = math.sqrt(2 * (o + 1 - nu)) * BConst * beta[-1]
        V = (2 * nu + 1) * self.c_bw ** (2 * nu + 1) * V_V
        return V, B, scale * (2 * (o + 1 - nu)) * BWreg, 1 / (2 * o + 3)


def pooled_sd(cnt, mean, m2):
    n = cnt.sum()
    mu = (cnt * mean).sum() / n
    return math.sqrt((m2.sum() + (cnt * (mean - mu) ** 2).sum()) / (n - 1))


def quantile_type2_blocks(xu, cum, prob):
    """Type-2 quantile of the expanded data from distinct values and cumulative counts."""
    n = int(cum[-1])
    hq = n * prob
    j = int(math.floor(hq))
    at = lambda i: xu[int(np.searchsorted(cum, i + 1))]      # 0-based order statistic
    return 0.5 * (at(j - 1) + at(j)) if hq == j else at(j)


def rd_fit_blocks(xu, cnt, ybar, m2, c=0.0, p=1, q=2, nnmatch=3, scaleregul=1.0):
    """xu ascending distinct values with their counts, means and M2."""
    split = int(np.searchsorted(xu, c, side="left"))
    rev = slice(None, None, -1)
    left = Blocks(*(np.ascontiguousarray(a[:split][rev]) for a in (xu, cnt, ybar, m2)), c)
    right = Blocks(xu[split:], cnt[split:], ybar[split:], m2[split:], c)

    cntf = cnt.astype(np.float64)
    x_sd = pooled_sd(cntf, xu, np.zeros_like(xu))
    y_sd = pooled_sd(cntf, ybar, m2)
    cum = np.cumsum(cnt)
    x_iq = quantile_type2_blocks(xu, cum, 0.75) - quantile_type2_blocks(xu, cum, 0.25)
    BWp = min(1.0, (x_iq / x_sd) / 1.349)
    x_min, x_max = float(xu[0]), float(xu[-1])
    range_l, range_r = abs(c - x_min) / x_sd, abs(c - x_max) / x_sd
    M_l, M_r = left.M, right.M
    bwcheck = 10 if (1 - M_l / left.n >= 0.2 or 1 - M_r / right.n >= 0.2) else None
    sl, sr = left.scaled(x_sd, y_sd, c), right.scaled(x_sd, y_sd, c)
    cs = c / x_sd
    c_bw = C_TRIANGULAR * BWp * (M_l + M_r) ** (-1 / 5)
    bw_max = max(abs(cs - x_min / x_sd), abs(cs - x_max / x_sd))
    c_bw = min(c_bw, bw_max)
    if bwcheck is not None:
        bw_min_l = sl.d[min(bwcheck, M_l) - 1] * (1 + SQRT_EPS)
        bw_min_r = sr.d[min(bwcheck, M_r) - 1] * (1 + SQRT_EPS)
        c_bw = max(c_bw, bw_min_l, bw_min_r)

    bl, br = BandwidthBlocks(sl, c_bw, nnmatch), BandwidthBlocks(sr, c_bw, nnmatch)
    C_d_l = bl.bw(q + 1, q + 1, q + 2, range_l * (1 + SQRT_EPS), 0)
    C_d_r = br.bw(q + 1, q + 1, q + 2, range_r * (1 + SQRT_EPS), 0)
    d_bw = min(((C_d_l[0] + C_d_r[0]) / (C_d_r[1] - C_d_l[1]) ** 2) ** C_d_l[3], bw_max)
    if bwcheck is not None:
        d_bw = max(d_bw, bw_min_l, bw_min_r)
    C_b_l = bl.bw(q, p + 1, q + 1, d_bw, scaleregul)
    C_b_r = br.bw(q, p + 1, q + 1, d_bw, scaleregul)
    b_bw = min(((C_b_l[0] + C_b_r[0]) / ((C_b_r[1] - C_b_l[1]) ** 2
                + scaleregul * (C_b_r[2] + C_b_l[2]))) ** C_b_l[3], bw_max)
    C_h_l = bl.bw(p, 0, q, b_bw, scaleregul)
    C_h_r = br.bw(p, 0, q, b_bw, scaleregul)
    h_bw = min(((C_h_l[0] + C_h_r[0]) / ((C_h_r[1] - C_h_l[1]) ** 2
                + scaleregul * (C_h_r[2] + C_h_l[2]))) ** C_h_l[3], bw_max)
    h, b = x_sd * h_bw, x_sd * b_bw

    out = {}
    P = p + 1
    for name, s in (("l", left), ("r", right)):
        kh, kb = window_len(s.t, h), window_len(s.t, b)
        k = max(kh, kb)
        ss = block_nn_ss(s.d, s.cnt, s.sy, s.ybar, s.m2, k, nnmatch)
        Gq_inv = inv_spd(gram(moments(s.t, s.cntf, kb, b, 1, 2 * q, True), q))
        mp = moments(s.t, s.cntf, kh, h, 1, 2 * p + 1 + p, True)
        Gp_inv = inv_spd(gram(mp, p))
        L = np.array([mp[j + p + 1] for j in range(P)]) / h ** (p + 1)
        sc = final_scores(s.t, s.sy, np.sqrt(ss), k, h, b, p, q,
                          Gq_inv[p + 1, :].copy(), L)
        out[name] = dict(beta_p=Gp_inv @ sc[:P], beta_bc=Gp_inv @ sc[P:2 * P],
                         V_cl=Gp_inv @ sc[2 * P:2 * P + P * P].reshape(P, P) @ Gp_inv,
                         V_rb=Gp_inv @ sc[2 * P + P * P:].reshape(P, P) @ Gp_inv,
                         n_h=s.nobs(kh))
    tau_cl = out["r"]["beta_p"][0] - out["l"]["beta_p"][0]
    tau_bc = out["r"]["beta_bc"][0] - out["l"]["beta_bc"][0]
    se_cl = math.sqrt(out["l"]["V_cl"][0, 0] + out["r"]["V_cl"][0, 0])
    se_rb = math.sqrt(out["l"]["V_rb"][0, 0] + out["r"]["V_rb"][0, 0])
    return {"coef": [tau_cl, tau_bc, tau_bc], "se": [se_cl, se_cl, se_rb],
            "h": [h, h], "b": [b, b], "n_h": [out["l"]["n_h"], out["r"]["n_h"]],
            "n_distinct": int(xu.shape[0])}


def collapse(path, variant):
    """One streaming pass: count, mean and M2 of y for each distinct x."""
    q = (pl.scan_parquet(path)
         .group_by(variant)
         .agg(n=pl.len(), ybar=pl.col("y").mean(),
              m2=((pl.col("y") - pl.col("y").mean()) ** 2).sum())
         .sort(variant))
    df = q.collect(engine="streaming")
    return (df[variant].to_numpy(), df["n"].to_numpy().astype(np.int64),
            df["ybar"].to_numpy(), df["m2"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--variant", default="xd")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    xw = np.round(rng.uniform(-1, 1, 5000) * 20) / 20                     # warm the kernels
    u, inv = np.unique(xw, return_inverse=True)
    cw = np.bincount(inv)
    yw = xw + (xw >= 0) + rng.normal(size=5000)
    rd_fit_blocks(u, cw, np.bincount(inv, yw) / cw, np.ones(u.shape[0]) * cw)
    t0 = time.perf_counter()
    xu, cnt, ybar, m2 = collapse(a.data, a.variant)
    t1 = time.perf_counter()
    r = rd_fit_blocks(xu, cnt, ybar, m2)
    t2 = time.perf_counter()
    r.update(collapse_s=t1 - t0, fit_s=t2 - t1, total_s=t2 - t0, peak_gb=hwm_gb(),
             threads=get_num_threads())
    print("RESULT " + json.dumps(r, default=float))


if __name__ == "__main__":
    main()
