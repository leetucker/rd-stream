"""rdrobust's estimator, over one side of the cutoff at a time.

`Side` holds one side's units in order of distance from the cutoff (see
_kernels.py). `select_bandwidths` is rdbwselect's three-step plug-in
(Calonico, Cattaneo and Titiunik 2014; Calonico, Cattaneo and Farrell 2020)
for all ten selectors, and `estimate` the local-polynomial fit with its
conventional and robust bias-corrected variances, both as R's rdrobust 4.1.1
computes them, for sharp designs without covariates.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from . import _cluster
from . import _kernels as K
from ._moments import kernel_poly, moments_mixed

SQRT_EPS = math.sqrt(np.finfo(float).eps)
SELECTORS = ("mserd", "msetwo", "msesum", "msecomb1", "msecomb2",
             "cerrd", "certwo", "cersum", "cercomb1", "cercomb2")
_C_C = {K.TRIANGULAR: 2.576, K.EPANECHNIKOV: 2.34, K.UNIFORM: 1.843}
_DUMMY_I = np.zeros(1, np.int64)
_DUMMY_F = np.zeros(1)
_RADIUS_MARGIN = 1 - 1e-10       # held rows cover d <= radius; windows stay a little inside


class NeedWider(Exception):
    """A step needs rows beyond those held (``memory="window"``): `radius`,
    in units of the running variable, is the distance from the cutoff it
    needs (None: more, by how much unknown)."""

    def __init__(self, radius=None):
        super().__init__(radius)
        self.radius = radius


class FullSums:
    """Power sums over one whole side, for fits whose window holds every
    unit of the side and that need no residuals (rdbwselect's range fit for
    C_d), when the rows held cover only a window near the cutoff.

    Sw[m] = sum f u^m and SD[m, j] = sum f u^m D_j with u = d / D, D the
    side's largest distance from the cutoff in the units of the side's t."""

    def __init__(self, Sw, SD, D, d_max, sgn):
        self.Sw, self.SD, self.D, self.d_max, self.sgn = Sw, SD, D, d_max, sgn

    def scaled(self, x_sd, y_sd):
        SD = self.SD.copy()
        SD[:, 0] /= y_sd
        return FullSums(self.Sw, SD, self.D / x_sd, self.d_max / x_sd, self.sgn)

    def columns(self, cols):
        return FullSums(self.Sw, np.ascontiguousarray(self.SD[:, cols]), self.D, self.d_max,
                        self.sgn)

    def moments_w(self, h, kern, A):
        from ._moments import _combine
        return _combine(self.Sw[:, None], kernel_poly(kern, h, self.D), self.D, A,
                        self.sgn)[:, 0]

    def moments_D(self, h, kern, A):
        from ._moments import _combine
        return _combine(self.SD, kernel_poly(kern, h, self.D), self.D, A, self.sgn)


class Side:
    """One side's units, nearest the cutoff first.

    Rows: x with the response matrix D (columns y, then the treatment take-up
    for a fuzzy design), optional sampling weights f and cluster codes cl.
    Collapsed units: x with counts cnt, column sums SD and within-value
    cross-products M2 (units x columns x columns). `sy` and `m2` are the
    current combined response D s (`combine`), which the variance passes use.
    """

    def __init__(self, x, c, *, D=None, f=None, cl=None, n_clusters=0,
                 cnt=None, SD=None, M2=None, x_sd=1.0, y_sd=1.0, dT=0):
        self.dT = dT                  # 1 with a take-up column (fuzzy), after y
        # the window path (memory="window"): the units held are those within
        # radius_raw of the cutoff; n_side and M_side count the whole side
        self.x_scale = x_sd
        self.radius_raw = math.inf
        self.radius = math.inf
        self.full = None
        self.prefer_full = False      # a slab-backed side takes C_d's sums from `full`
        self.x = x
        self.c = c
        self.t = x / x_sd - c / x_sd if x_sd != 1.0 else x - c
        self.d = np.abs(self.t)
        self.rows = cnt is None
        if self.rows:
            D = D.reshape(x.shape[0], -1)
            if y_sd != 1.0:
                D = D.copy()
                D[:, 0] /= y_sd
            self.D = D
            self.M2 = None
            self.cnt, self.cntf = _DUMMY_I, _DUMMY_F
            self.nobs = x.shape[0]
        else:
            SD = SD.reshape(x.shape[0], -1)
            if y_sd != 1.0:
                SD = SD.copy()
                SD[:, 0] /= y_sd
                M2 = M2.copy()
                M2[:, 0, :] /= y_sd
                M2[:, :, 0] /= y_sd
            self.D, self.M2 = SD, M2
            self.cnt = cnt
            self.cntf = cnt.astype(np.float64)
            self.cum = np.cumsum(cnt)
            self.nobs = int(self.cum[-1]) if cnt.shape[0] else 0
        self.ncol = self.D.shape[1]
        self.dZ = self.ncol - 1 - dT  # covariates, after y and the take-up
        self.sy = np.ascontiguousarray(self.D[:, 0])
        self.m2 = _DUMMY_F if self.rows else np.ascontiguousarray(self.M2[:, 0, 0])
        self.f = f
        self.cl = cl
        self.n_clusters = n_clusters
        self._clo = [None]       # the rows ordered by cluster, shared with copies of the side
        self.run_start = K.runs(self.d)
        self.M = self.run_start.shape[0] - 1          # distinct values of x held
        self.n_units = x.shape[0]
        self.n_side, self.M_side = self.nobs, self.M  # the whole side's

    def held_window(self, radius_raw, full, n_side, M_side):
        """Mark the side as holding only the units within `radius_raw` of the
        cutoff (raw units), with the whole side's power sums and counts."""
        self.radius_raw = radius_raw
        self.radius = radius_raw / self.x_scale
        self.full = full
        self.n_side, self.M_side = int(n_side), int(M_side)
        return self

    def scaled(self, x_sd, y_sd):
        """The same side with x and y divided by their standard deviations
        (rdrobust's stdvars, for the bandwidth step; the take-up is not
        scaled)."""
        if self.rows:
            out = Side(self.x, self.c, D=self.D, f=self.f, cl=self.cl,
                       n_clusters=self.n_clusters, x_sd=x_sd, y_sd=y_sd, dT=self.dT)
        else:
            out = Side(self.x, self.c, cnt=self.cnt, SD=self.D, M2=self.M2, x_sd=x_sd,
                       y_sd=y_sd, dT=self.dT)
        out._clo = self._clo
        if math.isfinite(self.radius_raw):
            out.held_window(self.radius_raw,
                            None if self.full is None else self.full.scaled(x_sd, y_sd),
                            self.n_side, self.M_side)
        return out

    def columns(self, cols, dT=0):
        """The side with only the response columns `cols` (dT: whether the
        second of them is a take-up)."""
        out = copy.copy(self)
        out.dT = dT
        out.dZ = len(cols) - 1 - dT
        out.D = np.ascontiguousarray(self.D[:, cols])
        if not self.rows:
            out.M2 = np.ascontiguousarray(self.M2[:, cols][:, :, cols])
        out.ncol = len(cols)
        out.sy = np.ascontiguousarray(out.D[:, 0])
        out.m2 = _DUMMY_F if self.rows else np.ascontiguousarray(out.M2[:, 0, 0])
        if self.full is not None:
            out.full = self.full.columns(cols)
        return out

    def combine(self, s):
        """The side whose response is D s (the variance passes' variable)."""
        s = np.asarray(s, dtype=np.float64)
        if self.ncol == 1 and s.shape[0] == 1 and s[0] == 1.0:
            return self
        out = copy.copy(self)
        out.sy = np.ascontiguousarray(self.D @ s)
        if not self.rows:
            out.m2 = np.ascontiguousarray(np.einsum("nij,i,j->n", self.M2, s, s))
        return out

    # ---- sums over windows

    def nobs_in(self, k):
        if self.rows:
            return k
        return int(self.cum[k - 1]) if k > 0 else 0

    def distinct_in(self, k):
        """Distinct values of x among the first k units."""
        return int(np.searchsorted(self.run_start, k, side="left"))

    def window(self, h, kern):
        if h > self.radius * _RADIUS_MARGIN:
            raise NeedWider(h * self.x_scale)
        return K.window_len(self.t, h, kern)

    def covers_side(self, h):
        """Whether a window of half-width h holds every unit of the side,
        held or not (then `full` has its sums)."""
        return self.full is not None and h > self.full.d_max

    def _f(self):
        return (self.f, True) if self.f is not None else (_DUMMY_F, False)

    def moments_w(self, k, h, kern, A):
        """sum w t^a (w = f K/h; times the unit's count)."""
        f, use_f = self._f()
        return K.moments(self.t, self.cntf, not self.rows, f, use_f, k, h, kern, 1, A)

    def moments_D(self, k, h, kern, A):
        """sum w t^a D_j, (A + 1) x columns."""
        f, use_f = self._f()
        return np.column_stack([
            K.moments(self.t, np.ascontiguousarray(self.D[:, j]), True, f, use_f, k, h, kern, 1, A)
            for j in range(self.ncol)])

    def zwd(self, k, h, kern):
        """sum w z d' over the window: the covariates' weighted cross-products
        with every response column (dZ x columns)."""
        zc = slice(1 + self.dT, self.ncol)
        f, use_f = self._f()
        w = K.weights(self.t, f, use_f, k, h, kern)
        if self.rows:
            return (self.D[:k, zc] * w[:, None]).T @ self.D[:k]
        DD = self.M2[:k] + self.D[:k, :, None] * self.D[:k, None, :] / self.cntf[:k, None, None]
        return np.einsum("n,nij->ij", w, DD[:, zc, :])

    def moments_ss(self, ss, k, h, kern, A):
        f, use_f = self._f()
        return K.moments(self.t, ss, True, f, use_f, k, h, kern, 2, A)

    # ---- residuals and variance meat (of the combined response)

    def residuals(self, k, vce, matches, h=None, kern=None, beta=None, Ginv=None, d=0,
                  n_window=None):
        """(sum of squared residuals per unit, signed row residuals or None),
        `beta` the combined response's coefficients. `n_window`: the window's
        observations, when the units are part of it (a slab)."""
        signed = self.cl is not None
        if vce == "nn":
            nrun = int(np.searchsorted(self.run_start, k, side="left"))
            return K.nn_residuals(self.d, self.cnt, self.sy, self.m2, self.sy, self.rows,
                                  self.run_start, nrun, matches, signed)
        mode = {"hc0": 0, "hc1": 1, "hc2": 2, "hc3": 3, "cr2": 0, "cr3": 0}[vce]
        n = self.nobs_in(k) if n_window is None else n_window
        factor = 1.0
        if mode == 1 and self.cl is None:
            factor = math.sqrt(n / (n - d))
        f, use_f = self._f()
        return K.hc_residuals(self.t, self.cntf, self.sy, self.m2, self.rows, f, use_f, k, h,
                              kern, np.ascontiguousarray(beta, dtype=np.float64),
                              np.ascontiguousarray(Ginv), mode, factor, signed)

    def meat(self, k, h, kern, P, ss, res, Ginv=None, vce="hc1"):
        """sum (w r)(w r)' res^2 over the window, or its cluster-robust
        version (CRV1, CRV2 or CRV3), as rdrobust_vce computes it."""
        if self.cl is None:
            return gram(self.moments_ss(ss, k, h, kern, 2 * (P - 1)), P - 1)
        f, use_f = self._f()
        if vce in ("cr2", "cr3"):
            L, u, seen = self.cluster_sums(k, h, kern, P, res)
            return _cluster.meat(L, u, seen, Ginv, 2 if vce == "cr2" else 3)
        scores = K.poly_scores(self.t, f, use_f, k, h, kern, P, res)
        return self.cluster_meat(scores, k, P)

    def cluster_meat(self, scores, k, k_df):
        M, G = K.cluster_meat_ordered(scores, k, *self._cluster_order())
        n = k
        return M * ((n - 1) / (n - k_df)) * (G / (G - 1))

    # ---- per-cluster sums over the window [0, k), a cluster per thread

    def _cluster_order(self):
        if self._clo[0] is None:
            self._clo[0] = K.cluster_order(self.cl, self.n_clusters)
        return self._clo[0]

    def cluster_sums(self, k, h, kern, P, res):
        f, use_f = self._f()
        return K.cluster_sums_ordered(self.t, f, use_f, k, h, kern, P, res,
                                      *self._cluster_order())

    def cluster_sums_qq(self, k, b, kern, Pq, Q, res):
        f, use_f = self._f()
        return K.cluster_sums_qq_ordered(self.t, f, use_f, k, b, kern, Pq, Q, res,
                                         *self._cluster_order())

    def cluster_score_sums(self, scores, k):
        return K.cluster_score_sums_ordered(scores, k, *self._cluster_order())

    # ---- the window sums the steps take (a slab-backed side has the same)

    @property
    def clustered(self):
        return self.cl is not None

    def run_distance(self, j):
        """The distance from the cutoff of the j-th distinct value (1-based)."""
        return self.d[self.run_start[j - 1]]

    def residual_meat(self, svec, k, h, kern, P, vce, matches, beta, Ginv, d):
        """The variance meat of a pilot fit over the window of k units for the
        combined response D svec, its coefficients `beta`."""
        cs = self.combine(svec)
        ss, res = cs.residuals(k, vce, matches, h, kern, beta, Ginv, d)
        return cs.meat(k, h, kern, P, ss, res, Ginv, vce)

    def variance_sums(self, svec, fit, *, p, q, kern, vce, matches):
        """The final step's squared-residual moments without clusters: at h
        (from the residuals at h, and from those at b), mixed, and at b."""
        k, kh, kb, h, b = fit.k, fit.kh, fit.kb, fit.h, fit.b
        cs = self.combine(svec)
        if vce == "nn":
            ss_h, _ = cs.residuals(k, "nn", matches)
            ss_b = ss_h
        else:
            ss_h, _ = cs.residuals(k, vce, 0, h, kern, fit.beta_p @ svec, fit.Gp_inv, p + 1)
            ss_b, _ = cs.residuals(k, vce, 0, b, kern, fit.beta_q @ svec, fit.Gq_inv, q + 1)
        f, use_f = self._f()
        mh = self.moments_ss(ss_h, kh, h, kern, 2 * p)
        mh_b = mh if ss_b is ss_h else self.moments_ss(ss_b, kh, h, kern, 2 * p)
        mx = moments_mixed(self.t, ss_b, f, use_f, min(kh, kb), h, b, kern, p + q)
        mb = self.moments_ss(ss_b, kb, b, kern, 2 * q)
        return mh, mh_b, mx, mb

    def cluster_meats(self, svec, fit, *, p, q, kern, vce, matches, hb_match):
        """The final step's cluster-robust meats: (conventional, robust); the
        robust one is in rdrobust's q-order basis when h = b."""
        k, h, b = fit.k, fit.h, fit.b
        P, Pq = p + 1, q + 1
        cs = self.combine(svec)
        if vce == "nn":
            ss_h, res_h = cs.residuals(k, "nn", matches)
            ss_b, res_b = ss_h, res_h
        else:
            ss_h, res_h = cs.residuals(k, vce, 0, h, kern, fit.beta_p @ svec, fit.Gp_inv, P)
            ss_b, res_b = cs.residuals(k, vce, 0, b, kern, fit.beta_q @ svec, fit.Gq_inv, Pq)
        f, use_f = self._f()
        if vce in ("cr2", "cr3"):
            mode = 2 if vce == "cr2" else 3
            L, u, seen = self.cluster_sums(k, h, kern, P, res_h)
            meat_cl = _cluster.meat(L, u, seen, fit.Gp_inv, mode)
            if hb_match:
                L, u, seen = self.cluster_sums(k, h, kern, Pq, res_b)
                return meat_cl, _cluster.meat(L, u, seen, fit.Gq_inv, mode)
            _, Q = K.final_rows(self.t, f, use_f, k, h, b, kern, p, q, fit.g, fit.L)
            L, Pg, uQ, uq, seen = self.cluster_sums_qq(k, b, kern, Pq,
                                                       np.ascontiguousarray(Q), res_b)
            return meat_cl, _cluster.meat_qq(L, Pg, uQ, uq, seen, fit.Gq_inv, mode)
        RW, Q = K.final_rows(self.t, f, use_f, k, h, b, kern, p, q, fit.g, fit.L)
        meat_cl = self.cluster_meat(RW * res_h[:, None], k, P)
        if hb_match:
            Rq_h = K.poly_scores(self.t, f, use_f, k, h, kern, Pq, res_b)
            return meat_cl, self.cluster_meat(Rq_h, k, Pq)
        return meat_cl, self.cluster_meat(Q * res_b[:, None], k, Pq)


def gram(mom, o):
    idx = np.add.outer(np.arange(o + 1), np.arange(o + 1))
    return mom[idx]


def inv_chol(G):
    L = np.linalg.cholesky(G)
    Li = np.linalg.inv(L)
    return Li.T @ Li


def inv_chol_or_pinv(G):
    try:
        return inv_chol(G)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(G)


def partial_z(U, Gi, ZWD, dT):
    """ZWZ and ZWY of the covariate adjustment: the covariates' cross-products
    after partialling out the local polynomial."""
    colsZ = np.arange(1 + dT, U.shape[1])
    UiGU = U[:, colsZ].T @ Gi @ U
    return ZWD[:, colsZ] - UiGU[:, colsZ], ZWD[:, :1 + dT] - UiGU[:, :1 + dT]


def combination(beta_row, gamma, dT, scale_y=1.0, fac=1.0):
    """rdrobust's s: the weights on the response columns [y, (take-up),
    covariates] whose variance is the estimate's, with the jumps tau_Y,
    tau_T they imply. `beta_row` is a row of coefficients (or of jumps) over
    the columns; `gamma` the covariate coefficients (dZ x (1 + dT))."""
    dZ = gamma.shape[0]
    cz = slice(1 + dT, 1 + dT + dZ)
    tau_Y = scale_y * fac * (beta_row[0] - gamma[:, 0] @ beta_row[cz])
    if dT == 0:
        return np.concatenate([[1.0], -gamma[:, 0]]), tau_Y, None
    tau_T = fac * (beta_row[1] - gamma[:, 1] @ beta_row[cz])
    s = np.concatenate([[1 / tau_T, -(tau_Y / tau_T ** 2)],
                        -(1 / tau_T) * gamma[:, 0] + (tau_Y / tau_T ** 2) * gamma[:, 1]])
    return s, tau_Y, tau_T


# ---------------------------------------------------------------------------
# rdbwselect

class _BandwidthSide:
    """rdrobust_bw on one (standardized) side, with the pilot fits at c_bw
    computed once per (order, derivative). With a take-up column (fuzzy),
    the variances are of the delta-method combination D s."""

    def __init__(self, s, c_bw, kern, vce, matches):
        self.s, self.c_bw, self.kern, self.vce, self.matches = s, c_bw, kern, vce, matches
        self.kV = s.window(c_bw, kern)
        self._vcache = {}

    def _pilot(self, o, nu):
        key = (o, nu)
        if key in self._vcache:
            return self._vcache[key]
        s, h, k, kern = self.s, self.c_bw, self.kV, self.kern
        if s.distinct_in(k) < o + 1:
            out = (math.nan, math.nan, np.ones(1))
        else:
            mw = s.moments_w(k, h, kern, 2 * o + 1)
            Gi = inv_chol_or_pinv(gram(mw, o))
            U = s.moments_D(k, h, kern, o)                      # (o + 1) x columns
            beta = Gi @ U
            gamma = np.zeros((0, 1 + s.dT))
            if s.dZ:
                ZWZ, ZWY = partial_z(U, Gi, s.zwd(k, h, kern), s.dT)
                gamma = np.linalg.pinv(ZWZ) @ ZWY
            svec = combination(beta[nu], gamma, s.dT, fac=math.factorial(nu))[0]
            if s.dT == 0 and s.dZ == 0:
                svec = np.ones(1)
            meat = s.residual_meat(svec, k, h, kern, o + 1, self.vce, self.matches,
                                   beta @ svec, Gi, o + 1)
            V_V = (Gi @ meat @ Gi)[nu, nu]
            v = mw[o + 1:2 * o + 2] / h ** (o + 1)
            Hp = h ** np.arange(o + 1)
            out = (V_V, (Hp * (Gi @ v))[nu], svec)
        self._vcache[key] = out
        return out

    def bw(self, o, nu, o_B, h_B, scale):
        nan_out = (math.nan, math.nan, math.nan, 1 / (2 * o + 3))
        V_V, BConst, svec = self._pilot(o, nu)
        if not math.isfinite(V_V) or not (math.isfinite(h_B) and h_B > 0):
            return nan_out
        s, kern = self.s, self.kern
        if scale == 0 and s.covers_side(h_B) and (s.prefer_full or
                                                  h_B > s.radius * _RADIUS_MARGIN):
            # the whole side, beyond the rows held: from its power sums
            if s.M_side < o_B + 1:
                return nan_out
            Gi = inv_chol_or_pinv(gram(s.full.moments_w(h_B, kern, 2 * o_B), o_B))
            beta = Gi @ s.full.moments_D(h_B, kern, o_B)
        else:
            kB = s.window(h_B, kern)
            if s.distinct_in(kB) < o_B + 1:
                return nan_out
            Gi = inv_chol_or_pinv(gram(s.moments_w(kB, h_B, kern, 2 * o_B), o_B))
            beta = Gi @ s.moments_D(kB, h_B, kern, o_B)
        BWreg = 0.0
        if scale > 0:
            meat = s.residual_meat(svec, kB, h_B, kern, o_B + 1, self.vce, self.matches,
                                   beta @ svec, Gi, o_B + 1)
            V_B = (Gi @ meat @ Gi)[o + 1, o + 1]
            BWreg = 3 * BConst ** 2 * V_B
        B = math.sqrt(2 * (o + 1 - nu)) * BConst * float(beta[o + 1] @ svec)   # R: beta_B[o + 2, ]
        V = (2 * nu + 1) * self.c_bw ** (2 * nu + 1) * V_V
        R = scale * (2 * (o + 1 - nu)) * BWreg
        return V, B, R, 1 / (2 * o + 3)


@dataclass
class Bandwidths:
    """Main (h) and bias (b) bandwidths on each side, by selector."""
    table: dict            # selector -> (h_left, h_right, b_left, b_right)
    selector: str

    @property
    def chosen(self):
        return self.table[self.selector]


def quantile_type2_sorted(xs, prob):
    n = xs.shape[0]
    hq = n * prob
    j = int(math.floor(hq))
    if prob <= 0:
        return xs[0]
    if prob >= 1:
        return xs[-1]
    return 0.5 * (xs[j - 1] + xs[j]) if hq == j else xs[j]


def select_bandwidths(left, right, stats, *, p, q, deriv, kern, vce, matches, scaleregul,
                      masspoints, bwcheck, bwrestrict, stdvars, selector="mserd", all_=False):
    """rdbwselect. `stats` has the full sample's x_sd, y_sd, x_iq, x_min,
    x_max and N (and the cluster counts); returns `Bandwidths`."""
    c = left.c
    x_sd = stats["x_sd"] if stdvars else 1.0
    y_sd = stats["y_sd"] if stdvars else 1.0
    BWp = min(stats["x_sd"], stats["x_iq"] / 1.349)
    if stdvars:
        BWp = min(1.0, (stats["x_iq"] / x_sd) / 1.349)
    sl = left.scaled(x_sd, y_sd) if stdvars else left
    sr = right.scaled(x_sd, y_sd) if stdvars else right
    cs = c / x_sd
    x_min, x_max = stats["x_min"] / x_sd, stats["x_max"] / x_sd
    range_l, range_r = abs(cs - x_min), abs(cs - x_max)
    N = stats["N"]
    M_l, M_r = left.M_side, right.M_side
    if masspoints == "adjust" and bwcheck is None and stats["mass"]:
        bwcheck = 10
    M = M_l + M_r if masspoints in ("check", "adjust") else N
    C_c = _C_C[kern]
    c_bw = C_c * BWp * (M if masspoints == "adjust" else N) ** (-1 / 5)
    bw_max_l, bw_max_r = abs(cs - x_min), abs(cs - x_max)
    bw_max = max(bw_max_l, bw_max_r)
    if bwrestrict:
        c_bw = min(c_bw, bw_max)
    bw_min_l = bw_min_r = None
    if bwcheck is not None:
        if min(bwcheck, M_l) > sl.M or min(bwcheck, M_r) > sr.M:
            raise NeedWider()                 # fewer distinct values held than bwcheck
        bw_min_l = sl.run_distance(min(bwcheck, M_l)) * (1 + SQRT_EPS)
        bw_min_r = sr.run_distance(min(bwcheck, M_r)) * (1 + SQRT_EPS)
        c_bw = max(c_bw, bw_min_l, bw_min_r)

    bl = _BandwidthSide(sl, c_bw, kern, vce, matches)
    br = _BandwidthSide(sr, c_bw, kern, vce, matches)
    C_d_l = bl.bw(q + 1, q + 1, q + 2, range_l * (1 + SQRT_EPS), 0)
    C_d_r = br.bw(q + 1, q + 1, q + 2, range_r * (1 + SQRT_EPS), 0)

    def restrict(v, hi):
        return min(v, hi) if bwrestrict else v

    def floor(v, *lo):
        return max(v, *lo) if bwcheck is not None else v

    out = {}
    need = set(SELECTORS) if all_ else {selector}
    two = need & {"msetwo", "certwo", "msecomb2", "cercomb2"}
    summ = need & {"msesum", "cersum", "msecomb1", "msecomb2", "cercomb1", "cercomb2"}
    rd = need & {"mserd", "cerrd", "msecomb1", "msecomb2", "cercomb1", "cercomb2"}
    if two:
        d_l = floor(restrict((C_d_l[0] / C_d_l[1] ** 2) ** C_d_l[3], bw_max_l), bw_min_l or 0)
        d_r = floor(restrict((C_d_r[0] / C_d_r[1] ** 2) ** C_d_r[3], bw_max_r), bw_min_r or 0)
        C_b_l = bl.bw(q, p + 1, q + 1, d_l, scaleregul)
        C_b_r = br.bw(q, p + 1, q + 1, d_r, scaleregul)
        b_l = restrict((C_b_l[0] / (C_b_l[1] ** 2 + scaleregul * C_b_l[2])) ** C_b_l[3], bw_max_l)
        b_r = restrict((C_b_r[0] / (C_b_r[1] ** 2 + scaleregul * C_b_r[2])) ** C_b_r[3], bw_max_r)
        C_h_l = bl.bw(p, deriv, q, b_l, scaleregul)
        C_h_r = br.bw(p, deriv, q, b_r, scaleregul)
        h_l = restrict((C_h_l[0] / (C_h_l[1] ** 2 + scaleregul * C_h_l[2])) ** C_h_l[3], bw_max_l)
        h_r = restrict((C_h_r[0] / (C_h_r[1] ** 2 + scaleregul * C_h_r[2])) ** C_h_r[3], bw_max_r)
        out["msetwo"] = (x_sd * h_l, x_sd * h_r, x_sd * b_l, x_sd * b_r)
    if summ:
        d_s = ((C_d_l[0] + C_d_r[0]) / (C_d_r[1] + C_d_l[1]) ** 2) ** C_d_l[3]
        d_s = restrict(d_s, bw_max)
        if bwcheck is not None:
            d_s = max(d_s, bw_min_l, bw_min_r)
        C_b_l = bl.bw(q, p + 1, q + 1, d_s, scaleregul)
        C_b_r = br.bw(q, p + 1, q + 1, d_s, scaleregul)
        b_s = ((C_b_l[0] + C_b_r[0]) / ((C_b_r[1] + C_b_l[1]) ** 2
                                        + scaleregul * (C_b_r[2] + C_b_l[2]))) ** C_b_l[3]
        b_s = restrict(b_s, bw_max)
        C_h_l = bl.bw(p, deriv, q, b_s, scaleregul)
        C_h_r = br.bw(p, deriv, q, b_s, scaleregul)
        h_s = ((C_h_l[0] + C_h_r[0]) / ((C_h_r[1] + C_h_l[1]) ** 2
                                        + scaleregul * (C_h_r[2] + C_h_l[2]))) ** C_h_l[3]
        h_s = restrict(h_s, bw_max)
        out["msesum"] = (x_sd * h_s,) * 2 + (x_sd * b_s,) * 2
    if rd:
        d_d = ((C_d_l[0] + C_d_r[0]) / (C_d_r[1] - C_d_l[1]) ** 2) ** C_d_l[3]
        d_d = restrict(d_d, bw_max)
        if bwcheck is not None:
            d_d = max(d_d, bw_min_l, bw_min_r)
        C_b_l = bl.bw(q, p + 1, q + 1, d_d, scaleregul)
        C_b_r = br.bw(q, p + 1, q + 1, d_d, scaleregul)
        b_d = ((C_b_l[0] + C_b_r[0]) / ((C_b_r[1] - C_b_l[1]) ** 2
                                        + scaleregul * (C_b_r[2] + C_b_l[2]))) ** C_b_l[3]
        b_d = restrict(b_d, bw_max)
        C_h_l = bl.bw(p, deriv, q, b_d, scaleregul)
        C_h_r = br.bw(p, deriv, q, b_d, scaleregul)
        h_d = ((C_h_l[0] + C_h_r[0]) / ((C_h_r[1] - C_h_l[1]) ** 2
                                        + scaleregul * (C_h_r[2] + C_h_l[2]))) ** C_h_l[3]
        h_d = restrict(h_d, bw_max)
        out["mserd"] = (x_sd * h_d,) * 2 + (x_sd * b_d,) * 2
    if need & {"msecomb1", "cercomb1"}:
        h1 = min(out["mserd"][0], out["msesum"][0])
        b1 = min(out["mserd"][2], out["msesum"][2])
        out["msecomb1"] = (h1, h1, b1, b1)
    if need & {"msecomb2", "cercomb2"}:
        med = lambda *v: float(np.median(np.array(v)))
        out["msecomb2"] = (med(out["mserd"][0], out["msesum"][0], out["msetwo"][0]),
                           med(out["mserd"][1], out["msesum"][1], out["msetwo"][1]),
                           med(out["mserd"][2], out["msesum"][2], out["msetwo"][2]),
                           med(out["mserd"][3], out["msesum"][3], out["msetwo"][3]))
    cer_n = stats["G_l"] + stats["G_r"] if stats.get("clustered") else N
    cer_h = cer_n ** (-(p / ((3 + p) * (3 + 2 * p))))
    for cer, mse in (("cerrd", "mserd"), ("certwo", "msetwo"), ("cersum", "msesum"),
                     ("cercomb1", "msecomb1"), ("cercomb2", "msecomb2")):
        if cer in need:
            v = out[mse]
            out[cer] = (v[0] * cer_h, v[1] * cer_h, v[2], v[3])
    table = {k: out[k] for k in SELECTORS if k in out and k in need}
    for v in table.values():
        if not all(math.isfinite(b) and b > 0 for b in v):
            raise ValueError("a pilot bandwidth fit is not identified (too few distinct values "
                             "of the running variable near the cutoff); give the bandwidth "
                             "with bandwidth=")
    return Bandwidths(table, selector)


# ---------------------------------------------------------------------------
# rdrobust's estimate at given bandwidths

@dataclass
class SideFit:
    """One side's fit at bandwidths h and b: coefficients for every response
    column, and what the variances need."""
    beta_p: np.ndarray            # (p + 1) x columns
    beta_q: np.ndarray            # (q + 1) x columns
    beta_bc: np.ndarray           # (p + 1) x columns
    n_h: int
    n_b: int
    k: int                        # units within max(h, b)
    kh: int
    kb: int
    h: float
    b: float
    L: np.ndarray                 # sum w_h t^(p+1+j) / h^(p+1), j = 0..p
    g: np.ndarray                 # row p + 1 of Gq_inv
    Gp_inv: np.ndarray
    Gq_inv: np.ndarray
    U_p: np.ndarray               # sum w_h r_p d' over the columns
    n_h_units: int


def fit_side(s, h, b, *, p, q, kern, grid=None):
    """One side's local-polynomial coefficients at bandwidths h, b, from the
    window moments (rdrobust's Q_q enters through Q'D = U_p - h^(p+1) L g'M_b).
    `grid`: a GridSide whose tables cover the window, or None for direct sums."""
    kh, kb = s.window(h, kern), s.window(b, kern)
    k = max(kh, kb)
    src = grid if grid is not None and k <= grid.extent else s
    Gq_inv = inv_chol(gram(src.moments_w(kb, b, kern, 2 * q), q))
    mp = src.moments_w(kh, h, kern, 3 * p + 1)
    Gp_inv = inv_chol(gram(mp, p))
    L = mp[p + 1:2 * p + 2] / h ** (p + 1)
    g = np.ascontiguousarray(Gq_inv[p + 1, :])
    U_p = src.moments_D(kh, h, kern, p)
    M_b = src.moments_D(kb, b, kern, q)
    beta_p = Gp_inv @ U_p
    beta_q = Gq_inv @ M_b
    beta_bc = Gp_inv @ (U_p - h ** (p + 1) * np.outer(L, g @ M_b))
    return SideFit(beta_p, beta_q, beta_bc, s.nobs_in(kh), s.nobs_in(kb), k, kh, kb, h, b,
                   L, g, Gp_inv, Gq_inv, U_p, kh)


def _sq(coef):
    return np.convolve(coef, coef)


def side_variances(s, fit, svec, *, p, q, kern, vce, matches, hb_match, grid=None):
    """Conventional and robust variance matrices of one side's coefficients
    for the combined response D s.

    Without clusters, both are sums over the window:
    V_cl = Gp_inv (sum w_h^2 r r' e^2) Gp_inv and, with rdrobust's
    Q_i = w_h r_i - h^(p+1) w_b (g' r_q,i) L,
    Q' diag(e^2) Q = sum w_h^2 r r' e^2 - h^(p+1) (v L' + L v') + h^(2p+2) c L L',
    v = sum w_h w_b (g' r_q) r e^2 and c = sum w_b^2 (g' r_q)^2 e^2."""
    k, kh, kb, h, b = fit.k, fit.kh, fit.kb, fit.h, fit.b
    P, Pq = p + 1, q + 1
    Gp_inv, Gq_inv = fit.Gp_inv, fit.Gq_inv
    if not s.clustered:
        hp1 = h ** (p + 1)
        edge = None
        if grid is not None and vce == "nn" and k <= grid.extent:
            edge = grid.residuals_at(k)
        if edge is not None:
            kp_h, kp_b = kernel_poly(kern, h, grid.D), kernel_poly(kern, b, grid.D)
            mh = grid.ss_moments(edge, svec, kh, _sq(kp_h), 2 * p)
            mx = grid.ss_moments(edge, svec, min(kh, kb), np.convolve(kp_h, kp_b), p + q)
            mb = grid.ss_moments(edge, svec, kb, _sq(kp_b), 2 * q)
            mh_b = mh
        else:
            mh, mh_b, mx, mb = s.variance_sums(svec, fit, p=p, q=q, kern=kern, vce=vce,
                                               matches=matches)
        V_cl = Gp_inv @ gram(mh, p) @ Gp_inv
        g = fit.g
        v = np.array([g @ mx[a:a + Pq] for a in range(P)])
        c = g @ gram(mb, q) @ g
        meat_rb = (gram(mh_b, p) - hp1 * (np.outer(v, fit.L) + np.outer(fit.L, v))
                   + hp1 * hp1 * c * np.outer(fit.L, fit.L))
        V_rb = Gp_inv @ meat_rb @ Gp_inv
        return V_cl, V_rb
    meat_cl, meat_rb = s.cluster_meats(svec, fit, p=p, q=q, kern=kern, vce=vce,
                                       matches=matches, hb_match=hb_match)
    V_cl = Gp_inv @ meat_cl @ Gp_inv
    if hb_match:
        V_rb = (Gq_inv @ meat_rb @ Gq_inv)[:P, :P]
    else:
        V_rb = Gp_inv @ meat_rb @ Gp_inv
    return V_cl, V_rb


def estimate(left, right, h_l, h_r, b_l, b_r, *, p, q, deriv, kern, vce, matches,
             scalepar=1.0, grids=(None, None)):
    """Conventional and bias-corrected estimates with their conventional and
    robust standard errors. With a take-up column (fuzzy), the local Wald
    ratio and the first stage; with covariates, the adjustment of Calonico,
    Cattaneo, Farrell and Titiunik (2019), gamma from both sides at h."""
    hb_match = h_l == b_l and h_r == b_r
    gl, gr = grids
    L = fit_side(left, h_l, b_l, p=p, q=q, kern=kern, grid=gl)
    R = fit_side(right, h_r, b_r, p=p, q=q, kern=kern, grid=gr)
    dT, dZ = left.dT, left.dZ
    fac = math.factorial(deriv)
    gamma = np.zeros((0, 1 + dT))
    if dZ:
        Z_l = partial_z(L.U_p, L.Gp_inv, left.zwd(L.n_h_units, h_l, kern), dT)
        Z_r = partial_z(R.U_p, R.Gp_inv, right.zwd(R.n_h_units, h_r, kern), dT)
        gamma = np.linalg.pinv(Z_l[0] + Z_r[0]) @ (Z_l[1] + Z_r[1])
    jump_p = R.beta_p[deriv] - L.beta_p[deriv]
    jump_bc = R.beta_bc[deriv] - L.beta_bc[deriv]
    s_cl, tau_Y_cl, tau_T_cl = combination(jump_p, gamma, dT, scalepar, fac)
    _, tau_Y_bc, tau_T_bc = combination(jump_bc, gamma, dT, scalepar, fac)
    out = dict(side_l=L, side_r=R, fuzzy=dT == 1, gamma=gamma)
    kw = dict(p=p, q=q, kern=kern, vce=vce, matches=matches, hb_match=hb_match)
    if dT == 0:
        svec = s_cl if dZ else np.ones(1)
        tau_cl, tau_bc = tau_Y_cl, tau_Y_bc
    else:
        tau_cl = tau_Y_cl / tau_T_cl
        s2 = np.array([1 / tau_T_cl, -(tau_Y_cl / tau_T_cl ** 2)])
        tau_bc = tau_cl - float(s2 @ np.array([tau_Y_cl - tau_Y_bc, tau_T_cl - tau_T_bc]))
        svec = s_cl
        sV_T = np.concatenate([[0.0, 1.0], -gamma[:, 1]])
        VT_l = side_variances(left, L, sV_T, grid=gl, **kw)
        VT_r = side_variances(right, R, sV_T, grid=gr, **kw)
        out["first_stage"] = dict(
            tau_cl=tau_T_cl, tau_bc=tau_T_bc,
            se_cl=fac * math.sqrt((VT_l[0] + VT_r[0])[deriv, deriv]),
            se_rb=fac * math.sqrt((VT_l[1] + VT_r[1])[deriv, deriv]))
    V_l = side_variances(left, L, svec, grid=gl, **kw)
    V_r = side_variances(right, R, svec, grid=gr, **kw)
    V_cl = scalepar ** 2 * fac ** 2 * (V_l[0] + V_r[0])[deriv, deriv]
    V_rb = scalepar ** 2 * fac ** 2 * (V_l[1] + V_r[1])[deriv, deriv]
    out.update(tau_cl=tau_cl, tau_bc=tau_bc, se_cl=math.sqrt(V_cl), se_rb=math.sqrt(V_rb))
    return out
