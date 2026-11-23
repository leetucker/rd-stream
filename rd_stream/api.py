"""`rd_estimate()` and `rd_bandwidth()`: the entry points."""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from hdfe_stream.core.utils import as_lazy

from . import _kernels as K
from ._engine import SELECTORS, Side, estimate, quantile_type2_sorted, select_bandwidths
from .results import ROWS, RDMulti, RDResult

KERNELS = {"triangular": K.TRIANGULAR, "epanechnikov": K.EPANECHNIKOV, "uniform": K.UNIFORM}
VCOVS = ("nearest_neighbor", "hc0", "hc1", "hc2", "hc3")
MASS_POINTS = ("adjust", "check", "off")


# ---------------------------------------------------------------------------
# options

@dataclass(frozen=True)
class _Options:
    cutoff: float
    p: int
    q: int
    deriv: int
    kern: int
    kernel: str
    vce: str              # nn, hc0..hc3 (hc1 with clusters is CR1)
    vcov_name: str
    matches: int
    masspoints: str
    bwcheck: int | None
    bwrestrict: bool
    stdvars: bool
    scaleregul: float
    scalepar: float


def _options(cutoff, polynomial_order, bias_order, derivative, kernel, vcov, n_neighbors,
             cluster, mass_points, min_distinct, restrict_bandwidth, standardize,
             regularization, scale) -> _Options:
    p = int(polynomial_order)
    if p < 0:
        raise ValueError("polynomial_order must be 0 or more")
    q = p + 1 if bias_order is None else int(bias_order)
    if q <= p:
        raise ValueError("bias_order must be larger than polynomial_order")
    deriv = int(derivative)
    if not 0 <= deriv <= p:
        raise ValueError("derivative must be between 0 and polynomial_order")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {tuple(KERNELS)}, not {kernel!r}")
    if vcov not in VCOVS:
        raise ValueError(f"vcov must be one of {VCOVS}, not {vcov!r}")
    vce = "nn" if vcov == "nearest_neighbor" else vcov
    name = vcov
    if cluster is not None:
        if vcov not in ("nearest_neighbor", "hc0", "hc1"):
            raise NotImplementedError(
                "with clusters, the variance is CR1 (rdrobust's default); its CR2 and CR3 "
                "are not available yet")
        vce, name = "hc1", "CR1"
    if mass_points not in MASS_POINTS:
        raise ValueError(f"mass_points must be one of {MASS_POINTS}, not {mass_points!r}")
    if int(n_neighbors) < 1:
        raise ValueError("n_neighbors must be at least 1")
    return _Options(float(cutoff), p, q, deriv, KERNELS[kernel], kernel, vce, name,
                    int(n_neighbors), mass_points,
                    None if min_distinct is None else int(min_distinct),
                    bool(restrict_bandwidth), bool(standardize), float(regularization),
                    float(scale))


def _parse(fml: str) -> tuple[list[str], str]:
    if fml.count("~") != 1:
        raise ValueError(f"the formula is 'outcome ~ running' (or 'y1 + y2 ~ running'), not {fml!r}")
    lhs, rhs = (s.strip() for s in fml.split("~"))
    ys = [s.strip() for s in lhs.split("+") if s.strip()]
    if not ys or not rhs or any(ch in rhs for ch in "+|*:()"):
        raise ValueError(f"the formula is 'outcome ~ running' (or 'y1 + y2 ~ running'), not {fml!r}")
    return ys, rhs


def _pair(v, what):
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        v = (float(v), float(v))
    v = tuple(float(a) for a in v)
    if len(v) != 2 or not all(math.isfinite(a) and a > 0 for a in v):
        raise ValueError(f"{what} must be a positive number or a (left, right) pair")
    return v


# ---------------------------------------------------------------------------
# the data

class _Data:
    """The running variable sorted once, with every outcome and the cluster
    and weight columns in the same order."""

    def __init__(self, data, running, outcomes, cluster, weights, treatment=None,
                 covariates=()):
        cols = [running, *outcomes]
        for c in (cluster, weights, treatment, *covariates):
            if c is not None and c not in cols:
                cols.append(c)
        lf = as_lazy(data).select(cols)
        keep = pl.col(running).is_not_null() & pl.col(running).is_not_nan()
        for c in (cluster, weights):
            if c is not None:
                keep &= pl.col(c).is_not_null()
        if weights is not None:
            keep &= pl.col(weights) >= 0
        t0 = time.perf_counter()
        df = lf.filter(keep).collect(engine="streaming").sort(running, maintain_order=True)
        self.read_sort_s = time.perf_counter() - t0
        self.x = np.array(df[running].to_numpy(), dtype=np.float64)
        self.ys = {y: np.array(df[y].cast(pl.Float64).to_numpy()) for y in outcomes}
        self.cl, self.n_clusters = None, 0
        if cluster is not None:
            codes = df[cluster].rank("dense").to_numpy().astype(np.int64) - 1
            self.cl, self.n_clusters = codes, int(codes.max()) + 1 if codes.size else 0
        self.w = None if weights is None else np.array(df[weights].cast(pl.Float64).to_numpy())
        self.t = (None if treatment is None else
                  np.array(df[treatment].cast(pl.Float64).to_numpy()))
        self.Z = None
        self.covariates = list(covariates)
        if covariates:
            self.Z = np.column_stack([df[z].cast(pl.Float64).to_numpy() for z in covariates])


def _sides(x, D, w, cl, n_clusters, c, collapse, dT=0):
    """Left and right `Side`s (units nearest the cutoff first) for the
    response matrix D (n x columns)."""
    split = int(np.searchsorted(x, c, side="left"))
    rev = slice(None, None, -1)
    if collapse:
        start = K.runs(x)
        cnt, SD, M2 = K.collapse_runs_multi(x, np.ascontiguousarray(D), start)
        xu = x[start[:-1]]
        s = int(np.searchsorted(xu, c, side="left"))
        flip = lambda a: np.ascontiguousarray(a[:s][rev])
        left = Side(flip(xu), c, cnt=flip(cnt), SD=flip(SD), M2=flip(M2), dT=dT)
        right = Side(xu[s:], c, cnt=cnt[s:], SD=np.ascontiguousarray(SD[s:]),
                     M2=np.ascontiguousarray(M2[s:]), dT=dT)
        return left, right
    part = lambda a, side: (None if a is None else
                            (np.ascontiguousarray(a[:split][rev]) if side == 0
                             else np.ascontiguousarray(a[split:])))
    left = Side(part(x, 0), c, D=part(D, 0), f=part(w, 0), cl=part(cl, 0),
                n_clusters=n_clusters, dT=dT)
    right = Side(part(x, 1), c, D=part(D, 1), f=part(w, 1), cl=part(cl, 1),
                 n_clusters=n_clusters, dT=dT)
    return left, right


def _drop_collinear(Z, names, tol=1e-7):
    """Drop covariates that are linear combinations of the others (as
    rdrobust's covs_drop), with a warning naming them."""
    import warnings

    from scipy.linalg import qr

    _, R, piv = qr(Z, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    keep_cols = sorted(int(j) for j, v in zip(piv, diag, strict=True) if v > tol * diag.max())
    if len(keep_cols) < Z.shape[1]:
        dropped = [names[j] for j in range(Z.shape[1]) if j not in keep_cols]
        warnings.warn(f"covariates dropped as collinear: {dropped}", stacklevel=4)
        return Z[:, keep_cols]
    return Z


def _stats(x, y, left, right, cl):
    """The full sample's statistics the bandwidth step needs."""
    n = x.shape[0]
    stats = dict(x_sd=float(np.std(x, ddof=1)), y_sd=float(np.std(y, ddof=1)),
                 x_iq=quantile_type2_sorted(x, 0.75) - quantile_type2_sorted(x, 0.25),
                 x_min=float(x[0]), x_max=float(x[-1]), N=n)
    stats["mass"] = (1 - left.M / left.nobs >= 0.2) or (1 - right.M / right.nobs >= 0.2)
    if cl is not None:
        split = int(np.searchsorted(x, left.c, side="left"))
        stats.update(clustered=True, G_l=int(np.unique(cl[:split]).size),
                     G_r=int(np.unique(cl[split:]).size))
    return stats


@contextlib.contextmanager
def _threads(n):
    if not n:
        yield
        return
    from stream_limits import thread_scope

    with thread_scope(int(n)):
        yield


# ---------------------------------------------------------------------------
# the entry points

def rd_estimate(fml: str, data, *, cutoff: float = 0.0, polynomial_order: int = 1,
                bias_order: int | None = None, derivative: int = 0,
                kernel: str = "triangular", bandwidth="mserd", bias_bandwidth=None,
                bandwidth_ratio: float | None = None, vcov: str = "nearest_neighbor",
                n_neighbors: int = 3, cluster: str | None = None, weights: str | None = None,
                mass_points: str = "adjust", min_distinct: int | None = None,
                restrict_bandwidth: bool = True, standardize: bool = True,
                regularization: float = 1.0, scale: float = 1.0,
                treatment: str | None = None, sharp_bandwidth: bool = False,
                covariates: Sequence[str] | str | None = None,
                collapse: str = "auto", n_threads: int | None = None) -> RDResult | RDMulti:
    """Sharp regression discontinuity: local-polynomial estimates at the
    cutoff with robust bias-corrected inference (Calonico, Cattaneo and
    Titiunik 2014), as R's rdrobust 4.1.1 computes them.

    The data are sorted once by the running variable. Every bandwidth step
    and the final fit are then compiled, threaded passes over the
    observations within that step's bandwidth. With several outcomes
    (``"y1 + y2 ~ x"``), or several bandwidths (a list), all the fits share
    that one sort.

    Parameters
    ----------
    fml : str
        ``"y ~ x"``: the outcome and the running variable. Several outcomes:
        ``"y1 + y2 + y3 ~ x"``.
    data : str, path, polars.DataFrame or polars.LazyFrame
        A Parquet file or glob, or a frame.
    cutoff : float, default 0
        Observations with x >= cutoff are treated.
    polynomial_order : int, default 1
        The order p of the local polynomial (rdrobust's ``p``).
    bias_order : int, optional
        The order q of the bias-correction polynomial; default p + 1
        (rdrobust's ``q``).
    derivative : int, default 0
        The derivative at the cutoff to estimate: 1 for a regression kink
        design (rdrobust's ``deriv``).
    kernel : {"triangular", "epanechnikov", "uniform"}
    bandwidth : str, float, (left, right) pair, or list of them
        A selector (``"mserd"``, the default, or ``"msetwo"``, ``"msesum"``,
        ``"msecomb1"``, ``"msecomb2"``, ``"cerrd"``, ``"certwo"``,
        ``"cersum"``, ``"cercomb1"``, ``"cercomb2"``: rdrobust's
        ``bwselect``), or the main bandwidth h itself (rdrobust's ``h``). A
        list fits each bandwidth in turn, from one sort, and returns an
        `RDMulti`.
    bias_bandwidth : float or (left, right) pair, optional
        The bias bandwidth b with a given h (rdrobust's ``b``); default h,
        or h / bandwidth_ratio.
    bandwidth_ratio : float, optional
        rdrobust's ``rho``: b = h / rho.
    vcov : {"nearest_neighbor", "hc0", "hc1", "hc2", "hc3"}
        The variance estimator (rdrobust's ``vce``). With `cluster`, the
        variance is cluster-robust (CR1).
    n_neighbors : int, default 3
        Neighbors for the nearest-neighbor variance (rdrobust's ``nnmatch``).
    cluster : str, optional
        A cluster column.
    weights : str, optional
        Sampling weights, multiplying the kernel weights.
    mass_points : {"adjust", "check", "off"}
        rdrobust's ``masspoints``. With "adjust" (the default), when 20% or
        more of the observations on either side share a value of the running
        variable with another one, the pilot bandwidth counts distinct
        values and every bandwidth covers at least 10 of them.
    min_distinct : int, optional
        Every bandwidth covers at least this many distinct values on each
        side (rdrobust's ``bwcheck``).
    restrict_bandwidth : bool, default True
        Bandwidths no wider than the data (rdrobust's ``bwrestrict``).
    standardize : bool, default True
        Select bandwidths on x and y divided by their standard deviations
        (rdrobust's ``stdvars``).
    regularization : float, default 1
        The scale of the bandwidth selectors' regularization term
        (rdrobust's ``scaleregul``).
    scale : float, default 1
        Multiplies the estimate (rdrobust's ``scalepar``).
    treatment : str, optional
        The treatment take-up, for a fuzzy design (rdrobust's ``fuzzy``): the
        estimate is the ratio of the jumps in the outcome and in the take-up
        (a local Wald estimator), and the result has the first stage.
    sharp_bandwidth : bool, default False
        In a fuzzy design, select the bandwidths for the outcome's jump
        alone (rdrobust's ``sharpbw``). Always so when take-up does not vary
        on one side of the cutoff.
    covariates : str or list of str, optional
        Covariates for the adjustment of Calonico, Cattaneo, Farrell and
        Titiunik (2019) (rdrobust's ``covs``): they enter linearly with the
        same coefficients on both sides, which leaves the estimand unchanged
        when they are balanced at the cutoff and can shorten the intervals.
        Collinear covariates are dropped with a warning.
    collapse : {"auto", "always", "never"}
        Fit on one row per distinct value of the running variable, which is
        exact (without clusters and weights). "auto" collapses when there
        are at most half as many distinct values as observations.
    n_threads : int, optional
        Threads for the compiled passes (stream-limits' ``thread_scope``);
        default all.

    Returns
    -------
    RDResult, or RDMulti with several outcomes or bandwidths.
    """
    outcomes, running = _parse(fml)
    o = _options(cutoff, polynomial_order, bias_order, derivative, kernel, vcov, n_neighbors,
                 cluster, mass_points, min_distinct, restrict_bandwidth, standardize,
                 regularization, scale)
    if collapse not in ("auto", "always", "never"):
        raise ValueError("collapse must be 'auto', 'always' or 'never'")
    grid = bandwidth if isinstance(bandwidth, list) else [bandwidth]
    with _threads(n_threads):
        data = _Data(data, running, outcomes, cluster, weights, treatment,
                     _as_list(covariates) if covariates is not None else ())
        fits = []
        for y_name in outcomes:
            for bw in grid:
                fits.append(_fit_one(data, running, y_name, o, bw, bias_bandwidth,
                                     bandwidth_ratio, cluster, weights, collapse,
                                     treatment, sharp_bandwidth))
    _PREPARED.clear()
    if len(fits) == 1:
        return fits[0]
    if len(grid) > 1:
        for r in fits:
            r.fml = f"{r.fml} [h={r.bandwidth[0]:.6g}]"
    return RDMulti(fits)


_PREPARED: dict = {}


def _prepare(data, running, y_name, o, cluster, weights, collapse):
    key = (id(data), y_name, collapse)
    if key in _PREPARED:
        return _PREPARED[key]
    _PREPARED.clear()
    x, y, w, cl, t, Z = data.x, data.ys[y_name], data.w, data.cl, data.t, data.Z
    ok = ~np.isnan(y)
    if t is not None:
        ok &= ~np.isnan(t)
    if Z is not None:
        ok &= ~np.isnan(Z).any(axis=1)
    if not ok.all():
        x, y = x[ok], y[ok]
        w = None if w is None else w[ok]
        cl = None if cl is None else cl[ok]
        t = None if t is None else t[ok]
        Z = None if Z is None else Z[ok]
    cols = [y] + ([] if t is None else [t])
    if Z is not None:
        Z = _drop_collinear(Z, data.covariates)
        cols += [Z[:, j] for j in range(Z.shape[1])]
    D = np.column_stack(cols)
    c = o.cutoff
    if x.shape[0] == 0 or not (x[0] < c < x[-1]):
        raise ValueError("the cutoff must be strictly inside the range of the running variable")
    use_units = False
    if collapse != "never" and cluster is None and weights is None:
        n_distinct = K.runs(x).shape[0] - 1
        use_units = collapse == "always" or n_distinct <= x.shape[0] / 2
    elif collapse == "always":
        raise ValueError("collapse='always' needs a fit without clusters and weights")
    left, right = _sides(x, D, w, cl, data.n_clusters, c, use_units,
                         dT=0 if t is None else 1)
    for s, name in ((left, "left"), (right, "right")):
        if s.M < o.q + 1:
            raise ValueError(f"not enough distinct values of {running} on the {name} of the "
                             f"cutoff ({s.M}) for a polynomial of order {o.q}")
    stats = _stats(x, y, left, right, cl)
    if t is not None:
        split = int(np.searchsorted(x, c, side="left"))
        tl, tr = t[:split], t[split:]
        if np.var(tl) == 0 and np.var(tr) == 0 and abs(tl.mean() - tr.mean()) < 1.4901161193847656e-08:
            raise ValueError("the treatment take-up has no variation and no jump at the cutoff: "
                             "the fuzzy design is not identified")
        stats["perfect_compliance"] = bool(np.var(tl) == 0 or np.var(tr) == 0)
    _PREPARED[key] = (left, right, stats, use_units)
    return _PREPARED[key]


def _fit_one(data, running, y_name, o, bandwidth, bias_bandwidth, rho, cluster, weights,
             collapse, treatment=None, sharp_bandwidth=False):
    t0 = time.perf_counter()
    left, right, stats, use_units = _prepare(data, running, y_name, o, cluster, weights, collapse)
    t1 = time.perf_counter()
    if isinstance(bandwidth, str):
        if bandwidth not in SELECTORS:
            raise ValueError(f"bandwidth must be a number or one of {SELECTORS}, not {bandwidth!r}")
        if stats["N"] < 20:
            h = (max(abs(o.cutoff - stats["x_min"]), abs(stats["x_max"] - o.cutoff)),) * 2
            b = h if rho is None else (h[0] / rho, h[1] / rho)
            selector = "manual"
        else:
            bl, br = left, right
            if treatment is not None and (sharp_bandwidth or stats["perfect_compliance"]):
                keep = [0] + list(range(2, left.ncol))
                bl, br = left.columns(keep), right.columns(keep)
            bws = select_bandwidths(bl, br, stats, p=o.p, q=o.q, deriv=o.deriv,
                                    kern=o.kern, vce=o.vce, matches=o.matches,
                                    scaleregul=o.scaleregul, masspoints=o.masspoints,
                                    bwcheck=o.bwcheck, bwrestrict=o.bwrestrict,
                                    stdvars=o.stdvars, selector=bandwidth)
            h_l, h_r, b_l, b_r = bws.chosen
            h, b = (h_l, h_r), (b_l, b_r)
            if rho is not None:
                b = (h_l / rho, h_r / rho)
            selector = bandwidth
    else:
        h = _pair(bandwidth, "bandwidth")
        if rho is not None:
            b = (h[0] / rho, h[1] / rho)
        else:
            b = _pair(bias_bandwidth, "bias_bandwidth") or h
        selector = "manual"
    t2 = time.perf_counter()
    est = estimate(left, right, h[0], h[1], b[0], b[1], p=o.p, q=o.q, deriv=o.deriv,
                   kern=o.kern, vce=o.vce, matches=o.matches, scalepar=o.scalepar)
    t3 = time.perf_counter()
    V = np.diag([est["se_cl"] ** 2, est["se_cl"] ** 2, est["se_rb"] ** 2])
    n_clusters = {}
    if cluster is not None:
        n_clusters[cluster] = stats["G_l"] + stats["G_r"]
    L, R = est["side_l"], est["side_r"]
    fac = math.factorial(o.deriv) * o.scalepar
    return RDResult(
        fml=f"{y_name} ~ {running}" + (f" (fuzzy: {treatment})" if treatment else ""),
        depvar=y_name, coefnames=list(ROWS),
        beta=np.array([est["tau_cl"], est["tau_bc"], est["tau_bc"]]), vcov=V,
        vcov_type=o.vcov_name, df_t=stats["N"], n_obs=stats["N"],
        n_clusters=n_clusters, weights=weights, running=running, cutoff=o.cutoff,
        bandwidth=h, bias_bandwidth=b, bandwidth_selector=selector, polynomial_order=o.p,
        bias_order=o.q, derivative=o.deriv, kernel=o.kernel, n_left=left.nobs,
        n_right=right.nobs, n_bandwidth=(L.n_h, R.n_h), n_bias_bandwidth=(L.n_b, R.n_b),
        n_distinct=(left.M, right.M),
        side_estimates=dict(conventional=(fac * L.beta_p[o.deriv], fac * R.beta_p[o.deriv]),
                            bias_corrected=(fac * L.beta_bc[o.deriv], fac * R.beta_bc[o.deriv])),
        collapsed=use_units, treatment=treatment, first_stage=est.get("first_stage"),
        timings=dict(read_sort_s=data.read_sort_s, prepare_s=t1 - t0,
                     bandwidth_s=t2 - t1, estimate_s=t3 - t2))


@dataclass
class BandwidthResult:
    """Selected bandwidths: `table` has one row per selector."""
    table: pl.DataFrame
    selector: str
    outcome: str
    running: str
    cutoff: float


def rd_bandwidth(fml: str, data, *, cutoff: float = 0.0, selector: str = "mserd",
                 all_selectors: bool = False, polynomial_order: int = 1,
                 bias_order: int | None = None, derivative: int = 0,
                 kernel: str = "triangular", vcov: str = "nearest_neighbor",
                 n_neighbors: int = 3, cluster: str | None = None, weights: str | None = None,
                 mass_points: str = "adjust", min_distinct: int | None = None,
                 restrict_bandwidth: bool = True, standardize: bool = True,
                 regularization: float = 1.0, treatment: str | None = None,
                 sharp_bandwidth: bool = False, covariates: Sequence[str] | str | None = None,
                 collapse: str = "auto", n_threads: int | None = None) -> BandwidthResult:
    """Data-driven bandwidths, as R's rdbwselect computes them (all ten
    selectors with ``all_selectors=True``). The options are `rd_estimate`'s."""
    outcomes, running = _parse(fml)
    if len(outcomes) != 1:
        raise ValueError("rd_bandwidth takes one outcome")
    if selector not in SELECTORS:
        raise ValueError(f"selector must be one of {SELECTORS}, not {selector!r}")
    o = _options(cutoff, polynomial_order, bias_order, derivative, kernel, vcov, n_neighbors,
                 cluster, mass_points, min_distinct, restrict_bandwidth, standardize,
                 regularization, 1.0)
    with _threads(n_threads):
        d = _Data(data, running, outcomes, cluster, weights, treatment,
                  _as_list(covariates) if covariates is not None else ())
        left, right, stats, _ = _prepare(d, running, outcomes[0], o, cluster, weights, collapse)
        if treatment is not None and (sharp_bandwidth or stats["perfect_compliance"]):
            keep = [0] + list(range(2, left.ncol))
            left, right = left.columns(keep), right.columns(keep)
        bws = select_bandwidths(left, right, stats, p=o.p, q=o.q, deriv=o.deriv, kern=o.kern,
                                vce=o.vce, matches=o.matches, scaleregul=o.scaleregul,
                                masspoints=o.masspoints, bwcheck=o.bwcheck,
                                bwrestrict=o.bwrestrict, stdvars=o.stdvars,
                                selector=selector, all_=all_selectors)
    _PREPARED.clear()
    rows = [dict(selector=k, h_left=v[0], h_right=v[1], b_left=v[2], b_right=v[3])
            for k, v in bws.table.items()]
    return BandwidthResult(pl.DataFrame(rows), selector, outcomes[0], running, o.cutoff)


def _as_list(v) -> list:
    return list(v) if isinstance(v, Sequence) and not isinstance(v, str) else [v]
