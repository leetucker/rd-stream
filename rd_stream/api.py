"""`rd_estimate()` and `rd_bandwidth()`: the entry points."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from hdfe_stream.core.utils import as_lazy

from . import _kernels as K
from ._engine import (
    SELECTORS,
    NeedWider,
    Side,
    estimate,
    quantile_type2_sorted,
    select_bandwidths,
)
from .results import ROWS, RDMulti, RDResult

KERNELS = {"triangular": K.TRIANGULAR, "epanechnikov": K.EPANECHNIKOV, "uniform": K.UNIFORM}
VCOVS = ("nearest_neighbor", "HC0", "HC1", "HC2", "HC3", "hetero", "{'CRV1': col}",
         "{'CRV2': col}", "{'CRV3': col}")
_CLUSTERED = {"CRV1": "hc1", "CRV2": "cr2", "CRV3": "cr3"}   # rdrobust's vce with clusters


def _parse_vcov(vcov):
    """(rdrobust's vce, the name, the cluster column or None) of a `vcov`:
    hdfe-stream's spellings ("HC1" or "hetero", ``{"CRV1": col}`` or
    ``"CRV1:col"``), in any case."""
    if isinstance(vcov, dict):
        if len(vcov) != 1:
            raise ValueError(f"vcov takes one clustering, as {{'CRV1': col}}, not {vcov!r}")
        (kind, col), = vcov.items()
    elif isinstance(vcov, str) and ":" in vcov:
        kind, _, col = vcov.partition(":")
    elif isinstance(vcov, str):
        v = vcov.lower()
        if v == "nearest_neighbor":
            return "nn", "nearest_neighbor", None
        if v == "hetero":
            v = "hc1"
        if v in ("hc0", "hc1", "hc2", "hc3"):
            return v, v.upper(), None
        if v in ("crv1", "crv2", "crv3"):
            raise ValueError(f"vcov={vcov!r} needs its cluster column, as "
                             f"{{{vcov.upper()!r}: col}}")
        raise ValueError(f"vcov must be one of {VCOVS}, not {vcov!r}")
    else:
        raise ValueError(f"vcov must be one of {VCOVS}, not {vcov!r}")
    kind = str(kind).upper()
    if kind not in _CLUSTERED or not isinstance(col, str) or not col.strip():
        raise ValueError(f"vcov must be one of {VCOVS}, not {vcov!r}")
    if "+" in col:
        raise ValueError("rdrobust's cluster-robust variances are one-way: "
                         f"vcov={vcov!r} names several clusterings")
    return _CLUSTERED[kind], kind, col.strip()
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
    vce: str              # nn, hc0..hc3, cr2, cr3 (hc1 with clusters is CRV1)
    vcov_name: str
    matches: int
    masspoints: str
    bwcheck: int | None
    bwrestrict: bool
    stdvars: bool
    scaleregul: float
    scalepar: float


def _options(cutoff, polynomial_order, bias_order, derivative, kernel, vcov, n_neighbors,
             mass_points, min_distinct, restrict_bandwidth, standardize,
             regularization, scale) -> tuple[_Options, str | None]:
    """The options, and the cluster column `vcov` names (or None)."""
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
    vce, name, cluster = _parse_vcov(vcov)
    if mass_points not in MASS_POINTS:
        raise ValueError(f"mass_points must be one of {MASS_POINTS}, not {mass_points!r}")
    if int(n_neighbors) < 1:
        raise ValueError("n_neighbors must be at least 1")
    return _Options(float(cutoff), p, q, deriv, KERNELS[kernel], kernel, vce, name,
                    int(n_neighbors), mass_points,
                    None if min_distinct is None else int(min_distinct),
                    bool(restrict_bandwidth), bool(standardize), float(regularization),
                    float(scale)), cluster


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
    and weight columns in the same order; or, by value (``memory="by_value"``),
    each distinct value's count, column sums and cross-products per outcome."""

    def __init__(self, data, running, outcomes, cluster, weights, treatment=None,
                 covariates=(), memory="rows", window=None):
        self.by_value = memory == "by_value"
        self.windowed = memory == "window"
        self.covariates = list(covariates)
        self.t_name = treatment
        if self.windowed:
            from ._window import WindowData

            self.window = WindowData(data, running, outcomes, cluster, weights, treatment,
                                     self.covariates, **window)
            self.n_clusters = 0
            return
        if self.by_value:
            t0 = time.perf_counter()
            self.t_name = treatment
            self.units, self.t_range = _read_by_value(as_lazy(data), running, outcomes,
                                                      treatment, self.covariates)
            self.read_sort_s = time.perf_counter() - t0
            self.n_clusters = 0
            return
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
        self.t_name = treatment
        self.t = (None if treatment is None else
                  np.array(df[treatment].cast(pl.Float64).to_numpy()))
        self.Z = None
        if covariates:
            self.Z = np.column_stack([df[z].cast(pl.Float64).to_numpy() for z in covariates])


def _data_cleanup(self):
    """Remove the window path's slabs on disk, if any."""
    if getattr(self, "windowed", False):
        self.window.cleanup()


_Data.cleanup = _data_cleanup


def _data_subset(self, mask_of):
    """A view of the rows (or values) whose running variable `mask_of` keeps:
    a donut hole, or one side of the cutoff for a placebo cutoff. The rows
    and by-value paths only; the window path reads with a filter instead."""
    import copy

    v = copy.copy(self)
    if self.by_value:
        v.units, v.t_range = {}, {}
        for y, (xu, cnt, SD, M2) in self.units.items():
            m = mask_of(xu)
            v.units[y] = (xu[m], cnt[m], np.ascontiguousarray(SD[m]),
                          np.ascontiguousarray(M2[m]))
            if y in self.t_range:
                lo, hi = self.t_range[y]
                v.t_range[y] = (lo[m], hi[m])
        return v
    m = mask_of(self.x)
    v.x = self.x[m]
    v.ys = {y: a[m] for y, a in self.ys.items()}
    for name in ("cl", "w", "t", "Z"):
        a = getattr(self, name)
        setattr(v, name, None if a is None else a[m])
    return v


_Data.subset = _data_subset


@dataclass(frozen=True)
class _Variant:
    """One fit's sample and cutoff: the rows a donut hole or a placebo
    cutoff's side keeps (`mask(x)` on held data, `expr(running)` for the
    window path's reads; None: every row), labeled by `tag` in the formula."""
    cutoff: float
    mask: object = None
    expr: object = None
    tag: str = ""
    donut: tuple | None = None


def _donuts(donut):
    """A donut option as a list of (left, right) radii, or [None]."""
    if donut is None:
        return [None]
    many = donut if isinstance(donut, list) else [donut]
    out = []
    for d in many:
        if d is None:
            out.append(None)
            continue
        v = (d, d) if isinstance(d, (int, float, np.integer, np.floating)) else tuple(d)
        try:
            pair = tuple(float(a) for a in v)
        except (TypeError, ValueError):
            pair = ()
        if len(pair) != 2 or not all(math.isfinite(a) and a >= 0 for a in pair):
            raise ValueError(f"donut must be 0 or more, or a (left, right) pair of them, "
                             f"not {d!r}")
        out.append(None if pair == (0, 0) else (float(pair[0]), float(pair[1])))
    return out


def _donut_variant(c, d, label):
    """The rows outside a donut hole: on the left, x <= c - d_left; on the
    right (x >= c), x >= c + d_right."""
    if d is None:
        return _Variant(c, tag="donut=0" if label else "")
    dl, dr = d

    def mask(x):
        return np.where(x < c, c - x >= dl, x - c >= dr)

    def expr(running):
        x = pl.col(running).cast(pl.Float64)
        return pl.when(x < c).then(c - x >= dl).otherwise(x - c >= dr)

    tag = f"donut={dl:g}" if dl == dr else f"donut=({dl:g}, {dr:g})"
    return _Variant(c, mask, expr, tag if label else "", d)


def _finite(c):
    e = pl.col(c).cast(pl.Float64)
    return e.is_not_null() & e.is_not_nan()


def _read_by_value(lf, running, outcomes, treatment, covariates):
    """One streaming group-by by the running variable: per outcome, each
    value's count of complete rows, the sums of the response columns
    [y, (take-up), covariates] and their within-value cross-products."""
    extra = ([treatment] if treatment else []) + list(covariates)
    aggs = []
    for i, y in enumerate(outcomes):
        cols = [y, *extra]
        ok = pl.all_horizontal([_finite(c) for c in cols])
        n = ok.sum()
        aggs.append(n.alias(f"n{i}"))
        if treatment:                     # the take-up's range per value (perfect compliance)
            tc = pl.col(treatment).cast(pl.Float64).filter(ok)
            aggs += [tc.min().alias(f"tlo{i}"), tc.max().alias(f"thi{i}")]
        v = [pl.col(c).cast(pl.Float64).filter(ok) for c in cols]
        for j in range(len(cols)):
            aggs.append(v[j].sum().alias(f"s{i}_{j}"))
            for jj in range(j, len(cols)):
                m = (v[j].var(ddof=0) if j == jj else pl.cov(v[j], v[jj], ddof=0))
                aggs.append((m.fill_null(0.0).fill_nan(0.0) * n).alias(f"m{i}_{j}_{jj}"))
    keep = _finite(running)
    df = (lf.filter(keep).group_by(pl.col(running).cast(pl.Float64))
          .agg(aggs).collect(engine="streaming").sort(running))
    x = np.array(df[running].to_numpy(), dtype=np.float64)
    out, ranges = {}, {}
    nc = 1 + len(extra)
    for i, y in enumerate(outcomes):
        cnt = df[f"n{i}"].to_numpy().astype(np.int64)
        has = cnt > 0
        SD = np.column_stack([df[f"s{i}_{j}"].to_numpy() for j in range(nc)])
        M2 = np.empty((x.shape[0], nc, nc))
        for j in range(nc):
            for jj in range(j, nc):
                M2[:, j, jj] = M2[:, jj, j] = df[f"m{i}_{j}_{jj}"].to_numpy()
        out[y] = (x[has], cnt[has], np.ascontiguousarray(SD[has]),
                  np.ascontiguousarray(M2[has]))
        if treatment:
            ranges[y] = (df[f"tlo{i}"].to_numpy()[has], df[f"thi{i}"].to_numpy()[has])
    return out, ranges


def _sides_from_units(xu, cnt, SD, M2, c, dT=0):
    """Left and right collapsed `Side`s from the distinct values' sums."""
    s = int(np.searchsorted(xu, c, side="left"))
    rev = slice(None, None, -1)
    flip = lambda a: np.ascontiguousarray(a[:s][rev])
    left = Side(flip(xu), c, cnt=flip(cnt), SD=flip(SD), M2=flip(M2), dT=dT)
    right = Side(np.ascontiguousarray(xu[s:]), c, cnt=np.ascontiguousarray(cnt[s:]),
                 SD=np.ascontiguousarray(SD[s:]), M2=np.ascontiguousarray(M2[s:]), dT=dT)
    return left, right


def _pooled_sd(mean_u, cnt, m2_u):
    """The standard deviation (ddof 1) of the rows behind the units."""
    n = cnt.sum()
    mean = (cnt * mean_u).sum() / n
    ss = m2_u.sum() + (cnt * (mean_u - mean) ** 2).sum()
    return math.sqrt(ss / (n - 1))


def _quantile_units(xu, cnt, prob):
    """rdrobust's type-2 quantile of the rows behind sorted units."""
    cum = np.cumsum(cnt)
    n = int(cum[-1])
    at = lambda i: xu[int(np.searchsorted(cum, i + 1, side="left"))]   # the i-th row (0-based)
    hq = n * prob
    j = int(math.floor(hq))
    if prob <= 0:
        return xu[0]
    if prob >= 1:
        return xu[-1]
    return 0.5 * (at(j - 1) + at(j)) if hq == j else at(j)


def _drop_collinear_gram(ZZ, names, tol=1e-7):
    """_drop_collinear from Z'Z: a pivoted Cholesky factorization picks the
    columns a pivoted QR of Z would keep. Returns the kept columns' indices."""
    import warnings

    A = ZZ.copy()
    k = A.shape[0]
    piv, diag = [], []
    rest = list(range(k))
    for _ in range(k):
        j = max(rest, key=lambda i: A[i, i])
        d = A[j, j]
        if d <= 0:
            break
        piv.append(j)
        diag.append(math.sqrt(d))
        rest.remove(j)
        col = A[:, j] / math.sqrt(d)
        A = A - np.outer(col, col)
    top = max(diag) if diag else 0.0
    keep = sorted(j for j, v in zip(piv, diag, strict=True) if v > tol * top)
    if len(keep) < k:
        dropped = [names[j] for j in range(k) if j not in keep]
        warnings.warn(f"covariates dropped as collinear: {dropped}", stacklevel=4)
    return keep


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
    stats["mass"] = (1 - left.M_side / left.n_side >= 0.2) or (1 - right.M_side / right.n_side >= 0.2)
    if cl is not None:
        split = int(np.searchsorted(x, left.c, side="left"))
        # the codes are 0..G-1: count those present on each side in one linear pass
        # (np.unique sorted them: 19 s of a 34 s clustered fit at 10^8 rows)
        stats.update(clustered=True, G_l=int(np.count_nonzero(np.bincount(cl[:split]))),
                     G_r=int(np.count_nonzero(np.bincount(cl[split:]))))
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
                bandwidth_ratio: float | None = None, vcov: str | dict = "nearest_neighbor",
                n_neighbors: int = 3, weights: str | None = None,
                mass_points: str = "adjust", min_distinct: int | None = None,
                restrict_bandwidth: bool = True, standardize: bool = True,
                regularization: float = 1.0, scale: float = 1.0,
                treatment: str | None = None, sharp_bandwidth: bool = False,
                covariates: Sequence[str] | str | None = None,
                donut=None,
                memory: str = "auto",
                max_memory_gb: float | None = None, memory_reserve: float | None = None,
                workdir: str | None = None, n_threads: int | None = None,
                profile: bool = False) -> RDResult | RDMulti:
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
    vcov : str or dict, default "nearest_neighbor"
        The variance estimator (rdrobust's ``vce``), spelled as in
        hdfe-stream: "nearest_neighbor", "HC0", "HC1" (or "hetero"), "HC2"
        or "HC3", or clustered by a column as ``{"CRV1": "school"}`` (or
        ``"CRV1:school"``), ``{"CRV2": ...}`` (Bell and McCaffrey's
        bias-reduced linearization) or ``{"CRV3": ...}`` (the jackknife-like
        correction). The names are case-insensitive. The clustered
        variances are rdrobust's: CRV1 is rdrobust's CR1 (hdfe-stream's CRV1
        factor, G/(G-1) (N-1)/(N-k)), and CRV2 and CRV3 take no small-sample
        factor, as in rdrobust (hdfe-stream's CRV3 takes CRV1's).
    n_neighbors : int, default 3
        Neighbors for the nearest-neighbor variance (rdrobust's ``nnmatch``).
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
    donut : float, (left, right) pair, or list of them, optional
        A donut hole: the observations within this distance of the cutoff
        are left out (on the left, x > cutoff - donut; on the right,
        cutoff <= x < cutoff + donut, so a positive radius also drops
        x = cutoff), and the whole fit, bandwidth selection included, runs
        on the rest. A list gives one fit per radius, from one read (0 for
        no hole), labeled ``[donut=...]``.
    memory : {"auto", "rows", "by_value", "window"}
        How the data are held. "rows" reads and sorts the rows used, and
        fits on every row. "by_value" groups the rows by the value of the
        running variable as it reads them (one streaming pass), keeping only
        each value's count, sums and cross-products, and fits on one row per
        value, which gives the same results; memory grows with the number of
        distinct values rather than the rows, and the fit needs no clusters
        or weights. "window" takes the full sample's statistics
        in streaming passes (standard deviations, exact quartiles and
        distinct counts, and the power sums of rdbwselect's fit over each
        side's range) and holds only the rows near the cutoff; when a step's
        window does not fit in memory, the rows are written to `workdir` in
        slabs by distance from the cutoff, and each window sum reads them a
        slab at a time. It takes any fit, and gives the same results as
        "rows". "auto" (the default) groups by value when the fit allows it
        and the running variable has at most half as many distinct values
        as rows (an approximate count, in one pass over it), else holds the
        rows when they fit the memory budget, and otherwise takes the window
        path.
    max_memory_gb : float, optional
        The memory the fit may use (stream-limits' ``MemoryBudget``); default
        the machine's or the job's limit. When even a slab of rows does not
        fit, the fit stops with a MemoryError.
    memory_reserve : float, optional
        The share of `max_memory_gb` kept back for what the model of the
        fit's memory does not cover; default 0.15.
    workdir : str, optional
        Where the window path writes its slabs (a run directory, removed
        after the fit); default $HDFE_STREAM_WORKDIR, else the system
        temporary directory. The slabs take about 8 bytes a row for each
        column the fit reads.
    n_threads : int, optional
        Threads for the compiled passes (stream-limits' ``thread_scope``);
        default all.
    profile : bool, default False
        Return hdfe-stream's `Profile` of the fit instead of fitting: the
        rows and distinct values of the running variable (one streaming
        pass), the path ``memory`` would take, and its memory and disk. For
        the window path, its summary passes run too.

    Returns
    -------
    RDResult, or RDMulti with several outcomes or bandwidths; a Profile with
    ``profile=True``.
    """
    outcomes, running = _parse(fml)
    _PREPARED.clear()                     # a call that failed may have left some
    o, cluster = _options(cutoff, polynomial_order, bias_order, derivative, kernel, vcov,
                          n_neighbors, mass_points, min_distinct, restrict_bandwidth,
                          standardize, regularization, scale)
    # on one row per distinct value of x when that is shorter (the same results): not with
    # "rows", which fits on every row
    collapse = "never" if memory == "rows" else "auto"
    grid = bandwidth if isinstance(bandwidth, list) else [bandwidth]
    covs = _as_list(covariates) if covariates is not None else []
    path = _memory_path(data, running, outcomes, cluster, weights, treatment, covs, collapse,
                        memory, max_memory_gb, memory_reserve, n_threads, profile,
                        f"rd_estimate, {len(outcomes)} outcome(s) x {len(grid)} bandwidth(s)",
                        o=o, hint=_largest_bandwidth(grid, bias_bandwidth, bandwidth_ratio))
    if profile:
        return path
    donuts = _donuts(donut)
    variants = [_donut_variant(o.cutoff, d, len(donuts) > 1) for d in donuts]
    fits = _fit_variants(data, running, outcomes, o, cluster, weights, treatment, covs, path,
                         max_memory_gb, memory_reserve, workdir, n_threads, grid,
                         bias_bandwidth, bandwidth_ratio, collapse, sharp_bandwidth, variants)
    if len(fits) == 1:
        return fits[0]
    return RDMulti(fits)


def _fit_variants(source, running, outcomes, o, cluster, weights, treatment, covs, path,
                  max_memory_gb, memory_reserve, workdir, n_threads, grid, bias_bandwidth,
                  rho, collapse, sharp_bandwidth, variants, skipped=None):
    """The fits of every outcome x bandwidth, for each variant (cutoff and
    rows kept). The rows and by-value paths read once and take each
    variant's rows from what they hold; the window path reads each variant
    with its filter. With a `skipped` list, a variant and outcome whose fit
    raises ValueError (a cutoff too near the edge of its rows, say) is
    recorded there as (variant, outcome, message) instead."""
    fits = []
    with _threads(n_threads):
        base = None
        if path != "window":
            base = _Data(source, running, outcomes, cluster, weights, treatment, covs,
                         memory=path)
        for v in variants:
            ov = o if v.cutoff == o.cutoff else dataclasses.replace(o, cutoff=v.cutoff)
            if base is None:
                window = _window_options(path, ov, running, outcomes, cluster, treatment, covs,
                                         max_memory_gb, memory_reserve,
                                         _largest_bandwidth(grid, bias_bandwidth, rho), workdir)
                window["keep"] = None if v.expr is None else v.expr(running)
                try:
                    data = _Data(source, running, outcomes, cluster, weights, treatment, covs,
                                 memory=path, window=window)
                except ValueError as e:
                    if skipped is None:
                        raise
                    skipped += [(v, y_name, str(e)) for y_name in outcomes]
                    continue
            else:
                data = base if v.mask is None else base.subset(v.mask)
            new = []
            for y_name in outcomes:
                def fit_outcome(y_name=y_name, data=data, ov=ov):
                    grids = (None, None)
                    if len(grid) > 1 and cluster is None:
                        left, right, _, _ = _prepare(data, running, y_name, ov, cluster,
                                                     weights, collapse)
                        grids = _grid_sides(left, right, ov, grid, bias_bandwidth, rho)
                    return [_fit_one(data, running, y_name, ov, bw, bias_bandwidth, rho,
                                     cluster, weights, collapse, treatment, sharp_bandwidth,
                                     grids)
                            for bw in grid]
                try:
                    new += _widening(data, fit_outcome)
                except ValueError as e:
                    if skipped is None:
                        data.cleanup()
                        raise
                    skipped.append((v, y_name, str(e)))
                except BaseException:
                    data.cleanup()
                    raise
            data.cleanup()
            for r in new:
                r.donut = v.donut
                if len(grid) > 1:
                    r.fml = f"{r.fml} [h={r.bandwidth[0]:.6g}]"
                if v.tag:
                    r.fml = f"{r.fml} [{v.tag}]"
            fits += new
    _PREPARED.clear()
    return fits


_PREPARED: dict = {}


def _widening(data, fit):
    """fit(), reading the rows again out to a wider radius whenever a step
    needs rows beyond those held (the window path)."""
    while True:
        try:
            return fit()
        except NeedWider as e:
            if not data.windowed:
                raise
            _PREPARED.clear()
            data.window.widen(e.radius)


def _largest_bandwidth(grid, bias_bandwidth, rho):
    """The largest given bandwidth (h or b, either side), or None when the
    bandwidths are selected."""
    out = None
    for bw in grid:
        if isinstance(bw, str):
            return None
        h = _pair(bw, "bandwidth")
        b = (h[0] / rho, h[1] / rho) if rho is not None else (
            _pair(bias_bandwidth, "bias_bandwidth") or h)
        out = max(out or 0.0, *h, *b)
    return out


def _window_options(path, o, running, outcomes, cluster, treatment, covs, max_memory_gb,
                    memory_reserve, hint, workdir=None):
    """WindowData's options for the window path (None for the others)."""
    if path != "window":
        return None
    from . import _memory

    n_read = 1 + len(outcomes) + sum(c is not None for c in (cluster, treatment)) + len(covs)
    per_row = _memory.rows_bytes(1, n_read, 1 + (treatment is not None) + len(covs),
                                 cluster is not None)
    return dict(o=o, per_row_bytes=per_row, max_memory_gb=max_memory_gb,
                memory_reserve=memory_reserve, hint=hint, workdir=workdir)


def _memory_path(data, running, outcomes, cluster, weights, treatment, covs, collapse, memory,
                 max_memory_gb, memory_reserve, n_threads, profile, model, o=None, hint=None,
                 slabs=False):
    """The memory path ("rows", "by_value" or "window"), or the Profile.
    Without options `o` there is no window path; `slabs`: its rows always
    go to slabs (rd_plot)."""
    from stream_limits import detect_cpus

    from . import _memory

    if memory not in _memory.MEMORY_PATHS:
        raise ValueError(f"memory must be one of {_memory.MEMORY_PATHS}, not {memory!r}")
    lf = as_lazy(data)
    extra = [c for c in (cluster, weights, treatment, *covs) if c is not None]
    kw = dict(memory=memory, max_memory_gb=max_memory_gb, memory_reserve=memory_reserve,
              threads=int(n_threads) if n_threads else detect_cpus(), collapse=collapse,
              clustered=cluster is not None, weighted=weights is not None,
              n_columns=1 + (treatment is not None) + len(covs))
    if profile:
        window = None
        if o is not None:
            def window():
                from ._window import WindowData

                opts = _window_options("window", o, running, outcomes, cluster, treatment,
                                       covs, max_memory_gb, memory_reserve, hint)
                return WindowData(data, running, outcomes, cluster, weights, treatment, covs,
                                  read=False, **opts)
        return _memory.profile(lf, data, running=running, outcomes=outcomes, extra=extra,
                               model=model, window=window, slabs=slabs, **kw)
    path, _ = _memory.choose(lf, data, running=running, n_read=1 + len(outcomes) + len(extra),
                             n_outcomes=len(outcomes), window=o is not None, **kw)
    return path


def _prepare(data, running, y_name, o, cluster, weights, collapse):
    key = (data, y_name, collapse, o.cutoff)    # the object, so that its id is not reused
    if key in _PREPARED:
        return _PREPARED[key]
    _PREPARED.clear()
    if data.by_value:
        _PREPARED[key] = _prepare_units(data, running, y_name, o)
        return _PREPARED[key]
    if data.windowed:
        _PREPARED[key] = _prepare_window(data, running, y_name, o, cluster, weights, collapse)
        return _PREPARED[key]
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
        use_units = n_distinct <= x.shape[0] / 2
    left, right = _sides(x, D, w, cl, data.n_clusters, c, use_units,
                         dT=0 if t is None else 1)
    for s, name in ((left, "left"), (right, "right")):
        if s.M_side < o.q + 1:
            raise ValueError(f"not enough distinct values of {running} on the {name} of the "
                             f"cutoff ({s.M_side}) for a polynomial of order {o.q}")
    stats = _stats(x, y, left, right, cl)
    if t is not None:
        split = int(np.searchsorted(x, c, side="left"))
        tl, tr = t[:split], t[split:]
        # a constant take-up on a side: min = max, R's var() == 0 (np.var can leave a
        # rounding residue for a constant whose mean is inexact, such as 0.1)
        const_l, const_r = bool(tl.min() == tl.max()), bool(tr.min() == tr.max())
        if const_l and const_r and abs(tl.mean() - tr.mean()) < 1.4901161193847656e-08:
            raise ValueError("the treatment take-up has no variation and no jump at the cutoff: "
                             "the fuzzy design is not identified")
        stats["perfect_compliance"] = const_l or const_r
    _PREPARED[key] = (left, right, stats, use_units)
    return _PREPARED[key]


def _grid_sides(left, right, o, grid, bias_bandwidth, rho):
    """Block power-sum tables (`GridSide`) for each side, out to the widest
    window of the grid's given bandwidths; (None, None) when there are none."""
    from ._moments import GridSide

    if getattr(left, "slabbed", False):
        return None, None
    ext = [0, 0]
    for bw in grid:
        if isinstance(bw, str):
            continue
        h = _pair(bw, "bandwidth")
        b = (h[0] / rho, h[1] / rho) if rho is not None else (_pair(bias_bandwidth,
                                                                     "bias_bandwidth") or h)
        for j, side in enumerate((left, right)):
            ext[j] = max(ext[j], side.window(h[j], o.kern), side.window(b[j], o.kern))
    if not ext[0] or not ext[1]:
        return None, None
    return tuple(GridSide(side, e, p=o.p, q=o.q, vce=o.vce, matches=o.matches)
                 for side, e in zip((left, right), ext, strict=True))


def _prepare_window(data, running, y_name, o, cluster, weights, collapse):
    """_prepare for the window path: the sides from the rows held (those
    near the cutoff), with the full sample's statistics, counts and power
    sums from the streaming passes."""
    W = data.window
    st = W.stats(y_name)
    if W.slabbed:
        return _prepare_slabs(data, running, y_name, o, cluster, weights, collapse, st)
    x, y, w, cl, t, Z = W.x, W.ys[y_name], W.w, W.cl, W.t, W.Z
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
    dT = 0 if t is None else 1
    cols = [y] + ([] if t is None else [t])
    full_cols = list(range(1 + dT))
    if Z is not None:
        keep = _drop_collinear_gram(st["ZZ"], data.covariates)
        Z = Z[:, keep]
        cols += [Z[:, j] for j in range(Z.shape[1])]
        full_cols += [1 + dT + j for j in keep]
    D = np.column_stack(cols)
    c = o.cutoff
    use_units = False
    if collapse != "never" and cluster is None and weights is None:
        n_distinct = K.runs(x).shape[0] - 1
        use_units = n_distinct <= x.shape[0] / 2
    left, right = _sides(x, D, w, cl, W.n_clusters, c, use_units, dT=dT)
    for s, side in ((left, "l"), (right, "r")):
        s.held_window(math.inf if W.side_complete(side) else W.radius,
                      st["full"][side].columns(full_cols), st[f"n_{side}"], st[f"M_{side}"])
        if s.nobs == 0 and not W.side_complete(side):
            raise NeedWider(None)
    return left, right, _window_stats(left, right, st, o, running, cluster, dT), use_units


def _window_stats(left, right, st, o, running, cluster, dT):
    """The statistics dict `select_bandwidths` takes, and the checks of
    `_prepare`, from the window path's full-sample statistics."""
    for s, name in ((left, "left"), (right, "right")):
        if s.M_side < o.q + 1:
            raise ValueError(f"not enough distinct values of {running} on the {name} of the "
                             f"cutoff ({s.M_side}) for a polynomial of order {o.q}")
    stats = {k: st[k] for k in ("x_sd", "y_sd", "x_iq", "x_min", "x_max", "N")}
    stats["mass"] = (1 - left.M_side / left.n_side >= 0.2) or (1 - right.M_side / right.n_side >= 0.2)
    if cluster is not None:
        stats.update(clustered=True, G_l=st["G_l"], G_r=st["G_r"])
    if dT:
        (lo_l, hi_l), (lo_r, hi_r) = st["take_up"]["l"], st["take_up"]["r"]
        const_l, const_r = lo_l == hi_l, lo_r == hi_r
        if const_l and const_r and abs(lo_l - lo_r) < 1.4901161193847656e-08:
            raise ValueError("the treatment take-up has no variation and no jump at the cutoff: "
                             "the fuzzy design is not identified")
        stats["perfect_compliance"] = bool(const_l or const_r)
    return stats


def _prepare_slabs(data, running, y_name, o, cluster, weights, collapse, st):
    """_prepare for the window path with the rows in slabs on disk: a
    `SlabSide` for each side."""
    from ._slabs import SlabSide

    W = data.window
    dT = 1 if W.treatment else 0
    extra = ([W.treatment] if W.treatment else []) + list(W.covariates)
    full_cols = list(range(1 + dT))
    if W.covariates:
        full_cols += [1 + dT + j for j in _drop_collinear_gram(st["ZZ"], data.covariates)]
    collapsible = collapse != "never" and cluster is None and weights is None
    use_units = collapsible and st["M_l"] + st["M_r"] <= st["N"] / 2
    sides = [SlabSide(W.store, side, o.cutoff, y=y_name, extra=extra, dT=dT,
                      collapse=use_units, full=st["full"][side].columns(full_cols),
                      n_side=st[f"n_{side}"], M_side=st[f"M_{side}"], cols=full_cols,
                      weights=weights)
             for side in ("l", "r")]
    left, right = sides
    return left, right, _window_stats(left, right, st, o, running, cluster, dT), use_units


def _prepare_units(data, running, y_name, o):
    """_prepare for the by-value path: the sides and statistics from each
    distinct value's sums."""
    xu, cnt, SD, M2 = data.units[y_name]
    dT = 1 if data.t_name else 0
    if data.covariates:
        zc = list(range(1 + dT, SD.shape[1]))
        ZZ = (M2[:, zc][:, :, zc] + SD[:, zc, None] * SD[:, None, zc] / cnt[:, None, None]
              ).sum(axis=0)
        keep = list(range(1 + dT)) + [zc[j] for j in _drop_collinear_gram(ZZ, data.covariates)]
        SD, M2 = np.ascontiguousarray(SD[:, keep]), np.ascontiguousarray(M2[:, keep][:, :, keep])
    c = o.cutoff
    if xu.shape[0] == 0 or not (xu[0] < c < xu[-1]):
        raise ValueError("the cutoff must be strictly inside the range of the running variable")
    left, right = _sides_from_units(xu, cnt, SD, M2, c, dT)
    for s, name in ((left, "left"), (right, "right")):
        if s.M_side < o.q + 1:
            raise ValueError(f"not enough distinct values of {running} on the {name} of the "
                             f"cutoff ({s.M_side}) for a polynomial of order {o.q}")
    cntf = cnt.astype(np.float64)
    stats = dict(x_sd=_pooled_sd(xu, cntf, np.zeros_like(cntf)),
                 y_sd=_pooled_sd(SD[:, 0] / cntf, cntf, M2[:, 0, 0]),
                 x_iq=_quantile_units(xu, cnt, 0.75) - _quantile_units(xu, cnt, 0.25),
                 x_min=float(xu[0]), x_max=float(xu[-1]), N=int(cnt.sum()))
    stats["mass"] = (1 - left.M_side / left.n_side >= 0.2) or (1 - right.M_side / right.n_side >= 0.2)
    if dT:
        split = int(np.searchsorted(xu, c, side="left"))
        tlo, thi = data.t_range[y_name]

        def constant(sl):
            # min = max over the side, as on the rows path (R's var() == 0)
            return bool(tlo[sl].min() == thi[sl].max())

        def mean(sl):
            return float(SD[sl, 1].sum() / cntf[sl].sum())

        L_, R_ = slice(0, split), slice(split, None)
        if constant(L_) and constant(R_) and abs(mean(L_) - mean(R_)) < 1.4901161193847656e-08:
            raise ValueError("the treatment take-up has no variation and no jump at the cutoff: "
                             "the fuzzy design is not identified")
        stats["perfect_compliance"] = constant(L_) or constant(R_)
    return left, right, stats, True


def _fit_one(data, running, y_name, o, bandwidth, bias_bandwidth, rho, cluster, weights,
             collapse, treatment=None, sharp_bandwidth=False, grids=(None, None)):
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
                   kern=o.kern, vce=o.vce, matches=o.matches, scalepar=o.scalepar,
                   grids=grids)
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
        bias_order=o.q, derivative=o.deriv, kernel=o.kernel, n_left=left.n_side,
        n_right=right.n_side, n_bandwidth=(L.n_h, R.n_h), n_bias_bandwidth=(L.n_b, R.n_b),
        n_distinct=(left.M_side, right.M_side),
        side_estimates=dict(conventional=(fac * L.beta_p[o.deriv], fac * R.beta_p[o.deriv]),
                            bias_corrected=(fac * L.beta_bc[o.deriv], fac * R.beta_bc[o.deriv])),
        collapsed=use_units, memory=("by_value" if data.by_value else
                                     "window" if data.windowed else "rows"),
        treatment=treatment, first_stage=est.get("first_stage"),
        timings=dict(read_sort_s=data.read_sort_s if not data.windowed
                     else data.window.stats_s + data.window.read_s, prepare_s=t1 - t0,
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
                 kernel: str = "triangular", vcov: str | dict = "nearest_neighbor",
                 n_neighbors: int = 3, weights: str | None = None,
                 mass_points: str = "adjust", min_distinct: int | None = None,
                 restrict_bandwidth: bool = True, standardize: bool = True,
                 regularization: float = 1.0, treatment: str | None = None,
                 sharp_bandwidth: bool = False, covariates: Sequence[str] | str | None = None,
                 donut=None,
                 memory: str = "auto",
                 max_memory_gb: float | None = None, memory_reserve: float | None = None,
                 workdir: str | None = None, n_threads: int | None = None,
                 profile: bool = False) -> BandwidthResult:
    """Data-driven bandwidths, as R's rdbwselect computes them (all ten
    selectors with ``all_selectors=True``). The options are `rd_estimate`'s;
    `donut` takes one radius or (left, right) pair here."""
    outcomes, running = _parse(fml)
    _PREPARED.clear()                     # a call that failed may have left some
    if len(outcomes) != 1:
        raise ValueError("rd_bandwidth takes one outcome")
    if selector not in SELECTORS:
        raise ValueError(f"selector must be one of {SELECTORS}, not {selector!r}")
    o, cluster = _options(cutoff, polynomial_order, bias_order, derivative, kernel, vcov,
                          n_neighbors, mass_points, min_distinct, restrict_bandwidth,
                          standardize, regularization, 1.0)
    collapse = "never" if memory == "rows" else "auto"
    covs = _as_list(covariates) if covariates is not None else []
    path = _memory_path(data, running, outcomes, cluster, weights, treatment, covs, collapse,
                        memory, max_memory_gb, memory_reserve, n_threads, profile,
                        "rd_bandwidth", o=o)
    if profile:
        return path
    if isinstance(donut, list):
        raise ValueError("rd_bandwidth takes one donut: a radius or a (left, right) pair")
    v = _donut_variant(o.cutoff, _donuts(donut)[0], False)
    with _threads(n_threads):
        window = _window_options(path, o, running, outcomes, cluster, treatment, covs,
                                 max_memory_gb, memory_reserve, None, workdir)
        if window is not None:
            window["keep"] = None if v.expr is None else v.expr(running)
        d = _Data(data, running, outcomes, cluster, weights, treatment, covs, memory=path,
                  window=window)
        if v.mask is not None and path != "window":
            d = d.subset(v.mask)

        def select():
            left, right, stats, _ = _prepare(d, running, outcomes[0], o, cluster, weights,
                                             collapse)
            if treatment is not None and (sharp_bandwidth or stats["perfect_compliance"]):
                keep = [0] + list(range(2, left.ncol))
                left, right = left.columns(keep), right.columns(keep)
            return select_bandwidths(left, right, stats, p=o.p, q=o.q, deriv=o.deriv,
                                     kern=o.kern, vce=o.vce, matches=o.matches,
                                     scaleregul=o.scaleregul, masspoints=o.masspoints,
                                     bwcheck=o.bwcheck, bwrestrict=o.bwrestrict,
                                     stdvars=o.stdvars, selector=selector, all_=all_selectors)
        try:
            bws = _widening(d, select)
        finally:
            d.cleanup()
    _PREPARED.clear()
    rows = [dict(selector=k, h_left=v[0], h_right=v[1], b_left=v[2], b_right=v[3])
            for k, v in bws.table.items()]
    return BandwidthResult(pl.DataFrame(rows), selector, outcomes[0], running, o.cutoff)


def _as_list(v) -> list:
    return list(v) if isinstance(v, Sequence) and not isinstance(v, str) else [v]
