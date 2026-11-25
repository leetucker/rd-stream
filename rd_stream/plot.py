"""`rd_plot()`: R's rdplot. Binned means of the outcome on each side of the
cutoff, with the number of bins chosen by Calonico, Cattaneo and
Titiunik's (2015) IMSE-optimal or mimicking-variance rules, and a global
polynomial fit on each side.

Every quantity is a sum over the sorted rows, or over the distinct values
of the running variable with their counts:

- the global polynomial: kernel-weighted power moments (the same window
  sums as `rd_estimate`'s);
- the bin selection: a fourth-order polynomial regression of y and y^2 on
  x, evaluated at every row and at the midpoints between consecutive rows,
  and, for the spacings estimators ("es", "esmv", "qs", "qsmv"), the
  differences between consecutive rows' outcomes, in rdplot's (stable)
  order;
- the bins: contiguous runs of the sorted rows.

The spacings estimators need the rows in order, so they need the rows
path; the polynomial-regression ones (the "pr" variants, which
``mass_points="adjust"`` switches to when the running variable has mass
points) also work by value.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numba import njit, prange

from . import _kernels as K
from ._engine import gram

BIN_METHODS = ("es", "espr", "esmv", "esmvpr", "qs", "qspr", "qsmv", "qsmvpr")
_DESCRIBE = {
    "es": "IMSE-optimal evenly-spaced method using spacings estimators",
    "espr": "IMSE-optimal evenly-spaced method using polynomial regression",
    "esmv": "mimicking variance evenly-spaced method using spacings estimators",
    "esmvpr": "mimicking variance evenly-spaced method using polynomial regression",
    "qs": "IMSE-optimal quantile-spaced method using spacings estimators",
    "qspr": "IMSE-optimal quantile-spaced method using polynomial regression",
    "qsmv": "mimicking variance quantile-spaced method using spacings estimators",
    "qsmvpr": "mimicking variance quantile-spaced method using polynomial regression",
}
_KERNELS = {"uniform": K.UNIFORM, "triangular": K.TRIANGULAR, "epanechnikov": K.EPANECHNIKOV}
N_PLOT = 500


# ---------------------------------------------------------------------------
# compiled passes

@njit(parallel=True, cache=True)
def _power_sums(x, V, A):
    """sum_i x_i^a V[i, c], a = 0..A (raw powers, as rdplot's k = 4 fit)."""
    n, nv = V.shape
    nchunk = max(1, min(512, n // 4096))
    step = (n + nchunk - 1) // nchunk if n else 0
    part = np.zeros((nchunk, A + 1, nv))
    for c in prange(nchunk):
        lo = c * step
        hi = min(n, lo + step)
        acc = np.zeros((A + 1, nv))
        for i in range(lo, hi):
            p = 1.0
            for a in range(A + 1):
                for j in range(nv):
                    acc[a, j] += p * V[i, j]
                p *= x[i]
        part[c] = acc
    out = np.zeros((A + 1, nv))
    for c in range(nchunk):
        out += part[c]
    return out


@njit(cache=True, inline="always")
def _poly(g, x):
    out = 0.0
    p = 1.0
    for j in range(g.shape[0]):
        out += g[j] * p
        p *= x
    return out


@njit(cache=True, inline="always")
def _dpoly(g, x):
    out = 0.0
    p = 1.0
    for j in range(1, g.shape[0]):
        out += j * g[j] * p
        p *= x
    return out


@njit(parallel=True, cache=True)
def _selection_sums(x, cnt, rows, g1, g2, var_y, y):
    """The bin selection's sums on one side (x ascending): sum mu1^2 and
    sum sigma^2 over the rows; over consecutive units, sum dx sigma^2 and
    sum dx^2 mu1^2 at their midpoints; and, with the rows (`y` non-empty),
    the spacings sums sum dx dy^2 and sum dy^2."""
    n = x.shape[0]
    nchunk = max(1, min(512, n // 4096))
    step = (n + nchunk - 1) // nchunk if n else 0
    part = np.zeros((nchunk, 6))
    spacings = y.shape[0] > 0
    for c in prange(nchunk):
        lo = c * step
        hi = min(n, lo + step)
        acc = np.zeros(6)
        for i in range(lo, hi):
            m = 1.0 if rows else cnt[i]
            xi = x[i]
            mu1 = _dpoly(g1, xi)
            s2 = _poly(g2, xi) - _poly(g1, xi) ** 2
            if s2 < 0:
                s2 = var_y
            acc[0] += m * mu1 * mu1
            acc[1] += m * s2
            if i + 1 < n:
                dx = x[i + 1] - xi
                xb = 0.5 * (x[i + 1] + xi)
                s2b = _poly(g2, xb) - _poly(g1, xb) ** 2
                if s2b < 0:
                    s2b = var_y
                mub = _dpoly(g1, xb)
                acc[2] += dx * s2b
                acc[3] += dx * dx * mub * mub
                if spacings:
                    dy = y[i + 1] - y[i]
                    acc[4] += dx * dy * dy
                    acc[5] += dy * dy
        part[c] = acc
    out = np.zeros(6)
    for c in range(nchunk):
        out += part[c]
    return out


@njit(cache=True)
def _segment_stats(x, cnt, sy, m2, rows, starts, ends):
    """Per segment [starts, ends) of units: rows, sum x, sum y and the
    within-segment sum of squares of y."""
    nseg = starts.shape[0]
    out = np.zeros((nseg, 4))
    for s in range(nseg):
        n = 0.0
        sx = 0.0
        syy = 0.0
        for i in range(starts[s], ends[s]):
            m = 1.0 if rows else cnt[i]
            n += m
            sx += m * x[i]
            syy += sy[i]
        mean = syy / n
        q = 0.0
        for i in range(starts[s], ends[s]):
            if rows:
                e = sy[i] - mean
                q += e * e
            else:
                e = sy[i] / cnt[i] - mean
                q += m2[i] + cnt[i] * e * e
        out[s, 0], out[s, 1], out[s, 2], out[s, 3] = n, sx, syy, q
    return out


# ---------------------------------------------------------------------------
# R's helpers

def _xxinv(G):
    """rdrobust's qrXXinv: the inverse from a Cholesky factorization, else
    the generalized inverse (MASS::ginv's tolerance)."""
    try:
        L = np.linalg.cholesky(G)
        Li = np.linalg.inv(L)
        return Li.T @ Li
    except np.linalg.LinAlgError:
        return np.linalg.pinv(G, rcond=math.sqrt(np.finfo(float).eps))


def _seq_by(start, stop, by):
    """R's seq(start, stop, by) for by > 0."""
    n = int((stop - start) / by + 1e-10)
    return np.minimum(start + np.arange(n + 1) * by, stop)


def _seq_len(start, stop, n):
    """R's seq(start, stop, length.out = n)."""
    by = (stop - start) / (n - 1)
    out = start + np.arange(n) * by
    out[-1] = stop
    return out


def _quantile7(xs, cnt, probs):
    """R's type-7 quantile of the rows behind sorted units (cnt None: rows)."""
    if cnt is None:
        at = lambda i: xs[i]                                  # 0-based row i
        n = xs.shape[0]
    else:
        cum = np.cumsum(cnt)
        n = int(cum[-1])
        at = lambda i: xs[int(np.searchsorted(cum, i + 1, side="left"))]
    out = np.empty(len(probs))
    for k, p in enumerate(probs):
        index = 1 + max(n - 1, 0) * p
        lo = math.floor(index)
        hi = math.ceil(index)
        q = at(lo - 1)
        xh = at(hi - 1)
        if index > lo and xh != q:
            h = index - lo
            q = (1 - h) * q + h * xh
        out[k] = q
    return out


def _J(B, V, n):
    """rdrobust's J.fun: the IMSE-optimal number of bins."""
    return math.ceil(((2 * B / V) * n) ** (1 / 3))


# ---------------------------------------------------------------------------
# the result

@dataclass
class RDPlot:
    """rd_plot's result: the bins, the global polynomial fit and the bin
    selection, as rdplot reports them; `plot()` draws them.

    Attributes
    ----------
    bins : polars.DataFrame
        One row per non-empty bin, left to right: ``bin`` (-J..-1 on the
        left, 1..J on the right), ``x_mid`` (the bin's midpoint, where
        rdplot draws it), ``x_mean``, ``y_mean``, ``x_low``, ``x_high`` (the
        bin's edges), ``n``, ``se`` (of the mean) and ``ci_low``,
        ``ci_high`` (the `ci` interval, from the t distribution).
    fit : polars.DataFrame
        The global polynomial on each side: ``side``, ``x``, ``y`` at 500
        points from the cutoff to the bandwidth's edge.
    coef : polars.DataFrame
        The polynomial's coefficients on each side (``power``, ``left``,
        ``right``), in powers of x - cutoff.
    n_bins, n_bins_imse, n_bins_mimicking : tuple of int
        The bins used, the IMSE-optimal number and the mimicking-variance
        number, (left, right) (rdplot's J, J_IMSE and J_MV).
    bin_scale, relative_scale : tuple of float
        The scale applied, and the bins used over the IMSE-optimal number
        (rdplot's scale and rscale).
    bin_length_mean, bin_length_median : tuple of float
    bin_method : str
        The bin selection, described as rdplot describes it.
    """

    outcome: str
    running: str
    cutoff: float
    bins: pl.DataFrame
    fit: pl.DataFrame
    coef: pl.DataFrame
    n_bins: tuple
    n_bins_imse: tuple
    n_bins_mimicking: tuple
    bin_scale: tuple
    relative_scale: tuple
    bin_length_mean: tuple
    bin_length_median: tuple
    polynomial_order: int
    bandwidth: tuple
    n: tuple
    n_bandwidth: tuple
    bin_method: str
    kernel: str
    ci: float
    covariate_coef: np.ndarray | None = None
    memory: str = "rows"
    donut: tuple | None = None
    donut_closed: str | None = None
    timings: dict = field(default_factory=dict)

    def plot(self, *, ax=None, ci: bool = False, shade: bool = False, title: str | None = None,
             xlabel: str | None = None, ylabel: str | None = None, color_points="C0",
             color_lines="C3"):
        """Draw the binned means, the polynomial fits and the cutoff, as
        rdplot draws them. Needs matplotlib, which is not a dependency
        (``pip install rd-stream[plot]``).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Draw here instead of on a new figure.
        ci : bool, default False
            Error bars: each bin's `ci` interval.
        shade : bool, default False
            The intervals as a shaded band.
        title, xlabel, ylabel : str, optional
            Default: no title, the running variable and the outcome.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
        else:
            fig = ax.figure
        b = self.bins
        xm, ym = b["x_mid"].to_numpy(), b["y_mean"].to_numpy()
        if shade:
            for side in (b.filter(pl.col("bin") < 0), b.filter(pl.col("bin") > 0)):
                ax.fill_between(side["x_mid"].to_numpy(), side["ci_low"].to_numpy(),
                                side["ci_high"].to_numpy(), color=color_points, alpha=0.2,
                                linewidth=0)
        if ci:
            ax.errorbar(xm, ym, yerr=[ym - b["ci_low"].to_numpy(), b["ci_high"].to_numpy() - ym],
                        fmt="none", ecolor=color_points, elinewidth=0.8, capsize=2)
        ax.scatter(xm, ym, s=14, color=color_points, zorder=3)
        for side in ("left", "right"):
            f = self.fit.filter(pl.col("side") == side)
            ax.plot(f["x"].to_numpy(), f["y"].to_numpy(), color=color_lines, linewidth=1.5)
        ax.axvline(self.cutoff, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel if xlabel is not None else self.running)
        ax.set_ylabel(ylabel if ylabel is not None else self.outcome)
        if title:
            ax.set_title(title)
        return fig


class RDPlotMulti:
    """rd_plot's result for several outcomes, from one read and sort of the
    data: an `RDPlot` per outcome, each with its own bins. Iterate over it,
    take one by outcome or position (``plots["y2"]``, ``plots[0]``), or draw
    them side by side with `plot()`; ``len()`` counts them."""

    def __init__(self, plots):
        self.all_plots = {p.outcome: p for p in plots}

    def __len__(self):
        return len(self.all_plots)

    def __iter__(self):
        return iter(self.all_plots.values())

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return list(self.all_plots.values())[key]
        return self.all_plots[key]

    @property
    def outcomes(self) -> list:
        return list(self.all_plots)

    def plot(self, *, outcomes: Sequence[str] | None = None, ncols: int = 2,
             panel_size: tuple = (5.0, 3.5), sharex: bool = True, ci: bool = False,
             shade: bool = False, xlabel: str | None = None, titles: bool = True,
             color_points="C0", color_lines="C3"):
        """Draw the outcomes as small multiples: a panel each, `ncols` to a
        row, with a shared x axis (`sharex`) and each outcome's own y axis.
        Needs matplotlib (``pip install rd-stream[plot]``).

        Parameters
        ----------
        outcomes : list of str, optional
            The outcomes to draw, in this order; default all.
        ncols : int, default 2
        panel_size : (width, height), default (5, 3.5)
            Inches per panel.
        sharex : bool, default True
        ci, shade, color_points, color_lines
            As in `RDPlot.plot`.
        xlabel : str, optional
            Default: the running variable, under the bottom row.
        titles : bool, default True
            Each panel titled by its outcome (its y axis is then unlabeled).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        names = list(self.all_plots) if outcomes is None else list(outcomes)
        unknown = [n for n in names if n not in self.all_plots]
        if unknown:
            raise ValueError(f"no plot for outcome(s) {unknown}; there are {self.outcomes}")
        if not names:
            raise ValueError("outcomes is empty")
        ncols = max(1, min(int(ncols), len(names)))
        nrows = math.ceil(len(names) / ncols)
        fig, axes = plt.subplots(nrows, ncols, sharex=sharex, squeeze=False,
                                 figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
        for i, name in enumerate(names):
            ax = axes[i // ncols][i % ncols]
            p = self.all_plots[name]
            last_row = i + ncols >= len(names)       # no panel below this one
            p.plot(ax=ax, ci=ci, shade=shade, title=name if titles else None,
                   xlabel=(xlabel if xlabel is not None else p.running) if last_row else "",
                   ylabel="" if titles else name, color_points=color_points,
                   color_lines=color_lines)
            if last_row:              # above an empty slot, the shared axis's labels too
                ax.xaxis.set_tick_params(labelbottom=True)
        for j in range(len(names), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# the entry point

def _pair(v, what, positive=True):
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        v = (v, v)
    v = tuple(v)
    if len(v) != 2 or (positive and not all(a > 0 for a in v)):
        raise ValueError(f"{what} must be a positive number or a (left, right) pair")
    return v


def rd_plot(fml: str, data, *, cutoff: float = 0.0, polynomial_order: int = 4,
            n_bins=None, bin_method: str = "esmv", bin_scale=None, kernel: str = "uniform",
            bandwidth=None, weights: str | None = None,
            covariates: Sequence[str] | str | None = None, covariates_at: str = "mean",
            support=None, mass_points: str = "adjust", ci: float = 95.0, donut=None,
            donut_closed: str = "none",
            memory: str = "auto", max_memory_gb: float | None = None,
            memory_reserve: float | None = None, workdir: str | None = None,
            n_threads: int | None = None, profile: bool = False) -> RDPlot | RDPlotMulti:
    """RD plot data, as R's rdplot computes them: the outcome's means in bins
    of the running variable on each side of the cutoff, and a global
    polynomial fit on each side. Draw it with ``.plot()``.

    Parameters
    ----------
    fml : str
        ``"y ~ x"``: the outcome and the running variable; several outcomes
        as ``"y1 + y2 ~ x"``, from one read and sort, each with its own bins
        and fit (on its own complete rows).
    data : str, path, polars.DataFrame or polars.LazyFrame
    cutoff : float, default 0
    polynomial_order : int, default 4
        The global polynomial's order (rdplot's ``p``).
    n_bins : int or (left, right) pair, optional
        The number of evenly spaced bins, instead of a selected number
        (rdplot's ``nbins``).
    bin_method : str, default "esmv"
        rdplot's ``binselect``: "es" or "qs", evenly spaced or quantile
        spaced bins, IMSE-optimal; "esmv" or "qsmv", the number of bins that
        mimics the outcome's variance. With "pr" appended ("espr", ...) the
        variance comes from a polynomial regression instead of spacings.
        With ``mass_points="adjust"`` and mass points, the "pr" variant is
        used.
    bin_scale : float or (left, right) pair, optional
        Multiplies the selected number of bins (rdplot's ``scale``); the
        result is rounded up to a whole number.
    kernel : {"uniform", "triangular", "epanechnikov"}, default "uniform"
        Weights the global polynomial fit.
    bandwidth : float or (left, right) pair, optional
        The global polynomial's window around the cutoff (rdplot's ``h``);
        default the whole range.
    weights : str, optional
        Sampling weights for the global polynomial fit (as in rdplot, the
        bins are unweighted).
    covariates : str or list of str, optional
        Covariates for the polynomial fit (rdplot's ``covs``), which then
        shows the covariate-adjusted fit at the covariates' means
        (``covariates_at="mean"``) or at zero (``"zero"``). The binned
        means are the outcome's, as in rdplot.
    support : (low, high), optional
        Extends the range the bins cover (rdplot's ``support``).
    mass_points : {"adjust", "check", "off"}
    ci : float, default 95
        The level of the bins' intervals: in percent (95), or as a fraction
        below 1 (0.95), which is the same.
    donut : float or (left, right) pair, optional
        A donut hole, as in `rd_estimate`: the observations within this
        distance of the cutoff are left out of the bins and the fits.
    donut_closed : {"none", "left", "right", "both"}, default "none"
        As in `rd_estimate`.
    memory, max_memory_gb, memory_reserve, workdir, n_threads, profile
        As in `rd_estimate`. The by-value path takes the "pr" bin methods
        only. The window path writes the rows to slabs on disk and reads them
        a slab at a time, in order (every bin method needs each side's rows
        in order), with the same results as holding them.

    Returns
    -------
    RDPlot, RDPlotMulti with several outcomes, or a Profile with
    ``profile=True``.
    """
    import time

    from . import api

    outcomes, running = api._parse(fml)
    p = int(polynomial_order)
    if p < 0:
        raise ValueError("polynomial_order must be 0 or more")
    if bin_method not in BIN_METHODS:
        raise ValueError(f"bin_method must be one of {BIN_METHODS}, not {bin_method!r}")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {tuple(_KERNELS)}, not {kernel!r}")
    if mass_points not in ("adjust", "check", "off"):
        raise ValueError("mass_points must be 'adjust', 'check' or 'off'")
    if covariates_at not in ("mean", "zero"):
        raise ValueError("covariates_at must be 'mean' or 'zero'")
    ci = float(ci)
    if 0 < ci < 1:
        ci *= 100.0
    if not 0 < ci < 100:
        raise ValueError(f"ci is a level in percent (95) or as a fraction (0.95), not {ci}")
    scale = _pair(bin_scale, "bin_scale") or (1.0, 1.0)
    nbins = _pair(n_bins, "n_bins")
    if nbins is not None:
        nbins = tuple(int(v) for v in nbins)
    h_given = _pair(bandwidth, "bandwidth")
    covs = api._as_list(covariates) if covariates is not None else []
    kern = _KERNELS[kernel]
    c = float(cutoff)

    # the spacings bin methods need the rows, unless mass points switch them to "pr"
    # (mass_points="adjust", which a running variable "auto" groups by value always has)
    by_value_ok = memory != "auto" or bin_method.endswith("pr") or mass_points == "adjust"
    # the window path's summary passes take rdrobust's options; the plot uses only its
    # counts, distinct values, covariates' cross-products and slabs
    o, _ = api._options(c, 1, None, 0, "triangular", "nearest_neighbor", 3, "adjust", None,
                        True, True, 1.0, 1.0)
    path = api._memory_path(data, running, outcomes, None, weights, None, covs,
                            "auto" if by_value_ok else "never", memory,
                            max_memory_gb, memory_reserve, n_threads, profile, "rd_plot",
                            o=o, slabs=True)
    if profile:
        return path
    if isinstance(donut, list):
        raise ValueError("rd_plot takes one donut: a radius or a (left, right) pair")
    v = api._donut_variant(c, api._donuts(donut)[0], False, donut_closed)
    with api._threads(n_threads):
        if path == "window":
            from ._plot_slabs import slab_plot_sides

            opts = api._window_options(path, o, running, outcomes, None, None, covs,
                                       max_memory_gb, memory_reserve, None, workdir)
            opts.update(slabs=True, keep=None if v.expr is None else v.expr(running))
            d = api._Data(data, running, outcomes, None, weights, None, covs, memory=path,
                          window=opts)
            read_s = d.window.stats_s + d.window.read_s

            def sides_of(y_name):
                return slab_plot_sides(d.window, y_name, covs, c, weights)
        else:
            d = api._Data(data, running, outcomes, None, weights, None, covs, memory=path)
            read_s = d.read_sort_s
            if v.mask is not None:
                d = d.subset(v.mask)

            def sides_of(y_name):
                return _plot_sides(d, y_name, covs, c)
        plots = []
        try:
            for y_name in outcomes:
                t1 = time.perf_counter()
                sides = sides_of(y_name)
                t2 = time.perf_counter()
                out = _rd_plot(sides, y_name, running, c, p, nbins, bin_method, scale, kern,
                               kernel, h_given, covs, covariates_at, support, mass_points, ci)
                out.memory = path
                out.donut = v.donut
                out.donut_closed = v.donut_closed if v.donut is not None else None
                out.timings = dict(read_sort_s=read_s, prepare_s=t2 - t1,
                                   plot_s=time.perf_counter() - t2)
                plots.append(out)
        finally:
            d.cleanup()
    api._PREPARED.clear()
    return plots[0] if len(plots) == 1 else RDPlotMulti(plots)


@dataclass
class _PlotSide:
    """One side, x ascending: units (rows, or values with counts), y sums and
    within-unit sums of squares, the rows' y (for the spacings), and the
    `Side` (nearest the cutoff first) for the polynomial fit."""
    x: np.ndarray
    cnt: np.ndarray | None
    sy: np.ndarray
    m2: np.ndarray
    y_rows: np.ndarray
    side: object
    n: int
    zsum: np.ndarray | None

    # what `_rd_plot` takes from a side (`_plot_slabs.SlabRows` gives the same from
    # slabs on disk)

    @property
    def x_lo(self) -> float:
        return float(self.x[0])

    @property
    def x_hi(self) -> float:
        return float(self.x[-1])

    @property
    def units(self) -> bool:
        """Whether the side holds distinct values with counts, not rows."""
        return self.cnt is not None

    def var_y(self) -> float:
        return _var_y(self)

    def power_sums(self, A):
        """sum x^a [1, y, y^2] over the rows, a = 0..A."""
        if self.cnt is None:
            V = np.column_stack([np.ones_like(self.x), self.sy, self.sy * self.sy])
        else:
            cntf = self.cnt.astype(np.float64)
            V = np.column_stack([cntf, self.sy, self.m2 + self.sy * self.sy / cntf])
        return _power_sums(self.x, V, A)

    def selection(self, g1, g2, var_y):
        cnt = self.cnt if self.cnt is not None else np.zeros(1, np.int64)
        return _selection_sums(self.x, cnt, self.cnt is None, g1, g2, var_y, self.y_rows)

    def quantile7(self, probs):
        return _quantile7(self.x, self.cnt, probs)

    def segments(self, e, J):
        """Per bin [e[b], e[b+1]) of the J (the last open above): which are
        non-empty, and their rows, sum x, sum y and sum of squares of y."""
        starts = np.searchsorted(self.x, e[:J], side="left")
        ends = np.append(starts[1:], self.x.shape[0])
        nz = ends > starts
        cnt = self.cnt if self.cnt is not None else np.zeros(1, np.int64)
        st = _segment_stats(self.x, cnt, self.sy, self.m2 if self.m2.size else np.zeros(1),
                            self.cnt is None, starts[nz], ends[nz])
        return nz, st


def _plot_sides(d, y_name, covs, c):
    from . import api

    if d.by_value:
        xu, cnt, SD, M2 = d.units[y_name]
        if covs:
            zc = list(range(1, SD.shape[1]))
            ZZ = (M2[:, zc][:, :, zc] + SD[:, zc, None] * SD[:, None, zc]
                  / cnt[:, None, None]).sum(axis=0)
            keep = [0] + [zc[j] for j in api._drop_collinear_gram(ZZ, covs)]
            SD, M2 = np.ascontiguousarray(SD[:, keep]), np.ascontiguousarray(M2[:, keep][:, :, keep])
        if xu.shape[0] == 0 or not (xu[0] < c < xu[-1]):
            raise ValueError("the cutoff must be strictly inside the range of the running variable")
        left, right = api._sides_from_units(xu, cnt, SD, M2, c)
        s = int(np.searchsorted(xu, c, side="left"))
        out = []
        for sl, side in ((slice(0, s), left), (slice(s, None), right)):
            out.append(_PlotSide(xu[sl], cnt[sl], np.ascontiguousarray(SD[sl, 0]),
                                 np.ascontiguousarray(M2[sl, 0, 0]), np.zeros(0), side,
                                 int(cnt[sl].sum()),
                                 SD[sl, 1:].sum(axis=0) if SD.shape[1] > 1 else None))
        return out
    x, y, w, Z = d.x, d.ys[y_name], d.w, d.Z
    ok = ~np.isnan(y)
    if Z is not None:
        ok &= ~np.isnan(Z).any(axis=1)
    if not ok.all():
        x, y = x[ok], y[ok]
        w = None if w is None else w[ok]
        Z = None if Z is None else Z[ok]
    if x.shape[0] == 0 or not (x[0] < c < x[-1]):
        raise ValueError("the cutoff must be strictly inside the range of the running variable")
    cols = [y]
    if Z is not None:
        Z = api._drop_collinear(Z, covs)
        cols += [Z[:, j] for j in range(Z.shape[1])]
    D = np.column_stack(cols)
    left, right = api._sides(x, D, w, None, 0, c, False)
    s = int(np.searchsorted(x, c, side="left"))
    out = []
    for sl, side in ((slice(0, s), left), (slice(s, None), right)):
        ys = np.ascontiguousarray(y[sl])
        out.append(_PlotSide(np.ascontiguousarray(x[sl]), None, ys, np.zeros(0), ys, side,
                             ys.shape[0], Z[sl].sum(axis=0) if Z is not None else None))
    return out


def _var_y(ps):
    n = ps.n
    if ps.cnt is None:
        return float(np.var(ps.sy, ddof=1)) if n > 1 else math.nan
    cnt = ps.cnt.astype(np.float64)
    mean_u = ps.sy / cnt
    mean = ps.sy.sum() / n
    ss = ps.m2.sum() + (cnt * (mean_u - mean) ** 2).sum()
    return ss / (n - 1) if n > 1 else math.nan


def _rd_plot(sides, y_name, running, c, p, nbins, bin_method, scale, kern, kernel, h_given,
             covs, covariates_at, support, mass_points, ci):
    from scipy import special

    from ._engine import partial_z

    L, R = sides
    x_min, x_max = L.x_lo, R.x_hi
    if support is not None:
        lo, hi = support
        x_min, x_max = min(x_min, float(lo)), max(x_max, float(hi))
    range_l, range_r = c - x_min, x_max - c
    n_l, n_r = L.n, R.n
    n = n_l + n_r
    if n < 20:
        raise ValueError("rd_plot needs at least 20 observations")

    method = bin_method
    if mass_points in ("check", "adjust"):
        mass = (1 - L.side.M_side / n_l >= 0.2) or (1 - R.side.M_side / n_r >= 0.2)
        if mass:
            warnings.warn("mass points in the running variable" +
                          ("; mass_points='adjust' selects the bins by polynomial regression"
                           if mass_points == "adjust" and not method.endswith("pr") else ""),
                          stacklevel=3)
            if mass_points == "adjust" and not method.endswith("pr"):
                method += "pr"
    if not method.endswith("pr") and L.units and nbins is None:
        raise ValueError(f"bin_method={method!r} (spacings estimators) needs the rows in order: "
                         "use memory='rows', or a 'pr' bin method")

    # the global polynomial of order p, weighted by the kernel at h
    h = h_given or (range_l, range_r)
    fits, U, Gi, Ns = [], [], [], []
    for ps, hh in ((L, h[0]), (R, h[1])):
        s = ps.side
        k = s.window(hh, kern)
        Ns.append(s.nobs_in(k))
        G = gram(s.moments_w(k, hh, kern, 2 * p), p)
        Ginv = _xxinv(G)
        Up = s.moments_D(k, hh, kern, p)
        fits.append(Ginv @ Up)
        U.append(Up)
        Gi.append(Ginv)
    gamma_z = None
    shift = 0.0
    if covs:
        Zs = []
        for ps, hh, Up, Ginv in ((L, h[0], U[0], Gi[0]), (R, h[1], U[1], Gi[1])):
            s = ps.side
            k = s.window(hh, kern)
            Zs.append(partial_z(Up, Ginv, s.zwd(k, hh, kern), 0))
        ZWZ, ZWY = Zs[0][0] + Zs[1][0], Zs[0][1] + Zs[1][1]
        gamma_z = np.linalg.pinv(ZWZ, rcond=1e-15) @ ZWY
        s_Y = np.concatenate([[1.0], -gamma_z[:, 0]])
        fits = [b @ s_Y for b in fits]
        if covariates_at == "mean":
            zbar = (L.zsum + R.zsum) / n
            shift = float(zbar @ gamma_z[:, 0])
    else:
        fits = [b[:, 0] for b in fits]
    xp_l, xp_r = _seq_len(c - h[0], c, N_PLOT), _seq_len(c, c + h[1], N_PLOT)
    yhat_l = np.polynomial.polynomial.polyval(xp_l - c, fits[0]) + shift
    yhat_r = np.polynomial.polynomial.polyval(xp_r - c, fits[1]) + shift

    # the bin selection: a fourth-order polynomial of y and y^2 in x
    kq = 4
    sel = []
    var_y = []
    for ps in (L, R):
        S = ps.power_sums(2 * kq)
        Gk = gram(S[:, 0], kq)
        Ginv = _xxinv(Gk)
        g1, g2 = Ginv @ S[:kq + 1, 1], Ginv @ S[:kq + 1, 2]
        vy = ps.var_y()
        var_y.append(vy)
        sel.append(ps.selection(g1, g2, vy))
    (mu1_l, s2_l, dxs2_l, dx2mu_l, dxdy_l, dy_l), (mu1_r, s2_r, dxs2_r, dx2mu_r, dxdy_r, dy_r) = sel
    B_es = ((range_l ** 2 / (12 * n)) * mu1_l, (range_r ** 2 / (12 * n)) * mu1_r)
    V_es_hat = ((0.5 / range_l) * dxdy_l, (0.5 / range_r) * dxdy_r)
    V_es_chk = ((1 / range_l) * dxs2_l, (1 / range_r) * dxs2_r)
    B_qs = ((n_l ** 2 / (24 * n)) * dx2mu_l, (n_r ** 2 / (24 * n)) * dx2mu_r)
    V_qs_hat = ((1 / (2 * n_l)) * dy_l, (1 / (2 * n_r)) * dy_r)
    V_qs_chk = ((1 / n_l) * s2_l, (1 / n_r) * s2_r)
    mvf = n / math.log(n) ** 2

    def dw(B, V):
        return tuple(_safe(lambda b=b, v=v: _J(b, v, n)) for b, v in zip(B, V, strict=True))

    def mv(V):
        return tuple(_safe(lambda vy=vy, v=v: math.ceil((vy / v) * mvf))
                     for vy, v in zip(var_y, V, strict=True))

    es = method.startswith("es")
    pr = method.endswith("pr")
    if es:
        V = V_es_chk if pr else V_es_hat
        J_imse, J_mv = dw(B_es, V), mv(V)
    else:
        V = V_qs_chk if pr else V_qs_hat
        J_imse, J_mv = dw(B_qs, V), mv(V)
    J_orig = J_mv if "mv" in method else J_imse
    describe = _DESCRIBE[method]
    J = [math.ceil(scale[0] * J_orig[0]), math.ceil(scale[1] * J_orig[1])]
    if nbins is not None:
        J = list(nbins)
        describe = "manually evenly spaced"
        es = True
    for j, vy in enumerate(var_y):
        if vy == 0:
            J[j] = 1
            warnings.warn("not enough variability in the outcome on the "
                          f"{'left' if j == 0 else 'right'} of the cutoff: one bin", stacklevel=3)
    rscale = (J[0] / J_imse[0], J[1] / J_imse[1])

    # the bins
    if es:
        edges = (_seq_by(x_min, c, range_l / J[0]), _seq_by(c, x_max, range_r / J[1]))
    else:
        edges = tuple(ps.quantile7(_seq_by(0.0, 1.0, 1.0 / J[j]))
                      for j, ps in enumerate((L, R)))
    rows = []
    lengths = []
    tq = []
    for j, (ps, e) in enumerate(zip((L, R), edges, strict=True)):
        Jj = J[j]
        e = e[:Jj + 1]
        lengths.append(np.diff(e))
        nz, st = ps.segments(e, Jj)
        ids = np.arange(1, Jj + 1)[nz]
        nb = st[:, 0]
        sd = np.where(nb > 1, np.sqrt(st[:, 3] / np.maximum(nb - 1, 1)), 0.0)
        se = sd / np.sqrt(nb)
        # the t quantile (scipy.special: importing scipy.stats takes half a second)
        quant = special.stdtrit(np.maximum(nb - 1, 1), 1 - (1 - ci / 100) / 2)
        ym = st[:, 2] / nb
        rows.append(pl.DataFrame({
            "bin": (ids - Jj - 1) if j == 0 else ids,
            "x_mid": 0.5 * (e[:-1] + e[1:])[nz], "x_mean": st[:, 1] / nb, "y_mean": ym,
            "x_low": e[:-1][nz], "x_high": e[1:][nz], "n": nb.astype(np.int64), "se": se,
            "ci_low": ym - quant * se, "ci_high": ym + quant * se}))
        tq.append(None)
    bins = pl.concat(rows)
    fit = pl.DataFrame({"side": ["left"] * N_PLOT + ["right"] * N_PLOT,
                        "x": np.concatenate([xp_l, xp_r]), "y": np.concatenate([yhat_l, yhat_r])})
    coef = pl.DataFrame({"power": np.arange(p + 1), "left": fits[0], "right": fits[1]})
    return RDPlot(
        outcome=y_name, running=running, cutoff=c, bins=bins, fit=fit, coef=coef,
        n_bins=tuple(J), n_bins_imse=tuple(J_imse), n_bins_mimicking=tuple(J_mv),
        bin_scale=tuple(float(v) for v in scale), relative_scale=rscale,
        bin_length_mean=tuple(float(np.mean(v)) for v in lengths),
        bin_length_median=tuple(float(np.median(v)) for v in lengths), polynomial_order=p,
        bandwidth=tuple(float(v) for v in h), n=(n_l, n_r), n_bandwidth=tuple(Ns),
        bin_method=describe, kernel=kernel, ci=float(ci), covariate_coef=gamma_z)


def _safe(f):
    try:
        return f()
    except (ZeroDivisionError, OverflowError, ValueError):
        return 1
