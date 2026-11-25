"""`rd_placebo()`: estimates at placebo cutoffs, where there should be no
discontinuity.

At a placebo cutoff below the true one, only the observations below the
true cutoff are used, and above it only those above, so that the true
discontinuity cannot enter a placebo estimate (Cattaneo, Idrobo and
Titiunik 2020, section 5.2). Each placebo fit is otherwise an ordinary
`rd_estimate` fit at that cutoff, with its own bandwidth. The rows and
by-value paths read the data once for every cutoff; the window path reads
it once per cutoff.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from . import api
from .results import RDMulti


@dataclass
class PlaceboResult:
    """rd_placebo's result.

    Attributes
    ----------
    table : polars.DataFrame
        One row per outcome and cutoff: ``outcome``, ``cutoff``, ``kind``
        ("placebo", or "actual" for the true cutoff), ``side`` (the side of
        the true cutoff whose observations were used), ``estimate`` (the
        conventional estimate), ``se``, ``robust_se``, ``robust_low``,
        ``robust_high`` and ``robust_p`` (the robust bias-corrected interval
        and p-value), ``h_left``, ``h_right``, ``n_left``, ``n_right`` (the
        observations within h) and ``note`` (why a cutoff has no estimate).
    fits : RDMulti
        The fits, as `rd_estimate` returns them.
    cutoff : float
        The true cutoff.
    running : str
    """

    table: pl.DataFrame
    fits: RDMulti
    cutoff: float
    running: str

    @property
    def outcomes(self) -> list:
        return list(dict.fromkeys(self.table["outcome"].to_list()))

    def plot(self, *, outcomes: Sequence[str] | None = None, ncols: int = 2,
             panel_size: tuple = (5.0, 3.5), xlabel: str | None = None,
             color_placebo="C0", color_actual="C3"):
        """The estimates against the cutoff, with their robust intervals: a
        panel per outcome, the true cutoff's estimate in `color_actual`.
        Needs matplotlib (``pip install rd-stream[plot]``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        names = self.outcomes if outcomes is None else list(outcomes)
        unknown = [n for n in names if n not in self.outcomes]
        if unknown or not names:
            raise ValueError(f"outcomes must be among {self.outcomes}, not {names}")
        ncols = max(1, min(int(ncols), len(names)))
        nrows = math.ceil(len(names) / ncols)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
        for i, name in enumerate(names):
            ax = axes[i // ncols][i % ncols]
            t = self.table.filter((pl.col("outcome") == name) & pl.col("estimate").is_not_nan())
            for kind, color in (("placebo", color_placebo), ("actual", color_actual)):
                k = t.filter(pl.col("kind") == kind)
                if not k.height:
                    continue
                c, est = k["cutoff"].to_numpy(), k["estimate"].to_numpy()
                ax.errorbar(c, est, yerr=[est - k["robust_low"].to_numpy(),
                                          k["robust_high"].to_numpy() - est],
                            fmt="o", color=color, ecolor=color, elinewidth=0.9, capsize=3,
                            markersize=4, label=kind)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.axvline(self.cutoff, color="gray", linewidth=0.8, linestyle="--")
            ax.set_title(name)
            if i == 0:
                ax.legend(frameon=False, fontsize="small")
            ax.set_xlabel(xlabel if xlabel is not None else f"cutoff ({self.running})")
        for j in range(len(names), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.tight_layout()
        return fig


def rd_placebo(fml: str, data, placebo_cutoffs: Sequence[float], *, cutoff: float = 0.0,
               include_cutoff: bool = True, bandwidth="mserd",
               bias_bandwidth=None, bandwidth_ratio: float | None = None,
               polynomial_order: int = 1, bias_order: int | None = None, derivative: int = 0,
               kernel: str = "triangular", vcov: str | dict = "nearest_neighbor",
               n_neighbors: int = 3, weights: str | None = None, mass_points: str = "adjust",
               min_distinct: int | None = None, restrict_bandwidth: bool = True,
               standardize: bool = True, regularization: float = 1.0, scale: float = 1.0,
               treatment: str | None = None, sharp_bandwidth: bool = False,
               covariates: Sequence[str] | str | None = None, memory: str = "auto",
               max_memory_gb: float | None = None, memory_reserve: float | None = None,
               workdir: str | None = None, n_threads: int | None = None) -> PlaceboResult:
    """RD estimates at placebo cutoffs, where the outcome should not jump.

    A placebo cutoff below `cutoff` is fitted on the observations below it
    only, and one above on those above it only (x >= cutoff), so that the
    true discontinuity cannot leak into a placebo estimate. Each fit selects
    its own bandwidth unless one is given. With `include_cutoff`, the
    estimate at the true cutoff, on every observation, is included too.

    Parameters
    ----------
    fml : str
        ``"y ~ x"``, or several outcomes as ``"y1 + y2 ~ x"``.
    data : str, path, polars.DataFrame or polars.LazyFrame
    placebo_cutoffs : list of float
        The placebo cutoffs; none may equal `cutoff`. A cutoff too near the
        edge of its side's observations (or with too few distinct values
        between) has no estimate, and the table's ``note`` says why.
    cutoff : float, default 0
        The true cutoff.
    include_cutoff : bool, default True
        Also estimate at the true cutoff.
    bandwidth : str or float or (left, right) pair, default "mserd"
        A selector (each fit selects its own) or a bandwidth for every fit.
    The other options are `rd_estimate`'s.

    Returns
    -------
    PlaceboResult
        Its `table` has one row per outcome and cutoff; `plot()` draws the
        estimates and their robust intervals against the cutoff.
    """
    outcomes, running = api._parse(fml)
    api._PREPARED.clear()
    o, cluster = api._options(cutoff, polynomial_order, bias_order, derivative, kernel, vcov,
                              n_neighbors, mass_points, min_distinct, restrict_bandwidth,
                              standardize, regularization, scale)
    if isinstance(bandwidth, list):
        raise ValueError("rd_placebo takes one bandwidth (or selector) for every fit")
    cuts = [float(v) for v in placebo_cutoffs]
    if not cuts and not include_cutoff:
        raise ValueError("placebo_cutoffs is empty")
    c = o.cutoff
    if any(v == c for v in cuts):
        raise ValueError(f"a placebo cutoff equals the true cutoff ({c:g})")
    if len(set(cuts)) != len(cuts):
        raise ValueError("placebo_cutoffs repeats a cutoff")
    variants = []
    if include_cutoff:
        variants.append(api._Variant(c, tag=f"cutoff={c:g} (actual)"))
    for v in sorted(cuts):
        variants.append(_side_variant(c, v))
    collapse = "never" if memory == "rows" else "auto"
    covs = api._as_list(covariates) if covariates is not None else []
    path = api._memory_path(data, running, outcomes, cluster, weights, treatment, covs, collapse,
                            memory, max_memory_gb, memory_reserve, n_threads, False,
                            "rd_placebo", o=o,
                            hint=api._largest_bandwidth([bandwidth], bias_bandwidth,
                                                        bandwidth_ratio))
    skipped = []
    fits = api._fit_variants(data, running, outcomes, o, cluster, weights, treatment, covs,
                             path, max_memory_gb, memory_reserve, workdir, n_threads,
                             [bandwidth], bias_bandwidth, bandwidth_ratio, collapse,
                             sharp_bandwidth, variants, skipped=skipped)
    rows = []
    for r in fits:
        lo, hi = r.robust_ci
        tab = r.tidy()
        rows.append(dict(outcome=r.depvar, cutoff=r.cutoff,
                         kind="actual" if r.cutoff == c else "placebo",
                         side=_side(c, r.cutoff), estimate=float(r.beta[0]),
                         se=float(r.se[0]), robust_se=float(r.se[2]), robust_low=lo,
                         robust_high=hi, robust_p=float(tab["Pr(>|t|)"][2]),
                         h_left=r.bandwidth[0], h_right=r.bandwidth[1],
                         n_left=r.n_bandwidth[0], n_right=r.n_bandwidth[1], note=None))
    nan = float("nan")
    for v, y, msg in skipped:
        rows.append(dict(outcome=y, cutoff=v.cutoff,
                         kind="actual" if v.cutoff == c else "placebo",
                         side=_side(c, v.cutoff), estimate=nan, se=nan, robust_se=nan,
                         robust_low=nan, robust_high=nan, robust_p=nan, h_left=nan,
                         h_right=nan, n_left=0, n_right=0, note=msg))
    order = {y: i for i, y in enumerate(outcomes)}
    rows.sort(key=lambda d: (order[d["outcome"]], d["cutoff"]))
    table = pl.DataFrame(rows, schema_overrides={"note": pl.Utf8, "side": pl.Utf8})
    return PlaceboResult(table, RDMulti(fits), c, running)


def _side(c, v):
    return None if v == c else ("left" if v < c else "right")


def _side_variant(c, v):
    """A placebo cutoff's fit: the observations on its side of the true
    cutoff only."""
    if v < c:
        def mask(x):
            return x < c

        def expr(running):
            return pl.col(running).cast(pl.Float64) < c
    else:
        def mask(x):
            return x >= c

        def expr(running):
            return pl.col(running).cast(pl.Float64) >= c
    return api._Variant(v, mask, expr, f"cutoff={v:g} (placebo)")


