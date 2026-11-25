"""Results of a regression discontinuity fit."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from hdfe_stream.core.results_base import RegressionMulti, RegressionResult
from scipy import special

ROWS = ("Conventional", "Bias-corrected", "Robust")
# the conventional estimate with the robust standard error, test and interval (the
# interval and test are the robust ones, about the bias-corrected estimate), as papers
# often report it
CONVENTIONAL_ROBUST = "Conventional, robust inference"
_ROW_NAMES = {"conventional": 0, "bias-corrected": 1, "bias_corrected": 1, "robust": 2,
              CONVENTIONAL_ROBUST.lower(): 3, "conventional_robust": 3}


def _row_index(name) -> int:
    i = _ROW_NAMES.get(str(name).strip().lower())
    if i is None:
        raise ValueError(f"rows are {ROWS + (CONVENTIONAL_ROBUST,)} (or 'conventional_robust'), "
                         f"not {name!r}")
    return i


@dataclass(kw_only=True)
class RDResult(RegressionResult):
    """One outcome's RD estimate, as rdrobust reports it: three rows.

    - ``Conventional``: the local-polynomial estimate with its conventional
      standard error.
    - ``Bias-corrected``: the bias-corrected estimate with the conventional
      standard error.
    - ``Robust``: the bias-corrected estimate with the robust standard error
      (Calonico, Cattaneo and Titiunik 2014). Its interval is the one to report.

    `beta` holds the three estimates and `vcov` their variances on the
    diagonal. The off-diagonal entries are zero, because rdrobust does not
    compute the covariances. A Wald test across the rows is not meaningful.

    `with_rows` picks the rows to report (in `tidy`, `confint`, `summary`
    and `etable`), and adds a fourth, ``"Conventional, robust inference"``:
    the conventional estimate shown with the robust standard error, and the
    robust test and interval (which are about the bias-corrected estimate),
    as papers often report rdrobust's results.

    Attributes beyond `RegressionResult`'s
    ---------------------------------------
    running : str
        The running variable.
    cutoff : float
    bandwidth, bias_bandwidth : tuple of float
        h and b, (left, right).
    bandwidth_selector : str
        The selector ("mserd", ...), or "manual".
    polynomial_order, bias_order, derivative : int
    kernel : str
    n_left, n_right : int
        Observations on each side.
    n_bandwidth : tuple of int
        Observations within h on each side.
    n_bias_bandwidth : tuple of int
        Observations within b on each side.
    n_distinct : tuple of int
        Distinct values of the running variable on each side.
    side_estimates : dict
        The intercepts (or derivatives) on each side: conventional and
        bias-corrected.
    donut : (left, right) or None
        The donut hole's radii, when observations near the cutoff were left
        out.
    collapsed : bool
        Whether the fit ran on one row per distinct value of the running
        variable (which gives the same results; see `memory`).
    memory : str
        How the data were held: "rows" (the sorted rows), "by_value"
        (grouped by the running variable as they were read) or "window"
        (the rows near the cutoff, or slabs of them on disk).
    """

    running: str
    cutoff: float
    bandwidth: tuple
    bias_bandwidth: tuple
    bandwidth_selector: str
    polynomial_order: int
    bias_order: int
    derivative: int
    kernel: str
    n_left: int
    n_right: int
    n_bandwidth: tuple
    n_bias_bandwidth: tuple
    n_distinct: tuple
    side_estimates: dict = field(default_factory=dict)
    collapsed: bool = False
    memory: str = "rows"
    treatment: str | None = None
    first_stage: dict | None = None
    donut: tuple | None = None
    timings: dict = field(default_factory=dict)
    # per row, the value its test and interval are about (None: the estimate itself)
    inference_center: np.ndarray | None = field(default=None, repr=False)

    _normal_inference = True

    def _center(self) -> np.ndarray:
        return np.asarray(self.beta if self.inference_center is None else self.inference_center,
                          dtype=float)

    def with_rows(self, rows: str | Sequence[str]) -> RDResult:
        """This fit, reporting only `rows`, in that order: any of
        "Conventional", "Bias-corrected", "Robust" and "Conventional, robust
        inference" (or "conventional_robust"; the names are case-insensitive).
        The new result's `tidy`, `confint`, `summary` and `etable` show those
        rows. It takes a fit with rdrobust's three rows."""
        if list(self.coefnames) != list(ROWS):
            raise ValueError("with_rows takes a fit with rdrobust's three rows, not one "
                             "whose rows were already chosen")
        rows = [rows] if isinstance(rows, str) else list(rows)
        if not rows:
            raise ValueError("rows is empty")
        idx = [_row_index(r) for r in rows]
        if len(set(idx)) != len(idx):
            raise ValueError(f"rows repeats a row: {rows}")
        b, v = np.asarray(self.beta, float), np.diag(np.asarray(self.vcov, float))
        names = list(ROWS) + [CONVENTIONAL_ROBUST]
        beta = np.array([b[0] if i == 3 else b[i] for i in idx])
        var = np.array([v[2] if i == 3 else v[i] for i in idx])
        center = np.array([b[2] if i == 3 else b[i] for i in idx])
        return dataclasses.replace(self, beta=beta, vcov=np.diag(var),
                                   coefnames=[names[i] for i in idx],
                                   inference_center=None if np.array_equal(center, beta)
                                   else center)

    def tidy(self) -> pl.DataFrame:
        out = super().tidy()
        if self.inference_center is None:
            return out
        se = self.se
        t = np.divide(self._center(), se, out=np.full_like(se, np.nan), where=se > 0)
        return out.with_columns(pl.Series("t value", t),
                                pl.Series("Pr(>|t|)", 2 * special.ndtr(-np.abs(t))))

    def confint(self, alpha: float = 0.05, keep=None, drop=None, exact_match: bool = False,
                joint: bool = False, reps: int = 10_000, seed: int | None = None) -> pl.DataFrame:
        out = super().confint(alpha, keep, drop, exact_match, joint, reps, seed)
        if self.inference_center is None:
            return out
        shift = dict(zip(self.coefnames, self._center() - np.asarray(self.beta, float),
                         strict=True))
        d = pl.Series([shift[c] for c in out["Coefficient"]])
        lo, hi = out.columns[1], out.columns[2]
        return out.with_columns(pl.col(lo) + d, pl.col(hi) + d)

    @property
    def estimate(self) -> float:
        """The conventional point estimate (the first row's estimate, once
        `with_rows` has chosen the rows)."""
        return float(self.beta[0])

    @property
    def robust_ci(self) -> tuple:
        """The robust bias-corrected 95% interval."""
        names = list(self.coefnames)
        row = next((n for n in ("Robust", CONVENTIONAL_ROBUST) if n in names), None)
        if row is None:
            raise ValueError("this result's rows have no robust interval")
        lo, hi = self.confint().row(names.index(row))[1:3]
        return (lo, hi)

    def _summary_lines(self) -> list[str]:
        h, b = self.bandwidth, self.bias_bandwidth
        fmt = lambda v: f"{v[0]:.6g}" if v[0] == v[1] else f"{v[0]:.6g} (left), {v[1]:.6g} (right)"
        lines = [
            f"running variable: {self.running}, cutoff {self.cutoff:g}",
            f"obs: {self.n_obs:,} ({self.n_left:,} left, {self.n_right:,} right); "
            f"within h: {self.n_bandwidth[0]:,} left, {self.n_bandwidth[1]:,} right",
            f"bandwidth h: {fmt(h)}; bias bandwidth b: {fmt(b)} ({self.bandwidth_selector})",
            f"polynomial order {self.polynomial_order}, bias order {self.bias_order}"
            + (f", derivative {self.derivative}" if self.derivative else "")
            + f"; {self.kernel} kernel; vcov: {self.vcov_type}",
        ]
        if self.treatment:
            fs = self.first_stage
            lines.append(f"fuzzy: take-up {self.treatment}; first-stage jump {fs['tau_cl']:.6g} "
                         f"(se {fs['se_cl']:.4g}; robust se {fs['se_rb']:.4g})")
        if self.donut is not None:
            lines.append(f"donut hole: {self.donut[0]:g} left, {self.donut[1]:g} right of the "
                         f"cutoff left out")
        if self.n_clusters:
            lines.append(f"clusters: {dict(self.n_clusters)}")
        if self.weights:
            lines.append(f"weights: {self.weights}")
        return lines

    def _summary_fields(self) -> dict:
        out = super()._summary_fields()
        out.update(running=self.running, cutoff=self.cutoff, bandwidth=list(self.bandwidth),
                   bias_bandwidth=list(self.bias_bandwidth),
                   bandwidth_selector=self.bandwidth_selector,
                   polynomial_order=self.polynomial_order, bias_order=self.bias_order,
                   derivative=self.derivative, kernel=self.kernel, n_left=self.n_left,
                   n_right=self.n_right, n_bandwidth=list(self.n_bandwidth),
                   n_bias_bandwidth=list(self.n_bias_bandwidth),
                   n_distinct=list(self.n_distinct), collapsed=self.collapsed, memory=self.memory,
                   treatment=self.treatment, first_stage=self.first_stage,
                   donut=None if self.donut is None else list(self.donut))
        return out

    def _pyfixest_attrs(self) -> dict:
        out = super()._pyfixest_attrs()
        out["_method"] = "rd"
        if self.inference_center is not None:
            tab, ci = self.tidy(), self.confint()
            out.update(_tstat=tab["t value"].to_numpy(), _pvalue=tab["Pr(>|t|)"].to_numpy(),
                       _conf_int=np.vstack([ci[:, 1].to_numpy(), ci[:, 2].to_numpy()]))
        return out


class RDMulti(RegressionMulti):
    """Several RD fits from one read and sort of the data (several outcomes,
    or several bandwidths)."""

    def with_rows(self, rows: str | Sequence[str]) -> RDMulti:
        """Every fit with only `rows` (see `RDResult.with_rows`)."""
        return RDMulti(r.with_rows(rows) for r in self)

    def estimates(self) -> pl.DataFrame:
        """One row per fit: outcome, bandwidths, estimate, robust interval
        (from the fits' three rows)."""
        rows = []
        for r in self:
            if list(r.coefnames) != list(ROWS):
                raise ValueError("estimates() takes fits with rdrobust's three rows")
            lo, hi = r.robust_ci
            rows.append(dict(outcome=r.depvar, h_left=r.bandwidth[0], h_right=r.bandwidth[1],
                             b_left=r.bias_bandwidth[0], b_right=r.bias_bandwidth[1],
                             estimate=r.estimate, se=float(r.se[0]),
                             robust_se=float(r.se[2]), robust_low=lo, robust_high=hi,
                             n_bandwidth=sum(r.n_bandwidth)))
        return pl.DataFrame(rows)


def etable(models, rows: str | Sequence[str] | None = None, **kwargs):
    """A regression table (hdfe-stream's `etable`, which is pyfixest's) of RD
    fits, with other models if wanted.

    Parameters
    ----------
    models : RDResult, RDMulti, or a list of them and other models
    rows : str or list of str, optional
        The rows to show for each RD fit, in this order (see
        `RDResult.with_rows`): any of "Conventional", "Bias-corrected",
        "Robust" and "Conventional, robust inference" (or
        "conventional_robust"). Default: rdrobust's three rows.
    **kwargs
        Passed to ``pf.etable``: ``type``, ``model_stats``, ``labels``, ...

    Examples
    --------
    >>> etable([fit1, fit2], rows="Robust")
    >>> etable(fits, rows="conventional_robust", labels={"Conventional, robust inference": "RD"})
    """
    from hdfe_stream.core.reporting import etable as _etable

    if rows is not None:
        many = models if isinstance(models, (list, tuple)) else [models]
        models = [m.with_rows(rows) if isinstance(m, (RDResult, RDMulti)) else m for m in many]
    return _etable(models, **kwargs)
