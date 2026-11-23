"""Results of a regression discontinuity fit."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
from hdfe_stream.core.results_base import RegressionMulti, RegressionResult

ROWS = ("Conventional", "Bias-corrected", "Robust")


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
    collapsed : bool
        Whether the fit ran on one row per distinct value of the running
        variable (exact; see the documentation).
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
    treatment: str | None = None
    first_stage: dict | None = None
    timings: dict = field(default_factory=dict)

    _normal_inference = True

    @property
    def estimate(self) -> float:
        """The conventional point estimate."""
        return float(self.beta[0])

    @property
    def robust_ci(self) -> tuple:
        """The robust bias-corrected 95% interval."""
        lo, hi = self.confint().row(2)[1:3]
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
                   n_distinct=list(self.n_distinct), collapsed=self.collapsed,
                   treatment=self.treatment, first_stage=self.first_stage)
        return out

    def _pyfixest_attrs(self) -> dict:
        out = super()._pyfixest_attrs()
        out["_method"] = "rd"
        return out


class RDMulti(RegressionMulti):
    """Several RD fits from one read and sort of the data (several outcomes,
    or several bandwidths)."""

    def estimates(self) -> pl.DataFrame:
        """One row per fit: outcome, bandwidths, estimate, robust interval."""
        rows = []
        for r in self:
            lo, hi = r.robust_ci
            rows.append(dict(outcome=r.depvar, h_left=r.bandwidth[0], h_right=r.bandwidth[1],
                             b_left=r.bias_bandwidth[0], b_right=r.bias_bandwidth[1],
                             estimate=r.estimate, se=float(r.se[0]),
                             robust_se=float(r.se[2]), robust_low=lo, robust_high=hi,
                             n_bandwidth=sum(r.n_bandwidth)))
        return pl.DataFrame(rows)
