"""Robustness checks for the Senate estimate: other bandwidths, donut holes,
placebo cutoffs and a placebo outcome (Cattaneo, Idrobo and Titiunik 2020,
chapter 5).

    python examples/robustness.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import polars as pl  # noqa: E402
from senate_data import load_senate  # noqa: E402

import rd_stream as rd  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"
senate = load_senate()
pl.Config.set_tbl_rows(20)
pl.Config.set_float_precision(3)

# ---- the estimate at other bandwidths, from one sort
grid = rd.rd_estimate("vote ~ margin", senate, bandwidth=[5.0, 10.0, 15.0, 20.0, 30.0])
print(grid.estimates().select("h_left", "estimate", "robust_low", "robust_high",
                              "n_bandwidth"))

# ---- donut holes: leave out the closest elections, then fit again
donuts = rd.rd_estimate("vote ~ margin", senate, donut=[0.0, 0.5, 1.0, 2.0])
print(donuts.estimates().select("h_left", "estimate", "robust_low", "robust_high",
                                "n_bandwidth")
      .with_columns(donut=pl.Series([0.0, 0.5, 1.0, 2.0])))

# ---- placebo cutoffs: no jump where nothing changes
placebo = rd.rd_placebo("vote ~ margin", senate, placebo_cutoffs=[-10, -5, -2.5, 2.5, 5, 10])
print(placebo.table.select("cutoff", "kind", "side", "estimate", "robust_low", "robust_high",
                           "robust_p"))
print(placebo.table.select("cutoff", "h_left", "n_left", "n_right"))

# the same with every cutoff at the true cutoff's bandwidth
h = rd.rd_estimate("vote ~ margin", senate).bandwidth[0]
common = rd.rd_placebo("vote ~ margin", senate, placebo_cutoffs=[-10, -5, -2.5, 2.5, 5, 10],
                       bandwidth=h)
print(common.table.select("cutoff", "kind", "estimate", "robust_low", "robust_high",
                          "robust_p", "n_left", "n_right"))
fig = common.plot(xlabel="cutoff (Democratic margin, percentage points)")
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES / "robustness_placebo.svg", bbox_inches="tight")

# ---- a placebo outcome: the previous election's vote share, set before the cutoff
lag = rd.rd_estimate("vote_lag1 ~ margin", senate)
lo, hi = lag.robust_ci
print(f"previous vote share: {lag.estimate:.3f} (robust interval [{lo:.3f}, {hi:.3f}])")
