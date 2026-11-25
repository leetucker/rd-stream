"""Quickstart: the incumbency advantage in U.S. Senate elections.

Does winning a Senate seat raise the party's vote share in the next election
for that seat? A Democrat who barely won and one who barely lost are alike in
all but the result, so the jump in the next vote share at a margin of 0 is
the effect of winning.

    python examples/quickstart.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from senate_data import load_senate  # noqa: E402

import rd_stream as rd  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"
senate = load_senate()

# ---- the RD plot: binned means and a global polynomial on each side
plot = rd.rd_plot("vote ~ margin", senate)
print(f"bins: {plot.n_bins[0]} left, {plot.n_bins[1]} right ({plot.bin_method})")
fig = plot.plot(xlabel="Democratic margin of victory (percentage points)",
                ylabel="Democratic vote share, next election")
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURES / "quickstart_rdplot.svg", bbox_inches="tight")

# ---- the estimate, with rdrobust's defaults
fit = rd.rd_estimate("vote ~ margin", senate)
print(fit.summary_text())
lo, hi = fit.robust_ci
print(f"robust 95% interval: [{lo:.3f}, {hi:.3f}]")

# ---- the bandwidth selectors
print(rd.rd_bandwidth("vote ~ margin", senate, all_selectors=True).table)

# ---- clustered by state, and with predetermined covariates
by_state = rd.rd_estimate("vote ~ margin", senate, vcov={"CRV1": "state"})
covs = rd.rd_estimate("vote ~ margin", senate, covariates=["vote_lag1", "vote_lag2"])

# ---- a table for a paper: the conventional estimate with robust inference
rd.etable([fit, by_state, covs], rows="conventional_robust", type="md",
          model_stats=["N"], labels={"Conventional, robust inference": "Winning the seat"})
