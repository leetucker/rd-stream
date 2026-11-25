# rd-stream

Regression discontinuity estimates with robust bias-corrected inference, on
data with millions to hundreds of millions of rows. It computes what R's
rdrobust package computes (local-polynomial estimates, the data-driven
bandwidth selectors, the conventional and robust standard errors, and RD
plots), and agrees with it to 1e-8 or better.

It reads Parquet or a Polars frame and sorts the running variable once. Each
bandwidth step and the final fit are then compiled, threaded passes over the
observations within that step's bandwidth. When the running variable takes
few distinct values (ages, test scores, dates), the fit runs on one row per
value, which is exact.

```python
from rd_stream import rd_estimate, rd_placebo, rd_plot

fit = rd_estimate("vote ~ margin", "senate.parquet")
fit.summary()

# several outcomes from one sort; a fuzzy design; covariates with clusters
rd_estimate("y1 + y2 + y3 ~ score", "data/*.parquet", cutoff=60)
rd_estimate("earnings ~ age", data, cutoff=65, treatment="enrolled")
rd_estimate("y ~ x", data, covariates=["z1", "z2"], vcov={"CRV2": "school"})

# a bandwidth grid, from one sort
rd_estimate("y ~ x", data, bandwidth=[0.1, 0.15, 0.2, 0.25]).estimates()

# robustness: donut holes, and placebo cutoffs on each side of the true one
rd_estimate("y ~ x", data, donut=[0, 0.01, 0.02])
rd_placebo("y ~ x", data, placebo_cutoffs=[-0.2, -0.1, 0.1, 0.2]).plot()

# binned means and polynomial fits, as rdplot draws them
rd_plot("vote ~ margin", "senate.parquet").plot(ci=True)
```

**Version 0.1.0.dev0**, in development.

## What it does

- `rd_estimate` covers sharp and fuzzy designs, regression kinks (`derivative=1`) and
  covariate adjustment.
  - It offers rdrobust's three kernels, any polynomial and bias order, and all ten
    bandwidth selectors, or bandwidths you give.
  - Variances: nearest-neighbor, HC0 to HC3, or clustered (CRV1, CRV2 or CRV3, written
    as in hdfe-stream: `vcov={"CRV1": "school"}`). It takes sampling weights.
  - It has rdrobust's mass-point adjustment, minimum-support check and bandwidth
    restriction.
  - A list of bandwidths costs little more than one: each side's sums come from block
    power sums, and the nearest-neighbor residuals are computed once.
  - `donut=` leaves out the observations within a distance of the cutoff (a radius, a
    left/right pair, or a list of them for one fit each).
- `rd_bandwidth` gives rdbwselect's bandwidths, one selector or all ten.
- `rd_placebo` estimates at placebo cutoffs, each from the observations on its side of
  the true cutoff only, and draws the estimates with their robust intervals.
- `rd_plot` gives rdplot's binned means, with its eight ways of choosing the bins, and the
  global polynomial fits. `.plot()` draws them with matplotlib. Several outcomes
  (`"y1 + y2 ~ x"`) share one read and sort, and `.plot()` draws them as a grid of panels.
- Results have hdfe-stream's reporting: `summary()`, `tidy()`, `confint()` and
  `summary_json()`. With several outcomes or bandwidths, `estimates()` gives one row
  per fit.
- A fit reports rdrobust's three rows (conventional, bias-corrected, robust).
  `etable(fits, rows=...)` and `fit.with_rows(...)` choose which to show, including
  `"conventional_robust"`: the conventional estimate with the robust standard error,
  test and interval, as many papers report it.
- `memory=` says how the data are held; the default, `"auto"`, picks the first that
  applies:
  - `"by_value"`: when the running variable has mass points (ages, scores, dates) and
    the fit has no clusters or weights, the data are grouped by its value as they are
    read. Memory grows with the number of distinct values, and the fit runs on one row
    per value, with the same results. At 10 million rows it is about 7 times faster than
    holding the rows.
  - `"rows"`: the rows are read and sorted once, when they fit the memory budget.
  - `"window"`: the full sample's statistics come from streaming passes, and only the
    rows near the cutoff are held (`rd_plot` reads its rows a slab at a time from disk,
    since its bins need each side's rows in order). When a step's window doesn't fit either, the rows
    are written to disk in slabs by distance from the cutoff, and each step reads them
    a slab at a time. The results are the same as with `"rows"`. 300 million rows took
    about 4.3 minutes under a 2 GB budget (4 threads).
- `max_memory_gb=` sets the budget, `workdir=` says where the slabs go, and
  `profile=True` reports the memory and disk a fit would need without running it.

## When to use it, and when not

Use it when an RD fit is slow or doesn't fit in memory. Two cases stand out:
- **Large samples.** R's rdrobust and diff-diff take minutes at tens of millions of
  rows.
- **A running variable with many ties.** rdrobust's nearest-neighbor variance slows
  sharply when many observations share each value.

For a few thousand observations, rdrobust is already fast. rd-stream agrees with it but
won't save you time there. rd-stream covers rdrobust, rdbwselect and rdplot; the rest of
the rdpackages family (rddensity, rdlocrand, rdmulti, rdpower) isn't covered. Fitting
from slabs on disk is about four times slower than in memory, so it's worth it only when
the rows don't fit. The documentation's conventions page maps rdrobust's option names to rd-stream's.

## Install

```bash
pip install -e .
```

Python 3.10 or later. It depends on hdfe-stream, stream-limits, Polars,
NumPy, numba, SciPy and pyarrow. Plots need matplotlib (`pip install -e ".[plot]"`).

A sibling of [hdfe-stream](https://github.com/leetucker/hdfe-stream),
[diff-stream](https://github.com/leetucker/diff-stream),
[match-stream](https://github.com/leetucker/match-stream) and
[penalized-stream](https://github.com/leetucker/penalized-stream).

## License

MIT.
