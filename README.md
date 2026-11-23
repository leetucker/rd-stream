# rd-stream

Regression discontinuity estimates with robust bias-corrected inference, on
data with millions to hundreds of millions of rows. It computes what R's
rdrobust computes (local-polynomial estimates, the data-driven bandwidth
selectors, and the conventional and robust standard errors), and agrees
with it to 1e-8 or better.

It reads Parquet or a Polars frame and sorts the running variable once. Each
bandwidth step and the final fit are then compiled, threaded passes over the
observations within that step's bandwidth. When the running variable takes
few distinct values (ages, test scores, dates), the fit runs on one row per
value, which is exact.

```python
from rd_stream import rd_estimate

fit = rd_estimate("vote ~ margin", "senate.parquet")
fit.summary()

# several outcomes from one sort; a fuzzy design; covariates
rd_estimate("y1 + y2 + y3 ~ score", "data/*.parquet", cutoff=60)
rd_estimate("earnings ~ age", data, cutoff=65, treatment="enrolled")
rd_estimate("y ~ x", data, covariates=["z1", "z2"], cluster="school")

# a bandwidth grid, from one sort
rd_estimate("y ~ x", data, bandwidth=[0.1, 0.15, 0.2, 0.25]).estimates()
```

**Version 0.1.0.dev0**, in development.

## What it does

- `rd_estimate` covers sharp and fuzzy designs, regression kinks (`derivative=1`) and
  covariate adjustment.
  - It offers rdrobust's three kernels, any polynomial and bias order, and all ten
    bandwidth selectors, or bandwidths you give.
  - Variances: nearest-neighbor, HC0 to HC3, or clustered (CR1). It takes sampling
    weights.
  - It has rdrobust's mass-point adjustment, minimum-support check and bandwidth
    restriction.
- `rd_bandwidth` gives rdbwselect's bandwidths, one selector or all ten.
- Results have hdfe-stream's reporting: `summary()`, `tidy()`, `confint()` and
  `summary_json()`. With several outcomes or bandwidths, `estimates()` gives one row
  per fit.

## When to use it, and when not

Use it when an RD fit is slow or doesn't fit in memory. Two cases stand out:
- **Large samples.** R's rdrobust and diff-diff take minutes at tens of millions of
  rows.
- **A running variable with many ties.** rdrobust's nearest-neighbor variance slows
  sharply when many observations share each value.

For a few thousand observations, rdrobust is already fast. rd-stream agrees with it but
won't save you time there. rdrobust also has things rd-stream doesn't have yet: CR2 and
CR3 cluster variances, and RD plots (`rdplot`). The documentation's conventions page
maps rdrobust's option names to rd-stream's.

## Install

```bash
pip install -e .
```

Python 3.10 or later. It depends on hdfe-stream, stream-limits, Polars,
NumPy, numba, SciPy and pyarrow.

A sibling of [hdfe-stream](https://github.com/leetucker/hdfe-stream),
[diff-stream](https://github.com/leetucker/diff-stream),
[match-stream](https://github.com/leetucker/match-stream) and
[penalized-stream](https://github.com/leetucker/penalized-stream).

## License

MIT.
