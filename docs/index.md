# rd-stream

Regression discontinuity estimates with robust bias-corrected inference, for
data with millions to hundreds of millions of rows. rd-stream computes what
R's rdrobust package computes (local-polynomial estimates, the ten
data-driven bandwidth selectors, conventional and robust standard errors, and
RD plots), and agrees with it to 1e-8 or better.

```python
from rd_stream import rd_estimate, rd_placebo, rd_plot

fit = rd_estimate("vote ~ margin", "senate.parquet")   # a path, a glob, or a Polars frame
fit.summary()

rd_plot("vote ~ margin", "senate.parquet").plot()      # binned means and polynomial fits

# robustness checks
rd_estimate("vote ~ margin", "senate.parquet", bandwidth=[5.0, 10.0, 20.0]).estimates()
rd_estimate("vote ~ margin", "senate.parquet", donut=[0.0, 0.5, 1.0]).estimates()
rd_placebo("vote ~ margin", "senate.parquet", placebo_cutoffs=[-5, 5]).plot()
```

The options are rdrobust's, with spelled-out names (`polynomial_order`,
`bandwidth`, `vcov`, ...); [Implementations](implementations.md) maps every
one. Results use hdfe-stream's reporting: `summary()`, `tidy()`,
`confint()` and `etable()` tables for papers.

## When to use this, and when not to

Use rd-stream when an RD fit is slow or does not fit in memory:

- **Large data.** From about a million observations, rdrobust in R takes
  minutes and Python's rdrobust needs several copies of the data in memory;
  rd-stream sorts the running variable once and makes compiled, threaded
  passes over each bandwidth's observations.
- **A running variable with mass points** (ages, test scores, dates):
  rd-stream fits on one row per distinct value, which is exact, so the time
  hardly grows with the rows.
- **Many fits on the same data:** several outcomes, a grid of bandwidths,
  donut holes or placebo cutoffs, from one read and sort. A further bandwidth
  costs well under a millisecond.
- **Data larger than memory:** `memory="window"` holds only the rows near
  the cutoff, and writes the rest to disk in slabs when a step needs them.

Use something else when:

- **The data are small** (under about 100,000 observations). rdrobust takes
  under a second there, and rd-stream's compiled kernels add a fraction of a
  second the first time a process uses them.
- **You need what rdrobust's family does beyond rdrobust, rdbwselect and
  rdplot:** the manipulation test (rddensity), local randomization
  inference (rdlocrand), multiple cutoffs and scores (rdmulti), power
  calculations (rdpower), or honest confidence intervals (RDHonest). These
  are not in rd-stream yet.

*Placeholder: one sentence with the headline numbers from the EC2 scale
benchmark (rd-stream against R's and Python's rdrobust at 10 million rows),
linking to [Benchmarks](benchmarks/index.md).*

## How it is fast

- **One sort, then window sums.** Every step of rdbwselect and rdrobust is a
  sum over the observations within a bandwidth of the cutoff. With the rows
  sorted by distance from the cutoff, each window is a prefix, summed by a
  compiled, threaded pass.
- **Nearest-neighbor residuals once.** rdrobust's default variance needs each
  observation's nearest neighbors in the running variable; after the sort
  they are neighbors in the array.
- **Mass points collapsed exactly.** With few distinct values, each value's
  count, sums and cross-products are all the fit needs, grouped as the data
  are read.
- **A bandwidth grid from block sums.** Prefix power sums over blocks of the
  sorted rows give every window's moments without another pass.

See [How it works](how_it_works.md).

```{toctree}
:hidden:
:maxdepth: 2

examples/index
how_it_works
implementations
resources
benchmarks/index
reference/index
changelog
```
