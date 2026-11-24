# Changelog

## 0.1.0 (unreleased)

- `rd_estimate`: sharp and fuzzy regression discontinuity, regression kinks and covariate
  adjustment, with rdrobust's estimates, bandwidth selectors and variances (nearest-neighbor,
  HC0 to HC3, and CRV1, CRV2 and CRV3 clusters as `vcov={"CRV1": col}`) and sampling weights. Several outcomes, or a
  list of bandwidths, from one sort.
- A list of bandwidths is fitted from block power sums of each side, so each further
  bandwidth costs well under a millisecond, whatever the number of observations in its
  window.
- `rd_bandwidth`: rdbwselect's bandwidths, one selector or all ten.
- `rd_plot`: rdplot's binned means (the eight bin selection methods, or a given number of
  bins) and global polynomial fits, drawn with `.plot()`; several outcomes from one read,
  drawn as a grid of panels.
- `memory="auto" | "rows" | "by_value" | "window"`, `max_memory_gb=` and
  `memory_reserve=`. With `"by_value"`, which "auto" takes when the running variable has
  at most half as many distinct values as rows (and the fit has no clusters or weights),
  the data are grouped by its value as they are read, and the fit runs on one row per
  value with the same results: several times faster than holding the rows, with memory
  that grows with the distinct values rather than the rows.
- `memory="window"`, for fits whose rows do not fit and whose running variable is
  continuous, or that have clusters or weights: the full sample's statistics come from
  streaming passes, including exact quartiles and exact distinct counts
  (`hdfe_stream.core.distinct`, or counted from the slabs when they do not fit in one
  pass). Only the rows near the cutoff are held. When a step's
  window does not fit, the rows go to disk (`workdir=`) in slabs by distance from the
  cutoff, read a slab at a time. "auto" takes it when neither other path fits. The
  results are the same as with `"rows"` (all the R reference cases agree within 1e-8 through
  it).
- `profile=True`: the rows, the distinct values of the running variable, the memory path
  and its memory and disk, without fitting.
- Agreement with R's rdrobust 4.1.1 within 1e-8 on 104 option combinations of rdrobust and
  rdbwselect, and 27 of rdplot (tests/).
