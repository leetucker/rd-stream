# Changelog

## 0.1.0 (unreleased)

- `rd_estimate`: sharp and fuzzy regression discontinuity, regression kinks and covariate
  adjustment, with rdrobust's estimates, bandwidth selectors and variances (nearest-neighbor,
  HC0 to HC3, CR1 clusters) and sampling weights. Several outcomes, or a list of bandwidths,
  from one sort.
- `rd_bandwidth`: rdbwselect's bandwidths, one selector or all ten.
- An exact fit on one row per distinct value of the running variable, chosen automatically.
- Agreement with R's rdrobust 4.1.1 within 1e-8 on 80 option combinations (tests/).
