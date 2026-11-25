# Implementations and conventions

rd-stream follows R's rdrobust 4.1.1 wherever packages differ. This page
lists what it is tested against, maps rdrobust's option names to
rd-stream's, and lists the few places where rd-stream does something
differently, so that a difference in someone's output can be traced.

## What is tested against what

| Implementation | Version | Compared on | Status |
|---|---|---|---|
| R rdrobust (`rdrobust`, `rdbwselect`) | 4.1.1 | 104 cases: sharp and fuzzy designs, kinks (`deriv = 1`), covariates, weights, clusters (CR1, CR2, CR3), HC0–HC3 and nearest-neighbor variances, the three kernels, polynomial orders 0–2 (and a bias order of 3), given and selected bandwidths (all ten selectors), mass points, `bwcheck`, `bwrestrict`, `stdvars`, `scaleregul` | estimates, standard errors and bandwidths to 1e-8 or better, through every memory path |
| R rdrobust (`rdplot`) | 4.1.1 | 27 cases: the eight bin selection methods, given numbers of bins, `scale`, `h`, `support`, covariates, weights, kernels, mass points | bins, numbers of bins and polynomial fits to 1e-8 or better, through every memory path |
| Python rdrobust | 2.1.1 | the same estimators | agrees with R, and so with rd-stream, except when the bias order exceeds the polynomial order by more than one (below) |

The R scripts that produce the references are in the repository's
`tests/` folder (`make_test_data.py` and `benchmarks/external/`), and their
outputs in `tests/data/`.

## Option names

rd-stream spells options out, and uses hdfe-stream's names where the
packages share a concept.

| rdrobust | rd-stream |
|---|---|
| `c` | `cutoff` |
| `p`, `q`, `deriv` | `polynomial_order`, `bias_order`, `derivative` |
| `h`, `b`, `rho` | `bandwidth`, `bias_bandwidth`, `bandwidth_ratio` |
| `bwselect` | `bandwidth` (a selector's name, such as `"mserd"`) |
| `vce`, `cluster` | `vcov`: `"nearest_neighbor"`, `"HC0"` to `"HC3"` (or `"hetero"` for HC1), or `{"CRV1": col}`, `{"CRV2": col}`, `{"CRV3": col}` |
| `nnmatch` | `n_neighbors` |
| `masspoints`, `bwcheck`, `bwrestrict` | `mass_points`, `min_distinct`, `restrict_bandwidth` |
| `stdvars`, `scaleregul`, `scalepar` | `standardize`, `regularization`, `scale` |
| `fuzzy`, `sharpbw`, `covs` | `treatment`, `sharp_bandwidth`, `covariates` |
| `weights` | `weights` |
| rdbwselect's `all = TRUE` | `rd_bandwidth(..., all_selectors=True)` |
| rdplot's `nbins`, `binselect`, `scale` | `n_bins`, `bin_method`, `bin_scale` |
| rdplot's `h`, `covs`, `covs_eval`, `support` | `bandwidth`, `covariates`, `covariates_at`, `support` |
| rdplot's `p`, `kernel`, `masspoints`, `ci` | `polynomial_order`, `kernel`, `mass_points`, `ci` |

The selectors' names (`"mserd"`, `"msetwo"`, `"cerrd"`, ...) and rdplot's
bin methods (`"esmv"`, `"qspr"`, ...) are rdrobust's.

**Cluster-robust variances.** rdrobust's CR1 is hdfe-stream's CRV1, with the
same small-sample factor, G/(G − 1) · (N − 1)/(N − k). CRV2 (Bell and
McCaffrey's bias-reduced linearization) and CRV3 (the jackknife-like
correction) are rdrobust's CR2 and CR3, with no small-sample factor, as in
rdrobust; hdfe-stream's CRV3 multiplies by CRV1's factor. In rdrobust, `vce`
and `cluster` together decide the variance (`vce = "hc2"` with clusters is
CR2); in rd-stream, the clustered `vcov` names its type.

## Where rd-stream differs

- **Singular matrices.** rdrobust inverts with R's `solve()` (an LU
  factorization, which stops when its estimate of the condition number is
  too large) and falls back to `MASS::ginv`. rd-stream uses a symmetric
  eigendecomposition, with the same cutoff for the generalized inverse. The
  two agree except for matrices that are singular to rounding, such as a
  cluster that holds nearly all of a window's weight.
- **rdplot's scale.** `bin_scale` times the selected number of bins is
  rounded up to a whole number; R leaves it fractional, which gives
  fractional bin indices.
- **rdplot's confidence level.** `ci` takes a percent (95) or a fraction
  (0.95); R takes the percent only.
- **rdplot with covariates.** The binned means are the outcome's own. R
  computes them from a regression of the outcome on the covariates and bin
  indicators, whose fitted values average, within each bin, to the same
  means; rd-stream skips the regression.
- **Additions.** Several outcomes or bandwidths per call, donut holes
  (`donut=`), placebo cutoffs (`rd_placebo`), the choice of rows in tables
  (`rows=`), and the memory paths are not in rdrobust.

## Python rdrobust and diff-diff

- **Python rdrobust 2.1.1** takes the last coefficient of the bias fit where
  R takes the coefficient of order p + 1. They agree when the bias order is
  one more than the polynomial order (the default); with `bias_order` two or
  more above it, Python's estimates differ from R's and rd-stream's.
- **diff-diff 4.0.0**'s RD port follows rdrobust 4.0.0's comparison of
  nearest-neighbor distances, which counts as ties only exactly equal
  distances. rdrobust 4.1.1, and rd-stream, count distances within a relative
  tolerance as ties, so the three can differ where distances nearly tie.
  diff-diff supports the nearest-neighbor variance only.
