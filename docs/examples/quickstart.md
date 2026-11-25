# Quickstart: the incumbency advantage

[`examples/quickstart.py`](https://github.com/leetucker/rd-stream/blob/main/examples/quickstart.py)
estimates the effect of winning a U.S. Senate seat on the party's vote share
in the next election for that seat, with the data of Cattaneo, Frandsen and
Titiunik (2015) that rdrobust's documentation uses. A Democrat who barely won
and one who barely lost are alike in all but the result, so the jump in the
next vote share at a margin of 0 is the effect of winning.

## Data

`load_senate()` (in `examples/senate_data.py`) writes the 1,390 elections to
a Parquet file the first time and returns its path. They come with Python's
rdrobust package (`pip install rdrobust`).

```python
from senate_data import load_senate
import rd_stream as rd

senate = load_senate()
```

`margin` is the Democratic margin of victory in percentage points (the
running variable; the cutoff is 0), and `vote` the Democratic vote share in
the next election for the seat.

## The RD plot

```python
plot = rd.rd_plot("vote ~ margin", senate)
plot.plot(xlabel="Democratic margin of victory (percentage points)",
          ylabel="Democratic vote share, next election")
```

```text
bins: 15 left, 35 right (mimicking variance evenly-spaced method using spacings estimators)
```

```{image} ../figures/quickstart_rdplot.svg
:class: benchmark
:alt: Binned means of the next vote share against the margin, with a fourth-order polynomial on each side, jumping at 0
```

The dots are the outcome's means in bins of the margin, as many as mimic the
outcome's variance (rdplot's default); the lines are fourth-order
polynomials fitted on each side. The jump at 0 is visible before any
estimation.

## The estimate

```python
fit = rd.rd_estimate("vote ~ margin", senate)
print(fit.summary_text())
```

```text
running variable: margin, cutoff 0
obs: 1,297 (595 left, 702 right); within h: 360 left, 323 right
bandwidth h: 17.7544; bias bandwidth b: 28.0281 (mserd)
polynomial order 1, bias order 2; triangular kernel; vcov: nearest_neighbor
┌────────────────┬──────────┬────────────┬──────────┬───────────┐
│ Coefficient    ┆ Estimate ┆ Std. Error ┆ t value  ┆ Pr(>|t|)  │
╞════════════════╪══════════╪════════════╪══════════╪═══════════╡
│ Conventional   ┆ 7.414131 ┆ 1.458716   ┆ 5.082642 ┆ 3.7222e-7 │
│ Bias-corrected ┆ 7.506502 ┆ 1.458716   ┆ 5.145966 ┆ 2.6615e-7 │
│ Robust         ┆ 7.506502 ┆ 1.741258   ┆ 4.310964 ┆ 0.000016  │
└────────────────┴──────────┴────────────┴──────────┴───────────┘
robust 95% interval: [4.094, 10.919]
```

These are rdrobust's defaults: a local linear fit with a triangular kernel,
the MSE-optimal bandwidth (`mserd`), and a nearest-neighbor variance.
Winning the seat raises the next vote share by about 7.4 percentage points.
The three rows are rdrobust's:

- **Conventional**: the local linear estimate and its standard error.
- **Bias-corrected**: the estimate less an estimate of its bias.
- **Robust**: the bias-corrected estimate with a standard error that accounts
  for the bias correction (Calonico, Cattaneo and Titiunik 2014). Its
  interval is the one to report.

## The bandwidth selectors

```python
rd.rd_bandwidth("vote ~ margin", senate, all_selectors=True).table
```

```text
┌──────────┬───────────┬───────────┬───────────┬───────────┐
│ selector ┆ h_left    ┆ h_right   ┆ b_left    ┆ b_right   │
╞══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
│ mserd    ┆ 17.754398 ┆ 17.754398 ┆ 28.028089 ┆ 28.028089 │
│ msetwo   ┆ 16.16982  ┆ 18.126469 ┆ 27.10389  ┆ 29.343562 │
│ msesum   ┆ 18.365456 ┆ 18.365456 ┆ 31.318519 ┆ 31.318519 │
│ msecomb1 ┆ 17.754398 ┆ 17.754398 ┆ 28.028089 ┆ 28.028089 │
│ msecomb2 ┆ 17.754398 ┆ 18.126469 ┆ 28.028089 ┆ 29.343562 │
│ cerrd    ┆ 12.406776 ┆ 12.406776 ┆ 28.028089 ┆ 28.028089 │
│ certwo   ┆ 11.299473 ┆ 12.666779 ┆ 27.10389  ┆ 29.343562 │
│ cersum   ┆ 12.833784 ┆ 12.833784 ┆ 31.318519 ┆ 31.318519 │
│ cercomb1 ┆ 12.406776 ┆ 12.406776 ┆ 28.028089 ┆ 28.028089 │
│ cercomb2 ┆ 12.406776 ┆ 12.666779 ┆ 28.028089 ┆ 29.343562 │
└──────────┴───────────┴───────────┴───────────┴───────────┘
```

`bandwidth="cerrd"` (or any of these) chooses the bandwidth for a fit; the
`cer` selectors give intervals with better coverage, the `mse` ones better
point estimates.

## Clusters, covariates and a table

```python
by_state = rd.rd_estimate("vote ~ margin", senate, vcov={"CRV1": "state"})
covs = rd.rd_estimate("vote ~ margin", senate, covariates=["vote_lag1", "vote_lag2"])
rd.etable([fit, by_state, covs], rows="conventional_robust", type="md",
          model_stats=["N"], labels={"Conventional, robust inference": "Winning the seat"})
```

```text
|                              | ('vote', '(1)')   | ('vote', '(2)')   | ('vote', '(3)')   |
|:-----------------------------|:------------------|:------------------|:------------------|
| ('coef', 'Winning the seat') | 7.414***          | 7.395***          | 7.167***          |
|                              |  (1.741)          |  (1.796)          |  (1.745)          |
| ('stats', 'Observations')    | 1,297             | 1,297             | 1,215             |
```

`vcov={"CRV1": "state"}` clusters by state (CRV2 and CRV3 are there too).
Predetermined covariates, here the party's vote shares in the two previous
elections for the seat, can shorten the interval without changing what is
estimated (Calonico, Cattaneo, Farrell and Titiunik 2019).

`rows="conventional_robust"` shows each fit as many papers report rdrobust's
results: the conventional estimate, with the robust standard error, and stars
from the robust test. `rows=` also takes any of the three rows above.
