# Function reference

Everything here is importable from `rd_stream`.

| Name | What it is |
|---|---|
| **[Estimation](estimation.md)** | |
| `rd_estimate` | RD estimates with robust bias-corrected inference (rdrobust) |
| `rd_bandwidth` | the data-driven bandwidths (rdbwselect) |
| `rd_placebo` | estimates at placebo cutoffs |
| **[Plots](plots.md)** | |
| `rd_plot` | binned means and global polynomial fits (rdplot) |
| **[Results](results.md)** | |
| `RDResult`, `RDMulti` | one fit, and several (outcomes, bandwidths, donut holes) |
| `BandwidthResult` | the selected bandwidths |
| `PlaceboResult` | the placebo estimates, as a table and a figure |
| `RDPlot`, `RDPlotMulti` | one RD plot's data, and several outcomes' |
| **[Tables](tables.md)** | |
| `etable` | fits side by side, with a choice of rows |

```{toctree}
:hidden:

estimation
plots
results
tables
```
