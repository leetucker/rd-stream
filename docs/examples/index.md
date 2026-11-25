# Examples

Each page goes with a script in the repository's
[`examples/`](https://github.com/leetucker/rd-stream/tree/main/examples)
folder and shows its output. The first two use the U.S. Senate elections data
of Cattaneo, Frandsen and Titiunik (2015), which rdrobust's documentation
uses (1,390 elections; they come with Python's rdrobust package). The third
simulates ten million observations.

| Page | What it shows |
|---|---|
| [Quickstart](quickstart.md) | The incumbency advantage: the RD plot, the estimate, the bandwidth selectors, clusters, covariates and a table. Start here. |
| [Robustness checks](robustness.md) | Other bandwidths, donut holes, placebo cutoffs and a placebo outcome. |
| [Large data](large_data.md) | Ten million rows: the profile, the same fit within 1 GB, a discrete running variable, several outcomes. |

```{toctree}
:hidden:

quickstart
robustness
large_data
```
