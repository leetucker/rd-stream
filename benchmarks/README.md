# Benchmarks

Two suites measure rd-stream's time and peak memory. Each configuration runs
in its own process, so that peak memory is its own, and writes one CSV row to
`benchmarks/results/` as soon as it finishes (`_harness.py`).

```bash
python benchmarks/scale_benchmark.py                 # against rdrobust, 10^4 to 10^8 rows
python benchmarks/scale_benchmark.py --model sharp 1000000 10000000
python benchmarks/threads_benchmark.py               # rd-stream at 1 to 16 threads
python benchmarks/plot_benchmarks.py                 # the figures, into docs/figures/
```

- **scale** (`scale_benchmark.py`, `results/scale.csv`): rd-stream against
  rdrobust in R (4.1.1) and Python (2.1.1, with its numba kernels), on four
  models: the default sharp fit, a clustered fit, a discrete running variable
  and the RD plot. Every library starts from the same Parquet file. The
  estimates are compared with rd-stream's (column `max_abs_diff`).
- **threads** (`threads_benchmark.py`, `results/threads.csv`): rd-stream
  alone, with numba's and Polars' thread counts set per configuration, on a
  continuous, a clustered, a discrete and an out-of-core (`memory="window"`)
  fit. Each configuration fits three times; the row reports the median, the
  fastest and the slowest.

The data are simulated (`_data.py`) and written to `benchmarks/data/` on
first use, 10 million rows at a time.

**R.** The R configurations need R with rdrobust 4.1.1, arrow and jsonlite in
`benchmarks/external/.r`:

```bash
micromamba create -p benchmarks/external/.r -f benchmarks/external/environment.yml
benchmarks/external/.r/bin/Rscript -e 'install.packages("rdrobust",
    repos = "https://packagemanager.posit.co/cran/2026-10-15")'
```

or set `RD_BENCH_RSCRIPT` to another Rscript. Without R, those configurations
are skipped.

**Environment variables.** `RD_BENCH_RESULTS` (where the CSVs go),
`RD_BENCH_DATA` (the simulated data) and `RD_BENCH_FIGURES` (the figures)
move those folders, for a trial run that should not touch the published
numbers.

The published numbers come from EC2 runs (`benchmarks/ec2`, which is not in
the repository: its settings name an AWS account).
