# Benchmarks

The benchmarks compare rd-stream with rdrobust in R (4.1.1) and in Python
(2.1.1, with its numba kernels) on simulated data, from 10,000 to 100 million
observations:

| Model | What is fitted |
|---|---|
| Sharp | rdrobust's defaults: a local linear fit, the `mserd` bandwidths, the nearest-neighbor variance |
| Clustered | the same, with clusters of about 50 observations (CR1, rd-stream's CRV1) |
| Discrete | a running variable with 500 values (mass points) |
| RD plot | rdplot's defaults, without drawing |

The running variable is 2·Beta(2, 4) − 1 (the cutoff at 0, about 30% of the
observations to its right), and the outcome a cubic in it with a jump of 0.3
and normal noise. Each configuration runs in its own process, from a Parquet
file of the n rows. The time includes reading the data: the rdrobust packages
read the columns into memory first, and rd-stream is given the path. The
memory is the process's peak resident memory. A configuration that runs out
of memory or exceeds an hour is not run at larger sizes. Every estimate is
compared with rd-stream's (every one agrees to within 6e-11). The scripts are in
the repository's `benchmarks/` folder.

## Machine

An AWS EC2 c8id.4xlarge: an Intel Xeon 6975P-C with 16 hardware threads on 8
cores, 32 GiB of memory (each process limited to 96% of it), and a local NVMe
disk for the data. The software: Python 3.14.8 with rd-stream 0.1.0,
hdfe-stream 0.13.1, stream-limits 0.3.0, Polars 2.1.0, NumPy 2.5.5, numba
0.68.1 and Python rdrobust 2.1.1; R 4.5.3 with rdrobust 4.1.1 and arrow
25.0.1. Measured on November 25, 2026. Times are in seconds and memory in GB
(10^9 bytes); each table cell gives the time and the peak memory.

## Sharp design

```{image} ../figures/scale_sharp.light.svg
:class: only-light benchmark
:alt: Time and peak memory against the number of observations, one line per implementation, log-log
```
```{image} ../figures/scale_sharp.dark.svg
:class: only-dark benchmark
:alt: Time and peak memory against the number of observations, one line per implementation, log-log
```

| Rows | rd-stream | Python rdrobust | R rdrobust |
|---|---|---|---|
| 10,000 | 0.20 s, 0.3 GB | 0.60 s, 0.4 GB | 0.18 s, 0.3 GB |
| 100,000 | 0.20 s, 0.3 GB | 0.30 s, 0.4 GB | 0.78 s, 0.3 GB |
| 1,000,000 | 0.23 s, 0.5 GB | 0.81 s, 0.7 GB | 5.57 s, 0.5 GB |
| 10,000,000 | 0.89 s, 2.1 GB | 6.58 s, 3.2 GB | 47.83 s, 2.7 GB |
| 30,000,000 | 2.24 s, 3.4 GB | 19.85 s, 8.6 GB | 139 s, 7.3 GB |
| 100,000,000 | 7.17 s, 10.8 GB | out of memory | 506 s, 23.5 GB |

## Clustered

```{image} ../figures/scale_clustered.light.svg
:class: only-light benchmark
:alt: Time and peak memory against the number of observations, one line per implementation, log-log
```
```{image} ../figures/scale_clustered.dark.svg
:class: only-dark benchmark
:alt: Time and peak memory against the number of observations, one line per implementation, log-log
```

| Rows | rd-stream | Python rdrobust | R rdrobust |
|---|---|---|---|
| 10,000 | 0.18 s, 0.2 GB | 0.71 s, 0.3 GB | 0.12 s, 0.2 GB |
| 100,000 | 0.19 s, 0.3 GB | 0.32 s, 0.3 GB | 0.26 s, 0.3 GB |
| 1,000,000 | 0.28 s, 0.5 GB | 1.87 s, 0.6 GB | 2.49 s, 0.5 GB |
| 10,000,000 | 1.38 s, 2.1 GB | 17.51 s, 3.2 GB | 20.88 s, 3.0 GB |
| 30,000,000 | 4.47 s, 4.3 GB | 53.25 s, 8.9 GB | 67.12 s, 8.7 GB |
| 100,000,000 | 16.29 s, 14.6 GB | 193 s, 28.3 GB | 248 s, 29.4 GB |

## Discrete running variable

```{image} ../figures/scale_discrete.light.svg
:class: only-light benchmark
:alt: Time and peak memory against the number of observations, one line per implementation, log-log
```
```{image} ../figures/scale_discrete.dark.svg
:class: only-dark benchmark
:alt: Time and peak memory against the number of observations, one line per implementation, log-log
```

| Rows | rd-stream | Python rdrobust | R rdrobust |
|---|---|---|---|
| 10,000 | 0.17 s, 0.3 GB | 0.48 s, 0.3 GB | 0.14 s, 0.3 GB |
| 100,000 | 0.19 s, 0.3 GB | 0.33 s, 0.4 GB | 0.74 s, 0.3 GB |
| 1,000,000 | 0.20 s, 0.3 GB | 4.12 s, 0.6 GB | 22.54 s, 0.6 GB |
| 10,000,000 | 0.23 s, 0.6 GB | 244 s, 2.9 GB | 1,254 s, 2.8 GB |
| 30,000,000 | 0.25 s, 1.0 GB | 2,033 s, 7.9 GB | over an hour |
| 100,000,000 | 0.45 s, 2.1 GB | over an hour | not run |

R's fit at 30 million rows ran past the hour, so it was not run at 100
million.

## RD plot

```{image} ../figures/scale_plot.light.svg
:class: only-light benchmark
:alt: Time and peak memory against the number of observations, one line per implementation, log-log
```
```{image} ../figures/scale_plot.dark.svg
:class: only-dark benchmark
:alt: Time and peak memory against the number of observations, one line per implementation, log-log
```

| Rows | rd-stream | Python rdplot | R rdplot |
|---|---|---|---|
| 10,000 | 0.20 s, 0.3 GB | 0.18 s, 0.4 GB | 0.21 s, 0.2 GB |
| 100,000 | 0.16 s, 0.3 GB | 0.28 s, 0.4 GB | 0.56 s, 0.3 GB |
| 1,000,000 | 0.25 s, 0.5 GB | 1.35 s, 0.8 GB | 4.02 s, 0.7 GB |
| 10,000,000 | 0.85 s, 2.1 GB | 13.26 s, 5.0 GB | 38.65 s, 5.0 GB |
| 30,000,000 | 2.23 s, 3.3 GB | 40.78 s, 14.2 GB | 126 s, 14.1 GB |
| 100,000,000 | 7.16 s, 9.2 GB | out of memory | out of memory |

## What the results show

- **At 10 million rows**, rd-stream is 7 times faster than Python's rdrobust
  and 54 times faster than R's on the sharp design, 13 and 15 times faster
  with clusters, and 16 and 45 times faster for the RD plot. At 1 million
  rows the gaps are 3.5 and 24 times (sharp) and 5 and 16 times (the plot).
- **With a discrete running variable**, rd-stream fits on one row per value,
  so its time hardly grows with the rows: 0.23 seconds at 10 million rows,
  against 244 seconds for Python's rdrobust (about 1,000 times longer) and 21
  minutes for R's (about 5,000 times). rdrobust's time grows faster than the
  rows here; neither package finished within an hour at the largest size it
  was given.
- **Memory.** From 30 million rows, rd-stream uses about half the memory
  of either package on the estimates, and a quarter of it for the plot. At 100 million rows,
  Python's rdrobust ran out of memory on the sharp design and both packages
  on the RD plot; R's clustered fit took 29.4 GB of the 32. rd-stream took
  at most 14.6 GB, and its window path (`memory="window"`) fits larger data
  in less.
- **Small data.** Under 100,000 rows, every implementation takes well under
  a second. R's rdrobust estimates fastest at 10,000 rows (0.12 to 0.18
  seconds against rd-stream's 0.17 to 0.20, which is mostly loading its
  compiled kernels).

## Threads

On the same machine, rd-stream's time at 1 to 16 threads (the median of
three fits):

| Model, rows | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| sharp, 10^7 | 1.94 s | 1.00 s | 0.75 s | 0.68 s |
| sharp, 10^8 | 22.7 s | 10.9 s | 7.80 s | 6.78 s |
| discrete, 10^8 | 1.69 s | 0.54 s | 0.31 s | 0.27 s |
| window (4 GB budget), 10^8 | 89.5 s | 54.4 s | 45.4 s | 47.3 s |

Using every hardware thread (the default) is fastest or within a few percent
of it. The window path, bound by reading and writing its slabs, levels off at
the number of cores.

## Reproducing

```bash
pip install -e ".[benchmark]"
micromamba create -p benchmarks/external/.r -f benchmarks/external/environment.yml
benchmarks/external/.r/bin/Rscript -e 'install.packages("rdrobust",
    repos = "https://packagemanager.posit.co/cran/2026-10-15")'
python benchmarks/scale_benchmark.py        # every model, every size
python benchmarks/threads_benchmark.py      # thread counts
python benchmarks/plot_benchmarks.py        # the figures
```
