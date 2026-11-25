# Benchmarks

```{note}
These benchmarks are being measured on EC2. The tables and figures below are
placeholders until the results are in.
```

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
compared with rd-stream's (they agree to 1e-8 or better). The scripts are in
the repository's `benchmarks/` folder.

## Machine

*Placeholder: the EC2 instance (c8id.4xlarge: 16 vCPUs on 8 cores, 32 GiB),
its CPU, and the library versions, from `benchmarks/results/machine.json`.*

## Sharp design

*Placeholder: the figure (`docs/figures/scale_sharp.light.svg`) and the table
of time and peak memory by size and library.*

## Clustered

*Placeholder: the figure (`docs/figures/scale_clustered.light.svg`) and the
table.*

## Discrete running variable

*Placeholder: the figure (`docs/figures/scale_discrete.light.svg`) and the
table.*

## RD plot

*Placeholder: the figure (`docs/figures/scale_plot.light.svg`) and the
table.*

## What the results show

*Placeholder: rd-stream's speed-up over each package at 10^6 and 10^7 rows;
the sizes at which R and Python ran out of memory or time; where rdrobust is
as fast (small data).*

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
