# Large data

[`examples/large_data.py`](https://github.com/leetucker/rd-stream/blob/main/examples/large_data.py)
simulates ten million observations (about 400 MB of Parquet, written a
million rows at a time) and fits them three ways: holding the rows, within a
1 GB memory budget, and grouped by value. The data have a continuous running
variable `x`, a discrete one `xd` (x rounded down to a grid of 1/250),
clusters of about 50 rows, and three outcomes that jump by 0.3, 0.1 and 0 at
the cutoff.

## What a fit needs, before fitting

```python
print(rd.rd_estimate("y1 ~ x", path, profile=True))
```

```text
Profile: rd-stream: rd_estimate, 1 outcome(s) x 1 bandwidth(s), rows
  10,000,000 rows kept of 10,000,000 (scan 0.3s); 8 threads
  memory     2.16 GB expected      2.53 GB upper bound   (this process holds 0.53 GB now)
  disk       0.00 GB expected      0.00 GB upper bound
  request: ncpus=8  mem=3gb  disk=1gb   (select=1:ncpus=8:mem=3gb)
  components (GB expected / upper):
    sorted rows and the fits' arrays     1.480 /    1.850  ram    fit
    compiled kernels and libraries       0.150 /    0.150  ram    fit
```

`profile=True` reads the data once, without fitting, and reports which way
the fit would hold the data, the memory and disk it expects, and a request
for a cluster's scheduler.

## Holding the rows, and a 1 GB budget

```python
fit = rd.rd_estimate("y1 ~ x", path)
small = rd.rd_estimate("y1 ~ x", path, max_memory_gb=1.0, workdir="scratch")
```

```text
rows: estimate 0.3020, robust interval [0.2964, 0.3022], h = 0.2263
window: estimate 0.3020, robust interval [0.2964, 0.3022], h = 0.2263
largest difference from holding the rows: 5.3e-11 (relative)
```

By default (`memory="auto"`) the rows are read and sorted once when they fit
the memory budget. Within 1 GB they do not, so the fit takes the full
sample's statistics in streaming passes, holds the rows near the cutoff, and
writes the rest to `workdir` in slabs by distance from the cutoff, reading
them a slab at a time when a step needs them. The results are the same, to
rounding. The slabs are removed when the fit ends.

## A discrete running variable

```python
disc = rd.rd_estimate("y1 ~ xd", path)
```

```text
by_value: 494 distinct values, estimate 0.3000
```

With at most half as many distinct values as rows, the data are grouped by
value as they are read: each value's count, sums and cross-products are all
the fit needs, and the fit on them is exact. Memory and time then grow with
the distinct values, not the rows.

## Several outcomes, clustered

```python
multi = rd.rd_estimate("y1 + y2 + y3 ~ x", path, vcov={"CRV1": "cl"})
multi.estimates()
```

```text
┌─────────┬───────────┬────────────┬─────────────┐
│ outcome ┆ estimate  ┆ robust_low ┆ robust_high │
╞═════════╪═══════════╪════════════╪═════════════╡
│ y1      ┆ 0.302009  ┆ 0.296373   ┆ 0.302159    │
│ y2      ┆ 0.099675  ┆ 0.093558   ┆ 0.103767    │
│ y3      ┆ -0.000406 ┆ -0.004212  ┆ 0.002804    │
└─────────┴───────────┴────────────┴─────────────┘
```

Every outcome shares one read and sort, and each selects its own bandwidth.
