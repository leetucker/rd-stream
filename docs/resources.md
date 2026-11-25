# Memory, disk and threads

## How the data are held

`memory=` chooses how a fit holds its data. The default, `"auto"`, takes the
first of these that applies:

| `memory=` | When "auto" takes it | Memory grows with | Disk |
|---|---|---|---|
| `"by_value"` | the running variable has at most half as many distinct values as rows, and the fit has no clusters or weights | the distinct values | none |
| `"rows"` | the rows fit the memory budget | the rows: about 85 bytes a row, plus 29 a column read | none |
| `"window"` | otherwise | the rows near the cutoff, or a few slabs | the rows, about 8 bytes a row per column read, when a step's window does not fit |

All three give the same results, to rounding. `"rows"` fits on every row;
the others fit on one row per distinct value where that is shorter, which is
exact.

`profile=True` reports, without fitting, which path a fit would take, its
expected and largest memory and disk, and a resource request for a cluster's
scheduler. It reads the data once (a streaming pass).

## The memory budget

The budget is stream-limits': the machine's memory, or the job's limit under
a scheduler or container, less a reserve (15%, `memory_reserve=`), less what
the process already holds. `max_memory_gb=` sets the limit directly. When
even the window path's smallest slab does not fit, the fit stops with a
MemoryError before reading the data.

## Disk

The window path writes its slabs into a run directory under `workdir=`, or
`$HDFE_STREAM_WORKDIR`, or the system's temporary directory, and removes it
when the fit ends. The temporary directory may be in memory (a tmpfs, as
`/tmp` is on many Linux systems), so a fit that writes slabs should be given
a `workdir` on a disk.

## Threads

The compiled passes and Polars' reading use every CPU stream-limits detects
(the job's allocation under a scheduler). `n_threads=` sets the count for
one call. On servers, using every hardware thread was fastest in our tests.
On laptops with efficiency cores, fewer threads (the performance cores) can
be faster, since each pass waits for its slowest thread; there, try
`n_threads=` set to the number of performance cores.
