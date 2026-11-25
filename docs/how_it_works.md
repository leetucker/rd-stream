# How it works

rdrobust's estimate, its bandwidth selectors and its standard errors are all
sums over the observations within a bandwidth of the cutoff. rd-stream
arranges the data so that each of those sums is one compiled, threaded pass
over a contiguous block of memory.

## One sort, then window sums

The rows are read once (only the columns a fit uses) and sorted by the
running variable. On each side of the cutoff, the observations within a
bandwidth h are then a prefix of the side's rows, taken in order of distance
from the cutoff. A local-polynomial fit at h needs, from that prefix, the
kernel-weighted power sums Σ K(d/h) d^a and Σ K(d/h) d^a y, which one pass
over the prefix computes. Each step of rdbwselect (the pilot fit, the bias
fits, the variance fits) and the final fit is one such pass, threaded with
numba.

## The nearest-neighbor residuals

rdrobust's default variance replaces each observation's residual with the
gap between its outcome and the mean of its nearest neighbors in the running
variable (three by default, all of a tied value's rows together). After the
sort, an observation's neighbors are next to it in the array, so the
residuals take one linear pass, which rd-stream computes once per side and
reuses for every step whose window holds them.

## Mass points

When the running variable takes few distinct values, each value's count,
outcome sum and within-value sums of squares and cross-products are all that
any of the sums above needs: rdrobust's nearest-neighbor residuals treat a
value's rows as one block too. rd-stream then fits on one row per value,
which gives the same results. With `memory="auto"`, a running variable with
at most half as many distinct values as rows (counted approximately, in one
pass) is grouped by value as it is read (`memory="by_value"`), so the rows
are never held at all.

## Several bandwidths

For a list of bandwidths, rd-stream builds, per side, prefix power sums over
blocks of 256 sorted observations, out to the widest window. A window's
moments are then a block prefix plus a short direct tail, and the
nearest-neighbor residuals are computed once and corrected only near each
window's edge. Each further bandwidth costs well under a millisecond,
whatever the number of observations in its window. Clustered fits, whose
variance needs each cluster's sums, take a pass per bandwidth instead.

## Clustered variances

Each side's rows are ordered by cluster once (a counting sort). Every
window is a prefix of the rows, so each cluster's rows in any window are a
leading run of its block, and threads take whole clusters. CRV2 and CRV3 then
adjust each cluster's score with small matrix computations (an
eigendecomposition or inverse of a matrix the size of the polynomial), which
run compiled and in parallel over clusters.

## Data larger than memory

`memory="window"` is for data whose rows do not fit in memory, with a
continuous running variable or a fit that cannot be grouped by value:

1. **Streaming passes** give the full sample's statistics: the counts, the
   standard deviations, exact quartiles (from a histogram of distances to the
   cutoff, then the rows of the quartile's bin), exact counts of distinct
   values and clusters, and the power sums that rdbwselect's fit over each
   side's whole range needs.
2. **The rows near the cutoff** are read and sorted, out to a few pilot
   bandwidths, and the fit runs on them as in memory. A step that needs rows
   further out reads them again, further out.
3. **Slabs.** When a step's window does not fit the memory budget, the rows
   are written once to Parquet files in `workdir`, by side and by distance
   from the cutoff, and every window sum is taken a slab at a time, a few
   slabs in memory. Nearest-neighbor residuals take their neighbors from the
   adjacent slabs, so the results are the same as holding the rows.

`rd_plot`'s window path always uses slabs, since every bin method needs each
side's rows in order: read in turn, the slabs give them.

The memory budget is stream-limits': the machine's or job's limit (or
`max_memory_gb=`), less a reserve. See [Memory, disk and threads](resources.md).
