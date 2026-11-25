"""How a fit holds its data, and what that costs: the memory model behind
``memory="auto"``, ``max_memory_gb=`` and ``profile=True``.

Three paths:

- ``"rows"``: the running variable, the outcomes and any take-up,
  covariate, cluster and weight columns are read and sorted once. Peak
  memory grows with the rows: about 85 bytes a row, plus 29 for each column
  read and 5 for each column of a fit's response (the outcome, take-up
  and covariates), plus 12 for cluster codes. Fitted to peaks measured on
  10^7 rows (1 to 20 outcomes, clusters, covariates, a take-up), which it
  meets or exceeds; the upper bound adds a quarter.
- ``"by_value"``: one streaming group-by of the rows by the value of the
  running variable gives each value's count, column sums and
  within-value cross-products, which are all the fit needs (it is exact).
  Memory grows with the number of
  distinct values, not the rows, plus the streaming read's buffers. It
  needs a fit without clusters or weights.

- ``"window"``: streaming passes for the full-sample statistics, then only
  the rows near the cutoff, read again further out when a step needs them
  (see _window.py); when a step's window does not fit, the rows are written
  to disk in slabs by distance from the cutoff and every window sum reads
  them a slab at a time (_slabs.py). Memory grows with the rows held (a
  window, or a few slabs), plus the exact distinct counts' hash tables
  (about 56 bytes a distinct value, split into passes to fit).

``memory="auto"`` takes the by-value path when the fit has no clusters or
weights and the running variable has at most half as many distinct values
as rows (an approximate count, one streaming pass over it): it is several
times faster than holding the rows. Else the rows path when its peak
fits stream-limits' `MemoryBudget` (the limit, less a reserve, less what
the process holds), else the window path.
"""

from __future__ import annotations

import math

import polars as pl
from hdfe_stream.core.profile import (
    Component,
    Profile,
    baseline_gb,
    scan_source,
    source_row_group_rows,
)

MEMORY_PATHS = ("auto", "rows", "by_value", "window")
_BY_VALUE = True             # tests: False keeps "auto" off the by-value path
_ROW_BASE, _PER_READ, _PER_COLUMN, _CLUSTER = 85, 29, 5, 12
_UPPER = 1.25
_LIBRARIES_GB = 0.15          # the compiled kernels, NumPy, Polars and SciPy


def rows_bytes(n, n_read, n_columns, clustered):
    """Expected peak bytes of the rows path for n rows."""
    per = _ROW_BASE + _PER_READ * n_read + _PER_COLUMN * n_columns
    return n * (per + (_CLUSTER if clustered else 0))


def units_bytes(n_values, n_outcomes, n_columns):
    """Expected peak bytes of the by-value table: per value and outcome a
    count, the column sums and the cross-products, held about four times
    (the group-by's state, the result, the sorted copy and the sides)."""
    per = 8 * (2 + n_columns + n_columns * n_columns)
    return 4 * n_values * n_outcomes * per


def read_buffer_bytes(source, n_read, threads):
    """The streaming read's buffers: a row group per thread, twice."""
    rg = source_row_group_rows(source) or 1_000_000
    return 2 * threads * rg * n_read * 8


def count_rows(lf: pl.LazyFrame) -> int:
    """Rows in the source (Parquet metadata, or the frame's height)."""
    return int(lf.select(pl.len()).collect(engine="streaming").item())


def approx_distinct(lf: pl.LazyFrame, running: str) -> int:
    """HyperLogLog count of the running variable's distinct values (about 2%)."""
    return int(lf.select(pl.col(running).drop_nulls().approx_n_unique())
               .collect(engine="streaming").item() or 0)


def choose(lf, source, *, memory, running, n_read, n_columns, n_outcomes, clustered,
           weighted, collapse, threads, max_memory_gb, memory_reserve, window=True):
    """The path for a fit: ("rows" | "by_value" | "window", notes).
    `window`: whether the caller has a window path."""
    from stream_limits import MemoryBudget

    collapsible = not clustered and not weighted and collapse != "never"
    if memory == "by_value":
        if not collapsible:
            raise ValueError("memory='by_value' needs a fit without clusters and weights")
        return "by_value", {}
    if memory == "rows":
        return "rows", {}
    if memory == "window":
        if not window:
            raise ValueError("memory='window' is not available here")
        return "window", {}
    budget = MemoryBudget(limit_gb=max_memory_gb, reserve=memory_reserve)
    n = count_rows(lf)
    need = rows_bytes(n, n_read, n_columns, clustered) / 1e9
    notes = {"rows": n, "rows_gb": round(need, 3),
             "headroom_gb": round(budget.headroom_gb(), 3)}
    if collapsible and _BY_VALUE:
        m = approx_distinct(lf, running)
        notes["distinct_values"] = m
        need_u = (units_bytes(m, n_outcomes, n_columns)
                  + read_buffer_bytes(source, n_read, threads)) / 1e9
        notes["by_value_gb"] = round(need_u, 3)
        if m <= n / 2 and budget.fits(need_u * _UPPER):
            return "by_value", notes
    if budget.fits(need * _UPPER):
        return "rows", notes
    if window:
        return "window", notes
    hint = ("" if collapsible else
            " The by-value path, whose memory grows with the distinct values of the "
            "running variable, needs a fit without clusters and weights.")
    raise MemoryError(
        f"the fit needs about {need:.1f} GB to hold and sort {n:,} rows, more than the "
        f"{budget.headroom_gb():.1f} GB available (max_memory_gb, less memory_reserve, less "
        f"what this process holds); fit fewer outcomes per call, or give max_memory_gb if "
        f"the machine has more memory than detected.{hint}")


def profile(lf, source, *, running, outcomes, extra, n_columns, clustered, weighted,
            collapse, memory, threads, max_memory_gb, memory_reserve, model, window=None,
            slabs=False):
    """`Profile` of a fit: one streaming pass for the rows kept and the
    distinct values of the running variable, and the components of the path
    ``memory`` would take. `window`: a function returning the window path's
    `WindowData` without its rows (its summary passes run), or None when
    the caller has no window path."""
    from stream_limits import MemoryBudget

    present = [pl.col(c) for c in [running, *extra]]
    scan = scan_source(lf, present=present, distinct={running: [running]}, minmax=[running])
    n, m = scan.n_rows, scan.distinct.get(running, 0)
    n_read = 1 + len(outcomes) + len(extra)
    budget = MemoryBudget(limit_gb=max_memory_gb, reserve=memory_reserve)
    rows_gb = rows_bytes(n, n_read, n_columns, clustered)
    collapsible = not clustered and not weighted and collapse != "never"
    path = memory
    if path == "auto":
        if collapsible and _BY_VALUE and m <= n / 2 and budget.fits(
                (units_bytes(m, len(outcomes), n_columns)
                 + read_buffer_bytes(source, n_read, threads)) * _UPPER / 1e9):
            path = "by_value"
        elif budget.fits(rows_gb * _UPPER / 1e9):
            path = "rows"
        elif window is not None:
            path = "window"
        else:
            path = "rows"
    if path == "window" and window is None:
        raise ValueError("memory='window' is not available here")
    comps = [Component("compiled kernels and libraries", _LIBRARIES_GB * 1e9,
                       phases=("fit",))]
    warnings = []
    if path == "rows":
        per = rows_gb / max(n, 1)
        comps.append(Component("sorted rows and the fits' arrays", rows_gb, rows_gb * _UPPER,
                               phases=("fit",),
                               formula=f"{n:,} rows x {per:.0f} B ({n_read} columns read)"))
        if not budget.fits(rows_gb * _UPPER / 1e9):
            warnings.append("holding the rows does not fit the memory budget"
                            + ("" if collapsible else
                               " (the by-value path needs a fit without clusters and weights)"))
    elif path == "window":
        from hdfe_stream.core.distinct import distinct_bytes

        W = window()
        per = rows_bytes(1, n_read, n_columns, clustered)
        held = 0 if slabs else W.rows_within(W.radius)
        if not slabs:
            comps.append(Component("rows within the first window", held * per,
                                   held * per * _UPPER, phases=("fit",),
                                   formula=f"{held:,} rows within {W.radius:.6g} of the "
                                           f"cutoff x {per:.0f} B"))
        from ._slabs import _HELD

        slab_rows = max(int(W.affordable_rows() / (_HELD + 1)), 0)
        if W.distinct is not None:
            top = max(W.distinct.values(), default=0)
            comps.append(Component("exact distinct counts", distinct_bytes(top),
                                   distinct_bytes(top) * _UPPER, phases=("stats",),
                                   formula=f"{top:,} distinct values, in one pass"))
        else:
            top = max(W.approx_distinct.values(), default=0)
            comps.append(Component("exact distinct counts", distinct_bytes(slab_rows),
                                   distinct_bytes(slab_rows) * _UPPER, phases=("fit",),
                                   formula=f"about {top:,} distinct values, counted from the "
                                           f"slabs a slab at a time"))
            warnings.append("the exact distinct counts do not fit in one pass: the rows go "
                            "to slabs on disk, and the counts come from them")
        disk = W.n_keep * 8 * (len(W.value_columns) + (W.cluster is not None))
        comps.append(Component("slabs on disk" if slabs else
                               "slabs on disk (when a step's window does not fit in memory)",
                               disk, disk * _UPPER, where="disk", phases=("fit",),
                               formula=f"{W.n_keep:,} rows x {len(W.value_columns)} columns "
                                       f"x 8 B; read {slab_rows:,} rows a slab"))
        if slabs:
            held_b = (_HELD + 1) * slab_rows * per
            comps.append(Component("slabs held in memory", held_b, held_b * _UPPER,
                                   phases=("fit",),
                                   formula=f"{_HELD + 1} slabs x {slab_rows:,} rows x "
                                           f"{per:.0f} B"))
        elif math.isfinite(W.r_max) and W.radius >= W.r_max:
            warnings.append("the first window reaches the memory budget's limit: the wider "
                            "steps will read the rows a slab at a time from disk")
    else:
        ub = units_bytes(m, len(outcomes), n_columns)
        comps.append(Component("by-value table", ub, ub * _UPPER, phases=("fit",),
                               formula=f"{m:,} distinct values x {len(outcomes)} outcome(s)"))
        rb = read_buffer_bytes(source, n_read, threads)
        comps.append(Component("streaming read buffers", rb, rb * 2, phases=("fit",),
                               formula=f"{threads} threads x a row group x {n_read} columns"))
    notes = {"rows": n, "distinct_values": m, "memory_path": path,
             "collapses": bool(collapsible and m <= n / 2), "outcomes": list(outcomes)}
    if path == "window":
        notes.update(window_radius=W.radius, window_rows=held, largest_radius=W.r_max,
                     slab_rows=slab_rows)
    return Profile(model=f"rd-stream: {model}, {path}", components=comps, scan=scan,
                   threads=threads, baseline_gb=baseline_gb(), memory_limit_gb=budget.limit_gb,
                   memory_reserve=budget.reserve, warnings=warnings, notes=notes)
