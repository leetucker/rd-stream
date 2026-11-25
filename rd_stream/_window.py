"""The window path (``memory="window"``): a fit that holds only the rows
near the cutoff, for data whose rows do not fit in memory and whose running
variable has too many distinct values to group by value.

Every step of rdbwselect and rdrobust is a sum over the units within a
bandwidth of the cutoff, except for a few full-sample quantities:

- the standard deviations of x and y, the quartiles of x (type 2), its range
  and the counts on each side: streaming sums, and exact order statistics
  from a histogram of distances to the cutoff and one filtered read of the
  rows in the quartiles' bins;
- the distinct values of x on each side (M, which ``mass_points="adjust"``
  puts in the pilot bandwidth) and of the clusters: exact counts
  (hdfe_stream.core.distinct) in one pass when their hash tables fit the
  memory budget, else counted from the slabs (`SlabStore.distinct`), which
  hold each distance in one slab: one read of the slabs instead of a pass
  over the source per hash partition. Until then the pilot radius takes the
  approximate (HyperLogLog) counts;
- rdbwselect's fit over each side's whole range for C_d: it needs no
  residuals, so it is a function of power sums over the side, taken in one
  streaming pass (`FullSums`).

The rows within a radius of the cutoff are then read (one filtered,
streaming pass) and sorted, and the fit runs as in memory on them. A step
whose window reaches past the radius raises `NeedWider`; the rows are read
again out to the distance it needs, while they fit the memory budget, and
the fit starts over. The first radius is a few pilot bandwidths, so that
most fits read the rows once.

The results are the rows path's: the same rows enter every window sum
in the same order, and the full-sample statistics are the same numbers,
up to the order of summation.
"""

from __future__ import annotations

import math
import time

import numpy as np
import polars as pl

from ._engine import _C_C, FullSums

_BINS = 4096                 # distance bins per side
_COLLECT_ROWS = 2_000_000    # an order statistic collects at most this many rows of a bin
_START = 3.0                 # the first radius, in pilot bandwidths
_GROW = 1.25                 # a wider read goes this far past the distance asked for
_SLABS_ALWAYS = False        # tests: slabs whatever the budget
_SLAB_ROWS = None            # tests: rows a slab, instead of the budget's share
_MIN_SLAB_ROWS = 1000        # fewer rows a slab than this: the budget is too small
_DISTINCT_SHARE = 0.5        # the exact counts in one pass take at most this share of the headroom


def _finite(e):
    return e.is_not_null() & e.is_not_nan()


class WindowData:
    """The full-sample statistics of each outcome, and the rows within the
    current radius of the cutoff; or, when a window the fit needs does not
    fit in memory, the rows in slabs on disk (`store`, see _slabs.py)."""

    def __init__(self, data, running, outcomes, cluster, weights, treatment, covariates, o,
                 *, per_row_bytes, max_memory_gb, memory_reserve, hint=None, read=True,
                 workdir=None, slabs=False, keep=None):
        from hdfe_stream.core.utils import as_lazy

        self.running, self.outcomes = running, list(outcomes)
        self.cluster, self.weights, self.treatment = cluster, weights, treatment
        self.covariates = list(covariates)
        self.o, self.c = o, o.cutoff
        self.per_row = per_row_bytes
        self.max_memory_gb, self.memory_reserve = max_memory_gb, memory_reserve
        cols = [running, *outcomes]
        for c in (cluster, weights, treatment, *covariates):
            if c is not None and c not in cols:
                cols.append(c)
        keep_rows = keep
        keep = _finite(pl.col(running).cast(pl.Float64))
        for c in (cluster, weights):
            if c is not None:
                keep &= pl.col(c).is_not_null()
        if weights is not None:
            keep &= pl.col(weights) >= 0
        if keep_rows is not None:             # a donut hole, or a placebo cutoff's side
            keep &= keep_rows
        self.data, self.columns, self.keep, self.workdir = data, cols, keep, workdir
        self.value_columns = [c for c in cols if c != cluster]
        self.store, self.slabbed = None, False
        self.lf = as_lazy(data).select(cols).filter(keep)
        self.X = pl.col(running).cast(pl.Float64)
        self.dist = (self.X - self.c).abs()
        self.left = self.X < self.c
        t0 = time.perf_counter()
        self._summarize()
        self.stats_s = time.perf_counter() - t0
        self.radius = None
        self.read_s = 0.0
        self._start(hint, read, slabs)

    # ---- the full-sample statistics

    def _masks(self, row):
        """Each outcome's rows (complete y, take-up and covariates): the
        common mask "c" when the outcome has no missing values there."""
        self.mask_of = {}
        for i, y in enumerate(self.outcomes):
            self.mask_of[y] = "c" if row[f"n_ok{i}"] == row["n_common"] else f"y{i}"
        self.masks = {}
        others = [_finite(pl.col(c).cast(pl.Float64))
                  for c in ([self.treatment] if self.treatment else []) + self.covariates]
        common = pl.all_horizontal(others) if others else pl.lit(True)
        for y in self.outcomes:
            m = self.mask_of[y]
            if m not in self.masks:
                self.masks[m] = (common if m == "c"
                                 else common & _finite(pl.col(y).cast(pl.Float64)))

    def _summarize(self):
        c, X, lf = self.c, self.X, self.lf
        others = [_finite(pl.col(v).cast(pl.Float64))
                  for v in ([self.treatment] if self.treatment else []) + self.covariates]
        common = pl.all_horizontal(others) if others else pl.lit(True)
        aggs = [pl.len().alias("n_keep"), X.min().alias("x_lo"), X.max().alias("x_hi"),
                common.sum().alias("n_common")]
        for i, y in enumerate(self.outcomes):
            aggs.append((common & _finite(pl.col(y).cast(pl.Float64))).sum().alias(f"n_ok{i}"))
        row = lf.select(aggs).collect(engine="streaming").row(0, named=True)
        self.n_keep = int(row["n_keep"])
        if self.n_keep == 0 or row["x_lo"] is None or not (row["x_lo"] < c < row["x_hi"]):
            raise ValueError("the cutoff must be strictly inside the range of the running variable")
        self.Dn = {"l": c - row["x_lo"], "r": row["x_hi"] - c}      # the sides' largest distances
        self._masks(row)

        # pass A: per mask, counts, means, ranges; the take-up's range on each side; Z'Z
        tcol = None if self.treatment is None else pl.col(self.treatment).cast(pl.Float64)
        zs = [pl.col(z).cast(pl.Float64) for z in self.covariates]
        aggs = []
        for m, M in self.masks.items():
            aggs += [M.sum().alias(f"{m}:n"), (M & self.left).sum().alias(f"{m}:nl"),
                     X.filter(M).mean().alias(f"{m}:xm"), X.filter(M).min().alias(f"{m}:lo"),
                     X.filter(M).max().alias(f"{m}:hi")]
            if tcol is not None:
                for s, S in (("l", self.left), ("r", ~self.left)):
                    aggs += [tcol.filter(M & S).min().alias(f"{m}:t{s}lo"),
                             tcol.filter(M & S).max().alias(f"{m}:t{s}hi")]
            for a in range(len(zs)):
                for b in range(a, len(zs)):
                    aggs.append((zs[a] * zs[b]).filter(M).sum().alias(f"{m}:zz{a}_{b}"))
        for i, y in enumerate(self.outcomes):
            M = self.masks[self.mask_of[y]]
            aggs.append(pl.col(y).cast(pl.Float64).filter(M).mean().alias(f"ym{i}"))
        A = lf.select(aggs).collect(engine="streaming").row(0, named=True)
        self.A = A
        for m in self.masks:
            if not A[f"{m}:n"] or A[f"{m}:lo"] is None or not (A[f"{m}:lo"] < c < A[f"{m}:hi"]):
                raise ValueError("the cutoff must be strictly inside the range of the running "
                                 "variable")

        # pass B: centered sums of squares, the power sums over each side, the histogram
        o = self.o
        Mw, MD = 2 * o.q + 6, o.q + 4            # rdbwselect's range fit: order q + 2, kernel deg 2
        f = (pl.lit(1.0) if self.weights is None else pl.col(self.weights).cast(pl.Float64))
        aggs = []
        for m, M in self.masks.items():
            aggs.append(((X - A[f"{m}:xm"]) ** 2).filter(M).sum().alias(f"{m}:xss"))
            for s, S, D in (("l", self.left, c - A[f"{m}:lo"]), ("r", ~self.left,
                                                                 A[f"{m}:hi"] - c)):
                u = self.dist / D
                for a in range(Mw + 1):
                    aggs.append((f * u ** a).filter(M & S).sum().alias(f"{m}:{s}w{a}"))
                for j, col in enumerate(self._columns(m)):
                    for a in range(MD + 1):
                        aggs.append((f * u ** a * col).filter(M & S).sum()
                                    .alias(f"{m}:{s}D{j}_{a}"))
        for i, y in enumerate(self.outcomes):
            M = self.masks[self.mask_of[y]]
            aggs.append(((pl.col(y).cast(pl.Float64) - A[f"ym{i}"]) ** 2).filter(M).sum()
                        .alias(f"yss{i}"))
        self.B = lf.select(aggs).collect(engine="streaming").row(0, named=True)
        self.Mw, self.MD = Mw, MD

        # the histogram of distances: per side and bin, the rows, and per mask their x range
        bin_ = (pl.when(self.left).then(self.dist / self.Dn["l"]).otherwise(self.dist / self.Dn["r"])
                * _BINS).floor().clip(0, _BINS - 1).cast(pl.Int32)
        aggs = [pl.len().alias("n"), self.dist.min().alias("dlo"), self.dist.max().alias("dhi")]
        for m, M in self.masks.items():
            aggs += [M.sum().alias(f"{m}:n"), X.filter(M).min().alias(f"{m}:lo"),
                     X.filter(M).max().alias(f"{m}:hi")]
        self.hist = (lf.group_by(self.left.alias("left"), bin_.alias("bin")).agg(aggs)
                     .collect(engine="streaming"))
        self.bin_expr = bin_

        # exact distinct values of x (as distances, like the runs of the sorted rows) and of
        # the clusters, on each side
        from hdfe_stream.core.distinct import count_distinct, distinct_bytes
        from stream_limits import MemoryBudget
        keys, where = {}, {}
        for m, M in self.masks.items():
            for s, S in (("l", self.left), ("r", ~self.left)):
                keys[f"{m}:M{s}"] = self.dist
                where[f"{m}:M{s}"] = M & S
                if self.cluster is not None:
                    keys[f"{m}:G{s}"] = self.cluster
                    where[f"{m}:G{s}"] = M & S
        self.distinct_keys, self.distinct_where = keys, where
        self.approx_distinct = count_distinct(lf, keys, where=where)
        need = sum(distinct_bytes(v) for v in self.approx_distinct.values()) * 1.25
        room = MemoryBudget(limit_gb=self.max_memory_gb,
                            reserve=self.memory_reserve).headroom_gb() * 1e9
        self.distinct = None              # counted from the slabs (to_slabs)
        if need <= _DISTINCT_SHARE * room:
            self.distinct = count_distinct(lf, keys, exact=True, where=where, partitions=1)
        self._order = {}

    def _columns(self, m):
        """The response columns whose power sums mask m needs: its outcomes,
        the take-up and the covariates."""
        cols = [pl.col(y).cast(pl.Float64) for y in self.outcomes if self.mask_of[y] == m]
        if self.treatment:
            cols.append(pl.col(self.treatment).cast(pl.Float64))
        cols += [pl.col(z).cast(pl.Float64) for z in self.covariates]
        return cols

    # ---- exact order statistics of x

    def _quantile(self, m, prob):
        """rdrobust's type-2 quantile of x among mask m's rows."""
        n = int(self.A[f"{m}:n"])
        hq = n * prob
        j = int(math.floor(hq))
        if hq == j:
            return 0.5 * (self._kth(m, j - 1) + self._kth(m, j))
        return self._kth(m, j)

    def _kth(self, m, k):
        """The k-th smallest x (0-based) among mask m's rows: its bin from the
        histogram (x ascending: the left side's bins from the farthest, then
        the right side's from the nearest), then the bin's rows."""
        h = self.hist.filter(pl.col(f"{m}:n") > 0)
        left = h.filter(pl.col("left")).sort("bin", descending=True)
        right = h.filter(~pl.col("left")).sort("bin")
        order = pl.concat([left, right])
        cum = np.cumsum(order[f"{m}:n"].to_numpy())
        i = int(np.searchsorted(cum, k + 1, side="left"))
        before = int(cum[i - 1]) if i else 0
        row = order.row(i, named=True)
        lo, hi = row[f"{m}:lo"], row[f"{m}:hi"]
        if lo == hi:
            return float(lo)
        cond = (self.masks[m] & (self.left == row["left"]) & (self.bin_expr == row["bin"]))
        return self._kth_in(cond, k - before, lo, hi, int(row[f"{m}:n"]))

    def _kth_in(self, cond, k, lo, hi, n):
        """The k-th smallest x among the rows where `cond` holds, all in
        [lo, hi]: collected and sorted when there are few enough, else from a
        finer histogram of the range."""
        key = (str(cond), k)
        if key in self._order:
            return self._order[key]
        if lo == hi:
            return float(lo)
        if n <= _COLLECT_ROWS:
            xs = np.sort(self.lf.filter(cond).select(self.X).collect(engine="streaming")
                         .to_series().to_numpy())
            v = float(xs[k])
        else:
            sub = ((self.X - lo) / (hi - lo) * _BINS).floor().clip(0, _BINS - 1).cast(pl.Int32)
            hist = (self.lf.filter(cond).group_by(sub.alias("bin"))
                    .agg(pl.len().alias("n"), self.X.min().alias("lo"), self.X.max().alias("hi"))
                    .sort("bin").collect(engine="streaming"))
            cum = np.cumsum(hist["n"].to_numpy())
            i = int(np.searchsorted(cum, k + 1, side="left"))
            before = int(cum[i - 1]) if i else 0
            r = hist.row(i, named=True)
            v = self._kth_in(cond & (sub == r["bin"]), k - before, r["lo"], r["hi"], int(r["n"]))
        self._order[key] = v
        return v

    def _count(self, key):
        """An exact distinct count; before the slabs count them, the
        approximate one (only the pilot radius takes it)."""
        return (self.distinct if self.distinct is not None else self.approx_distinct)[key]

    def stats(self, y):
        """The full-sample statistics `select_bandwidths` and `_prepare` take
        for outcome y, with each side's power sums and counts."""
        i = self.outcomes.index(y)
        m = self.mask_of[y]
        A, B, c = self.A, self.B, self.c
        n, nl = int(A[f"{m}:n"]), int(A[f"{m}:nl"])
        out = dict(x_sd=math.sqrt(B[f"{m}:xss"] / (n - 1)),
                   y_sd=math.sqrt(B[f"yss{i}"] / (n - 1)),
                   x_iq=self._quantile(m, 0.75) - self._quantile(m, 0.25),
                   x_min=float(A[f"{m}:lo"]), x_max=float(A[f"{m}:hi"]), N=n,
                   n_l=nl, n_r=n - nl,
                   M_l=self._count(f"{m}:Ml"), M_r=self._count(f"{m}:Mr"))
        if self.cluster is not None:
            out.update(clustered=True, G_l=self._count(f"{m}:Gl"), G_r=self._count(f"{m}:Gr"))
        if self.treatment:
            out["take_up"] = {s: (A[f"{m}:t{s}lo"], A[f"{m}:t{s}hi"]) for s in ("l", "r")}
        nz = len(self.covariates)
        ZZ = np.empty((nz, nz))
        for a in range(nz):
            for b in range(a, nz):
                ZZ[a, b] = ZZ[b, a] = A[f"{m}:zz{a}_{b}"]
        out["ZZ"] = ZZ
        # the power sums of the columns [y, (take-up), covariates]
        mine = [yy for yy in self.outcomes if self.mask_of[yy] == m]
        cols = [mine.index(y)] + list(range(len(mine), len(self._columns(m))))
        full = {}
        for s, D, sgn in (("l", c - A[f"{m}:lo"], -1.0), ("r", A[f"{m}:hi"] - c, 1.0)):
            Sw = np.array([B[f"{m}:{s}w{a}"] for a in range(self.Mw + 1)])
            SD = np.array([[B[f"{m}:{s}D{j}_{a}"] for j in cols] for a in range(self.MD + 1)])
            full[s] = FullSums(Sw, SD, D, D, sgn)
        out["full"] = full
        return out

    # ---- the rows near the cutoff

    def _rows_within(self):
        """(lower edges, rows) of every bin of both sides, by edge: the rows
        within a radius r are at most the rows of the bins whose lower edge
        is at most r."""
        h = self.hist
        edge = np.where(h["left"].to_numpy(), self.Dn["l"], self.Dn["r"]) * (
            h["bin"].to_numpy() / _BINS)
        order = np.argsort(edge, kind="stable")
        return edge[order], np.cumsum(h["n"].to_numpy()[order])

    def affordable_rows(self):
        """The rows the memory budget's headroom holds."""
        from stream_limits import MemoryBudget

        budget = MemoryBudget(limit_gb=self.max_memory_gb, reserve=self.memory_reserve)
        return budget.headroom_gb() * 1e9 / (1.25 * self.per_row)

    def affordable_radius(self):
        """The largest radius whose rows fit the memory budget (inf: all)."""
        rows = self.affordable_rows()
        edge, cum = self._rows_within()
        over = np.nonzero(cum > rows)[0]
        if not over.size:
            return math.inf
        e = edge[over[0]]
        return float(np.nextafter(e, -math.inf)) if e > 0 else -math.inf

    def rows_within(self, r):
        edge, cum = self._rows_within()
        i = int(np.searchsorted(edge, r, side="right"))
        return int(cum[i - 1]) if i else 0

    def _pilot(self):
        """The largest pilot bandwidth c_bw over the outcomes, in x's units."""
        o, out = self.o, 0.0
        for y in self.outcomes:
            st = self.stats(y)
            M = st["M_l"] + st["M_r"] if o.masspoints == "adjust" else st["N"]
            out = max(out, _C_C[o.kern] * min(st["x_sd"], st["x_iq"] / 1.349) * M ** (-1 / 5))
        return out

    def _start(self, hint, read=True, slabs=False):
        """The first read: out to the given bandwidths, or a few pilot
        bandwidths (as far as the budget allows); to slabs on disk when the
        given bandwidths' rows do not fit, or with `slabs`. Without `read`,
        only the radius (for a profile)."""
        self.r_max = self.affordable_radius()
        r = hint * (1 + 1e-8) if hint is not None else min(_START * self._pilot(), self.r_max)
        if not read:
            self.radius = r
        elif slabs or _SLABS_ALWAYS or self.distinct is None or r > self.r_max or r <= 0:
            self.to_slabs()
        else:
            self._read(r)

    def widen(self, need):
        """Read the rows again, out to `need` (None: twice the radius), or
        write them to slabs when that does not fit."""
        r = self.radius
        target = 2 * r if need is None else max(need * _GROW, 1.5 * r)
        if need is not None and need > self.r_max or need is None and r >= self.r_max:
            self.to_slabs()
            return
        self._read(min(target, self.r_max))

    def to_slabs(self):
        """Write the rows to slabs on disk, each a share of the memory budget,
        and let go of the rows held."""
        from ._slabs import _HELD, SlabStore

        self.x = self.ys = self.w = self.cl = self.t = self.Z = None
        rows = int(self.affordable_rows() / (_HELD + 1))
        if _SLAB_ROWS:
            rows = _SLAB_ROWS
        elif rows < _MIN_SLAB_ROWS:
            raise MemoryError(
                f"the memory budget leaves room for {max(rows, 0):,} rows a slab (max_memory_gb, "
                f"less memory_reserve, less what this process holds), too few to fit "
                f"{self.n_keep:,} rows a slab at a time")
        t0 = time.perf_counter()
        self.store = SlabStore(self, rows, self.workdir)
        if self.distinct is None:
            self.distinct = self.store.distinct(self.masks)
        self.read_s += time.perf_counter() - t0
        self.slabbed, self.radius = True, math.inf

    def cleanup(self):
        if self.store is not None:
            self.store.cleanup()

    def _read(self, r):
        if r > self.r_max or r <= 0:
            raise MemoryError(
                f"the rows within {r:.6g} of the cutoff (about {self.rows_within(r):,} of "
                f"{self.n_keep:,}) do not fit the memory budget (max_memory_gb, less "
                f"memory_reserve, less what this process holds)")
        if r >= max(self.Dn.values()):
            r = math.inf
        t0 = time.perf_counter()
        lf = self.lf if math.isinf(r) else self.lf.filter(self.dist <= r)
        df = lf.collect(engine="streaming").sort(self.running, maintain_order=True)
        self.read_s += time.perf_counter() - t0
        self.radius = r
        self.x = np.array(df[self.running].to_numpy(), dtype=np.float64)
        self.ys = {y: np.array(df[y].cast(pl.Float64).to_numpy()) for y in self.outcomes}
        self.cl, self.n_clusters = None, 0
        if self.cluster is not None:
            codes = df[self.cluster].rank("dense").to_numpy().astype(np.int64) - 1
            self.cl, self.n_clusters = codes, int(codes.max()) + 1 if codes.size else 0
        self.w = (None if self.weights is None
                  else np.array(df[self.weights].cast(pl.Float64).to_numpy()))
        self.t = (None if self.treatment is None
                  else np.array(df[self.treatment].cast(pl.Float64).to_numpy()))
        self.Z = None
        if self.covariates:
            self.Z = np.column_stack([df[z].cast(pl.Float64).to_numpy() for z in self.covariates])

    def side_complete(self, side):
        """Whether the rows held cover every row of a side ("l" or "r")."""
        return self.radius >= self.Dn[side]
