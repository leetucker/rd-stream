"""Slabs: the window path's rows on disk, when the widest window a fit needs
does not fit in memory.

The rows are written once, in one pass over the source a batch at a time,
to Parquet files by side of the cutoff and by distance from it: consecutive
bins of the distance histogram, nearest first, as many rows to a slab as a
few slabs in memory leave room for. A `SlabSide` then stands in for a
`Side`: every window sum a step takes (moments, residuals and their
variance meats, cluster sums) is the sum over the slabs the window
reaches, each read, sorted and held in turn. Nearest-neighbor residuals
need each unit's neighbors, which for a slab's first and last units are
in the slabs before and after it: a run of equal values reaches at most
`n_neighbors` runs either way, so the slab's units are taken with that many
runs of the neighboring slabs on each side.

Memory then grows with a slab (a few held at once), and disk with the rows.
"""

from __future__ import annotations

import glob
import math
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import polars as pl

from . import _cluster
from . import _kernels as K
from ._engine import Side, gram
from ._moments import moments_mixed, nn_runs, run_sums

_BATCH_ROWS = 1_000_000
_HELD = 4                     # slabs held at once: the cache, and a slab's neighbors
_CODE = "__rd_cluster_code"


def _batches(data, columns, rows):
    """The source's `columns`, `rows` at a time, as DataFrames: Parquet files
    through pyarrow without pre-buffering (whose memory does not grow with
    the file), frames by slices."""
    if isinstance(data, pl.DataFrame):
        yield from data.select(columns).iter_slices(rows)
        return
    if isinstance(data, pl.LazyFrame):
        yield from data.select(columns).collect_batches(chunk_size=rows)
        return
    import pyarrow.parquet as pq

    srcs = data if isinstance(data, (list, tuple)) else [data]
    paths = []
    for s in srcs:
        s = os.fspath(s)
        if os.path.isdir(s):
            paths += sorted(glob.glob(os.path.join(s, "**", "*.parquet"), recursive=True))
        elif any(ch in s for ch in "*?["):
            paths += sorted(glob.glob(s, recursive=True))
        else:
            paths.append(s)
    for path in paths:
        pf = pq.ParquetFile(path, pre_buffer=False)
        try:
            for batch in pf.iter_batches(batch_size=rows, columns=list(columns)):
                yield pl.from_arrow(batch)
        finally:
            pf.close()


class SlabStore:
    """The rows of a `WindowData`'s source by side and distance, on disk."""

    def __init__(self, W, slab_rows, workdir=None):
        from hdfe_stream.core.workspace import Run

        base = workdir or os.environ.get("HDFE_STREAM_WORKDIR") or tempfile.gettempdir()
        self.run = Run(Path(base), auto_cleanup=True)
        self.W = W
        self.running = W.running
        self.slab_rows = int(slab_rows)
        # slabs by distance: cells of the histogram (bins, split by finer histograms
        # where a bin holds more rows than a slab), grouped nearest first; each
        # slab's lower edge is the smallest distance in its first cell
        self.lower, self.n_slabs = {}, {}
        for side, left in (("l", True), ("r", False)):
            cells = self._cells(left)
            lower, acc = [], 0
            for d0, n in cells:
                if not lower or acc + n > slab_rows:
                    lower.append(d0)
                    acc = 0
                acc += n
            self.lower[side] = np.array(lower or [0.0])
            self.n_slabs[side] = len(self.lower[side])
        self.codes = None
        if W.cluster is not None:
            self.codes = (W.lf.select(pl.col(W.cluster).unique())
                          .collect(engine="streaming").with_row_index(_CODE))
        self.n_clusters = 0 if self.codes is None else self.codes.height
        self._write()

    def _cells(self, left):
        """(smallest distance, rows) of each cell of a side, nearest first:
        the histogram's bins, a bin with more rows than a slab split by a
        finer histogram of its distances (again, to four levels)."""
        h = self.W.hist.filter(pl.col("left") == left).sort("bin")
        out = []
        for row in h.iter_rows(named=True):
            if row["n"] <= self.slab_rows:
                out.append((row["dlo"], row["n"]))
            else:
                out += self._split(self.W.left == left, self.W.bin_expr == row["bin"],
                                   row["dlo"], row["dhi"], row["n"], 1)
        return out

    def _split(self, side, cond, lo, hi, n, depth):
        from . import _window

        W, nb = self.W, _window._BINS
        if lo == hi or depth > 4:
            raise MemoryError(
                f"{n:,} rows are {'at the distance' if lo == hi else 'within'} {lo:.6g} "
                f"{'' if lo == hi else f'to {hi:.6g} '}from the cutoff, more than the "
                f"{self.slab_rows:,} rows a slab can hold in the memory budget; give more "
                f"memory (max_memory_gb)")
        sub = ((W.dist - lo) / (hi - lo) * nb).floor().clip(0, nb - 1).cast(pl.Int32)
        h = (W.lf.filter(side & cond).group_by(sub.alias("s"))
             .agg(pl.len().alias("n"), W.dist.min().alias("dlo"), W.dist.max().alias("dhi"))
             .sort("s").collect(engine="streaming"))
        out = []
        for row in h.iter_rows(named=True):
            if row["n"] <= self.slab_rows:
                out.append((row["dlo"], row["n"]))
            else:
                out += self._split(side, cond & (sub == row["s"]), row["dlo"], row["dhi"],
                                   row["n"], depth + 1)
        return out

    def path(self, side, j):
        return self.run.path / f"{side}{j:05d}.parquet"

    def _write(self):
        """One pass over the source: each batch's kept rows to their slabs'
        files, in the source's order."""
        import pyarrow.parquet as pq

        W = self.W
        src_cols = list(dict.fromkeys(W.columns))
        out_cols = [W.running, *[c for c in W.value_columns if c != W.running]]
        f64 = [pl.col(c).cast(pl.Float64) for c in out_cols]
        nl = self.n_slabs["l"]
        writers = {}
        try:
            for batch in _batches(W.data, src_cols, min(_BATCH_ROWS, self.slab_rows)):
                b = batch.filter(W.keep)
                if not b.height:
                    continue
                key = b.select(W.left.alias("l"), W.dist.alias("d"))
                left = key["l"].to_numpy()
                dist = key["d"].to_numpy()
                slab = np.where(
                    left, np.searchsorted(self.lower["l"], dist, side="right") - 1,
                    nl + np.searchsorted(self.lower["r"], dist, side="right") - 1)
                slab = np.maximum(slab, np.where(left, 0, nl))
                vals = b.select(f64)
                if self.codes is not None:
                    code = (b.select(W.cluster).join(self.codes, on=W.cluster, how="left",
                                                     maintain_order="left")[_CODE])
                    vals = vals.with_columns(code.cast(pl.Int64))
                order = np.argsort(slab, kind="stable")
                s_sorted = slab[order]
                starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
                ends = np.r_[starts[1:], s_sorted.shape[0]]
                taken = vals[order]
                for a, e in zip(starts, ends, strict=True):
                    sid = int(s_sorted[a])
                    tab = taken.slice(int(a), int(e - a)).to_arrow()
                    w = writers.get(sid)
                    if w is None:
                        side, j = ("l", sid) if sid < nl else ("r", sid - nl)
                        w = writers[sid] = pq.ParquetWriter(self.path(side, j), tab.schema)
                    w.write_table(tab)
        finally:
            for w in writers.values():
                w.close()
        self.present = {(("l", s) if s < nl else ("r", s - nl)) for s in writers}

    def distinct(self, masks):
        """Exact distinct counts of each side's distances and clusters per
        mask, keyed like `WindowData`'s ("{mask}:M{side}", "{mask}:G{side}"):
        a distance lies in one slab, so its counts add over the slabs, and
        the clusters' codes are marked in a table of the clusters a side
        has. One read of the slabs, a slab at a time."""
        W = self.W
        out = {f"{m}:M{s}": 0 for m in masks for s in ("l", "r")}
        seen = ({(m, s): np.zeros(self.n_clusters, dtype=bool) for m in masks
                 for s in ("l", "r")} if self.codes is not None else {})
        for side, j in sorted(self.present):
            df = pl.read_parquet(self.path(side, j))
            for m, M in masks.items():
                sub = df.filter(M)
                out[f"{m}:M{side}"] += int(sub.select(W.dist.n_unique()).item())
                if seen:
                    seen[m, side][sub[_CODE].to_numpy()] = True
        for (m, s), mark in seen.items():
            out[f"{m}:G{s}"] = int(mark.sum())
        return out

    def frame(self, side, j):
        """Slab j of a side, its rows in order of distance from the cutoff."""
        if (side, j) not in self.present:
            return None
        df = pl.read_parquet(self.path(side, j)).sort(self.running, maintain_order=True)
        return df.reverse() if side == "l" else df

    def cleanup(self):
        self.run.cleanup()


class SlabSide:
    """One side of the cutoff whose units are in a `SlabStore`'s slabs: the
    window sums of a `Side`, a slab at a time. The response columns are
    [y, (take-up), covariates] (`cols` of them), with x and y divided by
    x_sd and y_sd when the bandwidths are selected on standardized data."""

    slabbed = True
    prefer_full = True            # C_d's range fit from the power sums, not a pass over slabs
    radius = radius_raw = math.inf

    def __init__(self, store, side, c, *, y, extra, dT, collapse, full, n_side, M_side,
                 x_sd=1.0, y_sd=1.0, cols=None, weights=None):
        self.store, self.side, self.c = store, side, c
        self.y, self.extra, self.dT = y, list(extra), dT
        self.collapse, self.weights = collapse, weights
        self.x_sd, self.y_sd = x_sd, y_sd
        self.x_scale = x_sd
        ncol_all = 1 + len(self.extra)
        self.cols = list(range(ncol_all)) if cols is None else list(cols)
        self.ncol = len(self.cols)
        self.dZ = self.ncol - 1 - dT
        self.full = full
        self.n_side, self.M_side = int(n_side), int(M_side)
        self.M = self.M_side                      # every run is reachable
        self.nobs = self.n_side
        self.n_clusters = store.n_clusters
        self.n_slabs = store.n_slabs[side]
        self._cache = OrderedDict()
        self._meta = []                           # per slab: units, obs, runs, last t

    @property
    def clustered(self):
        return self.store.codes is not None

    def _copy(self, **kw):
        args = dict(y=self.y, extra=self.extra, dT=self.dT, collapse=self.collapse,
                    full=self.full, n_side=self.n_side, M_side=self.M_side, x_sd=self.x_sd,
                    y_sd=self.y_sd, cols=self.cols, weights=self.weights)
        args.update(kw)
        return SlabSide(self.store, self.side, self.c, **args)

    def scaled(self, x_sd, y_sd):
        return self._copy(x_sd=x_sd, y_sd=y_sd,
                          full=None if self.full is None else self.full.scaled(x_sd, y_sd))

    def columns(self, cols, dT=0):
        return self._copy(cols=[self.cols[j] for j in cols], dT=dT,
                          full=None if self.full is None else self.full.columns(cols))

    def covers_side(self, h):
        return self.full is not None and h > self.full.d_max

    # ---- the slabs

    def _local(self, j):
        """Slab j as a `Side` (None when empty), from the cache or disk."""
        if j in self._cache:
            self._cache.move_to_end(j)
            return self._cache[j]
        df = self.store.frame(self.side, j)
        loc = None
        if df is not None:
            names = [self.y, *self.extra]
            ok = pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_not_nan()
                                    for c in names])
            df = df.filter(ok)
            if df.height:
                x = df[self.store.running].to_numpy()
                D = np.column_stack([df[c].to_numpy() for c in names])[:, self.cols]
                D = np.ascontiguousarray(D)
                f = None if self.weights is None else df[self.weights].to_numpy()
                cl = None if not self.clustered else df[_CODE].to_numpy()
                if self.collapse:
                    xa = x if self.side == "r" else x[::-1]           # ascending
                    Da = D if self.side == "r" else D[::-1]
                    start = K.runs(xa)
                    cnt, SD, M2 = K.collapse_runs_multi(xa, np.ascontiguousarray(Da), start)
                    xu = xa[start[:-1]]
                    if self.side == "l":
                        xu, cnt, SD, M2 = xu[::-1], cnt[::-1], SD[::-1], M2[::-1]
                    loc = Side(np.ascontiguousarray(xu), self.c, cnt=np.ascontiguousarray(cnt),
                               SD=np.ascontiguousarray(SD), M2=np.ascontiguousarray(M2),
                               x_sd=self.x_sd, y_sd=self.y_sd, dT=self.dT)
                else:
                    loc = Side(np.ascontiguousarray(x), self.c, D=D, f=f, cl=cl,
                               n_clusters=self.n_clusters, x_sd=self.x_sd, y_sd=self.y_sd,
                               dT=self.dT)
        while len(self._meta) <= j:
            self._meta.append(None)
        self._meta[j] = ((0, 0, 0, 0.0) if loc is None else
                         (loc.n_units, loc.nobs_in(loc.n_units), loc.M, float(loc.t[-1])))
        self._cache[j] = loc
        while len(self._cache) > _HELD:
            self._cache.popitem(last=False)
        return loc

    def _info(self, j):
        if j >= len(self._meta) or self._meta[j] is None:
            self._local(j)
        return self._meta[j]

    def _starts(self, j):
        """(units, obs, runs) before slab j."""
        u = n = r = 0
        for i in range(j):
            m = self._info(i)
            u, n, r = u + m[0], n + m[1], r + m[2]
        return u, n, r

    def _upto(self, k):
        """(j, local window, units before) of each slab the first k units
        reach; the slab itself is read by the caller (`_local`), one at a time."""
        u = 0
        for j in range(self.n_slabs):
            if u >= k:
                return
            m = self._info(j)
            if m[0] == 0:
                continue
            yield j, min(k - u, m[0]), u
            u += m[0]

    def _count(self, k, field, local):
        """A count over the first k units: whole slabs from their metadata,
        the slab the window ends in from its units."""
        out = 0
        for j, lk, _ in self._upto(k):
            m = self._info(j)
            out += m[field] if lk == m[0] else local(self._local(j), lk)
        return out

    # ---- the sums over windows

    def window(self, h, kern):
        k = 0
        for j in range(self.n_slabs):
            m = self._info(j)
            if m[0] == 0:
                continue
            u = abs(m[3] / h)
            if (u <= 1.0) if kern == K.UNIFORM else (u < 1.0):
                k += m[0]
                continue
            return k + K.window_len(self._local(j).t, h, kern)
        return k

    def distinct_in(self, k):
        return self._count(k, 2, lambda loc, lk: loc.distinct_in(lk))

    def nobs_in(self, k):
        return self._count(k, 1, lambda loc, lk: loc.nobs_in(lk))

    def _sum(self, k, fn):
        out = None
        for j, lk, _ in self._upto(k):
            v = fn(self._local(j), lk)
            out = v if out is None else out + v
        return out

    def moments_w(self, k, h, kern, A):
        return self._sum(k, lambda loc, lk: loc.moments_w(lk, h, kern, A))

    def moments_D(self, k, h, kern, A):
        return self._sum(k, lambda loc, lk: loc.moments_D(lk, h, kern, A))

    def zwd(self, k, h, kern):
        return self._sum(k, lambda loc, lk: loc.zwd(lk, h, kern))

    def run_distance(self, j):
        before = 0
        for i in range(self.n_slabs):
            m = self._info(i)
            if before + m[2] >= j:
                return self._local(i).run_distance(j - before)
            before += m[2]
        raise IndexError("fewer distinct values than asked for")

    # ---- residuals

    def _residuals(self, j, loc, lk, svec, win, vce, matches, h, kern, beta, Ginv, d,
                   signed):
        """Slab j's units' squared residuals (and signed row residuals) in
        the window `win` = (observations, distinct values) of the first k
        units, for the combined response D svec."""
        n_window, nrun_g = win
        cs = loc.combine(svec)
        if vce != "nn":
            return cs.residuals(lk, vce, 0, h, kern, beta, Ginv, d, n_window=n_window)
        # the slab's runs in the window, with n_neighbors runs of the slabs either side
        _, _, r0 = self._starts(j)
        core_runs = loc.distinct_in(lk)
        parts_before, need, i = [], matches, j - 1
        while need > 0 and i >= 0:
            nb = self._local(i)
            if nb is not None:
                take = min(need, nb.M)
                parts_before.insert(0, self._tail(nb.combine(svec), take))
                need -= take
            i -= 1
        parts_after = []
        if lk == loc.n_units:
            need = min(matches, nrun_g - (r0 + core_runs))
            i = j + 1
            while need > 0 and i < self.n_slabs:
                nb = self._local(i)
                if nb is not None:
                    take = min(need, nb.M)
                    parts_after.append(self._head(nb.combine(svec), take))
                    need -= take
                i += 1
        core = (cs.d[:lk], cs.sy[:lk], cs.cnt[:lk] if not cs.rows else None,
                cs.m2[:lk] if not cs.rows else None)
        runs_before = sum(int(K.runs(p[0]).shape[0] - 1) for p in parts_before)
        dist = np.concatenate([p[0] for p in parts_before] + [core[0]]
                              + [p[0] for p in parts_after])
        sy = np.concatenate([p[1] for p in parts_before] + [core[1]]
                            + [p[1] for p in parts_after])
        if cs.rows:
            cnt = np.ones(dist.shape[0], np.int64)
        else:
            cnt = np.concatenate([p[2] for p in parts_before] + [core[2]]
                                 + [p[2] for p in parts_after])
        rs = K.runs(dist)
        rsum, rcnt = run_sums(sy[:, None], cnt, cs.rows, rs)
        a, cb, _ = nn_runs(dist, rsum, rcnt, rs, rs.shape[0] - 1, n_window, matches,
                           runs_before, runs_before + core_runs)
        crs = cs.run_start[:core_runs + 1]
        per = np.repeat(np.arange(core_runs), np.diff(crs))
        ar, cr = a[per], cb[per, 0]
        if cs.rows:
            e = ar * cs.sy[:lk] - cr
            return e * e, (e if signed else np.empty(0))
        ybar = cs.sy[:lk] / cs.cnt[:lk]
        e = ar * ybar - cr
        return ar * ar * cs.m2[:lk] + cs.cnt[:lk] * e * e, np.empty(0)

    @staticmethod
    def _tail(s, runs):
        lo = s.run_start[s.M - runs]
        return (s.d[lo:], s.sy[lo:], None if s.rows else s.cnt[lo:],
                None if s.rows else s.m2[lo:])

    @staticmethod
    def _head(s, runs):
        hi = s.run_start[runs]
        return (s.d[:hi], s.sy[:hi], None if s.rows else s.cnt[:hi],
                None if s.rows else s.m2[:hi])

    # ---- the variance meats

    def _window_counts(self, k):
        return self.nobs_in(k), self.distinct_in(k)

    def residual_meat(self, svec, k, h, kern, P, vce, matches, beta, Ginv, d):
        win = self._window_counts(k)
        signed = self.clustered
        out = np.zeros((2 * P - 1,)) if not signed else None
        sums = seen = L = u = None
        for j, lk, _ in self._upto(k):
            loc = self._local(j)
            ss, res = self._residuals(j, loc, lk, svec, win, vce, matches, h, kern, beta, Ginv,
                                      d, signed)
            if not signed:
                out += loc.moments_ss(ss, lk, h, kern, 2 * (P - 1))
                continue
            f, use_f = loc._f()
            if vce in ("cr2", "cr3"):
                Lj, uj, sj = K.cluster_sums(loc.t, f, use_f, lk, h, kern, P, res,
                                            loc.cl[:lk], self.n_clusters)
                L, u, seen = ((Lj, uj, sj) if L is None else (L + Lj, u + uj, seen | sj))
            else:
                scores = K.poly_scores(loc.t, f, use_f, lk, h, kern, P, res)
                sj_, seen_j = K.cluster_score_sums(scores, loc.cl[:lk], self.n_clusters)
                sums, seen = ((sj_, seen_j) if sums is None else (sums + sj_, seen | seen_j))
        if not signed:
            return gram(out, P - 1)
        if vce in ("cr2", "cr3"):
            return _cluster.meat(L, u, seen, Ginv, 2 if vce == "cr2" else 3)
        return _cr1(sums, seen, k, P)

    def variance_sums(self, svec, fit, *, p, q, kern, vce, matches):
        k, kh, kb, h, b = fit.k, fit.kh, fit.kb, fit.h, fit.b
        win = self._window_counts(k)
        mh = np.zeros(2 * p + 1)
        mh_b = np.zeros(2 * p + 1)
        mx = np.zeros(p + q + 1)
        mb = np.zeros(2 * q + 1)
        for j, lk, u0 in self._upto(k):
            loc = self._local(j)
            lh, lb = max(0, min(kh - u0, lk)), max(0, min(kb - u0, lk))
            if vce == "nn":
                ss_h, _ = self._residuals(j, loc, lk, svec, win, "nn", matches, h, kern, None,
                                          None, 0, False)
                ss_b = ss_h
            else:
                ss_h, _ = self._residuals(j, loc, lk, svec, win, vce, 0, h, kern,
                                          fit.beta_p @ svec, fit.Gp_inv, p + 1, False)
                ss_b, _ = self._residuals(j, loc, lk, svec, win, vce, 0, b, kern,
                                          fit.beta_q @ svec, fit.Gq_inv, q + 1, False)
            f, use_f = loc._f()
            mh += loc.moments_ss(ss_h, lh, h, kern, 2 * p)
            mh_b += loc.moments_ss(ss_b, lh, h, kern, 2 * p)
            mx += moments_mixed(loc.t, ss_b, f, use_f, min(lh, lb), h, b, kern, p + q)
            mb += loc.moments_ss(ss_b, lb, b, kern, 2 * q)
        return mh, (mh if vce == "nn" else mh_b), mx, mb

    def cluster_meats(self, svec, fit, *, p, q, kern, vce, matches, hb_match):
        k, h, b = fit.k, fit.h, fit.b
        P, Pq = p + 1, q + 1
        win = self._window_counts(k)
        acc = {}

        def add(name, *vals):
            if name not in acc:
                acc[name] = list(vals)
            else:
                acc[name] = [a | v if a.dtype == np.bool_ else a + v
                             for a, v in zip(acc[name], vals, strict=True)]

        for j, lk, _ in self._upto(k):
            loc = self._local(j)
            if vce == "nn":
                _, res_h = self._residuals(j, loc, lk, svec, win, "nn", matches, h, kern, None,
                                           None, 0, True)
                res_b = res_h
            else:
                _, res_h = self._residuals(j, loc, lk, svec, win, vce, 0, h, kern,
                                           fit.beta_p @ svec, fit.Gp_inv, P, True)
                _, res_b = self._residuals(j, loc, lk, svec, win, vce, 0, b, kern,
                                           fit.beta_q @ svec, fit.Gq_inv, Pq, True)
            f, use_f = loc._f()
            cl, G = loc.cl[:lk], self.n_clusters
            if vce in ("cr2", "cr3"):
                add("cl", *K.cluster_sums(loc.t, f, use_f, lk, h, kern, P, res_h, cl, G))
                if hb_match:
                    add("rb", *K.cluster_sums(loc.t, f, use_f, lk, h, kern, Pq, res_b, cl, G))
                else:
                    _, Q = K.final_rows(loc.t, f, use_f, lk, h, b, kern, p, q, fit.g, fit.L)
                    add("rb", *K.cluster_sums_qq(loc.t, f, use_f, lk, b, kern, Pq,
                                                 np.ascontiguousarray(Q), res_b, cl, G))
                continue
            RW, Q = K.final_rows(loc.t, f, use_f, lk, h, b, kern, p, q, fit.g, fit.L)
            add("cl", *K.cluster_score_sums(RW * res_h[:, None], cl, G))
            if hb_match:
                add("rb", *K.cluster_score_sums(
                    K.poly_scores(loc.t, f, use_f, lk, h, kern, Pq, res_b), cl, G))
            else:
                add("rb", *K.cluster_score_sums(Q * res_b[:, None], cl, G))
        if vce in ("cr2", "cr3"):
            mode = 2 if vce == "cr2" else 3
            meat_cl = _cluster.meat(*acc["cl"], fit.Gp_inv, mode)
            if hb_match:
                return meat_cl, _cluster.meat(*acc["rb"], fit.Gq_inv, mode)
            return meat_cl, _cluster.meat_qq(*acc["rb"], fit.Gq_inv, mode)
        return _cr1(*acc["cl"], k, P), _cr1(*acc["rb"], k, Pq)


def _cr1(sums, seen, n, k_df):
    """rdrobust's CRV1 meat from each cluster's score sums."""
    S = sums[seen]
    G = S.shape[0]
    return (S.T @ S) * ((n - 1) / (n - k_df)) * (G / (G - 1))
