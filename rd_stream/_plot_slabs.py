"""rd_plot's window path: each side's rows from the window path's slabs on
disk, a slab at a time, in rdplot's order.

The slabs hold each side's rows by distance from the cutoff, each slab in
the source's order; read in turn (the right side's from the cutoff out, the
left side's from the far end in) and each sorted stably by x, they give the
side's rows in ascending x with ties in the source's order, as rdplot sorts
them. Every quantity `_rd_plot` takes from a side is then a sum over that
sequence:

- the count, range, the outcome's variance (per-slab means and sums of
  squares, combined) and the covariates' sums;
- the fourth-order fit's power sums;
- the bin selection's sums over the rows and over consecutive rows (each
  slab is taken with the next slab's first row, whose own terms are taken
  off again);
- the quantile-spaced bins' edges (exact ranks, located by the slabs' row
  counts);
- each bin's rows, sums and sum of squares (combined across slabs).

The global polynomial fit is the window path's `SlabSide`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import plot as P
from ._slabs import SlabSide

_KQ = 4                      # the bin selection's polynomial order (rdplot's k)


def slab_plot_sides(W, y, covs, c, weights):
    """The left and right `SlabRows` of outcome y, from a `WindowData`
    whose rows are in slabs."""
    from . import api

    st = W.stats(y)
    keep = api._drop_collinear_gram(st["ZZ"], covs) if covs else []
    cols = [0] + [1 + j for j in keep]
    out = []
    for s in ("l", "r"):
        side = SlabSide(W.store, s, c, y=y, extra=covs, dT=0, collapse=False, full=None,
                        n_side=st[f"n_{s}"], M_side=st[f"M_{s}"], cols=cols, weights=weights)
        out.append(SlabRows(W.store, s, W.running, y, covs, keep, side))
    return out


class SlabRows:
    """One side's rows, x ascending, from its slabs: what `_rd_plot` takes
    from a side (see `plot._PlotSide`)."""

    units = False                # rows, so the spacings bin methods apply

    def __init__(self, store, s, running, y, covs, keep, side):
        self.store, self.s, self.running, self.y = store, s, running, y
        self.covs, self.keep, self.side = list(covs), list(keep), side
        order = range(store.n_slabs[s])
        self.order = [j for j in (order if s == "r" else reversed(order))
                      if (s, j) in store.present]
        self._summarize()

    def _chunks(self, slabs=None):
        """(slab, x, y, Z) of each slab (or of `slabs`), x ascending."""
        names = [self.y, *self.covs]
        ok = pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_not_nan()
                                for c in names])
        for j in self.order if slabs is None else slabs:
            df = (pl.read_parquet(self.store.path(self.s, j), columns=[self.running, *names])
                  .filter(ok).sort(self.running, maintain_order=True))
            if not df.height:
                continue
            Z = (np.column_stack([df[self.covs[k]].to_numpy() for k in self.keep])
                 if self.keep else None)
            yield (j, np.ascontiguousarray(df[self.running].to_numpy(), dtype=np.float64),
                   np.ascontiguousarray(df[self.y].to_numpy(), dtype=np.float64), Z)

    def _summarize(self):
        """One pass: the rows per slab, the range, the outcome's mean and sum
        of squares, the covariates' sums and the power sums."""
        self.counts = []
        n, mean, m2 = 0, 0.0, 0.0
        self.x_lo = self.x_hi = None
        self.zsum = np.zeros(len(self.keep)) if self.keep else None
        self._S = np.zeros((2 * _KQ + 1, 3))
        for j, x, y, Z in self._chunks():
            k = x.shape[0]
            self.counts.append((j, k))
            if self.x_lo is None:
                self.x_lo = float(x[0])
            self.x_hi = float(x[-1])
            mk = float(y.mean())
            qk = float(((y - mk) ** 2).sum())
            d = mk - mean
            tot = n + k
            mean += d * k / tot
            m2 += qk + d * d * n * k / tot
            n = tot
            if Z is not None:
                self.zsum += Z.sum(axis=0)
            V = np.column_stack([np.ones_like(x), y, y * y])
            self._S += P._power_sums(x, V, 2 * _KQ)
        self.n = n
        self._var = m2 / (n - 1) if n > 1 else float("nan")

    def var_y(self):
        return self._var

    def power_sums(self, A):
        if A != 2 * _KQ:
            raise ValueError("the slabs' power sums are of order 2 x 4")
        return self._S

    def selection(self, g1, g2, var_y):
        none = np.zeros(1, np.int64)
        out = np.zeros(6)
        prev = None
        for _, x, y, _ in self._chunks():
            if prev is not None:
                px, py = prev
                # the previous slab with this one's first row (the pair across the
                # boundary), less that row's own terms, which this slab counts
                out += P._selection_sums(np.append(px, x[0]), none, True, g1, g2, var_y,
                                         np.append(py, y[0]))
                out -= P._selection_sums(x[:1], none, True, g1, g2, var_y, y[:1])
            prev = (x, y)
        if prev is not None:
            out += P._selection_sums(prev[0], none, True, g1, g2, var_y, prev[1])
        return out

    def quantile7(self, probs):
        """R's type-7 quantiles of the rows (as `plot._quantile7`), reading
        only the slabs that hold the ranks needed."""
        n = self.n
        ranks = {}
        for p in probs:
            index = 1 + max(n - 1, 0) * p
            for r in (int(np.floor(index)) - 1, int(np.ceil(index)) - 1):
                ranks[r] = None
        starts, at = {}, 0
        for j, k in self.counts:
            starts[j] = at
            at += k
        need = {}
        for r in ranks:
            for j, k in self.counts:
                if starts[j] <= r < starts[j] + k:
                    need.setdefault(j, []).append(r)
                    break
        for j, x, _, _ in self._chunks([j for j, _ in self.counts if j in need]):
            for r in need[j]:
                ranks[r] = float(x[r - starts[j]])
        out = np.empty(len(probs))
        for i, p in enumerate(probs):
            index = 1 + max(n - 1, 0) * p
            lo, hi = int(np.floor(index)), int(np.ceil(index))
            q, xh = ranks[lo - 1], ranks[hi - 1]
            if index > lo and xh != q:
                h = index - lo
                q = (1 - h) * q + h * xh
            out[i] = q
        return out

    def segments(self, e, J):
        """Per bin: the rows, sum x, sum y and sum of squares of y about the
        bin's mean (per-slab sums of squares combined)."""
        N = np.zeros(J)
        SX = np.zeros(J)
        SY = np.zeros(J)
        M2 = np.zeros(J)
        edges = np.asarray(e[:J])
        for _, x, y, _ in self._chunks():
            b = np.searchsorted(edges, x, side="right") - 1
            inb = b >= 0
            b, xb, yb = b[inb], x[inb], y[inb]
            k = np.bincount(b, minlength=J).astype(np.float64)
            sx = np.bincount(b, weights=xb, minlength=J)
            sy = np.bincount(b, weights=yb, minlength=J)
            has = k > 0
            mk = np.where(has, sy / np.maximum(k, 1), 0.0)
            qk = np.bincount(b, weights=(yb - mk[b]) ** 2, minlength=J)
            prior = np.where(N > 0, SY / np.maximum(N, 1), 0.0)
            d = mk - prior
            tot = N + k
            M2 += np.where(has, qk + d * d * N * k / np.maximum(tot, 1), 0.0)
            N, SX, SY = tot, SX + sx, SY + sy
        nz = N > 0
        return nz, np.column_stack([N[nz], SX[nz], SY[nz], M2[nz]])
