"""The window path (memory="window"): only the rows near the cutoff are
held, with the full sample's statistics from streaming passes."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hdfe_stream.core.profile import Profile
from stream_limits import MemoryBudget

import rd_stream as rd
from rd_stream import _engine, _memory, _window
from rd_stream._engine import quantile_type2_sorted

SIM = Path(__file__).resolve().parent / "data" / "sim.parquet"


@pytest.fixture
def narrow(monkeypatch):
    """A first read at a third of the pilot bandwidth, and a count of the reads
    and of the range fits from power sums."""
    monkeypatch.setattr(_window, "_START", 0.3)
    seen = {"reads": [], "full": 0}
    read = _window.WindowData._read

    def spy_read(self, r):
        seen["reads"].append(r)
        return read(self, r)

    full = _engine.FullSums.moments_w

    def spy_full(self, *a):
        seen["full"] += 1
        return full(self, *a)

    monkeypatch.setattr(_window.WindowData, "_read", spy_read)
    monkeypatch.setattr(_engine.FullSums, "moments_w", spy_full)
    return seen


def _agree(a, b, rtol=1e-8):
    for r1, r2 in zip(a, b, strict=True):
        np.testing.assert_allclose(r1.beta, r2.beta, rtol=rtol)
        np.testing.assert_allclose(r1.se, r2.se, rtol=rtol)
        np.testing.assert_allclose(r1.bandwidth, r2.bandwidth, rtol=rtol)
        np.testing.assert_allclose(r1.bias_bandwidth, r2.bias_bandwidth, rtol=rtol)
        assert (r1.n_left, r1.n_right) == (r2.n_left, r2.n_right)
        assert tuple(r1.n_distinct) == tuple(r2.n_distinct)
        assert tuple(r1.n_bandwidth) == tuple(r2.n_bandwidth)
        assert r1.n_clusters == r2.n_clusters
        assert r1.memory == "window"


KW = [{}, {"vcov": "HC1"}, {"vcov": {"CRV1": "cl"}}, {"vcov": {"CRV2": "cl"}},
      {"treatment": "t"}, {"covariates": ["z1", "z2", "z3"]}, {"weights": "w"},
      {"bandwidth": "msetwo"}, {"bandwidth": "cercomb2"}, {"polynomial_order": 2},
      {"kernel": "uniform"}, {"kernel": "epanechnikov"}, {"min_distinct": 30},
      {"standardize": False}, {"mass_points": "off"}]


@pytest.mark.parametrize("x", ["x", "xd"])
@pytest.mark.parametrize("kw", KW, ids=[str(k) for k in KW])
def test_matches_in_memory(kw, x, narrow):
    a = rd.rd_estimate(f"y + y2 ~ {x}", SIM, memory="window", **kw)
    b = rd.rd_estimate(f"y + y2 ~ {x}", SIM, memory="in_memory", **kw)
    _agree(a, b)
    assert len(narrow["reads"]) >= 2          # read again further out
    assert narrow["full"] > 0                 # C_d's range fit from the power sums


def test_given_bandwidths_are_identical(narrow):
    a = rd.rd_estimate("y + y2 ~ x", SIM, memory="window", bandwidth=[0.2, 0.3, (0.25, 0.4)])
    b = rd.rd_estimate("y + y2 ~ x", SIM, memory="in_memory", bandwidth=[0.2, 0.3, (0.25, 0.4)])
    _agree(a, b, rtol=0)
    assert len(narrow["reads"]) == 1          # out to the widest bandwidth, once


def test_rd_bandwidth_all_selectors(narrow):
    a = rd.rd_bandwidth("y ~ x", SIM, memory="window", all_selectors=True).table
    b = rd.rd_bandwidth("y ~ x", SIM, memory="in_memory", all_selectors=True).table
    for col in ("h_left", "h_right", "b_left", "b_right"):
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(), rtol=1e-8)


def test_missing_values_per_outcome(narrow):
    df = pl.read_parquet(SIM)
    holes = np.random.default_rng(0).random(df.height) < 0.05
    dfm = df.with_columns(y2=pl.when(pl.Series(holes)).then(None).otherwise(pl.col("y2")),
                          y=pl.when(pl.Series(holes[::-1])).then(float("nan"))
                          .otherwise(pl.col("y")))
    _agree(rd.rd_estimate("y + y2 ~ x", dfm, memory="window", treatment="t"),
           rd.rd_estimate("y + y2 ~ x", dfm, memory="in_memory", treatment="t"))


@pytest.mark.parametrize("n", [4000, 4001, 4003])
def test_quantiles_from_refined_bins(monkeypatch, n):
    # bins of a few rows and a finer histogram whenever a bin holds more than 5
    monkeypatch.setattr(_window, "_BINS", 8)
    monkeypatch.setattr(_window, "_COLLECT_ROWS", 5)
    df = pl.read_parquet(SIM).head(n)
    W = _window.WindowData(df, "x", ["y"], None, None, None, [],
                           rd.api._options(0.0, 1, None, 0, "triangular", "nearest_neighbor", 3,
                                           "adjust", None, True, True, 1.0, 1.0)[0],
                           per_row_bytes=100, max_memory_gb=None, memory_reserve=None,
                           read=False)
    xs = np.sort(df.filter(pl.col("y").is_not_null() & pl.col("y").is_not_nan())["x"].to_numpy())
    m = W.mask_of["y"]
    for prob in (0.25, 0.5, 0.75):
        assert W._quantile(m, prob) == quantile_type2_sorted(xs, prob)
    st = W.stats("y")
    assert st["x_sd"] == pytest.approx(np.std(xs, ddof=1), rel=1e-14)
    assert st["M_l"] + st["M_r"] == len(np.unique(xs))


def test_auto_takes_the_window_when_the_rows_do_not_fit(monkeypatch):
    # room for about 1,600 rows: not the 4,000, but the windows near the cutoff
    room = MemoryBudget().headroom_gb() * 1e9
    real = _memory.rows_bytes
    monkeypatch.setattr(_memory, "rows_bytes",
                        lambda n, *a: n * room / 2000 if n > 1 else room / 2000)
    r = rd.rd_estimate("y ~ x", SIM, vcov={"CRV1": "cl"}, bandwidth=0.3)
    assert r.memory == "window"
    monkeypatch.setattr(_memory, "rows_bytes", real)
    _agree([r], [rd.rd_estimate("y ~ x", SIM, vcov={"CRV1": "cl"}, bandwidth=0.3)], rtol=0)


def test_a_budget_too_small_for_a_slab(monkeypatch):
    room = MemoryBudget().headroom_gb() * 1e9
    monkeypatch.setattr(_memory, "rows_bytes", lambda n, *a: max(n, 1) * room / 300)
    with pytest.raises(MemoryError, match="rows a slab"):
        rd.rd_estimate("y ~ x", SIM)
    with pytest.raises(MemoryError, match="rows a slab"):
        rd.rd_estimate("y ~ x", SIM, bandwidth=1.0)


@pytest.fixture
def slabs(monkeypatch, tmp_path):
    """Slabs of at most 300 rows (about 14 of them), whatever the budget."""
    monkeypatch.setattr(_window, "_SLABS_ALWAYS", True)
    monkeypatch.setattr(_window, "_SLAB_ROWS", 300)
    return tmp_path


SLAB_KW = [{}, {"vcov": "HC1"}, {"vcov": "HC3"}, {"vcov": {"CRV1": "cl"}},
           {"vcov": {"CRV2": "cl"}}, {"vcov": {"CRV3": "cl"}},
           {"treatment": "t"}, {"covariates": ["z1", "z2", "z3"]}, {"weights": "w"},
           {"bandwidth": "msetwo"}, {"polynomial_order": 2}, {"kernel": "uniform"},
           {"min_distinct": 30}, {"n_neighbors": 6}, {"bandwidth": 0.3},
           {"bandwidth": 0.3, "bias_bandwidth": 0.5, "vcov": {"CRV1": "cl"}},
           {"bandwidth": 0.3, "bias_bandwidth": 0.5, "vcov": {"CRV2": "cl"}},
           {"bandwidth": [0.2, 0.4]}]


@pytest.mark.parametrize("x", ["x", "xd"])
@pytest.mark.parametrize("kw", SLAB_KW, ids=[str(k) for k in SLAB_KW])
def test_slabs_match_in_memory(kw, x, slabs):
    a = rd.rd_estimate(f"y + y2 ~ {x}", SIM, memory="window", workdir=slabs, **kw)
    b = rd.rd_estimate(f"y + y2 ~ {x}", SIM, memory="in_memory", **kw)
    _agree(a, b)
    assert not any(slabs.iterdir())             # the slabs are removed after the fit


def test_slabs_from_frames_and_bandwidths(slabs):
    df = pl.read_parquet(SIM)
    for data in (df, df.lazy()):
        _agree([rd.rd_estimate("y ~ x", data, memory="window", workdir=slabs)],
               [rd.rd_estimate("y ~ x", SIM, memory="in_memory")])
    a = rd.rd_bandwidth("y ~ x", SIM, memory="window", all_selectors=True, workdir=slabs).table
    b = rd.rd_bandwidth("y ~ x", SIM, memory="in_memory", all_selectors=True).table
    for col in ("h_left", "h_right", "b_left", "b_right"):
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(), rtol=1e-8)


@pytest.mark.parametrize("x", ["x", "xd"])
@pytest.mark.parametrize("kw", [{}, {"vcov": {"CRV1": "cl"}}, {"treatment": "t", "vcov": {"CRV1": "cl"}}],
                         ids=str)
def test_distinct_counts_from_the_slabs(kw, x, monkeypatch, tmp_path):
    # the exact counts do not fit in one pass: they come from the slabs, per outcome's
    # rows (missing values in y2) and side, and the fit matches the one in memory
    monkeypatch.setattr(_window, "_DISTINCT_SHARE", 0.0)
    monkeypatch.setattr(_window, "_SLAB_ROWS", 300)
    df = pl.read_parquet(SIM)
    holes = np.random.default_rng(1).random(df.height) < 0.05
    dfm = df.with_columns(y2=pl.when(pl.Series(holes)).then(None).otherwise(pl.col("y2")))
    made = []
    distinct = _window.WindowData.to_slabs

    def spy(self):
        assert self.distinct is None
        distinct(self)
        made.append(dict(self.distinct))

    monkeypatch.setattr(_window.WindowData, "to_slabs", spy)
    a = rd.rd_estimate(f"y + y2 ~ {x}", dfm, memory="window", workdir=tmp_path, **kw)
    b = rd.rd_estimate(f"y + y2 ~ {x}", dfm, memory="in_memory", **kw)
    _agree(a, b)
    assert len(made) == 1 and len({k.split(":")[0] for k in made[0]}) == 2


def test_slabs_when_the_window_does_not_fit(monkeypatch, tmp_path):
    # room for about 2,500 rows: the held window is too small for the bias step, so the
    # rows go to slabs of a few hundred
    room = MemoryBudget().headroom_gb() * 1e9
    real = _memory.rows_bytes
    monkeypatch.setattr(_memory, "rows_bytes", lambda n, *a: max(n, 1) * room / 3100)
    monkeypatch.setattr(_window, "_MIN_SLAB_ROWS", 100)
    r = rd.rd_estimate("y ~ x", SIM, workdir=tmp_path)
    monkeypatch.setattr(_memory, "rows_bytes", real)
    assert r.memory == "window"
    _agree([r], [rd.rd_estimate("y ~ x", SIM)])


def test_profile_and_errors():
    p = rd.rd_estimate("y + y2 ~ x", SIM, memory="window", profile=True)
    assert isinstance(p, Profile) and p.notes["memory_path"] == "window"
    assert 0 < p.notes["window_rows"] < 4000
    assert [c.name for c in p.components][1] == "rows within the first window"
    disk = [c for c in p.components if c.where == "disk"]
    assert disk and disk[0].expected == 4000 * 8 * 3     # x, y, y2
    assert p.notes["slab_rows"] > 0
    with pytest.raises(ValueError, match="not rd_plot"):
        rd.rd_plot("y ~ x", SIM, memory="window")
    with pytest.raises(ValueError, match="strictly inside"):
        rd.rd_estimate("y ~ x", SIM, memory="window", cutoff=50.0)
