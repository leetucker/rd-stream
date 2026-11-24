"""The memory paths: by value against in memory, the automatic choice under
a memory budget, and profile=True."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hdfe_stream.core.profile import Profile

import rd_stream as rd
from rd_stream import _memory

SIM = Path(__file__).resolve().parent / "data" / "sim.parquet"


@pytest.mark.parametrize("kw", [{}, {"vcov": "HC2"}, {"treatment": "t"},
                                {"covariates": ["z1", "z2", "z3"], "treatment": "t"},
                                {"bandwidth": [0.2, 0.4]}, {"polynomial_order": 2}])
def test_by_value_matches_the_rows_collapsed(kw, monkeypatch):
    a = rd.rd_estimate("y + y2 ~ xd", SIM, memory="by_value", **kw)
    monkeypatch.setattr(_memory, "_BY_VALUE", False)     # the rows held, collapsed
    b = rd.rd_estimate("y + y2 ~ xd", SIM, **kw)
    for r1, r2 in zip(a, b, strict=True):
        assert r2.memory == "rows" and r2.collapsed
        np.testing.assert_allclose(r1.beta, r2.beta, rtol=1e-10)
        np.testing.assert_allclose(r1.se, r2.se, rtol=1e-10)
        assert r1.memory == "by_value" and r1.collapsed


def test_by_value_drops_missing_values_per_outcome(monkeypatch):
    df = pl.read_parquet(SIM)
    holes = np.random.default_rng(0).random(df.height) < 0.05
    dfm = df.with_columns(y2=pl.when(pl.Series(holes)).then(None).otherwise(pl.col("y2")),
                          y=pl.when(pl.Series(holes[::-1])).then(float("nan"))
                          .otherwise(pl.col("y")))
    a = rd.rd_estimate("y + y2 ~ xd", dfm, memory="by_value")
    monkeypatch.setattr(_memory, "_BY_VALUE", False)
    b = rd.rd_estimate("y + y2 ~ xd", dfm)
    for r1, r2 in zip(a, b, strict=True):
        assert r1.n_obs == r2.n_obs
        np.testing.assert_allclose(r1.se, r2.se, rtol=1e-10)


def test_by_value_needs_no_clusters_or_weights():
    with pytest.raises(ValueError, match="clusters and weights"):
        rd.rd_estimate("y ~ xd", SIM, memory="by_value", vcov={"CRV1": "cl"})
    with pytest.raises(ValueError, match="memory must be"):
        rd.rd_estimate("y ~ xd", SIM, memory="disk")


def test_auto_groups_by_value_when_the_rows_do_not_fit(monkeypatch):
    monkeypatch.setattr(_memory, "rows_bytes", lambda *a: 1e15)
    r = rd.rd_estimate("y ~ xd", SIM)
    assert r.memory == "by_value"
    # clusters, or a continuous running variable: the window path, whose rows do not fit
    # either here
    with pytest.raises(MemoryError, match="memory budget"):
        rd.rd_estimate("y ~ xd", SIM, vcov={"CRV1": "cl"})
    with pytest.raises(MemoryError, match="memory budget"):
        rd.rd_estimate("y ~ x", SIM)
    # rd_plot has no window path
    with pytest.raises(MemoryError, match="to hold and sort"):
        rd.rd_plot("y ~ x", SIM)


def test_auto_groups_a_discrete_running_variable_by_value():
    assert rd.rd_estimate("y ~ xd", SIM).memory == "by_value"
    assert rd.rd_estimate("y ~ x", SIM).memory == "rows"
    assert rd.rd_estimate("y ~ xd", SIM, vcov={"CRV1": "cl"}).memory == "rows"
    r = rd.rd_estimate("y ~ xd", SIM, memory="rows")
    assert r.memory == "rows" and not r.collapsed
    with pytest.raises(ValueError, match="memory must be"):
        rd.rd_estimate("y ~ xd", SIM, memory="in_memory")


def test_profile():
    p = rd.rd_estimate("y + y2 ~ xd", SIM, profile=True)
    assert isinstance(p, Profile)
    assert p.notes["rows"] == 4000 and p.notes["memory_path"] == "by_value"
    assert p.notes["collapses"]
    assert 50 < p.notes["distinct_values"] < 100
    assert "rd-stream" in str(p)
    q = rd.rd_estimate("y ~ xd", SIM, profile=True, memory="by_value")
    assert [c.name for c in q.components][1] == "by-value table"
    assert isinstance(rd.rd_bandwidth("y ~ x", SIM, profile=True), Profile)
