"""Donut holes (`donut=`) and placebo cutoffs (`rd_placebo`): each fit is
the ordinary fit on the rows the variant keeps, on every memory path."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import rd_stream as rd

SIM = Path(__file__).resolve().parent / "data" / "sim.parquet"


def _outside(df, x, dl, dr, c=0.0):
    return df.filter(pl.when(pl.col(x) < c).then(c - pl.col(x) >= dl)
                     .otherwise(pl.col(x) - c >= dr))


def _same(a, b, rtol):
    np.testing.assert_allclose(a.beta, b.beta, rtol=rtol, atol=1e-13)
    np.testing.assert_allclose(a.se, b.se, rtol=rtol)
    np.testing.assert_allclose(a.bandwidth, b.bandwidth, rtol=rtol)
    assert (a.n_obs, a.n_bandwidth) == (b.n_obs, b.n_bandwidth)


@pytest.mark.parametrize("x,memory,kw", [
    ("x", "rows", {}), ("x", "window", {}), ("x", "rows", {"vcov": {"CRV1": "cl"}}),
    ("x", "window", {"vcov": {"CRV2": "cl"}}), ("xd", "rows", {}), ("xd", "by_value", {}),
    ("x", "rows", {"treatment": "t", "covariates": ["z1"]})])
def test_donut_is_the_fit_without_those_rows(x, memory, kw):
    df = pl.read_parquet(SIM)
    rtol = 1e-12 if memory == "rows" else 1e-9
    for d, (dl, dr) in ((0.02, (0.02, 0.02)), ((0.01, 0.03), (0.01, 0.03))):
        a = rd.rd_estimate(f"y ~ {x}", df, donut=d, memory=memory, **kw)
        b = rd.rd_estimate(f"y ~ {x}", _outside(df, x, dl, dr), memory=memory, **kw)
        _same(a, b, rtol)
        assert a.donut == (dl, dr) and b.donut is None and a.memory == memory


def test_donut_drops_the_cutoff_itself_and_lists():
    df = pl.read_parquet(SIM).with_columns(xd0=pl.col("xd"))
    at = df.filter(pl.col("xd") == 0).height
    assert at > 0
    r0 = rd.rd_estimate("y ~ xd", df)
    r = rd.rd_estimate("y ~ xd", df, donut=(0.0, 1e-9))    # x = c is on the right
    assert r.n_obs == r0.n_obs - at
    m = rd.rd_estimate("y + y2 ~ x", SIM, donut=[0, 0.01, (0.02, 0.03)],
                       bandwidth=[0.3, 0.4])
    assert len(m) == 12
    assert [f.fml for f in m][:4] == ["y ~ x [h=0.3] [donut=0]", "y ~ x [h=0.4] [donut=0]",
                                      "y2 ~ x [h=0.3] [donut=0]", "y2 ~ x [h=0.4] [donut=0]"]
    one = rd.rd_estimate("y2 ~ x", SIM, donut=(0.02, 0.03), bandwidth=0.4)
    _same([f for f in m if f.fml == "y2 ~ x [h=0.4] [donut=(0.02, 0.03)]"][0], one, 1e-12)
    with pytest.raises(ValueError, match="donut must be"):
        rd.rd_estimate("y ~ x", SIM, donut=-0.1)
    with pytest.raises(ValueError, match="donut must be"):
        rd.rd_estimate("y ~ x", SIM, donut=(0.1, 0.2, 0.3))


def test_donut_in_rd_bandwidth_and_rd_plot():
    df = pl.read_parquet(SIM)
    keep = _outside(df, "x", 0.05, 0.05)
    assert rd.rd_bandwidth("y ~ x", df, donut=0.05).table.equals(
        rd.rd_bandwidth("y ~ x", keep).table)
    a, b = rd.rd_plot("y ~ x", df, donut=0.05), rd.rd_plot("y ~ x", keep)
    assert a.bins.equals(b.bins) and a.fit.equals(b.fit) and a.donut == (0.05, 0.05)
    with pytest.raises(ValueError, match="one donut"):
        rd.rd_bandwidth("y ~ x", df, donut=[0.01, 0.02])
    with pytest.raises(ValueError, match="one donut"):
        rd.rd_plot("y ~ x", df, donut=[0.01, 0.02])


@pytest.mark.parametrize("memory", ["rows", "window"])
def test_placebo_cutoffs_use_their_side_only(memory):
    df = pl.read_parquet(SIM)
    p = rd.rd_placebo("y + y2 ~ x", df, [0.5, -0.3, 0.2], memory=memory)
    t = p.table
    assert t["outcome"].to_list() == ["y"] * 4 + ["y2"] * 4
    assert t["cutoff"].to_list()[:4] == [-0.3, 0.0, 0.2, 0.5]
    assert t["kind"].to_list()[:4] == ["placebo", "actual", "placebo", "placebo"]
    assert t["side"].to_list()[:4] == ["left", None, "right", "right"]
    for c, rows in ((-0.3, df.filter(pl.col("x") < 0)), (0.2, df.filter(pl.col("x") >= 0))):
        one = rd.rd_estimate("y2 ~ x", rows, cutoff=c, memory=memory)
        fit = [f for f in p.fits if f.depvar == "y2" and f.cutoff == c][0]
        _same(fit, one, 1e-12 if memory == "rows" else 1e-9)
        row = t.filter((pl.col("outcome") == "y2") & (pl.col("cutoff") == c)).row(0, named=True)
        assert row["estimate"] == pytest.approx(one.estimate, rel=1e-12)
        assert (row["robust_low"], row["robust_high"]) == pytest.approx(one.robust_ci, rel=1e-12)
    actual = rd.rd_estimate("y ~ x", df, memory=memory)
    assert t.row(1, named=True)["estimate"] == pytest.approx(actual.estimate, rel=1e-12)


def test_placebo_edges_and_errors():
    p = rd.rd_placebo("y ~ x", SIM, [-0.2, 0.999], include_cutoff=False)
    t = p.table
    assert t.height == 2 and t["kind"].to_list() == ["placebo", "placebo"]
    edge = t.row(1, named=True)
    assert np.isnan(edge["estimate"]) and "strictly inside" in edge["note"]
    with pytest.raises(ValueError, match="equals the true cutoff"):
        rd.rd_placebo("y ~ x", SIM, [0.0])
    with pytest.raises(ValueError, match="one bandwidth"):
        rd.rd_placebo("y ~ x", SIM, [0.2], bandwidth=[0.2, 0.3])
    p = rd.rd_placebo("y ~ xd", SIM, [-0.2, 0.2], memory="by_value")
    assert {f.memory for f in p.fits} == {"by_value"}


def test_placebo_plot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = rd.rd_placebo("y + y2 ~ x", SIM, [-0.2, 0.2, 0.999])
    fig = p.plot()
    assert [ax.get_title() for ax in fig.axes] == ["y", "y2"]
    plt.close(fig)
    with pytest.raises(ValueError, match="outcomes must be"):
        p.plot(outcomes=["z"])


@pytest.mark.parametrize("x,memory", [("x", "rows"), ("x", "window"), ("xd", "rows"),
                                      ("xd", "by_value")])
def test_constant_take_up_is_perfect_compliance(x, memory):
    # a constant take-up of 0.1 on the left: np.var() can leave a rounding residue (it
    # does for the 3,256 rows left of x's cutoff), min = max (R's var() == 0) does not,
    # so the bandwidths are the sharp design's
    df = pl.read_parquet(SIM).with_columns(
        t=pl.when(pl.col(x) < 0).then(0.1).otherwise(pl.col("t").cast(pl.Float64)))
    a = rd.rd_estimate(f"y ~ {x}", df, treatment="t", memory=memory)
    b = rd.rd_estimate(f"y ~ {x}", df, treatment="t", memory=memory, sharp_bandwidth=True)
    np.testing.assert_allclose(a.bandwidth, b.bandwidth, rtol=1e-12)


@pytest.mark.parametrize("memory", ["rows", "by_value", "window"])
@pytest.mark.parametrize("closed", ["none", "left", "right", "both"])
def test_donut_closed_ends(closed, memory):
    # xd is on a grid of 1/40: the rows at the radius (0.05) are kept, or dropped at
    # the hole's closed ends, as in Polars' is_between
    df = pl.read_parquet(SIM)
    d, x = 0.05, pl.col("xd")
    left = (-x > d) if closed in ("left", "both") else (-x >= d)
    right = (x > d) if closed in ("right", "both") else (x >= d)
    keep = df.filter(pl.when(x < 0).then(left).otherwise(right))
    assert df.filter(x.abs() == d).height > 0
    a = rd.rd_estimate("y ~ xd", df, donut=d, donut_closed=closed, memory=memory)
    b = rd.rd_estimate("y ~ xd", keep, memory=memory)
    _same(a, b, 1e-12 if memory != "window" else 1e-9)
    assert a.donut_closed == closed
    p = rd.rd_plot("y ~ xd", df, donut=d, donut_closed=closed, memory="rows")
    assert p.n == rd.rd_plot("y ~ xd", keep, memory="rows").n


def test_donut_closed_errors_and_zero_radius():
    with pytest.raises(ValueError, match="donut_closed"):
        rd.rd_estimate("y ~ x", SIM, donut=0.01, donut_closed="open")
    df = pl.read_parquet(SIM)
    # a zero radius drops nothing on its side, closed or not: x = cutoff stays
    a = rd.rd_estimate("y ~ xd", df, donut=(0.05, 0.0), donut_closed="both")
    b = rd.rd_estimate("y ~ xd", df.filter((pl.col("xd") >= 0) | (pl.col("xd") < -0.05)))
    _same(a, b, 1e-12)
