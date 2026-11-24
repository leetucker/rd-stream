"""The entry points' behavior beyond agreement with R (test_references.py)."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import rd_stream as rd

DATA = Path(__file__).resolve().parent / "data"
SIM = DATA / "sim.parquet"


def test_several_outcomes_match_separate_fits():
    multi = rd.rd_estimate("y + y2 ~ x", SIM)
    assert isinstance(multi, rd.RDMulti) and len(multi) == 2
    for r in multi:
        one = rd.rd_estimate(f"{r.depvar} ~ x", SIM)
        np.testing.assert_allclose(r.beta, one.beta, rtol=1e-12)
        np.testing.assert_allclose(r.se, one.se, rtol=1e-12)
    est = multi.estimates()
    assert est.columns[:2] == ["outcome", "h_left"] and est.height == 2


def test_bandwidth_list_matches_manual_fits():
    grid = [0.15, 0.2, (0.25, 0.3)]
    multi = rd.rd_estimate("y ~ x", SIM, bandwidth=grid)
    assert len(multi) == 3
    for r, h in zip(multi, grid, strict=True):
        one = rd.rd_estimate("y ~ x", SIM, bandwidth=h)
        np.testing.assert_allclose(r.beta, one.beta, rtol=1e-12)
        assert r.bandwidth_selector == "manual"


def test_missing_values_are_dropped_per_outcome():
    df = pl.read_parquet(SIM)
    rng = np.random.default_rng(0)
    holes = rng.random(df.height) < 0.05
    dfm = df.with_columns(y2=pl.when(pl.Series(holes)).then(None).otherwise(pl.col("y2")))
    multi = rd.rd_estimate("y + y2 ~ x", dfm)
    y_fit, y2_fit = multi.fetch_model(0), multi.fetch_model(1)
    assert y_fit.n_obs == df.height
    assert y2_fit.n_obs == df.height - int(holes.sum())
    clean = rd.rd_estimate("y2 ~ x", dfm.filter(pl.col("y2").is_not_null()))
    np.testing.assert_allclose(y2_fit.beta, clean.beta, rtol=1e-12)


def test_frames_and_paths_agree():
    a = rd.rd_estimate("y ~ x", SIM)
    b = rd.rd_estimate("y ~ x", pl.read_parquet(SIM))
    c = rd.rd_estimate("y ~ x", pl.scan_parquet(SIM))
    np.testing.assert_array_equal(a.beta, b.beta)
    np.testing.assert_array_equal(a.beta, c.beta)


def test_collapse_is_exact():
    auto = rd.rd_estimate("y ~ xd", SIM)
    rows = rd.rd_estimate("y ~ xd", SIM, memory="rows")
    assert auto.collapsed and not rows.collapsed
    np.testing.assert_allclose(auto.beta, rows.beta, rtol=1e-10)
    np.testing.assert_allclose(auto.se, rows.se, rtol=1e-10)
    assert not rd.rd_estimate("y ~ x", SIM).collapsed
    assert not rd.rd_estimate("y ~ xd", SIM, vcov={"CRV1": "cl"}).collapsed
    assert not rd.rd_estimate("y ~ xd", SIM, weights="w").collapsed
    with pytest.raises(TypeError):
        rd.rd_estimate("y ~ xd", SIM, collapse="never")


def test_threads_do_not_change_the_result():
    a = rd.rd_estimate("y ~ x", SIM, n_threads=1)
    b = rd.rd_estimate("y ~ x", SIM, n_threads=4)
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-12)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-12)


@pytest.mark.parametrize("fml", ["y", "y ~ x + z", "~ x", "y ~ x ~ z"])
def test_bad_formulas(fml):
    with pytest.raises(ValueError, match="formula"):
        rd.rd_estimate(fml, SIM)


def test_bad_options():
    with pytest.raises(ValueError, match="cutoff"):
        rd.rd_estimate("y ~ x", SIM, cutoff=5.0)
    with pytest.raises(ValueError, match="kernel"):
        rd.rd_estimate("y ~ x", SIM, kernel="gaussian")
    with pytest.raises(ValueError, match="bias_order"):
        rd.rd_estimate("y ~ x", SIM, polynomial_order=2, bias_order=2)
    with pytest.raises(ValueError, match="bandwidth"):
        rd.rd_estimate("y ~ x", SIM, bandwidth="ik")
    with pytest.raises(ValueError, match="bandwidth"):
        rd.rd_estimate("y ~ x", SIM, bandwidth=-1.0)
    with pytest.raises(ValueError, match="cluster column"):
        rd.rd_estimate("y ~ x", SIM, vcov="CRV2")
    with pytest.raises(ValueError, match="vcov must be"):
        rd.rd_estimate("y ~ x", SIM, vcov="cr2")
    with pytest.raises(ValueError, match="one-way"):
        rd.rd_estimate("y ~ x", SIM, vcov={"CRV1": "cl+w"})
    with pytest.raises(ValueError, match="one clustering"):
        rd.rd_estimate("y ~ x", SIM, vcov={"CRV1": "cl", "CRV2": "cl"})
    with pytest.raises(TypeError):
        rd.rd_estimate("y ~ x", SIM, cluster="cl")


def test_vcov_spellings_follow_hdfe_stream():
    a = rd.rd_estimate("y ~ x", SIM, vcov={"CRV2": "cl"})
    for v in ("CRV2:cl", {"crv2": "cl"}, "crv2:cl"):
        np.testing.assert_array_equal(rd.rd_estimate("y ~ x", SIM, vcov=v).se, a.se)
    assert a.vcov_type == "CRV2"
    assert rd.rd_estimate("y ~ x", SIM, vcov={"CRV1": "cl"}).vcov_type == "CRV1"
    h = rd.rd_estimate("y ~ x", SIM, vcov="HC1")
    assert h.vcov_type == "HC1"
    for v in ("hc1", "hetero"):
        np.testing.assert_array_equal(rd.rd_estimate("y ~ x", SIM, vcov=v).se, h.se)


def test_bandwidth_table():
    bw = rd.rd_bandwidth("y ~ x", SIM, all_selectors=True)
    assert bw.table["selector"].to_list()[0] == "mserd" and bw.table.height == 10
    one = rd.rd_bandwidth("y ~ x", SIM)
    fit = rd.rd_estimate("y ~ x", SIM)
    assert one.table.row(0)[1] == pytest.approx(fit.bandwidth[0], rel=1e-12)


def test_summary_and_json():
    r = rd.rd_estimate("y ~ x", SIM, treatment="t")
    text = r.summary_text()
    assert "fuzzy: take-up t" in text and "Robust" in text
    d = json.loads(r.summary_json())
    assert d["bandwidth_selector"] == "mserd" and d["treatment"] == "t"
    lo, hi = r.robust_ci
    assert lo < r.beta[1] < hi


def test_small_sample_uses_the_whole_range():
    # rdrobust: with fewer than 20 observations, h is the whole range
    x = np.linspace(-1, 0.9, 18)
    df = pl.DataFrame({"x": x, "y": x + (x >= 0) + np.sin(7 * x)})
    r = rd.rd_estimate("y ~ x", df)
    assert r.bandwidth_selector == "manual"
    assert r.bandwidth == (1.0, 1.0)
