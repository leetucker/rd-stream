"""Agreement of rd_plot with R's rdplot (references from
benchmarks/external/rdplot_references.R on tests/data/plot_cases.json)."""

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import rd_stream as rd

DATA = Path(__file__).resolve().parent / "data"
CASES = {c["name"]: c for c in json.loads((DATA / "plot_cases.json").read_text())}
REFS = {r["name"]: r for r in json.loads((DATA / "plot_references.json").read_text())}
RTOL = 1e-8
_NAMES = {"c": "cutoff", "p": "polynomial_order", "kernel": "kernel", "binselect": "bin_method",
          "masspoints": "mass_points", "scale": "bin_scale"}


def _kwargs(case):
    kw = {_NAMES[k]: v for k, v in case.items() if k in _NAMES}
    for k, name in (("h", "bandwidth"), ("nbins", "n_bins"), ("support", "support")):
        if k in case:
            kw[name] = tuple(case[k]) if isinstance(case[k], list) else case[k]
    if case.get("weights"):
        kw["weights"] = "w"
    if "covs" in case:
        kw["covariates"] = case["covs"]
    return kw


def _close(a, b):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=RTOL, atol=1e-10)


def _same_frame(a, b):
    # equal to rounding: numba compiles for the CPU it runs on, so sums can round
    # differently between machines
    assert_frame_equal(a, b, rel_tol=1e-10, abs_tol=1e-12)


def _params():
    out = []
    for name, case in CASES.items():
        out.append(pytest.param(name, "rows", id=f"{name}-rows"))
        # the window path: the rows in slabs of at most 300 on disk, read in order
        out.append(pytest.param(name, "slabs", id=f"{name}-slabs"))
        if case.get("x") == "xd" and not case.get("weights") and case.get("masspoints") != "off":
            out.append(pytest.param(name, "by_value", id=f"{name}-by_value"))
    return out


@pytest.mark.parametrize("name,memory", _params())
def test_rdplot(name, memory, monkeypatch, tmp_path):
    case, ref = CASES[name], REFS[name]
    kw = _kwargs(case)
    if memory == "slabs":
        from rd_stream import _window

        monkeypatch.setattr(_window, "_SLAB_ROWS", 300)
        memory, kw["workdir"] = "window", tmp_path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = rd.rd_plot(f"{case.get('y', 'y')} ~ {case.get('x', 'x')}",
                       DATA / f"{case['data']}.parquet", memory=memory, **kw)
    assert r.memory == memory
    _close(r.coef["left"], ref["coef_left"])
    _close(r.coef["right"], ref["coef_right"])
    assert list(r.n_bins) == ref["J"]
    assert list(r.n_bins_imse) == ref["J_IMSE"]
    assert list(r.n_bins_mimicking) == ref["J_MV"]
    _close(r.relative_scale, ref["rscale"])
    _close(r.bin_length_mean, ref["bin_avg"])
    _close(r.bin_length_median, ref["bin_med"])
    assert list(r.n) == ref["N"] and list(r.n_bandwidth) == ref["N_h"]
    assert r.bin_method == ref["binselect"]
    b = r.bins
    _close(b["x_mid"], ref["rdplot_mean_bin"])
    _close(b["x_mean"], ref["rdplot_mean_x"])
    _close(b["y_mean"], ref["rdplot_mean_y"])
    _close(b["x_low"], ref["rdplot_min_bin"])
    _close(b["x_high"], ref["rdplot_max_bin"])
    _close(b["se"], ref["rdplot_se_y"])
    assert b["n"].to_list() == ref["rdplot_N"]
    _close(b["ci_low"], ref["rdplot_ci_l"])
    _close(b["ci_high"], ref["rdplot_ci_r"])
    _close(r.fit["x"], ref["poly_x"])
    _close(r.fit["y"], ref["poly_y"])
    if memory == "window":
        assert not any(tmp_path.iterdir())         # the slabs are removed after the plot


SIM = DATA / "sim.parquet"


def test_plot_draws():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    r = rd.rd_plot("y ~ x", SIM)
    fig = r.plot(ci=True, shade=True, title="y at the cutoff")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "y"
    assert len(ax.lines) >= 3                     # two fits and the cutoff
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_spacings_need_the_rows():
    with pytest.raises(ValueError, match="needs the rows in order"):
        rd.rd_plot("y ~ xd", SIM, memory="by_value", mass_points="off")
    r = rd.rd_plot("y ~ xd", SIM, memory="by_value", mass_points="off", bin_method="esmvpr")
    assert r.memory == "by_value"


def test_plot_options_and_profile():
    with pytest.raises(ValueError, match="bin_method"):
        rd.rd_plot("y ~ x", SIM, bin_method="ik")
    with pytest.raises(ValueError, match="ci"):
        rd.rd_plot("y ~ x", SIM, ci=100)
    a, b = rd.rd_plot("y ~ x", SIM, ci=0.9), rd.rd_plot("y ~ x", SIM, ci=90)
    assert a.ci == b.ci == 90.0
    _same_frame(a.bins, b.bins)
    from hdfe_stream.core.profile import Profile

    assert isinstance(rd.rd_plot("y ~ x", SIM, profile=True), Profile)
    r = rd.rd_plot("y ~ x", SIM, covariates=["z1"], covariates_at="zero")
    assert r.covariate_coef.shape == (1, 1)


@pytest.mark.parametrize("x,memory", [("x", "rows"), ("xd", "rows"), ("xd", "by_value")])
def test_several_outcomes_match_single_calls(x, memory):
    df = pl.read_parquet(SIM)
    holes = np.random.default_rng(2).random(df.height) < 0.05
    dfm = df.with_columns(y2=pl.when(pl.Series(holes)).then(None).otherwise(pl.col("y2")))
    kw = {"memory": memory, "bin_method": "esmvpr"} if memory == "by_value" else {}
    m = rd.rd_plot(f"y + y2 ~ {x}", dfm, **kw)
    assert isinstance(m, rd.RDPlotMulti) and len(m) == 2 and m.outcomes == ["y", "y2"]
    for name in ("y", "y2"):
        one = rd.rd_plot(f"{name} ~ {x}", dfm, **kw)
        _same_frame(m[name].bins, one.bins)
        _same_frame(m[name].fit, one.fit)
        assert m[name].n_bins == one.n_bins and m[name].memory == one.memory
    assert m[1] is m["y2"] and [p.outcome for p in m] == ["y", "y2"]


def test_several_outcomes_draw():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = rd.rd_plot("y + y2 + z1 ~ x", SIM)
    fig = m.plot(ci=True)                          # two columns: the fourth slot is empty
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title() for ax in visible] == ["y", "y2", "z1"]
    assert len(fig.axes) == 4 and visible[1].get_xlabel() == "x"
    plt.close(fig)
    fig = m.plot(outcomes=["y2"], ncols=3, titles=False)
    assert len(fig.axes) == 1 and fig.axes[0].get_ylabel() == "y2"
    plt.close(fig)
    with pytest.raises(ValueError, match="no plot"):
        m.plot(outcomes=["z"])
