"""Agreement of rd_plot with R's rdplot (references from
benchmarks/external/rdplot_references.R on tests/data/plot_cases.json)."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

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


def _params():
    out = []
    for name, case in CASES.items():
        out.append(pytest.param(name, "in_memory", id=f"{name}-rows"))
        if case.get("x") == "xd" and not case.get("weights") and case.get("masspoints") != "off":
            out.append(pytest.param(name, "by_value", id=f"{name}-by_value"))
    return out


@pytest.mark.parametrize("name,memory", _params())
def test_rdplot(name, memory):
    case, ref = CASES[name], REFS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = rd.rd_plot(f"{case.get('y', 'y')} ~ {case.get('x', 'x')}",
                       DATA / f"{case['data']}.parquet", memory=memory, **_kwargs(case))
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
    with pytest.raises(ValueError, match="one outcome"):
        rd.rd_plot("y + y2 ~ x", SIM)
    with pytest.raises(ValueError, match="ci"):
        rd.rd_plot("y ~ x", SIM, ci=0.95)
    from hdfe_stream.core.profile import Profile

    assert isinstance(rd.rd_plot("y ~ x", SIM, profile=True), Profile)
    r = rd.rd_plot("y ~ x", SIM, covariates=["z1"], covariates_at="zero")
    assert r.covariate_coef.shape == (1, 1)
