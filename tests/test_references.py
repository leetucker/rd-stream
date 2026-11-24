"""Agreement with R's rdrobust 4.1.1 (references from
benchmarks/external/rdrobust_references.R on tests/data/cases.json)."""

import json
from pathlib import Path

import numpy as np
import pytest

import rd_stream as rd

DATA = Path(__file__).resolve().parent / "data"
CASES = {c["name"]: c for c in json.loads((DATA / "cases.json").read_text())}
REFS = {r["name"]: r for r in json.loads((DATA / "references.json").read_text())}
RTOL = 1e-8

_NAMES = {"c": "cutoff", "p": "polynomial_order", "q": "bias_order", "deriv": "derivative",
          "kernel": "kernel", "bwcheck": "min_distinct", "bwrestrict": "restrict_bandwidth",
          "stdvars": "standardize", "scaleregul": "regularization", "nnmatch": "n_neighbors",
          "rho": "bandwidth_ratio", "scalepar": "scale", "masspoints": "mass_points"}


def _kwargs(case, variant):
    kw = {_NAMES[k]: v for k, v in case.items() if k in _NAMES}
    if "vce" in case:
        kw["vcov"] = case["vce"].upper()
    if "h" in case:
        kw["bandwidth"] = tuple(case["h"]) if isinstance(case["h"], list) else case["h"]
    elif "bwselect" in case:
        kw["bandwidth"] = case["bwselect"]
    if "b" in case:
        kw["bias_bandwidth"] = tuple(case["b"]) if isinstance(case["b"], list) else case["b"]
    if case.get("cluster"):
        # rdrobust's vce with clusters: hc0/hc1 (and the default) give CR1, hc2 and hc3 CR2
        # and CR3
        v = {"HC2": "CRV2", "HC3": "CRV3", "CR2": "CRV2", "CR3": "CRV3"}.get(
            kw.get("vcov"), "CRV1")
        kw["vcov"] = {v: "cl"}
    if case.get("weights"):
        kw["weights"] = "w"
    if "fuzzy" in case:
        kw["treatment"] = case["fuzzy"]
    if "covs" in case:
        kw["covariates"] = case["covs"]
    if case.get("sharpbw"):
        kw["sharp_bandwidth"] = True
    # "collapsed": the rows held, collapsed to one per value ("auto" kept off the by-value
    # path, as when its table does not fit)
    kw["memory"] = {"slabs": "window", "collapsed": "auto"}.get(variant, variant)
    return kw


def _fml(case):
    return f"{case.get('y', 'y')} ~ {case.get('x', 'x')}"


def _close(a, b):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=RTOL, atol=1e-12)


def _collapsible(case):
    return not case.get("cluster") and not case.get("weights")


PARAMS = []
for name, case in CASES.items():
    if case.get("all"):
        continue
    PARAMS.append(pytest.param(name, "rows", id=f"{name}-rows"))
    PARAMS.append(pytest.param(name, "window", id=f"{name}-window"))
    PARAMS.append(pytest.param(name, "slabs", id=f"{name}-slabs"))
    if case.get("x") == "xd" and _collapsible(case):
        PARAMS.append(pytest.param(name, "collapsed", id=f"{name}-collapsed"))
        PARAMS.append(pytest.param(name, "by_value", id=f"{name}-by_value"))


@pytest.fixture
def narrow_window(monkeypatch):
    """The window path's first read at a third of the pilot bandwidth, so
    that the fits read the rows again and the range fits use the power sums."""
    from rd_stream import _window

    monkeypatch.setattr(_window, "_START", 0.3)


@pytest.mark.parametrize("name,variant", PARAMS)
def test_rdrobust(name, variant, narrow_window, monkeypatch, tmp_path):
    case, ref = CASES[name], REFS[name]
    kw = _kwargs(case, variant)
    if variant == "collapsed":
        from rd_stream import _memory

        monkeypatch.setattr(_memory, "_BY_VALUE", False)
    if variant == "slabs":
        # the rows in slabs of at most 300 on disk
        from rd_stream import _window

        monkeypatch.setattr(_window, "_SLABS_ALWAYS", True)
        monkeypatch.setattr(_window, "_SLAB_ROWS", 300)
        kw["workdir"] = tmp_path
    r = rd.rd_estimate(_fml(case), DATA / f"{case['data']}.parquet", **kw)
    _close(r.beta, ref["coef"])
    _close(r.se, ref["se"])
    _close(r.bandwidth, ref["h"])
    _close(r.bias_bandwidth, ref["b"])
    assert list(r.n_bandwidth) == ref["n_h"]
    assert list(r.n_bias_bandwidth) == ref["n_b"]
    assert [r.n_left, r.n_right] == ref["n"]
    if variant not in ("window", "slabs"):
        assert r.collapsed == (variant in ("collapsed", "by_value"))
    assert r.memory == {"by_value": "by_value", "window": "window",
                        "slabs": "window"}.get(variant, "rows")
    if "fuzzy" in case:
        fs = r.first_stage
        _close([fs["tau_cl"], fs["tau_bc"]], ref["tau_T"][:2])
        _close([fs["se_cl"], fs["se_rb"]], [ref["se_T"][0], ref["se_T"][-1]])


@pytest.mark.parametrize("memory", ["auto", "window"])
@pytest.mark.parametrize("name", [n for n, c in CASES.items() if c.get("all")])
def test_rdbwselect_all(name, memory, narrow_window):
    case, ref = CASES[name], REFS[name]
    kw = _kwargs(case, memory)
    bw = rd.rd_bandwidth(_fml(case), DATA / f"{case['data']}.parquet", all_selectors=True, **kw)
    table = {row["selector"]: row for row in bw.table.iter_rows(named=True)}
    for row in ref["bws"]:
        got = table[row["selector"]]
        _close([got["h_left"], got["h_right"], got["b_left"], got["b_right"]],
               [row["2"], row["3"], row["4"], row["5"]])
