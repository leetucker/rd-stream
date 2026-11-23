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


def _kwargs(case, collapse):
    kw = {_NAMES[k]: v for k, v in case.items() if k in _NAMES}
    if "vce" in case:
        kw["vcov"] = case["vce"]
    if "h" in case:
        kw["bandwidth"] = tuple(case["h"]) if isinstance(case["h"], list) else case["h"]
    elif "bwselect" in case:
        kw["bandwidth"] = case["bwselect"]
    if "b" in case:
        kw["bias_bandwidth"] = tuple(case["b"]) if isinstance(case["b"], list) else case["b"]
    if case.get("cluster"):
        kw["cluster"] = "cl"
    if case.get("weights"):
        kw["weights"] = "w"
    if "fuzzy" in case:
        kw["treatment"] = case["fuzzy"]
    if "covs" in case:
        kw["covariates"] = case["covs"]
    if case.get("sharpbw"):
        kw["sharp_bandwidth"] = True
    kw["collapse"] = collapse
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
    PARAMS.append(pytest.param(name, "never", id=f"{name}-rows"))
    if case.get("x") == "xd" and _collapsible(case):
        PARAMS.append(pytest.param(name, "always", id=f"{name}-collapsed"))


@pytest.mark.parametrize("name,collapse", PARAMS)
def test_rdrobust(name, collapse):
    case, ref = CASES[name], REFS[name]
    kw = _kwargs(case, collapse)
    r = rd.rd_estimate(_fml(case), DATA / f"{case['data']}.parquet", **kw)
    _close(r.beta, ref["coef"])
    _close(r.se, ref["se"])
    _close(r.bandwidth, ref["h"])
    _close(r.bias_bandwidth, ref["b"])
    assert list(r.n_bandwidth) == ref["n_h"]
    assert list(r.n_bias_bandwidth) == ref["n_b"]
    assert [r.n_left, r.n_right] == ref["n"]
    assert r.collapsed == (collapse == "always")
    if "fuzzy" in case:
        fs = r.first_stage
        _close([fs["tau_cl"], fs["tau_bc"]], ref["tau_T"][:2])
        _close([fs["se_cl"], fs["se_rb"]], [ref["se_T"][0], ref["se_T"][-1]])


@pytest.mark.parametrize("name", [n for n, c in CASES.items() if c.get("all")])
def test_rdbwselect_all(name):
    case, ref = CASES[name], REFS[name]
    kw = _kwargs(case, "auto")
    kw.pop("collapse")
    bw = rd.rd_bandwidth(_fml(case), DATA / f"{case['data']}.parquet", all_selectors=True, **kw)
    table = {row["selector"]: row for row in bw.table.iter_rows(named=True)}
    for row in ref["bws"]:
        got = table[row["selector"]]
        _close([got["h_left"], got["h_right"], got["b_left"], got["b_right"]],
               [row["2"], row["3"], row["4"], row["5"]])
