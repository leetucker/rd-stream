"""Bandwidth grids: the fits from the block power-sum tables (_moments.py)
agree with separate fits from direct sums."""

from pathlib import Path

import numpy as np
import pytest

import rd_stream as rd

SIM = Path(__file__).resolve().parent / "data" / "sim.parquet"
GRID = [0.1, 0.17, (0.2, 0.3), 0.45, 0.8]
TOL = 1e-9


@pytest.mark.parametrize("fml,kw", [
    ("y ~ x", {}),
    ("y ~ x", {"vcov": "HC1"}),
    ("y ~ x", {"vcov": "HC3", "kernel": "epanechnikov"}),
    ("y ~ x", {"kernel": "uniform", "n_neighbors": 5}),
    ("y ~ x", {"polynomial_order": 2, "derivative": 1}),
    ("y ~ x", {"treatment": "t"}),
    ("y ~ x", {"covariates": ["z1", "z2"], "treatment": "t"}),
    ("y ~ x", {"weights": "w"}),
    ("y ~ x", {"bandwidth_ratio": 0.6}),
    ("y ~ xd", {"collapse": "never"}),
    ("y ~ xd", {"collapse": "always", "treatment": "t"}),
    ("y ~ xd", {"collapse": "always", "covariates": ["z1"]}),
])
def test_grid_matches_separate_fits(fml, kw):
    multi = rd.rd_estimate(fml, SIM, bandwidth=GRID, **kw)
    for r, h in zip(multi, GRID, strict=True):
        one = rd.rd_estimate(fml, SIM, bandwidth=h, **kw)
        scale = np.abs(one.beta).max()
        np.testing.assert_allclose(r.beta, one.beta, rtol=TOL, atol=TOL * scale)
        np.testing.assert_allclose(r.se, one.se, rtol=TOL)
        assert r.n_bandwidth == one.n_bandwidth


def test_grid_with_a_selector_and_numbers():
    multi = rd.rd_estimate("y ~ x", SIM, bandwidth=["mserd", 0.3, 0.6])
    sel = rd.rd_estimate("y ~ x", SIM)
    np.testing.assert_allclose(multi.fetch_model(0).se, sel.se, rtol=TOL)
    assert multi.fetch_model(0).bandwidth_selector == "mserd"
