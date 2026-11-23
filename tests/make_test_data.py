"""Write the reference tests' data (tests/data/*.parquet) and the option
cases (tests/data/cases.json). R's references come from
benchmarks/external/rdrobust_references.R."""
import json
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).resolve().parent / "data"


def main():
    DATA.mkdir(exist_ok=True)
    import rdrobust
    senate = pl.read_csv(Path(rdrobust.__file__).parent / "data" / "rdrobust_RDsenate.csv",
                         null_values="NA").select(x="margin", y="vote", cl="state")
    senate.write_parquet(DATA / "senate.parquet")

    rng = np.random.default_rng(2026)
    n = 4000
    x = 2 * rng.beta(2, 4, n) - 1
    y = 0.5 + 1.2 * x + 3 * x**2 - 2 * x**3 + 0.3 * (x >= 0) + rng.normal(0, 0.5, n)
    pl.DataFrame({
        "x": x, "y": y,
        "y2": np.sin(3 * x) + 0.2 * (x >= 0.2) + rng.normal(0, 0.3, n),
        "xd": np.round(x * 40) / 40,
        "cl": rng.integers(0, 150, n),
        "w": rng.uniform(0.5, 2.0, n),
        "t": np.where(x >= 0, rng.random(n) < 0.8, rng.random(n) < 0.15).astype(float),
        "t1": np.where(x >= 0, rng.random(n) < 0.7, 0.0).astype(float),
        "z1": (z1 := rng.normal(0, 1, n)),
        "z2": (z2 := 0.5 * x + rng.normal(0, 1, n)),
        "z3": z1 - 2 * z2,
    }).write_parquet(DATA / "sim.parquet")

    cases = []
    def add(name, data, **opts):
        cases.append({"name": name, "data": data, **opts})
    for d in ("senate", "sim"):
        add(f"{d}_default", d)
        add(f"{d}_all_selectors", d, all=True)
    for k in ("epanechnikov", "uniform"):
        add(f"sim_{k}", "sim", kernel=k)
        add(f"sim_{k}_hc2", "sim", kernel=k, vce="hc2")
    for v in ("hc0", "hc1", "hc2", "hc3"):
        add(f"sim_{v}", "sim", vce=v)
    for sel in ("msetwo", "msesum", "msecomb1", "msecomb2", "cerrd", "certwo", "cersum",
                "cercomb1", "cercomb2"):
        add(f"sim_{sel}", "sim", bwselect=sel)
    add("sim_p2", "sim", p=2)
    add("sim_p0", "sim", p=0)
    add("sim_p1_q3", "sim", p=1, q=3)
    add("sim_kink", "sim", p=2, deriv=1)
    add("sim_kink_p1", "sim", p=1, deriv=1)
    add("sim_bwcheck", "sim", bwcheck=30)
    add("sim_nobwrestrict", "sim", bwrestrict=False)
    add("sim_nostdvars", "sim", stdvars=False)
    add("sim_scaleregul0", "sim", scaleregul=0)
    add("sim_nnmatch5", "sim", nnmatch=5)
    add("sim_manual_h", "sim", h=0.2)
    add("sim_manual_hb", "sim", h=0.2, b=0.35)
    add("sim_manual_two", "sim", h=[0.15, 0.25], b=[0.3, 0.4])
    add("sim_rho", "sim", rho=0.5)
    add("sim_manual_rho", "sim", h=0.2, rho=0.8)
    add("sim_cutoff", "sim", c=0.2, y="y2")
    add("sim_scalepar", "sim", scalepar=2.5)
    add("sim_weights", "sim", weights=True)
    add("sim_weights_hc3", "sim", weights=True, vce="hc3")
    add("sim_cluster", "sim", cluster=True)
    add("sim_cluster_manual_hb", "sim", cluster=True, h=0.2)
    add("sim_cluster_manual_h_b", "sim", cluster=True, h=0.2, b=0.3)
    add("sim_cluster_weights", "sim", cluster=True, weights=True)
    add("sim_cluster_msetwo", "sim", cluster=True, bwselect="msetwo")
    add("senate_cluster", "senate", cluster=True)
    add("senate_hc1", "senate", vce="hc1")
    for extra in ({}, {"vce": "hc1"}, {"masspoints": "off"}, {"masspoints": "check"},
                  {"bwselect": "msetwo"}, {"kernel": "uniform"}, {"p": 2}, {"bwcheck": 25}):
        tag = "_".join(f"{k}{v}" for k, v in extra.items()) or "default"
        add(f"discrete_{tag}", "sim", x="xd", **extra)
    add("discrete_cluster", "sim", x="xd", cluster=True)
    add("discrete_all_selectors", "sim", x="xd", all=True)
    add("fuzzy_default", "sim", fuzzy="t")
    add("fuzzy_hc2", "sim", fuzzy="t", vce="hc2")
    add("fuzzy_cluster", "sim", fuzzy="t", cluster=True)
    add("fuzzy_sharpbw", "sim", fuzzy="t", sharpbw=True)
    add("fuzzy_one_sided", "sim", fuzzy="t1")
    add("fuzzy_msetwo_uniform", "sim", fuzzy="t", bwselect="msetwo", kernel="uniform")
    add("fuzzy_kink", "sim", fuzzy="t", p=2, deriv=1)
    add("fuzzy_scalepar", "sim", fuzzy="t", scalepar=2.0)
    add("fuzzy_discrete", "sim", fuzzy="t", x="xd")
    add("fuzzy_discrete_hc1", "sim", fuzzy="t", x="xd", vce="hc1")
    add("fuzzy_weights", "sim", fuzzy="t", weights=True)
    add("covs_default", "sim", covs=["z1", "z2"])
    add("covs_hc3", "sim", covs=["z1", "z2"], vce="hc3")
    add("covs_cluster", "sim", covs=["z1", "z2"], cluster=True)
    add("covs_msetwo", "sim", covs=["z1", "z2"], bwselect="msetwo")
    add("covs_collinear", "sim", covs=["z1", "z2", "z3"])
    add("covs_fuzzy", "sim", covs=["z1", "z2"], fuzzy="t")
    add("covs_fuzzy_sharpbw", "sim", covs=["z1", "z2"], fuzzy="t", sharpbw=True)
    add("covs_fuzzy_one_sided", "sim", covs=["z1", "z2"], fuzzy="t1")
    add("covs_discrete", "sim", covs=["z1", "z2"], x="xd")
    add("covs_kink", "sim", covs=["z1"], p=2, deriv=1)
    add("covs_weights", "sim", covs=["z1", "z2"], weights=True)
    add("covs_all_selectors", "sim", covs=["z1", "z2"], all=True)
    (DATA / "cases.json").write_text(json.dumps(cases, indent=1) + "\n")
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
