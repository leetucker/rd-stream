"""One fit by a Python competitor, timed.

    python run_py.py LIB data.parquet VARIANT VCE [--profile out.prof]

LIB is rdrobust (2.1.1, numba kernel) or diff_diff (4.0.0); VARIANT x or xd;
VCE nn, hc1 or cluster (hc1 clustered). Prints one line: RESULT {json}.
"""

import argparse
import json
import time

import numpy as np
import pyarrow.parquet as pq


def hwm_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024**2
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib")
    ap.add_argument("data")
    ap.add_argument("variant")
    ap.add_argument("vce")
    ap.add_argument("--profile")
    a = ap.parse_args()

    if a.lib == "rdrobust":
        import rdrobust
        # warm the numba kernel (cached on disk) outside the timing
        rng = np.random.default_rng(0)
        xw = rng.uniform(-1, 1, 500)
        rdrobust.rdrobust(xw + (xw >= 0) + rng.normal(size=500), xw)
    else:
        from diff_diff import RegressionDiscontinuity

    t0 = time.perf_counter()
    t = pq.read_table(a.data, columns=[a.variant, "y", "cl"])
    x = t.column(a.variant).to_numpy()
    y = t.column("y").to_numpy()
    cl = t.column("cl").to_numpy()
    del t
    t1 = time.perf_counter()

    prof = None
    if a.profile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    if a.lib == "rdrobust":
        if a.vce == "cluster":
            r = rdrobust.rdrobust(y, x, cluster=cl, vce="hc1")
        else:
            r = rdrobust.rdrobust(y, x, vce=a.vce)
        coef = r.coef.iloc[:, 0].tolist()
        se = r.se.iloc[:, 0].tolist()
        h = r.bws.iloc[0].tolist()
        b = r.bws.iloc[1].tolist()
        n_h = [int(v) for v in r.N_h]
    else:
        import pandas as pd
        if a.vce != "nn":
            raise SystemExit("diff_diff supports only vce='nn'")
        df = pd.DataFrame({"y": y, "x": x})
        res = RegressionDiscontinuity(cutoff=0.0).fit(df, outcome="y", running="x")
        coef = [res.att_conventional, res.att, res.att]
        se = [res.se_conventional, float("nan"), res.se_robust]
        h, b = [res.h_left, res.h_right], [res.b_left, res.b_right]
        n_h = [res.n_h_left, res.n_h_right]
    if prof:
        prof.disable()
        prof.dump_stats(a.profile)
    t2 = time.perf_counter()
    print("RESULT " + json.dumps({
        "read_s": t1 - t0, "fit_s": t2 - t1, "peak_gb": hwm_gb(),
        "coef": coef, "se": se, "h": h, "b": b, "n_h": n_h}, default=float))


if __name__ == "__main__":
    main()
