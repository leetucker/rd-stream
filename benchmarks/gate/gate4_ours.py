"""Gate 4, ours: K outcomes x G bandwidths from one read and one sort.
For each outcome: rdrobust's mserd selection, then G fits at h = h_sel * f,
b = h, for f evenly spaced in [0.5, 2].

    python gate4_ours.py data_kK.parquet K G
"""
import json, sys, time
import numpy as np, polars as pl
from numba import get_num_threads
from prototype import estimate, hwm_gb, make_sides, outcome_sides, select_bandwidth, warm

path, K, G = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
warm()
names = ["y"] + [f"y{k}" for k in range(2, K + 1)]
t0 = time.perf_counter()
df = pl.read_parquet(path, columns=["x"] + names).sort("x", maintain_order=True)
x = np.array(df["x"].to_numpy())
ys = {n: np.array(df[n].to_numpy()) for n in names}
del df
left, right, split = make_sides(x, ys["y"])
t1 = time.perf_counter()
fits, sel_s, grid_s = {}, 0.0, 0.0
for n in names:
    l, r = outcome_sides(left, right, split, ys[n])
    a = time.perf_counter()
    h, b = select_bandwidth(l, r, x, ys[n], 0.0, 1, 2, 3, 1.0)
    sel_s += time.perf_counter() - a
    a = time.perf_counter()
    grid = [estimate(l, r, h * f, h * f) for f in np.linspace(0.5, 2.0, G)]
    grid_s += time.perf_counter() - a
    fits[n] = {"h_sel": h, "b_sel": b, "grid": [(g["h"][0], g["coef"], g["se"]) for g in grid]}
t2 = time.perf_counter()
print("RESULT " + json.dumps({"read_sort_s": t1 - t0, "select_s": sel_s, "grid_s": grid_s,
                              "total_s": t2 - t0, "peak_gb": hwm_gb(), "threads": get_num_threads(),
                              "fits": fits}, default=float))
