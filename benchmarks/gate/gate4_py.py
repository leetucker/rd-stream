"""Gate 4, Python rdrobust: time one mserd fit and fixed-h fits. Args: data outcome h1,h2,..."""
import json, sys, time
import numpy as np, pyarrow.parquet as pq, rdrobust
rng = np.random.default_rng(0); xw = rng.uniform(-1, 1, 500)
rdrobust.rdrobust(xw + (xw >= 0) + rng.normal(size=500), xw)
t = pq.read_table(sys.argv[1], columns=["x", sys.argv[2]])
x, y = t.column(0).to_numpy(), t.column(1).to_numpy()
hs = [float(v) for v in sys.argv[3].split(",")]
a = time.perf_counter(); r = rdrobust.rdrobust(y, x); sel = time.perf_counter() - a
fx = []
for h in hs:
    a = time.perf_counter(); f = rdrobust.rdrobust(y, x, h=h)
    fx.append({"h": h, "s": time.perf_counter() - a, "coef": f.coef.iloc[:, 0].tolist(), "se": f.se.iloc[:, 0].tolist()})
print("RESULT " + json.dumps({"select_s": sel, "h_sel": float(r.bws.iloc[0, 0]), "fixed": fx}))
