"""One rdplot by Python rdrobust 2.1.1, timed, without drawing.

    python py_plot.py data.parquet VARIANT
"""
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import pyarrow.parquet as pq  # noqa: E402
import rdrobust  # noqa: E402


def hwm_gb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1e6
    return float("nan")


t0 = time.perf_counter()
t = pq.read_table(sys.argv[1], columns=[sys.argv[2], "y"])
x, y = t[sys.argv[2]].to_numpy(), t["y"].to_numpy()
t1 = time.perf_counter()
r = rdrobust.rdplot(y, x, hide=True)
t2 = time.perf_counter()
J = getattr(r, "J", None)
print("RESULT " + json.dumps({"read_s": t1 - t0, "fit_s": t2 - t1, "peak_rss_gb": hwm_gb(),
                              "J": None if J is None else [int(v) for v in list(J)]},
                             default=float), flush=True)
