"""Gate 5: numerical stress against R. Large offsets (epoch seconds), higher
order (p = 2, q = 3), integer-day running variable (direct and collapsed)."""
import json, os, subprocess
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from prototype import rd_fit, warm
from collapsed import rd_fit_blocks

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent.parent / "tests"))
from r_config import r_environment, rscript_command  # noqa: E402
env = r_environment()
R = (rscript_command() or ["Rscript"]) + [str(HERE / "gate5_r.R")]

def case(name, x, y, c, p):
    path = HERE / "data" / f"stress_{name}.parquet"
    pq.write_table(pa.table({"x": x, "y": y}), path)
    out = subprocess.run(R + [str(path), repr(c), str(p)], capture_output=True, text=True, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not lines:
        print(name, "R failed:", out.stderr[-1500:]); return
    ref = json.loads(lines[0][7:])
    o = np.argsort(x, kind="stable")
    res = {"direct": rd_fit(x[o], y[o], c=c, p=p, q=p + 1)}
    xu, inv = np.unique(x, return_inverse=True)
    if xu.shape[0] < 0.5 * x.shape[0]:
        cnt = np.bincount(inv)
        ybar = np.bincount(inv, y) / cnt
        m2 = np.bincount(inv, (y - ybar[inv]) ** 2)
        res["collapsed"] = rd_fit_blocks(xu, cnt, ybar, m2, c=c, p=p, q=p + 1)
    for k, r in res.items():
        rel = lambda a, b: max(abs(u - v) / abs(v) for u, v in zip(a, b))
        print(f"{name:22s} {k:9s} coef {rel(r['coef'], ref['coef']):.1e} se {rel(r['se'], ref['se']):.1e} "
              f"h {rel(r['h'], ref['h']):.1e} b {rel(r['b'], ref['b']):.1e} n_h {r['n_h']} vs {ref['n_h']}")

warm()
rng = np.random.default_rng(7)
n = 1_000_000
u = 2 * rng.beta(2, 4, n) - 1
y = 0.5 + 1.2 * u + 3 * u**2 - 2 * u**3 + 0.3 * (u >= 0) + rng.normal(0, 0.5, n)
c_sec = 1.7e9
case("epoch_seconds_p1", c_sec + u * 3.0e7, y, c_sec, 1)        # x ~ 1.7e9, a year either side
case("epoch_seconds_p2", c_sec + u * 3.0e7, y, c_sec, 2)
c_day = 19723.0                                                # 2024-01-01 in days since 1970
case("integer_days_p1", np.floor(c_day + u * 730), y, c_day, 1)   # about 1,460 distinct days
case("integer_days_p2", np.floor(c_day + u * 730), y, c_day, 2)
