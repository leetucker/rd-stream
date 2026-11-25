"""Large data: ten million simulated rows, how rd-stream holds them, and the
same fit under a 1 GB memory budget.

    python examples/large_data.py

The data (about 400 MB of Parquet, written to examples/data/ the first time,
a million rows at a time) have a continuous running variable x, a discrete
one xd (500 values), clusters of about 50 rows and three outcomes, each with
a jump at 0 (xd is x rounded down to a grid of 1/250, so each row keeps its side
of the cutoff).
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import rd_stream as rd

DATA = Path(__file__).resolve().parent / "data"
N = 10_000_000


def simulate(path, n=N, chunk=1_000_000):
    rng = np.random.default_rng(2026)
    with pq.ParquetWriter(path, pa.schema([(c, pa.float64()) for c in
                                           ("x", "xd", "y1", "y2", "y3")] +
                                          [("cl", pa.int64())])) as w:
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            x = 2.0 * rng.beta(2.0, 4.0, m) - 1.0
            d = x >= 0
            w.write_table(pa.table({
                "x": x, "xd": np.floor(x * 250.0) / 250.0,      # each row on its own side of 0
                "y1": 0.5 + 1.2 * x + 3.0 * x**2 + 0.3 * d + rng.normal(0, 0.5, m),
                "y2": np.sin(2 * x) + 0.1 * d + rng.normal(0, 0.6, m),
                "y3": x + 0.0 * d + rng.normal(0, 0.5, m),
                "cl": rng.integers(0, n // 50, m)}))


path = DATA / "rd_large.parquet"
if not path.exists():
    DATA.mkdir(parents=True, exist_ok=True)
    simulate(path)

# ---- what a fit would need, without fitting: one streaming pass
print(rd.rd_estimate("y1 ~ x", path, profile=True))

# ---- the default: the rows fit in memory here, so they are read and sorted once
fit = rd.rd_estimate("y1 ~ x", path)
print(f"{fit.memory}: estimate {fit.estimate:.4f}, robust interval "
      f"[{fit.robust_ci[0]:.4f}, {fit.robust_ci[1]:.4f}], h = {fit.bandwidth[0]:.4f}")

# ---- the same fit within 1 GB: the rows near the cutoff, the rest in slabs on disk
small = rd.rd_estimate("y1 ~ x", path, max_memory_gb=1.0, workdir=DATA)
print(f"{small.memory}: estimate {small.estimate:.4f}, robust interval "
      f"[{small.robust_ci[0]:.4f}, {small.robust_ci[1]:.4f}], h = {small.bandwidth[0]:.4f}")
print(f"largest difference from holding the rows: "
      f"{np.max(np.abs(np.asarray(small.se) / np.asarray(fit.se) - 1)):.1e} (relative)")

# ---- a discrete running variable is grouped by value as it is read
disc = rd.rd_estimate("y1 ~ xd", path)
print(f"{disc.memory}: {disc.n_distinct[0] + disc.n_distinct[1]} distinct values, "
      f"estimate {disc.estimate:.4f}")

# ---- three outcomes from one read and sort, clustered
multi = rd.rd_estimate("y1 + y2 + y3 ~ x", path, vcov={"CRV1": "cl"})
print(multi.estimates().select("outcome", "estimate", "robust_low", "robust_high"))
