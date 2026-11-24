"""Simulated data for the performance runs, written 10 million rows at a time
(so that 3·10^8 rows never sit in memory at once).

Kinds:
  rd            the gate's design (make_data.py): x = 2 Beta(2, 4) - 1, xd = x on a
                1/250 grid, cl about 50 rows a cluster, y a cubic plus a jump of
                0.3 plus N(0, 0.5^2) noise; with --outcomes K, y2..yK as well
  concentrated  the same y, with 90% of x within 1e-4 of the cutoff (x =
                +-1e-4 U) and the rest 2 Beta(2, 4) - 1: most rows in the first
                band of the distance histogram
  ints          one column g of random 63-bit integers (about n distinct values)

    python make_big.py KIND N [--outcomes K] [--out PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[1] / "data"
CHUNK = 10_000_000


def mu(x):
    return 0.5 + 1.2 * x + 3.0 * x**2 - 2.0 * x**3


def chunk(kind, n, outcomes, rng, n_total):
    if kind == "ints":
        return pa.table({"g": rng.integers(0, 2**63 - 1, n, dtype=np.int64)})
    x = 2.0 * rng.beta(2.0, 4.0, n) - 1.0
    if kind == "concentrated":
        near = rng.random(n) < 0.9
        x[near] = rng.uniform(-1e-4, 1e-4, int(near.sum()))
    treated = x >= 0
    cols = {"x": x, "xd": np.round(x * 250.0) / 250.0,
            "cl": rng.integers(0, max(50, n_total // 50), n).astype(np.int64),
            "y": mu(x) + 0.3 * treated + rng.normal(0.0, 0.5, n)}
    for k in range(2, outcomes + 1):
        cols[f"y{k}"] = np.sin(k * x) + 0.1 * k * treated + rng.normal(0.0, 0.5 + 0.05 * k, n)
    return pa.table(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["rd", "concentrated", "ints"])
    ap.add_argument("n", type=float)
    ap.add_argument("--outcomes", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    n = int(a.n)
    DATA.mkdir(parents=True, exist_ok=True)
    tag = f"_k{a.outcomes}" if a.outcomes > 1 else ""
    out = Path(a.out) if a.out else DATA / f"{a.kind}_{n}{tag}.parquet"
    writer = None
    done, i = 0, 0
    while done < n:
        m = min(CHUNK, n - done)
        t = chunk(a.kind, m, a.outcomes, np.random.default_rng([12345, i]), n)
        if writer is None:
            writer = pq.ParquetWriter(out, t.schema)
        writer.write_table(t, row_group_size=1_000_000)
        done, i = done + m, i + 1
    writer.close()
    print(out)


if __name__ == "__main__":
    main()
