"""Simulated sharp RD data for the go/no-go gate.

x = 2 * Beta(2, 4) - 1 (the running variable of Calonico, Cattaneo and
Titiunik's simulations), cutoff 0. y is a cubic in x, plus a jump of 0.3 at
the cutoff, plus N(0, 0.5^2) noise. xd is x rounded to a grid of 1/250
(about 500 distinct values on [-1, 1]), the discrete variant. cl is a
cluster id with about 50 rows per cluster. y2..yK are extra outcomes for
the multi-outcome check.

    python make_data.py N [N ...] [--outcomes K]
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parent / "data"


def mu(x):
    return 0.5 + 1.2 * x + 3.0 * x**2 - 2.0 * x**3


def make(n, outcomes=1, seed=12345):
    rng = np.random.default_rng(seed)
    x = 2.0 * rng.beta(2.0, 4.0, n) - 1.0
    treated = x >= 0
    cols = {
        "x": x,
        "xd": np.round(x * 250.0) / 250.0,
        "cl": rng.integers(0, max(50, n // 50), n).astype(np.int64),
        "y": mu(x) + 0.3 * treated + rng.normal(0.0, 0.5, n),
    }
    for k in range(2, outcomes + 1):
        cols[f"y{k}"] = (np.sin(k * x) + 0.1 * k * treated
                         + rng.normal(0.0, 0.5 + 0.05 * k, n))
    return pa.table(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="+", type=float)
    ap.add_argument("--outcomes", type=int, default=1)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    for n in (int(v) for v in a.n):
        suffix = f"_k{a.outcomes}" if a.outcomes > 1 else ""
        path = DATA / f"rd_{n}{suffix}.parquet"
        if path.exists():
            print("exists", path)
            continue
        pq.write_table(make(n, a.outcomes), path, row_group_size=1 << 20)
        print("wrote", path)


if __name__ == "__main__":
    main()
