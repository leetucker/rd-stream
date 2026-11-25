"""Simulated regression discontinuity data for the benchmarks: the gate's
design (benchmarks/gate/make_data.py), written 10 million rows at a time so
that large files never sit in memory, and kept in benchmarks/data (or
RD_BENCH_DATA) for later steps.

Columns:
  x   the running variable, 2 Beta(2, 4) - 1 (cutoff 0, about 30% of the rows
      to its right)
  xd  x rounded to a grid of 1/250: a discrete running variable (500 values)
  cl  clusters of about 50 rows
  y   a cubic in x, a jump of 0.3 at the cutoff, and N(0, 0.5^2) noise
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("RD_BENCH_DATA", HERE / "data"))
CHUNK = 10_000_000


def _mu(x):
    return 0.5 + 1.2 * x + 3.0 * x**2 - 2.0 * x**3


def _chunk(n, n_total, rng):
    import pyarrow as pa

    x = 2.0 * rng.beta(2.0, 4.0, n) - 1.0
    return pa.table({
        "x": x,
        "xd": np.round(x * 250.0) / 250.0,
        "cl": rng.integers(0, max(50, n_total // 50), n).astype(np.int64),
        "y": _mu(x) + 0.3 * (x >= 0) + rng.normal(0.0, 0.5, n),
    })


def rd_path(n: int) -> Path:
    """The Parquet file of n simulated rows, written on first use (the same
    rows every time: the seed depends on the chunk, not the run)."""
    import pyarrow.parquet as pq

    n = int(n)
    path = DATA_DIR / f"rd_{n}.parquet"
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    writer, done, i = None, 0, 0
    try:
        while done < n:
            m = min(CHUNK, n - done)
            t = _chunk(m, n, np.random.default_rng([12345, i]))
            if writer is None:
                writer = pq.ParquetWriter(part, t.schema)
            writer.write_table(t, row_group_size=1_000_000)
            done, i = done + m, i + 1
    finally:
        if writer is not None:
            writer.close()
    os.replace(part, path)
    return path
