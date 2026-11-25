"""The U.S. Senate elections data of Cattaneo, Frandsen and Titiunik (2015),
which rdrobust's documentation and replication files use: 1,390 elections,
1914 to 2010.

    margin           the Democratic party's margin of victory in the election,
                     in percentage points (the running variable; cutoff 0)
    vote             its vote share in the next election for the same seat
    state, year      the state and the election's year
    vote_lag1        its vote share in the previous election for the seat
    vote_lag2        and in the one before
    pres_vote_lag1   the Democratic presidential candidate's vote share in the
                     state, at the previous presidential election
    population       the state's population

The lagged vote shares are determined before the election, so they should
not jump at the cutoff: a placebo outcome, or covariates.

The data come with Python's rdrobust package (``pip install rdrobust``);
`load_senate()` writes them to examples/data/senate.parquet the first time
and returns that path.
"""

from __future__ import annotations

from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def load_senate() -> Path:
    path = DATA / "senate.parquet"
    if path.exists():
        return path
    import polars as pl

    try:
        import rdrobust
    except ImportError as exc:
        raise ImportError("the Senate data come with Python's rdrobust package: "
                          "pip install rdrobust") from exc
    src = Path(rdrobust.__file__).parent / "data" / "rdrobust_RDsenate.csv"
    DATA.mkdir(parents=True, exist_ok=True)
    (pl.read_csv(src, null_values="NA", infer_schema_length=None)
     .select("margin", "vote", "state", "year",
             pl.col("demvoteshlag1").alias("vote_lag1"),
             pl.col("demvoteshlag2").alias("vote_lag2"),
             pl.col("presdemvoteshlag1").alias("pres_vote_lag1"), "population")
     .write_parquet(path))
    return path
