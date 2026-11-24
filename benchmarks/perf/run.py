"""The deferred performance runs (DECISIONS.md, "Performance tests to run
later"), each case in its own process under a 10 GB cap.

    python run.py T1 T2 ...      (or: all)

Appends one JSON line per case to results/perf.jsonl and prints a summary.
Data are in benchmarks/data/ (make_big.py) and benchmarks/gate/data/
(make_data.py). The window path's slabs go to benchmarks/data/slabs, on
disk: the system temporary directory may be in memory (tmpfs).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from r_config import r_libs, rscript_command  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
GATE = ROOT / "benchmarks" / "gate" / "data"
BIG = ROOT / "benchmarks" / "data"
SLABS = BIG / "slabs"
OUT = HERE / "results" / "perf.jsonl"
RSCRIPT = rscript_command() or ["Rscript"]
CAP = ["systemd-run", "--user", "--scope", "-p", "MemoryMax=10G", "-p", "MemorySwapMax=0",
       "--quiet"]


def data(n, kind="rd", k=1):
    if kind == "rd" and k == 1 and n <= 30_000_000:
        return str(GATE / f"rd_{n}.parquet")
    if kind == "rd" and k == 20 and n == 10_000_000:
        return str(GATE / "rd_10000000_k20.parquet")
    return str(BIG / f"{kind}_{n}{f'_k{k}' if k > 1 else ''}.parquet")


def run(test, tag, cmd, threads=4, limit=3600, env_extra=None):
    env = dict(os.environ, NUMBA_NUM_THREADS=str(threads), POLARS_MAX_THREADS=str(threads),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               R_LIBS_USER=r_libs() or "")
    env.update(env_extra or {})
    load = os.getloadavg()[0]
    t0 = time.perf_counter()
    p = subprocess.run(CAP + ["timeout", str(limit)] + cmd, capture_output=True, text=True,
                       env=env, cwd=HERE)
    rec = {"test": test, "tag": tag, "threads": threads, "load_before": round(load, 2),
           "wall_s": round(time.perf_counter() - t0, 2),
           "status": {0: "ok", 124: "timeout", 137: "out_of_memory"}.get(p.returncode,
                                                                          f"exit_{p.returncode}")}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            rec.update(json.loads(line[7:]))
    if p.returncode:
        rec["stderr"] = p.stderr[-1500:]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
    keys = ("status", "wall_s", "total_s", "fit_s", "read_s", "peak_anon_gb", "peak_rss_gb",
            "memory")
    print(test, tag, {k: rec[k] for k in keys if k in rec}, flush=True)
    return rec


def ours(test, tag, case, path, *args, threads=4, limit=3600):
    a = [json.dumps(x) if isinstance(x, dict) else str(x) for x in args]
    return run(test, tag, [PY, str(HERE / "bench.py"), case, path, *a], threads, limit)


def r(test, tag, script, *args, limit=1800):
    return run(test, tag, RSCRIPT + [str(HERE / script), *map(str, args)], 1, limit)


def py(test, tag, script, *args, limit=1800):
    return run(test, tag, [PY, str(HERE / script), *map(str, args)], 1, limit)


def T1():
    SLABS.mkdir(parents=True, exist_ok=True)
    for mem in ("auto", "in_memory"):
        ours("T1", f"20 outcomes x (mserd + 50 bandwidths), 1e7, {mem}", "multi",
             data(10**7, k=20), 20, 50, {"memory": mem, "workdir": str(SLABS)})


def T2():
    for n in (10**7, 3 * 10**7):
        for tag, fml, kw in (("continuous NN", "y ~ x", {}),
                             ("continuous HC1", "y ~ x", {"vcov": "HC1"}),
                             ("continuous CR1", "y ~ x", {"vcov": {"CRV1": "cl"}}),
                             ("discrete NN", "y ~ xd", {})):
            ours("T2", f"{tag}, {n:.0e}", "fit", data(n), fml, kw)


def T3():
    for n in (10**7, 3 * 10**7):
        for v in ("crv2", "crv3"):
            ours("T3", f"ours {v}, {n:.0e}", "fit", data(n), "y ~ x",
                 {"vcov": {v.upper(): "cl"}})
    for n in (10**6, 10**7):
        for v in ("cr1", "cr2", "cr3"):
            r("T3", f"R {v}, {n:.0e}", "r_fit.R", data(n), "x", v)


def T4():
    for n in (10**7, 3 * 10**7, 10**8):
        for mem in ("by_value", "in_memory"):
            ours("T4", f"discrete {mem}, {n:.0e}", "fit", data(n), "y ~ xd", {"memory": mem})
        ours("T4", f"HyperLogLog pre-pass on x, {n:.0e}", "distinct", data(n), "x",
             {"exact": False})


def T5():
    n = 10**7
    k20 = data(n, k=20)
    for tag, path, fml, kw in (
            ("1 outcome", data(n), "y ~ x", {}),
            ("5 outcomes", k20, "y + y2 + y3 + y4 + y5 ~ x", {}),
            ("clusters", data(n), "y ~ x", {"vcov": {"CRV1": "cl"}}),
            ("covariates 3", k20, "y ~ x", {"covariates": ["y2", "y3", "y4"]}),
            ("fuzzy+covariates", k20, "y ~ x", {"treatment": "y9", "covariates": ["y2"]})):
        ours("T5", f"profile {tag}", "profile", path, fml, {**kw, "memory": "in_memory"})
        ours("T5", f"measured {tag}", "fit", path, fml, {**kw, "memory": "in_memory"})
    ours("T5", "profile by_value 1e8", "profile", data(10**8), "y ~ xd", {"memory": "by_value"})
    ours("T5", "measured by_value 1e8", "fit", data(10**8), "y ~ xd", {"memory": "by_value"})


def T6():
    for n in (10**7, 3 * 10**7):
        for v in ("x", "xd"):
            ours("T6", f"ours rd_plot {v}, {n:.0e}", "plot", data(n), f"y ~ {v}", {})
            py("T6", f"Python rdplot {v}, {n:.0e}", "py_plot.py", data(n), v)
            r("T6", f"R rdplot {v}, {n:.0e}", "r_plot.R", data(n), v)


def T7():
    for threads in (4, 8):
        for chunk in (None, 16):
            for tag, kw in (("NN", {}), ("CR1", {"vcov": {"CRV1": "cl"}})):
                kw2 = dict(kw, **({"_chunksize": chunk} if chunk else {}))
                for rep in range(3):
                    ours("T7", f"{tag} 1e7, {threads} threads, chunk {chunk}, run {rep}",
                         "fit", data(10**7), "y ~ x", kw2, threads=threads)


def T8():
    for n in (10**7, 3 * 10**7):
        ours("T8", f"tables vs direct, {n:.0e}", "tables", data(n), "y ~ x", {})


def T9(sizes=(10**8, 3 * 10**8)):
    SLABS.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        for gb in (2.0, 4.0):
            kw = {"max_memory_gb": gb, "workdir": str(SLABS)}
            ours("T9", f"window auto {gb} GB, {n:.0e}", "fit", data(n), "y ~ x", kw,
                 limit=7200)
        ours("T9", f"phase stats, {n:.0e}", "phase", data(n), "y ~ x",
             {"phase": "stats", "max_memory_gb": 4.0, "workdir": str(SLABS)})
        ours("T9", f"phase write, {n:.0e}", "phase", data(n), "y ~ x",
             {"phase": "write", "max_memory_gb": 4.0, "workdir": str(SLABS)})
    ours("T9", "in_memory 1e8 (expected out of memory)", "fit", data(10**8), "y ~ x",
         {"memory": "in_memory"})
    ours("T9", "auto under the 10 GB cap, 1e8", "fit", data(10**8), "y ~ x",
         {"workdir": str(SLABS)})


def T13():
    """The reruns after the memory model's refit and the distinct counts from slabs."""
    ours("T13", "continuous CR1, 3e+07", "fit", data(3 * 10**7), "y ~ x", {"vcov": {"CRV1": "cl"}})
    for n, gb in ((10**8, 2.0), (3 * 10**8, 2.0), (3 * 10**8, 4.0)):
        ours("T13", f"window auto {gb} GB, {n:.0e}", "fit", data(n), "y ~ x",
             {"max_memory_gb": gb, "workdir": str(SLABS)}, limit=7200)
    T1()


def T14():
    """Ours from T2, T3, T6, T7 and T12 again (they first ran with OMP_WAIT_POLICY=passive)."""
    T2()
    for n in (10**7, 3 * 10**7):
        for v in ("crv2", "crv3"):
            ours("T14", f"ours {v}, {n:.0e}", "fit", data(n), "y ~ x",
                 {"vcov": {v.upper(): "cl"}})
        for v in ("x", "xd"):
            ours("T14", f"ours rd_plot {v}, {n:.0e}", "plot", data(n), f"y ~ {v}", {})
    T7()
    T12()


def T10():
    SLABS.mkdir(parents=True, exist_ok=True)
    n = 10**8
    for held in (2, 4, 8):
        for batch in (250_000, 1_000_000):
            kw = {"max_memory_gb": 4.0, "workdir": str(SLABS),
                  "_set": {"rd_stream._slabs._HELD": held,
                           "rd_stream._slabs._BATCH_ROWS": batch}}
            ours("T10", f"held {held}, batch {batch}, 1e8", "fit", data(n), "y ~ x", kw,
                 limit=7200)


def T11():
    for path, col, tag in ((data(10**8), "x", "1e8 x"), (data(3 * 10**8), "x", "3e8 x"),
                           (data(10**9, kind="ints"), "g", "1e9 ints")):
        for k in (1, 4, 16):
            ours("T11", f"exact, {k} partitions, {tag}", "distinct", path, col,
                 {"partitions": k}, limit=7200)
        ours("T11", f"approximate, {tag}", "distinct", path, col, {"exact": False})


def T12():
    SLABS.mkdir(parents=True, exist_ok=True)
    path = data(10**7, kind="concentrated")
    ours("T12", "concentrated in_memory", "fit", path, "y ~ x", {"memory": "in_memory"})
    for gb in (1.0, 2.0):
        ours("T12", f"concentrated window {gb} GB", "fit", path, "y ~ x",
             {"memory": "window", "max_memory_gb": gb, "workdir": str(SLABS)})


if __name__ == "__main__":
    names = sys.argv[1:]
    if names == ["all"]:
        names = [f"T{i}" for i in range(1, 13)]
    for name in names:
        globals()[name]()
