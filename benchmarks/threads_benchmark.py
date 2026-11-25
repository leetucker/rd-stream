"""How rd-stream's time depends on its thread counts: the evidence for its
default (DECISIONS.md, item 16).

    python benchmarks/threads_benchmark.py                       # every model, default sizes
    python benchmarks/threads_benchmark.py --model continuous 10000000

Each configuration sets numba's and Polars' thread counts (NUMBA_NUM_THREADS,
POLARS_MAX_THREADS), and some numba's chunk size, and fits one model on the
simulated data (_data.py):

    continuous  rd_estimate("y ~ x"), the default fit (nearest-neighbor
                variance, mserd bandwidths), holding the rows
    clustered   the same with vcov={"CRV1": "cl"}
    discrete    rd_estimate("y ~ xd"), grouped by value
    window      rd_estimate("y ~ x", memory="window", max_memory_gb=4): the
                rows near the cutoff, and slabs on disk

On a machine with simultaneous multithreading, the counts up to the number
of physical cores and up to the number of logical CPUs show whether the
second hardware thread of each core helps; on a hybrid CPU, whether the
efficiency cores do. Each configuration fits REPEATS times in its own
process, after an untimed warm-up of the kernels; the row reports the
median (t_fit), the fastest and the slowest. A configuration asking for
more threads than the machine has is skipped.

Results go to benchmarks/results/threads.csv.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

from _data import rd_path
from _harness import RESULTS_DIR, run_child, run_configs, write_machine_info

SCRIPT = Path(__file__).resolve()
REPEATS = 3

CONFIGS = {
    "continuous": {
        "rd-stream (1 thread)": "t1",
        "rd-stream (2 threads)": "t2",
        "rd-stream (4 threads)": "t4",
        "rd-stream (8 threads)": "t8",
        "rd-stream (16 threads)": "t16",
        "rd-stream (8 threads, chunks of 4096)": "t8c4096",
        "rd-stream (16 threads, chunks of 4096)": "t16c4096",
        "rd-stream (8 threads, Polars 16)": "t8p16",
    },
    "clustered": {
        "rd-stream (1 thread)": "t1",
        "rd-stream (2 threads)": "t2",
        "rd-stream (4 threads)": "t4",
        "rd-stream (8 threads)": "t8",
        "rd-stream (16 threads)": "t16",
        "rd-stream (8 threads, chunks of 4096)": "t8c4096",
        "rd-stream (16 threads, chunks of 4096)": "t16c4096",
        "rd-stream (8 threads, Polars 16)": "t8p16",
    },
    "discrete": {
        "rd-stream (1 thread)": "t1",
        "rd-stream (2 threads)": "t2",
        "rd-stream (4 threads)": "t4",
        "rd-stream (8 threads)": "t8",
        "rd-stream (16 threads)": "t16",
        "rd-stream (8 threads, Polars 16)": "t8p16",
    },
    "window": {
        "rd-stream (1 thread)": "t1",
        "rd-stream (2 threads)": "t2",
        "rd-stream (4 threads)": "t4",
        "rd-stream (8 threads)": "t8",
        "rd-stream (16 threads)": "t16",
        "rd-stream (8 threads, chunks of 4096)": "t8c4096",
        "rd-stream (16 threads, chunks of 4096)": "t16c4096",
        "rd-stream (8 threads, Polars 16)": "t8p16",
    },
}
DEFAULT_SIZES = [1_000_000, 10_000_000, 30_000_000, 100_000_000]
TIMEOUT_S = 3600
WINDOW_GB = 4.0


def _parse(lib):
    """(numba threads, Polars threads, chunk size or None) of a configuration."""
    rest = lib[1:]
    chunk = polars = None
    if "c" in rest:
        rest, chunk = rest.split("c")
    if "p" in rest:
        rest, polars = rest.split("p")
    return int(rest), int(polars or rest), int(chunk) if chunk else None


def _fit(model, path, workdir):
    import rd_stream as rd

    if model == "continuous":
        return rd.rd_estimate("y ~ x", str(path), memory="rows")
    if model == "clustered":
        return rd.rd_estimate("y ~ x", str(path), vcov={"CRV1": "cl"}, memory="rows")
    if model == "discrete":
        return rd.rd_estimate("y ~ xd", str(path), memory="by_value")
    return rd.rd_estimate("y ~ x", str(path), memory="window", max_memory_gb=WINDOW_GB,
                          workdir=str(workdir))


def runner(config, n, path, workdir):
    model, lib = config.split(":", 1)
    _, _, chunk = _parse(lib)
    import numba

    if chunk:
        numba.set_parallel_chunksize(chunk)
    times, fits = [], []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        r = _fit(model, path, workdir)
        times.append(time.perf_counter() - t0)
        fits.append(r)
    mid = sorted(range(REPEATS), key=times.__getitem__)[REPEATS // 2]
    r = fits[mid]
    return {"t_read": 0.0, "t_fit": statistics.median(times), "t_fit_min": min(times),
            "t_fit_max": max(times), "n_obs": int(r.n_obs), "converged": True,
            "numba_threads": numba.get_num_threads(), "memory_path": r.memory,
            "coefs": {"tau_cl": float(r.beta[0]), "se_rb": float(r.se[2])},
            **{k: v for k, v in r.timings.items()}}


def warmup_data(kind):
    return 100_000, rd_path(100_000)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int)
    ap.add_argument("--model", choices=list(CONFIGS), action="append")
    ap.add_argument("--only", nargs="+", help="configuration labels to run")
    ap.add_argument("--timeout", type=float, help=f"seconds (default {TIMEOUT_S})")
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "threads.csv"
    write_machine_info(RESULTS_DIR / "machine.json")
    cpus = os.cpu_count() or 1

    def env(label):
        lib = next(v for m in CONFIGS.values() for k, v in m.items() if k == label)
        threads, polars, _ = _parse(lib)
        return {"NUMBA_NUM_THREADS": str(threads), "POLARS_MAX_THREADS": str(polars),
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def command(label, kind):
        threads, polars, _ = _parse(kind.split(":", 1)[1])
        if max(threads, polars) > cpus:
            return f"skipped: this machine has {cpus} logical CPUs"
        return None

    for model in args.model or list(CONFIGS):
        for n in args.sizes or DEFAULT_SIZES:
            path = rd_path(n)
            configs = {label: f"{model}:{lib}" for label, lib in CONFIGS[model].items()}
            run_configs(SCRIPT, out, model, n, path, configs, command=command,
                        timeout=args.timeout or TIMEOUT_S, only=args.only,
                        reference="rd-stream (1 thread)", env=env, warmup=warmup_data,
                        extra=lambda label: {"threads": env(label)["NUMBA_NUM_THREADS"],
                                             "polars_threads": env(label)["POLARS_MAX_THREADS"]})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, config, n, path, workdir = sys.argv
        run_child(runner, config, n, path, workdir)
    else:
        main()
