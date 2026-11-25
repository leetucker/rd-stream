"""Time and peak memory of rd-stream against rdrobust (R 4.1.1 and Python
2.1.1, with its numba kernels) as the data grow from 10,000 to 100 million
rows.

    python benchmarks/scale_benchmark.py                        # every model, default sizes
    python benchmarks/scale_benchmark.py --model sharp 1000000 10000000

The data are simulated (_data.py). The models:

    sharp      the default fit: rdrobust(y, x) (nearest-neighbor variance,
               mserd bandwidths), rd_estimate("y ~ x")
    clustered  the same with clusters of about 50 rows (CR1 / CRV1)
    discrete   y on a running variable with 500 values (mass points)
    plot       rdplot(y, x) / rd_plot("y ~ x"), without drawing

Every library starts from a Parquet file: rdrobust's reads the columns into
memory (timed as "read"); rd-stream is given the path, with its default
memory="auto". Each configuration runs in its own process (_harness.py), and
one that fails, runs out of memory or exceeds the timeout (an hour) is not
run at larger sizes. The estimates are compared with rd-stream's (column
max_abs_diff: the conventional estimate, the robust standard error and the
bandwidth h; for the plot, the numbers of bins).

Results go to benchmarks/results/scale.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _data import rd_path
from _harness import (
    RESULTS_DIR,
    external_command,
    external_missing,
    run_child,
    run_configs,
    write_machine_info,
)

SCRIPT = Path(__file__).resolve()

CONFIGS = {
    "sharp": {"rd-stream": "rd", "rdrobust (Python)": "py", "rdrobust (R)": "r"},
    "clustered": {"rd-stream": "rd", "rdrobust (Python)": "py", "rdrobust (R)": "r"},
    "discrete": {"rd-stream": "rd", "rdrobust (Python)": "py", "rdrobust (R)": "r"},
    "plot": {"rd-stream": "rd", "rdplot (Python)": "py", "rdplot (R)": "r"},
}
DEFAULT_SIZES = [10_000, 100_000, 1_000_000, 10_000_000, 30_000_000, 100_000_000]
TIMEOUT_S = 3600
X = {"sharp": "x", "clustered": "x", "discrete": "xd", "plot": "x"}


def run_rd(model, path, workdir):
    import rd_stream as rd

    x = X[model]
    t0 = time.perf_counter()
    if model == "plot":
        r = rd.rd_plot(f"y ~ {x}", str(path))
        coefs = {"J_left": int(r.n_bins[0]), "J_right": int(r.n_bins[1])}
        n_obs = int(sum(r.n))
    else:
        kw = {"vcov": {"CRV1": "cl"}} if model == "clustered" else {}
        r = rd.rd_estimate(f"y ~ {x}", str(path), workdir=str(workdir), **kw)
        coefs = {"tau_cl": float(r.beta[0]), "se_rb": float(r.se[2]),
                 "h": float(r.bandwidth[0])}
        n_obs = int(r.n_obs)
    t_fit = time.perf_counter() - t0
    return {"t_read": 0.0, "t_fit": t_fit, "n_obs": n_obs, "converged": True,
            "memory_path": r.memory, "coefs": coefs}


def run_py(model, path):
    import numpy as np
    import pyarrow.parquet as pq

    x = X[model]
    t0 = time.perf_counter()
    cols = [x, "y"] + (["cl"] if model == "clustered" else [])
    t = pq.read_table(path, columns=cols)
    xs, ys = t[x].to_numpy(), t["y"].to_numpy()
    cl = t["cl"].to_numpy() if model == "clustered" else None
    del t
    t_read = time.perf_counter() - t0
    import rdrobust

    t0 = time.perf_counter()
    if model == "plot":
        import matplotlib

        matplotlib.use("Agg")
        r = rdrobust.rdplot(ys, xs, hide=True)
        J = np.ravel(np.asarray(r.J))
        coefs = {"J_left": int(J[0]), "J_right": int(J[1])}
    else:
        r = (rdrobust.rdrobust(ys, xs, cluster=cl) if cl is not None
             else rdrobust.rdrobust(ys, xs))
        coefs = {"tau_cl": float(r.coef.iloc[0, 0]), "se_rb": float(r.se.iloc[2, 0]),
                 "h": float(r.bws.iloc[0, 0])}
    t_fit = time.perf_counter() - t0
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(ys)), "converged": True,
            "coefs": coefs}


def runner(config, n, path, workdir):
    model, lib = config.split(":", 1)
    if lib == "rd":
        return run_rd(model, path, workdir)
    if lib == "py":
        return run_py(model, path)
    raise ValueError(config)


def warmup_data(kind):
    return 100_000, rd_path(100_000)


def r_command(model, path):
    def command(label, kind):
        if not kind.endswith(":r"):
            return None
        spec = dict(fit="rdplot" if model == "plot" else "rdrobust", path=path, x=X[model])
        if model == "clustered":
            spec["cluster"] = "cl"
        return external_command("R", **spec) or external_missing("R")
    return command


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int)
    ap.add_argument("--model", choices=list(CONFIGS), action="append")
    ap.add_argument("--only", nargs="+", help="configuration labels to run")
    ap.add_argument("--timeout", type=float, help=f"seconds (default {TIMEOUT_S})")
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "scale.csv"
    write_machine_info(RESULTS_DIR / "machine.json")
    for model in args.model or list(CONFIGS):
        for n in args.sizes or DEFAULT_SIZES:
            path = rd_path(n)
            configs = {label: f"{model}:{lib}" for label, lib in CONFIGS[model].items()}
            run_configs(SCRIPT, out, model, n, path, configs, command=r_command(model, path),
                        timeout=args.timeout or TIMEOUT_S, only=args.only,
                        reference="rd-stream", warmup=warmup_data)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, config, n, path, workdir = sys.argv
        run_child(runner, config, n, path, workdir)
    else:
        main()
