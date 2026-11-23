"""Gate 1: competitor scaling. Each fit runs in its own process under a hard
memory cap and a time limit; a configuration that times out or runs out of
memory is not run at larger sizes. Results are appended to
results/scaling.jsonl, one line per fit.

    python run_scaling.py [--sizes 1e4 1e5 ...] [--libs R rdrobust diff_diff]
                          [--variants x xd] [--vces nn hc1 cluster]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results" / "scaling.jsonl"
PY = str(HERE.parent.parent / ".venv" / "bin" / "python")
sys.path.insert(0, str(HERE.parent.parent / "tests"))
from r_config import r_libs, rscript_command  # noqa: E402

RSCRIPT = rscript_command() or ["Rscript"]
R_LIBS = r_libs() or ""
CAP = ["systemd-run", "--user", "--scope", "-p", "MemoryMax=10G",
       "-p", "MemorySwapMax=0", "--quiet"]


def command(lib, data, variant, vce):
    if lib == "R":
        return RSCRIPT + [str(HERE / "run_r.R"), str(data), variant, vce]
    return [PY, str(HERE / "run_py.py"), lib, str(data), variant, vce]


def run(lib, n, variant, vce, limit):
    data = DATA / f"rd_{n}.parquet"
    env = dict(os.environ, R_LIBS_USER=R_LIBS, OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    load = os.getloadavg()[0]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(CAP + ["timeout", str(limit)] + command(lib, data, variant, vce),
                           capture_output=True, text=True, env=env)
        status = {0: "ok", 124: "timeout", 137: "out_of_memory", 143: "out_of_memory"}.get(
            p.returncode, f"exit_{p.returncode}")
        out = p.stdout
        err = p.stderr[-2000:]
    except Exception as e:                      # noqa: BLE001
        status, out, err = "error", "", repr(e)
    wall = time.perf_counter() - t0
    rec = {"lib": lib, "n": n, "variant": variant, "vce": vce, "status": status,
           "wall_s": wall, "load_before": load}
    for line in out.splitlines():
        if line.startswith("RESULT "):
            rec.update(json.loads(line[7:]))
    if status != "ok":
        rec["stderr"] = err
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=float, default=[1e4, 1e5, 1e6, 1e7, 3e7])
    ap.add_argument("--libs", nargs="+", default=["rdrobust", "R", "diff_diff"])
    ap.add_argument("--variants", nargs="+", default=["x", "xd"])
    ap.add_argument("--vces", nargs="+", default=["nn", "hc1", "cluster"])
    ap.add_argument("--limit", type=int, default=1800)
    a = ap.parse_args()
    OUT.parent.mkdir(exist_ok=True)
    for lib in a.libs:
        for variant in a.variants:
            for vce in a.vces:
                if lib == "diff_diff" and vce != "nn":
                    continue
                done = []                     # (n, fit seconds) of finished sizes
                for n in (int(v) for v in a.sizes):
                    if len(done) >= 2 and done[-1][1] > 5:
                        (n0, t0), (n1, t1) = done[-2], done[-1]
                        growth = max(1.0, math.log(t1 / t0) / math.log(n1 / n0))
                        predicted = t1 * (n / n1) ** growth
                        if predicted > a.limit:
                            rec = {"lib": lib, "n": n, "variant": variant, "vce": vce,
                                   "status": "skipped_predicted_timeout",
                                   "predicted_s": predicted, "growth_exponent": growth}
                            with OUT.open("a") as f:
                                f.write(json.dumps(rec) + "\n")
                            print(f"{lib:9s} {variant:2s} {vce:7s} n={n:>9d} skipped: predicted "
                                  f"{predicted:.0f}s (growth n^{growth:.2f})", flush=True)
                            break
                    rec = run(lib, n, variant, vce, a.limit)
                    with OUT.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{lib:9s} {variant:2s} {vce:7s} n={n:>9d} {rec['status']:13s} "
                          f"wall={rec['wall_s']:8.1f}s fit={rec.get('fit_s', float('nan')):8.2f}s "
                          f"peak={rec.get('peak_gb', float('nan')):5.2f}GB load={rec['load_before']:.1f}",
                          flush=True)
                    if rec["status"] != "ok":
                        break
                    done.append((n, rec["fit_s"]))


if __name__ == "__main__":
    main()
