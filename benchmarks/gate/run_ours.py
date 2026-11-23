"""Gates 2 and 3, timed: the prototype (x and xd) and the collapsed path (xd)
at each size, each in its own capped process. Appends to results/ours.jsonl.

    python run_ours.py [--sizes ...] [--threads 8]
"""
import argparse, json, os, subprocess, time
from pathlib import Path
from run_scaling import CAP, PY, HERE, DATA

def run(script, n, variant, threads, limit=1800):
    env = dict(os.environ, NUMBA_NUM_THREADS=str(threads), POLARS_MAX_THREADS=str(threads))
    load = os.getloadavg()[0]
    t0 = time.perf_counter()
    p = subprocess.run(CAP + ["timeout", str(limit), PY, str(HERE / script), str(DATA / f"rd_{n}.parquet"),
                              "--variant", variant], capture_output=True, text=True, env=env)
    rec = {"lib": script.removesuffix(".py"), "n": n, "variant": variant, "threads": threads,
           "status": {0: "ok", 124: "timeout", 137: "out_of_memory"}.get(p.returncode, f"exit_{p.returncode}"),
           "wall_s": time.perf_counter() - t0, "load_before": load}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            rec.update(json.loads(line[7:]))
    if rec["status"] != "ok":
        rec["stderr"] = p.stderr[-2000:]
    return rec

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", nargs="+", type=float, default=[1e4, 1e5, 1e6, 1e7, 3e7])
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
out = HERE / "results" / "ours.jsonl"
for script, variant in (("prototype.py", "x"), ("prototype.py", "xd"), ("collapsed.py", "xd")):
    for n in (int(v) for v in a.sizes):
        rec = run(script, n, variant, a.threads)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{rec['lib']:10s} {variant:2s} n={n:>9d} {rec['status']} total={rec.get('total_s', float('nan')):7.2f}s "
              f"peak={rec.get('peak_gb', float('nan')):5.2f}GB load={load if (load:=rec['load_before']) else 0:.1f}", flush=True)
