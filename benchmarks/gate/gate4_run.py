"""Gate 4 driver at 1e7: ours (20 outcomes x 50 bandwidths), then R and Python
rdrobust timing one mserd fit and 3 fixed-h fits (from our grid) for two
outcomes, to extrapolate. Appends to results/gate4.jsonl."""
import json, os, subprocess
from run_scaling import CAP, PY, HERE, RSCRIPT, R_LIBS
data = str(HERE / "data" / "rd_10000000_k20.parquet")
out = HERE / "results" / "gate4.jsonl"
def go(cmd, tag, env=None):
    p = subprocess.run(CAP + ["timeout", "3600"] + cmd, capture_output=True, text=True, env=env)
    rec = {"tag": tag, "rc": p.returncode}
    for l in p.stdout.splitlines():
        if l.startswith("RESULT "): rec.update(json.loads(l[7:]))
    if p.returncode: rec["stderr"] = p.stderr[-2000:]
    with out.open("a") as f: f.write(json.dumps(rec) + "\n")
    return rec
ours = go([PY, str(HERE / "gate4_ours.py"), data, "20", "50"], "ours")
print("ours", {k: ours.get(k) for k in ("read_sort_s", "select_s", "grid_s", "total_s", "peak_gb")}, flush=True)
env = dict(os.environ, R_LIBS_USER=R_LIBS, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
for y in ("y", "y7"):
    g = ours["fits"][y]["grid"]
    hs = ",".join(repr(g[i][0]) for i in (0, 24, 49))
    for tag, cmd in (("python_rdrobust", [PY, str(HERE / "gate4_py.py"), data, y, hs]),
                     ("R", RSCRIPT + [str(HERE / "gate4_r.R"), data, y, hs])):
        r = go(cmd, f"{tag}:{y}", env)
        print(tag, y, "select", r.get("select_s"), "fixed", [f["s"] for f in r.get("fixed", [])], flush=True)
