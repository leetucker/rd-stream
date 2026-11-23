import json, subprocess, sys
PY = sys.executable
R = {(r["n"], r["variant"]): r for r in map(json.loads, open("results/smoke.jsonl"))
     if r["lib"] == "R" and r["vce"] == "nn"}
for (n, v), q in sorted(R.items()):
    out = subprocess.run([PY, "prototype.py", f"data/rd_{n}.parquet", "--variant", v],
                         capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        print(out.stderr[-3000:]); sys.exit(1)
    r = json.loads(line[0][7:])
    rel = lambda a, b: max(abs(x - y) / max(abs(y), 1e-300) for x, y in zip(a, b))
    print(f"n={n:>7d} {v:2s} rel diff: coef {rel(r['coef'], q['coef']):.1e} se {rel(r['se'], q['se']):.1e} "
          f"h {rel(r['h'], q['h']):.1e} b {rel(r['b'], q['b']):.1e} n_h {r['n_h']} vs {q['n_h']}")
