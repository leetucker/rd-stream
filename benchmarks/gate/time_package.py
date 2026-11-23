"""Time rd_stream.rd_estimate (the package) after an in-process warm-up.
    python time_package.py data.parquet variant -> RESULT json"""
import json, sys, time
from pathlib import Path
import rd_stream as rd
from hdfe_stream.core.resources import peak_rss_gb
sim = Path(__file__).resolve().parents[2] / "tests" / "data" / "sim.parquet"
for v in ("x", "xd"):
    rd.rd_estimate(f"y ~ {v}", sim)
t = time.perf_counter()
r = rd.rd_estimate(f"y ~ {sys.argv[2]}", sys.argv[1])
print("RESULT " + json.dumps({"total_s": time.perf_counter() - t, **r.timings, "collapsed": r.collapsed,
                              "coef": r.beta.tolist(), "peak_gb": peak_rss_gb()}))
