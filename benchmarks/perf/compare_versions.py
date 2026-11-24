"""One fit with whichever rd_stream is first on the path (PYTHONPATH picks an
older checkout), after a warm-up: its time and step timings.

    python compare_versions.py DATA FML [KWARGS-JSON]
"""
import json
import sys
import time
from pathlib import Path

import rd_stream as rd

SIM = Path(__file__).resolve().parents[2] / "tests" / "data" / "sim.parquet"
path, fml = sys.argv[1], sys.argv[2]
kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
x = fml.split("~")[1].strip()
rd.rd_estimate(f"y ~ {x}", SIM, **kw)
t = time.perf_counter()
r = rd.rd_estimate(fml, path, **kw)
print("RESULT " + json.dumps({"version": str(Path(rd.__file__).parent), "total_s":
                              time.perf_counter() - t, "timings": r.timings,
                              "se": [float(v) for v in r.se]}), flush=True)
