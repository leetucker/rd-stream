"""One timed case, in this process, after a warm-up that compiles the kernels.

    python bench.py CASE DATA [ARGS...]

CASE:
  fit FML KWARGS        rd_estimate(FML, DATA, **KWARGS)
  bandwidth FML KWARGS  rd_bandwidth
  plot FML KWARGS       rd_plot
  profile FML KWARGS    rd_estimate(..., profile=True): the profile's figures
  multi K G [KWARGS]    K outcomes (y, y2..yK) x (mserd + G fixed bandwidths from
                        0.5 to 2 times 0.089) from one read
  distinct COL KWARGS   hdfe_stream.core.distinct.count_distinct(exact=True, **KWARGS)
  phase FML KWARGS      the window path's summary passes ("stats"), or those and
                        the slab write ("write"): KWARGS has "phase"
  tables FML KWARGS     one fit at the selected bandwidth from direct sums, against
                        building the grid tables for it and fitting from them

KWARGS is a JSON object; "_chunksize" sets numba.set_parallel_chunksize, and
"_set" ({"module.NAME": value}) sets module constants (slab tuning). Prints
one line, RESULT {json}, with the time, the peak anonymous memory (RssAnon,
sampled every 2 ms; it leaves out the pages of memory-mapped files) and the
peak resident memory (VmHWM).
"""

import json
import sys
import threading
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2] / "tests" / "data" / "sim.parquet"
_peak = [0]
_stop = [False]


def _watch():
    while not _stop[0]:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("RssAnon"):
                    _peak[0] = max(_peak[0], int(line.split()[1]) * 1024)
        time.sleep(0.002)


def _hwm_gb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def _warm_kwargs(kw):
    """The case's options for the warm-up on the test data, whose covariates
    are z1..z3 and whose take-up is t."""
    drop = {"max_memory_gb", "memory_reserve", "workdir", "memory", "n_threads"}
    out = {k: v for k, v in kw.items() if k not in drop}
    if "covariates" in out:
        out["covariates"] = ["z1", "z2", "z3"][:len(out["covariates"])]
    if "treatment" in out:
        out["treatment"] = "t"
    return out


def main():
    case, data, *args = sys.argv[1:]
    import numba

    import rd_stream as rd

    kw = json.loads(args[-1]) if args and args[-1].startswith("{") else {}
    chunk = kw.pop("_chunksize", None)
    if chunk:
        numba.set_parallel_chunksize(int(chunk))
    import importlib
    for name, value in kw.pop("_set", {}).items():
        mod, attr = name.rsplit(".", 1)
        setattr(importlib.import_module(mod), attr, value)
    # warm-up: compile what the case runs, on the small test data
    if case in ("fit", "bandwidth", "plot", "phase", "profile", "tables"):
        fml = args[0]
        xw = fml.split("~")[1].strip()
        wkw = _warm_kwargs(kw)
        wkw.pop("phase", None)
        if case == "plot":
            rd.rd_plot(f"y ~ {xw}", SIM, **wkw)
        else:
            rd.rd_estimate(f"y + y2 ~ {xw}", SIM, **wkw)
            if kw.get("memory") == "window":
                from rd_stream import _window
                _window._SLABS_ALWAYS, _window._SLAB_ROWS = True, 500
                rd.rd_estimate(f"y + y2 ~ {xw}", SIM, memory="window", **wkw)
                _window._SLABS_ALWAYS, _window._SLAB_ROWS = False, None
    elif case == "multi":
        rd.rd_estimate("y + y2 ~ x", SIM, bandwidth=["mserd", 0.2, 0.3])
    th = threading.Thread(target=_watch, daemon=True)
    th.start()
    out = {}
    t0 = time.perf_counter()
    if case in ("fit", "bandwidth"):
        fn = rd.rd_estimate if case == "fit" else rd.rd_bandwidth
        r = fn(args[0], data, **kw)
        if case == "bandwidth":
            out["table"] = r.table.to_dicts()
        else:
            fits = list(r) if hasattr(r, "__iter__") else [r]
            f0 = fits[0]
            out.update(memory=f0.memory, collapsed=f0.collapsed, timings=f0.timings,
                       n_fits=len(fits), coef=[float(v) for v in f0.beta],
                       se=[float(v) for v in f0.se], h=list(f0.bandwidth),
                       b=list(f0.bias_bandwidth))
    elif case == "plot":
        r = rd.rd_plot(args[0], data, **kw)
        out.update(n_bins=list(r.n_bins), memory=r.memory, timings=r.timings,
                   coef=r.coef.to_dicts())
    elif case == "profile":
        p = rd.rd_estimate(args[0], data, profile=True, **kw)
        out.update(notes={k: v for k, v in p.notes.items() if k != "outcomes"},
                   components=[dict(name=c.name, where=c.where, expected_gb=c.expected / 1e9,
                                    upper_gb=c.upper / 1e9) for c in p.components],
                   baseline_gb=p.baseline_gb)
    elif case == "multi":
        K, G = int(args[0]), int(args[1])
        import numpy as np
        hs = [float(v) for v in 0.089 * np.linspace(0.5, 2.0, G)]
        names = ["y"] + [f"y{k}" for k in range(2, K + 1)]
        r = rd.rd_estimate(" + ".join(names) + " ~ x", data, bandwidth=["mserd", *hs],
                           **kw)
        fits = list(r)
        out.update(n_fits=len(fits), timings=fits[0].timings,
                   selected_h=[f.bandwidth[0] for f in fits[::G + 1]])
    elif case == "distinct":
        from hdfe_stream.core.distinct import count_distinct
        out["count"] = count_distinct(data, {"g": args[0]}, **{"exact": True, **kw})["g"]
    elif case == "phase":
        from rd_stream import _window
        from rd_stream.api import _options, _parse, _window_options
        phase = kw.pop("phase")
        ys, x = _parse(args[0])
        o, _ = _options(0.0, 1, None, 0, "triangular", "nearest_neighbor", 3, "adjust",
                        None, True, True, 1.0, 1.0)
        opts = _window_options("window", o, x, ys, None, None, [], kw.get("max_memory_gb"),
                               None, None, kw.get("workdir"))
        W = _window.WindowData(data, x, ys, None, None, None, [], read=False, **opts)
        out["stats_s"] = time.perf_counter() - t0
        if phase == "write":
            W.to_slabs()
            out["slabs"] = dict(W.store.n_slabs)
            out["slab_rows"] = W.store.slab_rows
            W.cleanup()
    elif case == "tables":
        from rd_stream import api
        from rd_stream._engine import estimate
        ys, x = api._parse(args[0])
        o, _ = api._options(0.0, 1, None, 0, "triangular", "nearest_neighbor", 3, "adjust",
                            None, True, True, 1.0, 1.0)
        d = api._Data(data, x, ys, None, None)
        a = time.perf_counter()
        r = api._fit_one(d, x, ys[0], o, "mserd", None, None, None, None, "auto")
        out["direct_fit_s"] = r.timings["estimate_s"]
        out["selection_s"] = r.timings["bandwidth_s"]
        out["prepare_s"] = r.timings["prepare_s"]
        left, right, _, _ = api._prepare(d, x, ys[0], o, None, None, "auto")
        h = r.bandwidth
        a = time.perf_counter()
        grids = api._grid_sides(left, right, o, [h, h], None, None)
        out["table_build_s"] = time.perf_counter() - a
        a = time.perf_counter()
        est = estimate(left, right, h[0], h[1], h[0], h[1], p=1, q=2, deriv=0, kern=o.kern,
                       vce=o.vce, matches=3, grids=grids)
        out["table_fit_s"] = time.perf_counter() - a
        a = time.perf_counter()
        est2 = estimate(left, right, h[0], h[1], h[0], h[1], p=1, q=2, deriv=0, kern=o.kern,
                        vce=o.vce, matches=3)
        out["direct_fit_again_s"] = time.perf_counter() - a
        out["agree"] = abs(est["se_rb"] / est2["se_rb"] - 1)
    out["total_s"] = time.perf_counter() - t0
    _stop[0] = True
    th.join()
    out.update(peak_anon_gb=_peak[0] / 1e9, peak_rss_gb=_hwm_gb(),
               threads=numba.get_num_threads())
    print("RESULT " + json.dumps(out, default=float), flush=True)


if __name__ == "__main__":
    main()
