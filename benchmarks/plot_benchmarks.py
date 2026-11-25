"""The benchmark figures, from benchmarks/results/scale.csv and threads.csv.

    python benchmarks/plot_benchmarks.py

Writes into docs/figures/ (or $RD_BENCH_FIGURES), light and dark versions:

- scale_<model>.svg: time and peak memory against rows. rd-stream is drawn
  solid, the other libraries dotted; a configuration that failed, ran out of
  memory or timed out at a size ends with an x at the last size it finished.
- threads_<model>.svg: the speed-up over one thread (the median of the
  repeats) against the thread count, a line per size; the configurations
  that change numba's chunk size or Polars' thread count are drawn as
  separate markers at their numba thread count.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("RD_BENCH_RESULTS", HERE / "results"))
FIGURES = Path(os.environ.get("RD_BENCH_FIGURES", HERE.parent / "docs" / "figures"))
TITLES = {"sharp": "Sharp RD, nearest-neighbor variance, mserd bandwidths",
          "clustered": "Sharp RD, cluster-robust variance (CR1 / CRV1)",
          "discrete": "Sharp RD, a running variable with 500 values",
          "plot": "RD plot (rdplot / rd_plot), without drawing",
          "continuous": "rd_estimate, holding the rows",
          "window": "rd_estimate, memory=\"window\" (4 GB budget)"}
THEMES = {"light": {"fg": "#1f2328", "grid": "#d0d7de", "bg": "white"},
          "dark": {"fg": "#e6edf3", "grid": "#30363d", "bg": "#0d1117"}}
PALETTE = ["#2f6fdf", "#d9622b", "#2a9d6f", "#9b59b6", "#c0392b", "#7f8c8d"]
PLAIN = re.compile(r"^rd-stream \((\d+) threads?\)$")


def rows(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _axes_style(ax, t):
    ax.set_facecolor(t["bg"])
    ax.grid(True, which="major", color=t["grid"], lw=0.6)
    ax.tick_params(colors=t["fg"], which="both")
    for spine in ax.spines.values():
        spine.set_color(t["grid"])


def _legend(fig, ax, t):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
                     frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    for text in leg.get_texts():
        text.set_color(t["fg"])


def _save(fig, name, theme, t):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.{theme}.svg", facecolor=t["bg"])
    plt.close(fig)


# --------------------------------------------------------------------- scale

def scale_series(data, model):
    out = {}
    for r in data:
        if r["model"] != model:
            continue
        s = out.setdefault(r["configuration"], {"ok": [], "failed": None,
                                                "order": int(r.get("order") or 0)})
        if r["status"] == "ok":
            s["ok"].append((int(r["n"]), float(r["wall_s"]), float(r["peak_memory_mb"]) / 1e3))
        elif not r["status"].startswith("skipped"):
            n = int(r["n"])
            s["failed"] = n if s["failed"] is None else min(s["failed"], n)
    for s in out.values():
        s["ok"].sort()
    return dict(sorted(out.items(), key=lambda kv: kv[1]["order"]))


def draw_scale(model, data, theme):
    t = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=t["bg"])
    for ax, (k, ylabel) in zip(axes, ((1, "seconds (read + fit)"), (2, "peak memory (GB)")),
                               strict=True):
        _axes_style(ax, t)
        for i, (label, s) in enumerate(data.items()):
            if not s["ok"]:
                continue
            xs = [p[0] for p in s["ok"]]
            ys = [p[k] for p in s["ok"]]
            mine = label.startswith("rd-stream")
            color = PALETTE[i % len(PALETTE)]
            ax.plot(xs, ys, "-" if mine else ":", lw=2.2 if mine else 1.6, marker="o",
                    ms=3.5, color=color, label=label)
            if s["failed"] is not None:
                ax.plot([xs[-1]], [ys[-1]], "x", ms=9, mew=2, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rows", color=t["fg"])
        ax.set_ylabel(ylabel, color=t["fg"])
    _legend(fig, axes[0], t)
    fig.suptitle(TITLES.get(model, model), color=t["fg"], fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    _save(fig, f"scale_{model}", theme, t)


# ------------------------------------------------------------------- threads

def draw_threads(model, data, theme):
    t = THEMES[theme]
    ok = [r for r in data if r["model"] == model and r["status"] == "ok"]
    sizes = sorted({int(r["n"]) for r in ok})
    if not sizes:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=t["bg"])
    _axes_style(ax, t)
    variants = sorted({r["configuration"] for r in ok if not PLAIN.match(r["configuration"])})
    markers = ["s", "^", "D", "v", "P"]
    for i, n in enumerate(sizes):
        at = {r["configuration"]: r for r in ok if int(r["n"]) == n}
        base = at.get("rd-stream (1 thread)")
        if base is None:
            continue
        one = float(base["t_fit"])
        plain = sorted((int(PLAIN.match(c).group(1)), one / float(r["t_fit"]))
                       for c, r in at.items() if PLAIN.match(c))
        color = PALETTE[i % len(PALETTE)]
        ax.plot([p[0] for p in plain], [p[1] for p in plain], "-", marker="o", ms=4,
                color=color, label=f"{n:,} rows")
        for j, label in enumerate(variants):
            if label in at:
                ax.plot([int(at[label]["threads"])], [one / float(at[label]["t_fit"])],
                        markers[j % len(markers)], ms=7, mfc="none", mew=1.5, color=color,
                        label=label if i == 0 else None)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("numba threads", color=t["fg"])
    ax.set_ylabel("speed-up over 1 thread", color=t["fg"])
    _legend(fig, ax, t)
    fig.suptitle(TITLES.get(model, model), color=t["fg"], fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    _save(fig, f"threads_{model}", theme, t)


def main():
    scale, threads = rows("scale.csv"), rows("threads.csv")
    for model in dict.fromkeys(r["model"] for r in scale):
        s = scale_series(scale, model)
        for theme in THEMES:
            draw_scale(model, s, theme)
    for model in dict.fromkeys(r["model"] for r in threads):
        for theme in THEMES:
            draw_threads(model, threads, theme)
    print(f"figures -> {FIGURES}")


if __name__ == "__main__":
    main()
