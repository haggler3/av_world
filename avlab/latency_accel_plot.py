"""GIF and figure for the constant-acceleration latency predictor (avlab.latency.Latency(accel=True)).

    python -m avlab.latency_accel_plot          # docs/img/latency_accel_lane.gif: the avlab.latency_plot lane at 2.0 s delay
    python -m avlab.latency_accel_plot --bench  # docs/img/latency_accel.png from results/latency/accel.json

The lane is the one of avlab.latency_plot (a leader 30 m ahead at 12 m/s brakes at 6 m/s^2 from t = 4 s); the three
followers are clean sensors, a 2.0 s delay with constant-velocity prediction, and the same delay with constant-acceleration
prediction.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avlab import latency_plot as LP  # noqa: E402
from avlab.patchyice_plot import BG, FG, GRID, MUTED  # noqa: E402

DELAY = 2.0
CV, CA = "#E69F00", "#56B4E9"


def lane_gif(path="docs/img/latency_accel_lane.gif"):
    LP.DELAY = DELAY
    LP.MODELS = {"clean sensors": ((0.0, False), "#009E73"),
                 f"{DELAY} s delay + constant-velocity prediction": ((DELAY, True), CV),
                 f"{DELAY} s delay + constant-acceleration prediction": ((DELAY, True, True), CA)}
    out = LP.lane_gif(path=path)
    mins = {nm: min(r["gap"] for r in LP.scripted(p)) for nm, (p, _) in LP.MODELS.items()}
    return out, mins


def bench_fig(src="results/latency/accel.json", path="docs/img/latency_accel.png"):
    res = json.load(open(src))
    cells, tests = res["cells"], res["tests"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=BG)
    ax.set_facecolor(BG)
    for i, d in enumerate((1.4, 2.0)):
        for j, (c, col, lab) in enumerate(((f"comp{d}", CV, "constant velocity"), (f"ca{d}", CA, "constant acceleration"))):
            if c not in cells:
                continue
            r = cells[c]
            x = i + (j - 0.5) * 0.36
            ax.bar(x, r["rate_per_h"], width=0.32, color=col)
            ax.plot([x, x], r["ci"], color=FG, lw=1.4)
            ax.text(x, r["ci"][1] * 1.04 + 0.005, f"{r['av_crash']:.0f} crashes\n{r['rate_per_h']:.3f}/h", color=col,
                    fontsize=8, ha="center", va="bottom")
            if i == 1:
                ax.text(x, 0.03, lab, color=BG, fontsize=7.5, ha="center", va="bottom", rotation=90, weight="bold")
    if "clean" in cells:
        r = cells["clean"]
        ax.axhline(r["rate_per_h"], color=MUTED, lw=1, ls="--")
        ax.text(0.5, r["rate_per_h"] + 0.012, f"clean sensors\n{r['rate_per_h']:.3f}/h", color=MUTED, fontsize=7.5, ha="center")
    ymax = max(cells[c]["ci"][1] for c in cells if c.startswith(("comp1.4", "comp2.0", "ca"))) * 1.3
    ax.set_ylim(0, ymax); ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["1.4 s delay", "2.0 s delay"])
    ax.set_ylabel("AV crashes per AV-hour (95 %)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    fig.text(0.02, 0.95, "Latency: predicting the leader's acceleration as well as its speed", color=FG, fontsize=11,
             weight="bold")
    sub = "Oakland, half AVs, IDM, envelope off, 16 copies x 600 s per bar."
    for n in ("H1", "S1"):
        if n in tests:
            t = tests[n]
            sub += f"  {n}: RR {t['rr']:.2f} ({t['lo']:.2f}-{t['hi']:.2f}), p {t['p']:.2f}"
    fig.text(0.02, 0.895, sub, color=MUTED, fontsize=7.2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.1)
    fig.savefig(path, dpi=110, facecolor=BG)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_fig()
    else:
        print(lane_gif())
