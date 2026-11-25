"""GIF and figure for avlab.rearpartner (2026-11-25).

    python -m avlab.rearpartner_plot gif   # hunt a human-on-AV rear-end at friction 0.4 -> docs/img/rearpartner.gif
    python -m avlab.rearpartner_plot fig   # pair counts and closing speeds from results/rearpartner/run.json
"""
import json
import os
import sys

os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")

import numpy as np  # noqa: E402

BG, FG, MUTED = "#0d1117", "#e6edf3", "#8b949e"
HUMAN, AV = "#E69F00", "#56B4E9"  # Okabe-Ito


def gif(out="docs/img/rearpartner.gif", seeds=range(12)):
    from avlab import blackbox as BB
    from examples import _common as C
    from xnet import network as NW
    from xnet import render as R
    from av_world import easy
    net, geo, info = easy._build("oakland")
    beh = C.calibrated(p_av=0.5, friction=0.4)
    for seed in seeds:
        frames, e = BB.hunt(net, beh, seed=seed, types=("rear-end",), budget=300.0)
        if frames is None:
            continue
        rep = BB.investigate(net, geo, frames, beh)
        k = rep["frame"]
        st, prev = frames[k], frames[k - 1]
        new = np.nonzero((st.wreck_t > 0) & (prev.wreck_t <= 0))[0]
        avs = st.av[new]
        if "rear-end" in rep["types"] and avs.any() and (~avs).any():
            # which one was in front: the larger position on the same track
            front = new[np.argmax(prev.s[new])]
            if not st.av[front]:
                continue
            clip = frames[max(0, k - 45):k + 20]
            R.animate(net, geo, info, clip, out, zoom=(tuple(rep["centre_xy"]), 40.0),
                      title="Rear-end at friction 0.4: a human driver hits a braking AV", fps=10)
            print("seed", seed, "copy", e, out, os.path.getsize(out))
            return rep
    print("no human-on-AV rear-end found")


def fig(res="results/rearpartner/run.json", out="docs/img/rearpartner.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from avlab import crashlog as CL
    d = json.load(open(res))
    a = d["analysis"]
    rows = {k: np.asarray(v) for k, v in d["rows"].items()}
    rp = CL.rear_pairs(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=BG, gridspec_kw={"width_ratios": [1.1, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_color(MUTED)
        ax.tick_params(colors=FG)
    p = a["pairs"]
    labels = ["human hits human", "human hits AV", "AV hits human", "AV hits AV"]
    vals = [p["human_on_human"], p["human_on_av"], p["av_on_human"], p["av_on_av"]]
    cols = [HUMAN, AV, AV, AV]
    hatch = ["", "", "//", "//"]
    y = np.arange(4)[::-1]
    for yi, v, c, h, lab in zip(y, vals, cols, hatch, labels):
        ax1.barh(yi, v, color=c if not h else BG, edgecolor=c, hatch=h, height=0.6)
        ax1.text(v + max(vals) * 0.02, yi, f"{v}", va="center", color=FG, fontsize=11)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, color=FG, fontsize=11)
    ax1.set_xlim(0, max(vals) * 1.18)
    ax1.set_xlabel("rear-end pairs (struck car not already a wreck)", color=FG)
    h1 = a["H1"]
    ax1.set_title(f"AV is the struck car in {100 * h1['share']:.0f}% of mixed pairs\n(95% {100 * h1['ci'][0]:.0f}-{100 * h1['ci'][1]:.0f}%, p = {h1['p']:.1g})",
                  color=FG, fontsize=11)
    dv = 3.6 * (rp["striker_v"] - rp["struck_v"])
    hs = rp["striker_av"] == 0
    bins = np.linspace(0, max(5.0, np.percentile(dv[hs], 98) if hs.any() else 5), 16)
    for m, c, lab in ((hs & (rp["struck_av"] == 1), AV, "AV in front"), (hs & (rp["struck_av"] == 0), HUMAN, "human in front")):
        if m.sum():
            ax2.hist(dv[m], bins=bins, color=c, alpha=0.55, density=True)
            med = np.median(dv[m])
            ax2.axvline(med, color=c, lw=2)
            ax2.text(med, ax2.get_ylim()[1] * (0.92 if c == AV else 0.78), f" {lab}\n median {med:.1f}", color=c, fontsize=10, va="top")
    ax2.set_xlabel("closing speed of a human striker (km/h)", color=FG)
    ax2.set_ylabel("density", color=FG)
    s2 = a["S2"]
    ax2.set_title(f"Closing speed by who was in front (Holm p = {s2['holm']:.2g})", color=FG, fontsize=11)
    fig.suptitle("Who rear-ends whom: half-AV Oakland, friction 0.4, 96 copies x 10 min", color=FG, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=110, facecolor=BG)
    print(out)


if __name__ == "__main__":
    {"gif": gif, "fig": fig}[sys.argv[1]]()
