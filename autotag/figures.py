"""Figures and GIFs for docs/AUTOTAG.md, from the DAG's cached outputs and results/autotag/eval.json.

    python -m autotag.figures              # -> docs/img/autotag/*.png, *.gif
"""
import glob
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

OUT = "docs/img/autotag"
R = "results/autotag"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
GRAY, INK, MUTED, SURFACE = "#8d8c86", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.7, "axes.axisbelow": True,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})


def cached(stage):
    from . import dag
    from . import pipeline  # noqa: F401
    keys = dag.keys_for(stage)
    with open(os.path.join(dag.CACHE, f"{stage}-{keys[stage]}.pkl"), "rb") as f:
        return pickle.load(f)


def label(ax, xs, ys, fmt="{:.2f}"):
    for x, y in zip(xs, ys):
        ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=8.5)


def fig_quality(r):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1.1, 1.3, 1.0]})
    names = ["lane_change", "cut_in"]
    w = 0.2
    for k, (src, which, c, lab) in enumerate((("rules_untuned", "precision", "#9db9e6", "precision, untuned"), ("rules", "precision", BLUE, "precision, tuned"),
                                              ("rules_untuned", "recall", "#f2b199", "recall, untuned"), ("rules", "recall", ORANGE, "recall, tuned"))):
        vals = [r[src][n][which] for n in names]
        ax[0].bar(np.arange(2) + (k - 1.5) * w, vals, w * 0.92, color=c, label=lab)
    ax[0].set_xticks(range(2), ["lane change", "cut-in"]), ax[0].set_ylim(0, 1.15), ax[0].legend(fontsize=8, ncol=2, loc="upper center")
    ax[0].set_ylabel("against simulator truth (held-out copies)"), ax[0].set_title("Rules, tuned on the golden set")
    lt = ["hidden_ahead", "distracted", "crash_soon", "tailgater"]
    au = [r["learned"][n]["auroc"] for n in lt] + [r["counterfactual"]["auroc_spread"]]
    nm = ["hidden road\nuser ahead", "distracted\ndriver", "crash within\n3 s", "tailgater\n(style)", "action-\nsensitive"]
    cols = [BLUE, BLUE, BLUE, BLUE, AQUA]
    ax[1].bar(range(5), au, 0.6, color=cols)
    label(ax[1], range(5), au, "{:.3f}")
    ax[1].axhline(0.5, color=INK, lw=1)
    ax[1].set_xticks(range(5), nm, fontsize=8.5), ax[1].set_ylim(0.4, 1.05), ax[1].set_ylabel("AUROC on the independent golden set")
    ax[1].set_title("Learned and counterfactual taggers")
    e0 = [r["learned"][n]["ece_raw"] for n in lt]
    e1 = [r["learned"][n]["ece_calibrated"] for n in lt]
    ax[2].bar(np.arange(4) - 0.2, e0, 0.38, color=GRAY, label="raw score")
    ax[2].bar(np.arange(4) + 0.2, e1, 0.38, color=BLUE, label="after isotonic calibration")
    ax[2].set_xticks(range(4), nm[:4], fontsize=8.5), ax[2].set_ylabel("expected calibration error"), ax[2].legend(fontsize=8)
    ax[2].set_title("Calibration")
    fig.tight_layout()
    fig.savefig(f"{OUT}/quality.png", dpi=150)


def fig_drift(r, dom):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    rates = r["tag_rates_per_scene"]
    names = [n for n in ("hard_brake", "low_ttc", "tailgating", "cut_in", "lane_change", "turning", "ped_close", "dense_traffic", "hidden_ahead", "risky_moment")
             if n in rates["sim"] or n in rates["av2"]]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [rates["sim"].get(n, 0) for n in names], 0.38, color=BLUE, label="simulated (Oakland)")
    ax[0].bar(x + 0.2, [rates["av2"].get(n, 0) for n in names], 0.38, color=ORANGE, label="real (Argoverse 2)")
    ax[0].set_yscale("log"), ax[0].set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8), ax[0].set_ylabel("tags per scene (log)")
    ax[0].set_title("The same taggers on real and simulated scenes"), ax[0].legend(fontsize=8.5)
    bins = np.linspace(0, max(np.quantile(dom["score_av2"], 0.98), dom["threshold"] * 1.5), 40)
    env = dom["env"]
    ax[1].hist(dom["score_sim"][env >= 6], bins=bins, color=BLUE, alpha=0.75, label="held-out simulated scenes", density=True)
    ax[1].hist(dom["score_av2"], bins=bins, color=ORANGE, alpha=0.65, label="real scenes", density=True)
    ax[1].axvline(dom["threshold"], color=INK, lw=1.2)
    ax[1].text(dom["threshold"], ax[1].get_ylim()[1] * 0.92, "  sim_gap threshold\n  (99th pct of sim)", fontsize=8, va="top")
    ax[1].set_xlabel("distance to the 5 nearest simulated scenes (standardised)"), ax[1].set_ylabel("density")
    ax[1].set_title(f"{100 * r['domain']['av2_share_sim_gap']:.0f} % of real scenes are outside the simulator"), ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/drift_domain.png", dpi=150)


def fig_fusion_active(r):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.0))
    lm = r["label_model"]
    names = list(lm["single"]) + ["majority vote", "label model (EM)"]
    vals = list(lm["single"].values()) + [lm["majority_auroc"], lm["label_model_auroc"]]
    cols = [GRAY] * len(lm["single"]) + ["#9db9e6", BLUE]
    ax[0].barh(range(len(names)), vals, color=cols)
    for i, v in enumerate(vals):
        ax[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax[0].axvline(0.5, color=INK, lw=1)
    ax[0].set_yticks(range(len(names)), names, fontsize=8.5), ax[0].invert_yaxis(), ax[0].set_xlim(0.3, 0.9)
    ax[0].set_xlabel("AUROC for a crash within 3 s (golden set)"), ax[0].set_title("Fusing taggers without labels")
    al = r["active_learning"]
    for k, c in (("random", GRAY), ("uncertainty", ORANGE), ("unc+div", BLUE)):
        n, a = zip(*al[k])
        ax[1].plot(n, a, marker="o", ms=4, lw=2, color=c, label=k)
    ax[1].set_xlabel("items reviewed (the simulator answers)"), ax[1].set_ylabel("held-out AUROC, tailgater")
    ax[1].set_title("Active learning: which items to review"), ax[1].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fusion_active.png", dpi=150)


# ---------------------------------------------------------------- GIFs: a scene and its tags, live
BG, PANEL, TXT, TMUTED = "#0d1117", "#141a1f", "#e6edf3", "#8b949e"
TAGCOL = {"hard_brake": "#e66767", "low_ttc": "#f0c35a", "tailgating": "#eb6834", "cut_in": "#e87ba4", "lane_change": "#9085e9", "turning": "#1baf7a",
          "queued": "#6e7681", "ped_close": "#5a9cf0", "hidden_ahead": "#3987e5", "risky_moment": "#ffa198", "dense_traffic": "#8b949e", "sim_gap": "#d2a8ff",
          "tailgater": "#ffb020", "distracted": "#ff7b72"}


def car(ax, x, y, h, L, fc, ec=None, lw=0.0, z=5):
    c, s = np.cos(h), np.sin(h)
    pts = np.array([[-L / 2, -0.95], [L / 2, -0.95], [L / 2, 0.95], [-L / 2, 0.95]])
    p = np.stack([x + pts[:, 0] * c - pts[:, 1] * s, y + pts[:, 0] * s + pts[:, 1] * c], 1)
    return ax.add_patch(Polygon(p, closed=True, fc=fc, ec=ec or fc, lw=lw, zorder=z))


def gif_scene(sc, tags, out, title, extra=None):
    """Left: the scene from above, trails faint, vehicles coloured by their active tag. Right: the tag timeline with a moving cursor."""
    T, dt, e = sc["T"], sc["dt"], sc["ego"]
    tg = [t for t in tags if t["status"] == "auto" and t["name"] not in ("queued",)]
    names = [n for n in TAGCOL if any(t["name"] == n for t in tg)]
    fig = plt.figure(figsize=(10.6, 5.6), facecolor=BG)
    ax = fig.add_axes([0.01, 0.04, 0.5, 0.82])
    tl = fig.add_axes([0.57, 0.1, 0.41, 0.74], facecolor=PANEL)
    ax.set_facecolor(PANEL)
    cx, cy = np.nanmean(sc["x"][e]), np.nanmean(sc["y"][e])
    r = 55
    for i in range(sc["x"].shape[0]):
        ax.plot(sc["x"][i], sc["y"][i], color="#30363d", lw=1.0, zorder=1)
    ax.set_xlim(cx - r, cx + r), ax.set_ylim(cy - r, cy + r), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for k, n in enumerate(names):
        for t in tg:
            if t["name"] == n:
                tl.barh(k, t["t_end"] - t["t_start"], left=t["t_start"], height=0.6, color=TAGCOL[n], alpha=0.85)
    tl.set_yticks(range(len(names)), [n.replace("_", " ") for n in names], color=TXT, fontsize=8.5)
    tl.set_xlim(0, T * dt), tl.set_ylim(len(names) - 0.5, -0.5), tl.tick_params(colors=TMUTED)
    tl.set_xlabel("seconds", color=TMUTED)
    for sp in tl.spines.values():
        sp.set_color("#30363d")
    tl.grid(color="#222831")
    cursor = tl.axvline(0, color=TXT, lw=1.2)
    fig.text(0.015, 0.95, title, color=TXT, fontsize=12.5, weight="bold")
    fig.text(0.015, 0.91, "white = ego; a vehicle takes the colour of its active tag; grey lines = where everyone drove", color=TMUTED, fontsize=8.5)
    hud = fig.text(0.57, 0.88, "", color=TXT, fontsize=9)
    arts = []

    def draw(k):
        nonlocal arts
        for a_ in arts:
            a_.remove()
        arts = []
        tnow = k * dt
        active = {}
        for t in tg:
            if t["agent"] >= 0 and t["t_start"] <= tnow <= t["t_end"]:
                active.setdefault(t["agent"], t["name"])
        for i in np.nonzero(sc["alive"][:, k])[0]:
            fc = "#ffffff" if i == e else TAGCOL.get(active.get(i), "#6e7681")
            arts.append(car(ax, sc["x"][i, k], sc["y"][i, k], sc["yaw"][i, k], sc["length"][i], fc, ec="#ffffff" if i in active and i != e else None, lw=1.0))
        if len(sc["px"]):
            m = np.isfinite(sc["px"][:, k])
            arts.append(ax.scatter(sc["px"][m, k], sc["py"][m, k], s=10, c="#ffa198", zorder=6))
        cursor.set_xdata([tnow, tnow])
        on = sorted({t["name"] for t in tg if t["t_start"] <= tnow <= t["t_end"]})
        hud.set_text(f"t = {tnow:4.1f} s   active: {', '.join(on) if on else '-'}")
        return []

    FuncAnimation(fig, draw, frames=T).save(out, writer=PillowWriter(fps=8), dpi=72, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", out, flush=True)


def pick(df, need, src_scenes):
    a = df[df.status == "auto"]
    ok = [sid for sid, g in a.groupby("scene_id") if set(need) <= set(g.name)]
    best = max(ok, key=lambda sid: a[a.scene_id == sid].name.nunique()) if ok else a.scene_id.value_counts().index[0]
    sc = next(s for s in src_scenes if s["id"] == best)
    return sc, a[a.scene_id == best].to_dict("records")


def main():
    os.makedirs(OUT, exist_ok=True)
    r = json.load(open(f"{R}/eval.json"))
    fig_quality(r)
    fig_fusion_active(r)
    fig_drift(r, cached("domain"))
    import pandas as pd
    av2 = cached("av2_scenes")
    sc, tg = pick(pd.read_parquet(f"{R}/tags_av2.parquet"), ("cut_in", "hidden_ahead", "hard_brake"), av2)
    gif_scene(sc, tg, f"{OUT}/av2_scene.gif", "A real Argoverse 2 scene, auto-tagged")
    sim = cached("sim_scenes")
    sc, tg = pick(pd.read_parquet(f"{R}/tags_sim.parquet"), ("lane_change", "hidden_ahead", "turning"), sim)
    gif_scene(sc, tg, f"{OUT}/sim_scene.gif", "A simulated scene, auto-tagged (truth known)")
    print("figures done")


if __name__ == "__main__":
    main()
