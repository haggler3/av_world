"""The world-model planner chart (docs/img/wm_planner.png) from results/wmplan/eval.json and model_eval.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avrl.ladder_plot import INK, INK2, SURFACE, style  # noqa: E402

from . import model as M  # noqa: E402

COL = {"rule": "#9da7b3", "ppo raw": "#ff6b6b", "ppo+cbf": "#f0a3a3", "wm-mpc raw": "#39d0ff", "wm-mpc+cbf": "#a5e6ff"}
TYPES = {"rear-end": "#3987e5", "angle": "#f0883e", "left-turn": "#f2cc60", "sideswipe": "#d2a8ff", "merge": "#ff7b72",
         "pedestrian": "#f778ba", "bicycle": "#3ddc84"}


def main(out="docs/img/wm_planner.png"):
    ev = json.load(open(f"{M.OUT}/eval.json"))
    me = json.load(open(f"{M.OUT}/model_eval.json"))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.3), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1, 1]})
    # A: what the model foresees
    ax = axs[0]
    style(ax, f"A  Latent model vs holding the scene still (crash AUROC {me['crash AUROC']:.3f}, red-run {me['red-light run AUROC']:.4f})")
    rows = [("speed m/s", "speed (m/s)"), ("stop margin", "stop-line margin"), ("leader margin", "leader margin"),
            ("junction margin", "junction margin")]
    t = [0, 1, 2, 3]
    for (key, lab), c in zip(rows[:1], ["#39d0ff"]):
        ax.plot(t, [me[f"{key} @{k}s"]["model"] for k in t], color=c, lw=2.2, marker="o", label=f"{lab}: latent model")
        ax.plot(t, [me[f"{key} @{k}s"]["hold"] for k in t], color=c, lw=1.4, ls=(0, (3, 2)), label=f"{lab}: hold still")
    ax.set_xlabel("seconds ahead", color=INK2, fontsize=8.5)
    ax.set_ylabel("RMS error, AV speed (m/s)", color=INK2, fontsize=8.5)
    ax.set_xticks(t)
    ax.set_ylim(0, 2.9)
    ax2 = ax.inset_axes([0.08, 0.6, 0.4, 0.3])
    labels = ["stop", "leader", "junction"]
    mv = [me[f"{k} @2s (RMS)"]["model"] for k in ("stop margin", "leader margin", "junction margin")]
    hv = [me[f"{k} @2s (RMS)"]["hold"] for k in ("stop margin", "leader margin", "junction margin")]
    x = np.arange(3)
    ax2.bar(x - 0.18, mv, 0.36, color="#39d0ff")
    ax2.bar(x + 0.18, hv, 0.36, color="#5b6670")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7, color=INK2)
    ax2.tick_params(colors=INK2, labelsize=6.5)
    ax2.set_facecolor(SURFACE)
    ax2.set_title("safety margins @2 s, RMS (blue: model)", fontsize=7, color=INK2)
    for s in ax2.spines.values():
        s.set_color("#383835")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False, labelcolor=INK2)
    # B: crashes per AV-hour, by type, without a filter
    ax = axs[1]
    style(ax, "B  AV crashes per AV-hour, no safety filter")
    cfg = ["rule", "ppo raw", "wm-mpc raw"]
    for i, c in enumerate(cfg):
        v = ev[c]
        left = 0.0
        for k, col in TYPES.items():
            w = v["types"][k] / v["av_hours"]
            if w:
                ax.barh(i, w, left=left, color=col, height=0.6, label=k if k not in [l.get_label() for l in ax.patches] else None)
            left += w
        ax.text(left + 0.01, i, f"{v['crash_per_h']:.2f}/h  ({int(v['crashes'])} in {v['av_hours']:.0f} AV-h)", va="center", color=INK2, fontsize=8)
    ax.set_yticks(range(len(cfg)))
    ax.set_yticklabels(["rule-based AV", "PPO policy (234k steps)", "latent planner (PPO prior)"], color=INK2, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(ev[c]["crash_per_h"] for c in cfg) * 1.7 + 0.05)
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), fontsize=7.5, loc="lower right", frameon=False, labelcolor=INK2)
    ax.text(0.98, 0.42, "with the barrier function:\nall three 0 crashes", transform=ax.transAxes, ha="right", color=INK, fontsize=8)
    # C: what it costs
    ax = axs[2]
    style(ax, "C  Speed and smoothness")
    cfg = ["rule", "ppo raw", "ppo+cbf", "wm-mpc raw", "wm-mpc+cbf"]
    for c in cfg:
        v = ev[c]
        ax.scatter(v["speed_kmh"], v["hard_per_h"], s=90, color=COL[c], edgecolor=SURFACE, lw=1.5, zorder=3)
        ax.annotate(c, (v["speed_kmh"], v["hard_per_h"]), textcoords="offset points", xytext=(8, 3), color=INK2, fontsize=8)
    ax.set_xlabel("AV mean speed (km/h)", color=INK2, fontsize=8.5)
    ax.set_ylabel("hard brakes per AV-hour", color=INK2, fontsize=8.5)
    fig.tight_layout(pad=1.6)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
