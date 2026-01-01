"""The lab's charts: docs/img/avlab_overview.png and docs/img/avlab_conditions.png (from results/avlab/summary.json)."""
import json

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avrl.ladder_plot import GRID, INK, INK2, SURFACE, style  # noqa: E402
from xnet import sim as X  # noqa: E402

from . import sweep as S  # noqa: E402

COL = {"rule": "#9da7b3", "rule+cbf": "#e6edf3", "ppo+idm": "#3987e5", "ppo+cbf": "#3ddc84", "ppo raw": "#ff6b6b"}
TYPE_COL = dict(zip(X.CRASH_NAMES, ("#3987e5", "#f0883e", "#f2cc60", "#d2a8ff", "#ff7b72", "#f778ba", "#3ddc84")))
LABEL = {"demand": "traffic demand", "risk": "human risk", "ped_scale": "pedestrians", "p_ped_violate": "jaywalking",
         "p_bike": "cyclists", "p_av": "AV share", "friction": "road friction", "perception_noise": "sensor noise",
         "breakdowns": "breakdowns"}
FLOOR = 2e-3


def overview(r, out):
    cfgs = [c for c in COL if c in r["configs"]]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.25, 1.05]})
    # A: safety vs speed
    ax = axs[0]
    style(ax, f"A  {r['n_conditions']} conditions pooled: crashes vs speed")
    hum = r["configs"]["rule"]["human_rate"][0]
    for c in cfgs:
        s = r["configs"][c]
        y, lo, hi = (max(v, FLOOR) for v in s["rate"])
        ax.errorbar([s["speed_kmh"]], [y], yerr=[[y - lo], [hi - y]], fmt="o", color=COL[c], ms=9, mec=SURFACE, mew=1.5, capsize=3, zorder=3)
        off = {"rule": (10, 4), "ppo+idm": (-10, -22)}.get(c, (9, -4))
        ax.annotate(f"{c}\n{s['rate'][0]:.3f}/h", (s["speed_kmh"], y), textcoords="offset points", xytext=off, color=INK2, fontsize=8,
                    ha="right" if off[0] < 0 else "left")
    ax.axhline(hum, color=INK2, lw=1, ls=(0, (2, 3)))
    ax.text(0.98, hum * 1.15, f"human drivers, same conditions: {hum:.3f} /veh-h", color=INK2, fontsize=8, ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xlabel("AV mean speed (km/h)", color=INK2, fontsize=8.5)
    ax.set_ylabel("AV crashes per AV-hour (95 % CI, log)", color=INK2, fontsize=8.5)
    # B: sensitivity
    ax = axs[1]
    style(ax, "B  What drives AV crashes: rate ratio, low -> high end of each factor")
    fs = list(S.FACTORS)
    show = [c for c in ("rule+cbf", "ppo+idm", "ppo+cbf", "ppo raw") if c in cfgs]
    for i, c in enumerate(show):
        rr = r["configs"][c]["rate_ratio"]
        y = np.arange(len(fs)) + (i - (len(show) - 1) / 2) * 0.17
        m = np.array([rr[f][0] for f in fs])
        lo = np.array([rr[f][1] for f in fs])
        hi = np.array([rr[f][2] for f in fs])
        ax.errorbar(np.clip(m, 1e-2, 1e3), y, xerr=[np.clip(m - lo, 0, None), np.clip(hi - m, 0, None)], fmt="o", color=COL[c], ms=5,
                    capsize=0, lw=1.2, label=c, zorder=3)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.004, 300)
    ax.set_yticks(range(len(fs)))
    ax.set_yticklabels([LABEL[f] for f in fs], color=INK2, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("x crash rate (Poisson regression, 95 % CI, log)", color=INK2, fontsize=8.5)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False, labelcolor=INK2)
    # C: crash types
    ax = axs[2]
    style(ax, "C  How the AV crashes: crash types (AV involved)")
    ax.grid(False)
    for i, c in enumerate(cfgs):
        t = r["configs"][c]["types"]
        tot = sum(t.values())
        left = 0.0
        for k in X.CRASH_NAMES:
            w = t[k] / tot if tot else 0
            ax.barh(i, w, left=left, color=TYPE_COL[k], height=0.62, label=k if i == 0 else None)
            if w > 0.08:
                ax.text(left + w / 2, i, f"{100 * w:.0f}%", ha="center", va="center", color="#0d1117", fontsize=7.5, weight="bold")
            left += w
        ax.text(1.02, i, f"{int(tot)}", va="center", color=INK2, fontsize=8)
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels(cfgs, color=INK2, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50 %", "100 %"])
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, labelcolor=INK2)
    fig.tight_layout(pad=1.6)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def conditions(r, g, out):
    cfgs = [c for c in ("rule", "rule+cbf", "ppo+idm", "ppo+cbf") if c in r["configs"]]
    fs = list(S.FACTORS)
    fig = plt.figure(figsize=(17, 7.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 0.15, 1.15, 1.15], hspace=0.55, wspace=0.35)
    for j, f in enumerate(fs):
        ax = fig.add_subplot(gs[j // 3, j % 3])
        style(ax, LABEL[f])
        ax.title.set_fontsize(9)
        for c in cfgs:
            dep = r["configs"][c]["dependence"][f]
            ax.plot([d["x"] for d in dep], [max(d["rate"][0], FLOOR) for d in dep], color=COL[c], lw=1.8, marker="o", ms=3.5, label=c)
        ax.set_yscale("log")
        ax.set_ylim(FLOOR * 0.8, 3)
        if S.FACTORS[f][2]:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if j == 0:
            ax.legend(fontsize=6.5, loc="upper left", frameon=False, labelcolor=INK2)
    fig.text(0.02, 0.965, "A  AV crashes per AV-hour across each factor (quartile bins, every other factor varying)",
             color=INK, fontsize=10.5)
    if g:
        cmap = plt.get_cmap("magma")
        for i, c in enumerate([c for c in ("ppo+idm", "ppo+cbf") if c in g["configs"]]):
            ax = fig.add_subplot(gs[:, 4 + i])
            rate = np.array(g["configs"][c]["rate"])
            im = ax.imshow(np.log10(np.maximum(rate, FLOOR)), origin="lower", cmap=cmap, vmin=np.log10(FLOOR), vmax=0.5, aspect="auto",
                           extent=(-0.5, len(g["noise"]) - 0.5, -0.5, len(g["friction"]) - 0.5))
            for a in range(rate.shape[0]):
                for b in range(rate.shape[1]):
                    ax.text(b, a, f"{rate[a, b]:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if rate[a, b] < 0.3 else "#0d1117")
            ax.set_xticks(range(len(g["noise"])))
            ax.set_xticklabels([f"{x:.0%}" for x in g["noise"]], fontsize=7.5, color=INK2)
            ax.set_yticks(range(len(g["friction"])))
            ax.set_yticklabels([f"{x:.2f}" for x in g["friction"]], fontsize=7.5, color=INK2)
            ax.set_xlabel("AV sensor noise (sigma of gap error)", color=INK2, fontsize=8.5)
            if i == 0:
                ax.set_ylabel("road friction", color=INK2, fontsize=8.5)
            ax.set_title(f"{c}", color=INK, fontsize=10, loc="left")
            for s_ in ax.spines.values():
                s_.set_visible(False)
        fig.text(0.615, 0.965, "B  The operating domain: AV crashes per AV-hour, friction x sensor noise (humans x5 risk)",
                 color=INK, fontsize=10.5)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    s = json.load(open(f"{S.OUT}/summary.json"))
    r = s["sweeps"]["oakland"]
    overview(r, "docs/img/avlab_overview.png")
    conditions(r, s.get("grid"), "docs/img/avlab_conditions.png")


if __name__ == "__main__":
    main()
