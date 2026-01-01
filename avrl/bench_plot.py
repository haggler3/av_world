"""The AVBench chart (docs/img/avbench.png) from results/bench/."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import bench as B  # noqa: E402
from .ladder_plot import GRID, INK, INK2, SURFACE, style  # noqa: E402

BAR = "#f0883e"


def pooled_all(names):
    """Counters summed over every scenario: {policy: {"raw": rates, "shielded": rates}}."""
    tot = {}
    for n in names:
        runs = json.load(open(f"{B.OUT}/{n}.json"))["runs"]
        for p, r in runs.items():
            p = B.RENAME.get(p, p)
            for mode, m in r.items():
                tot.setdefault(p, {}).setdefault(mode, 0)
                tot[p][mode] = tot[p][mode] + np.array(m).sum(0)
    return {p: {mode: B.pooled(m[None]) for mode, m in r.items()} for p, r in tot.items()}


def main(out="docs/img/avbench.png"):
    lb = json.load(open(f"{B.OUT}/leaderboard.json"))
    names, checks, board = lb["scenarios"], lb["checks"], lb["board"]
    learned = [p for p in B.POLICIES if B.POLICIES[p] and p in board]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.45, 1, 1]})

    # ---- A: checks x policies, scenarios passed
    ax = axs[0]
    style(ax, f"A  Scenarios where each policy clears the bar (of {len(names)})")
    ax.grid(False)
    M = np.array([[sum(board[p]["scenarios"][n]["checks"][c][0] for n in names) for c in checks] for p in learned])
    cmap = LinearSegmentedColormap.from_list("bar", ["#3a1f1f", "#7a3b2e", "#c9a227", "#2f7a4f", "#3ddc84"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=len(names), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]}", ha="center", va="center", color="#ffffff", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in checks], fontsize=8, color=INK2)
    ax.set_yticks(range(len(learned)))
    ax.set_yticklabels([f"{p}  ({board[p]['passed']}/{len(names) * len(checks)})" for p in learned], fontsize=8, color=INK2)
    ax.axvline(3.5, color=SURFACE, lw=3)
    ax.text(1.5, -0.85, "as deployed (with the safety envelope)", ha="center", color=INK, fontsize=8.5)
    ax.text(5.5, -0.85, "the raw policy (envelope off)", ha="center", color=INK, fontsize=8.5)
    ax.set_ylim(len(learned) - 0.5, -1.1)

    # ---- B: crash rates, deployed vs raw, pooled over every scenario
    ax = axs[1]
    style(ax, "B  AV crashes per AV-hour, all scenarios pooled")
    P = pooled_all(names)
    rule = P["rule-based AV"]["raw"]["av_crash_per_h"]
    human = P["rule-based AV"]["raw"]["human_crash_per_h"]
    y = np.arange(len(learned))
    floor = 3e-3
    raw = [max(P[p]["raw"]["av_crash_per_h"], floor) for p in learned]
    dep = [max(P[p]["shielded"]["av_crash_per_h"], floor) for p in learned]
    for i in range(len(learned)):
        ax.plot([dep[i], raw[i]], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(raw, y, color="#ff6b6b", s=46, zorder=3, label="raw policy (envelope off)")
    ax.scatter(dep, y, color="#3987e5", s=46, zorder=3, label="as deployed (0 plotted at the floor)")
    ax.axvline(rule, color=BAR, lw=1.5, ls=(0, (4, 3)))
    ax.text(rule * 1.15, -0.95, f"the bar: rule-based AV {rule:.3f} /h", color=BAR, fontsize=8)
    ax.axvline(human, color=INK2, lw=1, ls=(0, (2, 3)))
    ax.text(human / 1.15, -0.6, f"humans {human:.3f} /h", color=INK2, fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(len(learned) - 0.5, -1.1)
    ax.set_xlabel("crashes per AV-hour (log; rows as in A)", color=INK2, fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, labelcolor=INK2)

    # ---- C: the trade-off every policy has to win at once
    ax = axs[2]
    style(ax, "C  Two-sided bar: fast enough, safe unaided")
    ax.fill_between([0.97, 1.35], 0.05, 1.0, color="#3ddc84", alpha=0.08, zorder=0)
    ax.axvline(0.97, color=BAR, lw=1.2, ls=(0, (4, 3)))
    ax.axhline(1.0, color=BAR, lw=1.2, ls=(0, (4, 3)))
    base = {n: B.pooled(np.array(json.load(open(f"{B.OUT}/{n}.json"))["runs"]["rule-based AV"]["raw"])) for n in names}
    cols = plt.cm.cool(np.linspace(0, 1, len(learned)))
    for c, p in zip(cols, learned):
        xs, ys = [], []
        for n in names:
            sc = board[p]["scenarios"][n]
            xs.append(sc["shielded"]["av_speed_kmh"] / base[n]["av_speed_kmh"])
            ys.append(max(sc["raw"]["av_crash_per_h"], floor) / max(base[n]["av_crash_per_h"], floor))
        ax.scatter(xs, ys, color=c, s=10, alpha=0.45, zorder=2)
        X_ = P[p]["shielded"]["av_speed_kmh"] / P["rule-based AV"]["raw"]["av_speed_kmh"]
        Y_ = max(P[p]["raw"]["av_crash_per_h"], floor) / rule
        ax.scatter([X_], [Y_], color=c, s=70, edgecolor=SURFACE, lw=1.5, zorder=3)
        if Y_ < 500:  # the four short ladder runs sit together at the top: one label for them below
            off = {"PPO, 130k steps (xnet)": (-12, -2, "right")}.get(p, (8, -4, "left"))
            ax.annotate(p.replace("PPO, ", "").replace(" (", "\n("), (X_, Y_), textcoords="offset points", xytext=off[:2],
                        color=INK2, fontsize=7.5, ha=off[2])
    ax.annotate("the four 62k-step ladder policies:\nfloor it, let the envelope brake", (1.0, 2200), xytext=(0.42, 1500),
                color=INK2, fontsize=7.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.text(0.62, 0.3, "0 raw crashes,\nplotted at the floor", color=INK2, fontsize=7)
    ax.set_yscale("log")
    ax.set_xlim(0.4, 1.35)
    ax.set_ylim(0.05, 6000)
    ax.text(1.33, 0.07, "target", color="#3ddc84", fontsize=8.5, ha="right")
    ax.set_xlabel("AV speed as deployed / rule-based AV (dots: scenarios)", color=INK2, fontsize=8.5)
    ax.set_ylabel("raw crash rate / rule-based AV (log)", color=INK2, fontsize=8.5)
    fig.suptitle("AVBench: nine scenarios, eight checks each, all against the rule-based AV driving the same traffic",
                 color=INK, fontsize=12, x=0.01, ha="left", y=0.995)
    fig.tight_layout(pad=1.6, rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
