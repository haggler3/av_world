"""The dynamics-ladder chart (docs/img/dynamics_ladder.png) from results/ladder/scores.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
RAMP = {"1": "#184f95", "2": "#256abf", "3": "#3987e5", "4": "#6da7ec", "5": "#b7d3f6"}  # more dynamic = brighter
NAMES = {"1": "AVs only", "2": "+ human traffic", "3": "+ pedestrians", "4": "+ buses, vans, breakdowns", "5": "+ risky humans"}


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=10)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main(path="results/ladder/scores.json", out="docs/img/dynamics_ladder.png"):
    rows = json.load(open(path))
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.25, 1, 1]})
    # ---- A: score vs simulation steps, one line per training world
    ax = axs[0]
    style(ax, "A  Score in the full-dynamics test world vs simulation steps")
    for w in "12345":
        r = sorted([x for x in rows if x["run"] == f"w{w}"], key=lambda x: x["sim_steps"])
        xs, ys = np.array([x["sim_steps"] for x in r]) / 1000, [x["score"] for x in r]
        ax.plot(xs, ys, color=RAMP[w], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=f"{w}  {NAMES[w]}", zorder=3)
    bc = next(x["score"] for x in rows if x["run"] == "w1" and x["updates"] == 0)
    ax.axhline(bc, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(62.4, bc + 0.0006, "imitation start", color=INK2, fontsize=8, ha="right")
    ax.annotate("PPO's first updates shake\nthe imitated driver loose", xy=(7.8, 0.0425), xytext=(17, 0.0425), color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate("more dynamic worlds\nrecover faster", xy=(7.8, 0.0471), xytext=(14, 0.0505), color=INK, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate("every world converges:\nsimulation steps dominate", xy=(62.4, 0.0676), xytext=(33, 0.0525), color=INK, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("simulator steps (thousands of area-copy steps of 0.2 s)", color=INK2, fontsize=8.5)
    ax.set_ylabel("reward per AV decision", color=INK2, fontsize=8.5)
    leg = ax.legend(title="training world", fontsize=7.5, title_fontsize=8, loc="lower right", frameon=False, labelcolor=INK2)
    leg.get_title().set_color(INK2)
    # ---- B: gain of each rung over rung 1 at fixed budgets
    ax = axs[1]
    style(ax, "B  Gain over the 'AVs only' world at equal steps")
    budgets = [15, 30, 60, 120]
    shades = ["#b7d3f6", "#6da7ec", "#3987e5", "#1c5cab"]
    for b, c in zip(budgets, shades):
        base = next(x["score"] for x in rows if x["run"] == "w1" and x["updates"] == b)
        ys = [100 * (next(x["score"] for x in rows if x["run"] == f"w{w}" and x["updates"] == b) / base - 1) for w in "12345"]
        ax.plot(range(1, 6), ys, color=c, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, zorder=3)
        ax.text(5.12, ys[-1], f"{b} updates\n({b * 520 / 1000:.0f}k steps)", color=INK2, fontsize=7.5, va="center")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(1, 6))
    short = {"1": "AVs only", "2": "+ humans", "3": "+ pedestrians", "4": "+ buses,\nvans", "5": "+ risky\nhumans"}
    ax.set_xticklabels([f"{w}\n{short[w]}" for w in "12345"], fontsize=7.5, color=INK2)
    ax.set_xlim(0.7, 5.95)
    ax.set_ylabel("score vs world 1 [%]", color=INK2, fontsize=8.5)
    ax.annotate("+12 %: each source of dynamics\nadds situations the policy learns\nfrom in the first few thousand steps",
                xy=(5, 12.2), xytext=(1.05, 9.5), color=INK, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.text(1.05, -3.6, "with enough steps the simple world catches up (-1 to -3 %)", color=INK2, fontsize=8)
    # ---- C: from scratch, the world is the only teacher
    ax = axs[2]
    style(ax, "C  From scratch (no imitation start)")
    for w in "15":
        r = sorted([x for x in rows if x["run"] == f"s{w}"], key=lambda x: x["sim_steps"])
        xs, ys = np.array([x["sim_steps"] for x in r]) / 1000, [x["score"] for x in r]
        ax.plot(xs, ys, color=RAMP[w], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, zorder=3)
        ax.text(xs[-1] + 1.5, ys[-1] + (0.0022 if w == "1" else -0.0022), f"world {w}\n{NAMES[w]}", color=INK2, fontsize=7.5, va="center")
    s1 = {x["updates"]: x["score"] for x in rows if x["run"] == "s1"}
    s5 = {x["updates"]: x["score"] for x in rows if x["run"] == "s5"}
    ax.annotate(f"+{100 * (s5[15] / s1[15] - 1):.0f} % for the full world\nafter {15 * 520 / 1000:.1f}k steps", xy=(7.8, s5[15]),
                xytext=(18, 0.036), color=INK, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(0, 80)
    ax.set_xlabel("simulator steps (thousands)", color=INK2, fontsize=8.5)
    fig.suptitle("A more dynamic training world makes the driving policy learn faster early on; simulation steps set where it ends up",
                 color=INK, fontsize=12.5, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.005, "Same map (Oakland), same reward, same PPO budget. Score = reward per AV decision (progress, comfort, jerk, "
             "headway, speeding, safety-envelope corrections, near misses, red runs, crashes) in one held-out world with every "
             "source of dynamics on and humans 5x riskier than calibrated.", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
