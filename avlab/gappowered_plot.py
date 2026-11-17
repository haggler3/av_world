"""Figure for avlab.gappowered: gap-caution rate ratios at friction 0.4, one row per random stream plus the post hoc pooled row."""
import numpy as np

BG, FG, GRID_C = "#0d1117", "#e6edf3", "#30363d"
PAL = {"all": "#4c9be8", "merge": "#f2a541", "pooled": "#3fb68b"}


def plot(res, path="docs/img/gappowered.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from avlab.whocrashes import rate_ratio
    s = res["summary"]; P = s["posthoc_pooled_3_streams"]
    fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor=BG)
    rows = [("seed 6 (2026-11-15, 40 copies/cell)", "6"), ("seed 7 (2026-11-16, 40 copies/cell)", "7"),
            ("seed 8 (today, 120 copies/cell): pre-registered", "8")]
    yl = []
    for j, (key, lab) in enumerate((("av_crashes", "all AV crashes"), ("av_merges", "AV merge crashes"))):
        col = PAL["all" if j == 0 else "merge"]
        for i, (name, sd) in enumerate(rows):
            q = P[key]["per_stream"][sd]
            r = rate_ratio(q["gap"], q["gap_h"], q["blind"], q["blind_h"])
            y = -(i * 1.0) - j * 0.32
            ax.plot([max(r["lo"], 0.05), min(r["hi"], 4)], [y, y], color=col, lw=2)
            ax.plot([r["rr"]], [y], "o", color=col, ms=7 if sd == "8" else 5)
            ax.text(4.3, y, f"{r['rr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})  {q['gap']:.0f} vs {q['blind']:.0f}", color=col, fontsize=8.5, va="center")
            if j == 0:
                yl.append((y - 0.16, name))
        pr = P[key]["pooled"]; y = -3.3 - j * 0.32
        ax.plot([pr["lo"], pr["hi"]], [y, y], color=col, lw=3)
        ax.plot([pr["rr"]], [y], "D", color=col, ms=7)
        ax.text(4.3, y, f"{pr['rr']:.2f} ({pr['lo']:.2f}-{pr['hi']:.2f}) p {pr['p']:.1g}", color=col, fontsize=8.5, va="center")
    yl.append((-3.46, "three streams pooled (post hoc)"))
    ax.set_yticks([v for v, _ in yl]); ax.set_yticklabels([n for _, n in yl], color=FG, fontsize=9)
    ax.axvline(1, color=FG, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlim(0.08, 16)
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4"])
    ax.text(0.09, -3.95, "blue: all AV crashes   orange: merge crashes", color=FG, fontsize=8.5, alpha=0.8)
    ax.set_ylim(-4.15, 0.35)
    t = s["tests"]["gap@0.4 vs blind@0.4"]["av_rr"]
    ax.set_title(f"Gap-cautious vs blind AV on ice (friction 0.4, half-AV Oakland).\nToday's pre-registered test: rate ratio {t['rr']:.2f} "
                 f"({t['lo']:.2f}-{t['hi']:.2f}), p {t['p']:.2g}; the effect shrank from stream to stream", color=FG, fontsize=10, loc="left")
    ax.set_xlabel("AV crash-rate ratio, gap / blind (log; label: ratio, 95 %, crashes gap vs blind)", color=FG)
    ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(color=GRID_C, lw=0.6, axis="x")
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
