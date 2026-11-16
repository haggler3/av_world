"""Figure for avlab.gapconfirm: AV crash rates, blind vs gap-cautious, at three frictions; rate ratios and the speed cost."""
import numpy as np

BG, FG, GRID_C = "#0d1117", "#e6edf3", "#30363d"
PAL = {"blind": "#f2a541", "gap": "#4c9be8", "old": "#8b949e"}


def _style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=FG, which="both"); ax.grid(color=GRID_C, lw=0.6, axis="y")
    for sp in ax.spines.values():
        sp.set_color(GRID_C)


def plot(res, path="docs/img/gapconfirm.png", old_path="results/frictionmerge/frictionmerge.json"):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    s = res["summary"]
    mus = ["0.4", "0.6", "1.0"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), facecolor=BG)
    for j, m in enumerate(mus):
        for k, pol in enumerate(("blind", "gap")):
            c = s["cells"][f"{pol}@{m}"]
            n, h = c["av_crashes"], c["av_hours"]
            r = 1000 * n / h; lo = 1000 * max(n - 1.96 * np.sqrt(n), 0) / h; hi = 1000 * (n + 1.96 * np.sqrt(n)) / h
            x = j + (k - 0.5) * 0.32
            a1.bar(x, r, 0.3, color=PAL[pol], alpha=0.9)
            a1.errorbar(x, r, yerr=[[r - lo], [hi - r]], color=FG, lw=1, capsize=3)
            a1.text(x, hi + 1, f"{n:.0f}", color=FG, fontsize=8.5, ha="center")
    a1.set_xticks(range(3)); a1.set_xticklabels([f"friction {m}" for m in mus], color=FG)
    a1.set_ylabel("AV-involved crashes per 1000 AV-h", color=FG)
    a1.text(0.98, 0.95, "blind AV (gap scale 1.15)", color=PAL["blind"], transform=a1.transAxes, ha="right", fontsize=9)
    a1.text(0.98, 0.89, "gap-cautious AV (1.495)", color=PAL["gap"], transform=a1.transAxes, ha="right", fontsize=9)
    a1.text(0.98, 0.83, "numbers: AV crashes", color=FG, transform=a1.transAxes, ha="right", fontsize=8.5, alpha=0.7)
    a1.set_title(f"Fresh random stream (seed {res['seed']}), half-AV Oakland", color=FG, fontsize=10, loc="left")
    _style(a1)
    rows = []
    try:
        o = json.load(open(old_path))["summary"]["tests"]["gap vs blind"]
        rows.append(("0.4, 2026-11-15 (seed 6)", o["av_rr"], o["av_speed_ratio"], PAL["old"]))
    except (OSError, KeyError):
        pass
    pooled = s.get("pooled_with_2026_11_15_posthoc")
    for m in mus:
        tt = s["tests"][f"gap@{m} vs blind@{m}"]
        rows.append((f"{m}, today (seed {res['seed']})" + ("  [confirmatory]" if m == "0.4" else ""), tt["av_rr"], tt["av_speed_ratio"], PAL["gap"]))
    if pooled:
        rows.insert(2, ("0.4, both streams pooled (post hoc)", pooled["av_rr"], float("nan"), FG))
    for i, (lab, rr, sp, col) in enumerate(rows[::-1]):
        hi = min(rr["hi"], 20)
        a2.plot([max(rr["lo"], 0.02), hi], [i, i], color=col, lw=2.5)
        a2.plot([rr["rr"]], [i], "o", color=col, ms=7)
        a2.text(0.021, i + 0.22, f"{lab}: {rr['rr']:.2f} ({rr['lo']:.2f}-{rr['hi']:.2f}), p {rr['p']:.3f}" + ("" if sp != sp else f"; speed x{sp:.2f}"),
                color=col, fontsize=8.5)
    a2.axvline(1, color=FG, lw=1, ls=":")
    a2.set_xscale("log"); a2.set_xlim(0.02, 20)
    a2.set_xticks([0.1, 0.3, 1, 3, 10]); a2.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a2.set_ylim(-0.6, len(rows) - 0.1); a2.set_yticks([])
    a2.set_xlabel("AV crash rate ratio, gap / blind (95 %, log)", color=FG)
    a2.set_title("Rate ratio by friction", color=FG, fontsize=10, loc="left")
    _style(a2); a2.grid(color=GRID_C, lw=0.6, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
