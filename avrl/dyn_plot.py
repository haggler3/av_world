"""The 'make the simulator teach harder' chart (docs/img/dynamic_sim.png) from results/dyn/eval.json and
results/avlab/split/validate_*.json."""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avlab.split import OUT as SPLIT_OUT  # noqa: E402

from . import fork as FK  # noqa: E402
from .ladder_plot import INK, INK2, SURFACE, style  # noqa: E402

COL = {"base": "#9da7b3", "shape": "#f2cc60", "fork": "#39d0ff", "fork+shape": "#3ddc84", "mutate": "#d2a8ff"}
NAME = {"base": "barrier teacher only (baseline)", "shape": "+ world-model crash shaping", "fork": "+ fork at danger",
        "fork+shape": "+ both", "mutate": "+ mutating cities"}
STEPS = {0: 0, 25: 25, 50: 50, 100: 100}


def ev(res, tag, world):
    return res.get(f"{tag}|{world}")


def curve(res, name, world="oakland"):
    """{updates: [crash rate per seed ...]}, with the start policy at 0."""
    out = {0: [ev(res, "start", world)["crash_per_h"][0]]} if ev(res, "start", world) else {}
    for k, v in res.items():
        tag, w = k.split("|")
        if w != world or not tag.startswith(name + "_s"):
            continue
        c = int(tag.rsplit("_", 1)[1])
        out.setdefault(c, []).append((v["crash_per_h"][0], v["crash_per_h"][1], v["crash_per_h"][2], v["speed_kmh"][0], v["hard_per_h"][0]))
    return out


def main(out="docs/img/dynamic_sim.png"):
    res = json.load(open(f"{FK.OUT}/eval.json"))
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.2), facecolor=SURFACE)
    # A: crash rate vs training updates
    ax = axs[0, 0]
    style(ax, "A  Crashes per AV-hour with no safety filter, vs training")
    start = ev(res, "start", "oakland")
    for name in COL:
        cv = curve(res, name)
        xs = sorted(c for c in cv if c > 0)
        if not xs:
            continue
        m = [np.mean([r[0] for r in cv[c]]) for c in xs]
        ax.plot([0] + xs, ([start["crash_per_h"][0]] if start else [np.nan]) + m, color=COL[name], lw=2.2, marker="o", ms=5, mec=SURFACE,
                mew=1.4, label=NAME[name], zorder=3)
        for c in xs:
            ax.scatter([c] * len(cv[c]), [r[0] for r in cv[c]], color=COL[name], s=10, alpha=0.4, zorder=2)
    rule = ev(res, "rule", "oakland")
    if rule:
        ax.axhline(rule["crash_per_h"][0], color="#e6edf3", lw=1, ls=(0, (4, 3)))
        ax.text(100, rule["crash_per_h"][0] * 1.12, "rule-based AV", color=INK2, fontsize=8, ha="right")
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xticks([0, 5, 10, 15, 25, 50, 100])
    ax.set_xticklabels(["0", "5", "10", "15", "25", "50", "100"])
    ax.set_xlabel("PPO updates (520 simulator steps each)", color=INK2, fontsize=8.5)
    ax.set_ylabel("AV crashes per AV-hour (log; dots: seeds)", color=INK2, fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False, labelcolor=INK2)
    # B: safety vs speed at 100 updates
    ax = axs[0, 1]
    style(ax, "B  Safety against speed after 100 updates (mean of seeds)")
    for name in COL:
        cv = curve(res, name).get(100)
        if not cv:
            continue
        ax.scatter([np.mean([r[3] for r in cv])], [np.mean([r[0] for r in cv])], s=110, color=COL[name], edgecolor=SURFACE, lw=1.5, zorder=3)
        off = {"shape": (8, -13), "fork": (8, 5), "fork+shape": (-8, 7), "mutate": (8, -4), "base": (8, 5)}[name]
        ax.annotate(name, (np.mean([r[3] for r in cv]), np.mean([r[0] for r in cv])), textcoords="offset points", xytext=off,
                    color=INK2, fontsize=8, ha="right" if off[0] < 0 else "left")
    if start:
        ax.scatter([start["speed_kmh"][0]], [start["crash_per_h"][0]], s=110, color="#ff6b6b", edgecolor=SURFACE, lw=1.5)
        ax.annotate("start (234k-step policy)", (start["speed_kmh"][0], start["crash_per_h"][0]), textcoords="offset points", xytext=(8, 4),
                    color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("AV mean speed (km/h)", color=INK2, fontsize=8.5)
    ax.set_ylabel("AV crashes per AV-hour (log)", color=INK2, fontsize=8.5)
    # C: held-out cities
    ax = axs[1, 0]
    style(ax, "C  Cities never trained in (100 updates; humans x5 riskier)")
    worlds = [("oakland", "Oakland\n(training map)"), ("mutated", "mutated\nOakland, held out"), ("squirrel_hill", "Squirrel Hill"),
              ("downtown", "Downtown")]
    x = np.arange(len(worlds))
    for j, (name, off) in enumerate((("start", -0.3), ("base", -0.1), ("mutate", 0.1), ("rule", 0.3))):
        ys, lo, hi = [], [], []
        for w, _ in worlds:
            if name in ("start", "rule"):
                r = ev(res, name, w)
                vals = [r["crash_per_h"]] if r else []
            else:
                vals = [v["crash_per_h"] for k, v in res.items() if k.startswith(f"{name}_s") and k.endswith(f"_100|{w}")]
            m = np.mean([v[0] for v in vals]) if vals else np.nan
            ys.append(m)
            lo.append(m - np.mean([v[1] for v in vals]) if vals else 0)
            hi.append(np.mean([v[2] for v in vals]) - m if vals else 0)
        ax.bar(x + off, ys, 0.19, color={"start": "#ff6b6b", "base": COL["base"], "mutate": COL["mutate"], "rule": "#e6edf3"}[name], yerr=[lo, hi],
               error_kw=dict(ecolor=INK2, lw=1), label={"start": "start policy", "base": "trained on Oakland", "mutate": "trained on mutating cities",
                                                        "rule": "rule-based AV"}[name])
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in worlds], fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_ylabel("AV crashes per AV-hour (log; bars: 95 % CI)", color=INK2, fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    # D: the estimator
    ax = axs[1, 1]
    style(ax, "D  Estimating a rare crash rate: spread over replicates")
    vals = sorted(glob.glob(f"{SPLIT_OUT}/validate_*.json"))
    rows = [json.load(open(p)) for p in vals]
    for i, r in enumerate(rows):
        tag = r["policy"].split("/")[-2] + "\n" + f"{r['plain_crash_per_h']['mean']:.3f}/AV-h"
        for j, (m, c) in enumerate((("plain", "#9da7b3"), ("split", "#3ddc84"))):
            sd = r[f"{m}_crash_per_h"]["sd"]
            ax.bar(i + (j - 0.5) * 0.36, sd, 0.34, color=c, label={"plain": "plain Monte Carlo", "split": "splitting on danger"}[m] if i == 0 else None)
            ax.text(i + (j - 0.5) * 0.36, sd, f"{sd:.4f}", ha="center", va="bottom", color=INK2, fontsize=7.5)
        ax.text(i, -0.0004, tag, ha="center", va="top", color=INK2, fontsize=7.5)
        ax.text(i, max(r["plain_crash_per_h"]["sd"], r["split_crash_per_h"]["sd"]) * 1.12 + 0.012,
                f"splitting: variance x{1 / r['variance_ratio']:.1f} higher\nmean {r['split_crash_per_h']['mean']:.3f} vs {r['plain_crash_per_h']['mean']:.3f}",
                ha="center", color=INK, fontsize=8)
    if rows:
        t = rows[0]["tilt_ess"]
        ax.text(0.02, 0.97, f"raising human risk x{t['factor']:.0f} and reweighting: log-weight sd {t['log_weight_sd']:.1f},\n"
                            f"effective sample {t['ess_fraction']:.0e} of the copies", transform=ax.transAxes, ha="left", va="top",
                color=INK2, fontsize=7.5)
        ax.set_ylim(0, max(max(r["plain_crash_per_h"]["sd"], r["split_crash_per_h"]["sd"]) for r in rows) * 1.45 + 0.02)
    ax.set_xticks([])
    ax.set_ylabel("standard deviation of the crash-rate estimate (per AV-hour)", color=INK2, fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.9), frameon=False, labelcolor=INK2)
    fig.tight_layout(pad=1.8)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


def table():
    """Markdown: crash rate / speed / hard brakes per variant and checkpoint, mean over seeds (+ the seeds' range)."""
    res = json.load(open(f"{FK.OUT}/eval.json"))
    ck = sorted({int(k.split("|")[0].rsplit("_", 1)[1]) for k in res if "_s" in k.split("|")[0] and k.endswith("|oakland")})
    lines = ["| variant | " + " | ".join(f"{c} updates" for c in ck) + " |", "|---|" + "---|" * len(ck)]
    for name in COL:
        cv = curve(res, name)
        cells = []
        for c in ck:
            r = cv.get(c)
            cells.append(f"{np.mean([x[0] for x in r]):.3f} ({min(x[0] for x in r):.3f}-{max(x[0] for x in r):.3f})" if r else "")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    print()
    lines = ["| variant | crashes /AV-h | AV km/h | hard brakes /AV-h (100 updates) |", "|---|---|---|---|"]
    for name in COL:
        r = curve(res, name).get(100)
        if r:
            lines.append(f"| {name} | {np.mean([x[0] for x in r]):.3f} | {np.mean([x[3] for x in r]):.1f} | {np.mean([x[4] for x in r]):.1f} |")
    print("\n".join(lines))


if __name__ == "__main__":
    import sys

    table() if len(sys.argv) > 1 and sys.argv[1] == "table" else main()
