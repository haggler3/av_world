"""Figure for avlab.knee5: stream 5's new cells against what each four-stream model predicted, and the hinge profile 4 vs 5 streams."""
import numpy as np

from avlab.kneeprofile_plot import BG, FG, GRID_C, _style

PAL = {"power law": "#8b949e", "hinge 0.64": "#4c9be8", "hinge 0.7": "#3fb68b", "hinge 0.8": "#e86bb3", "obs": "#f2a541", "four": "#8b949e"}


def plot(out, path="docs/img/knee5.png", gif_path="docs/img/knee5.gif"):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=BG, gridspec_kw={"width_ratios": [1, 1.15]})
    o = out["stream5"]["rr_0.65_vs_0.75"]
    a1.axvspan(o["lo"], o["hi"], color=PAL["obs"], alpha=0.15, lw=0)
    a1.axvline(o["rr"], color=PAL["obs"], lw=2.2)
    a1.text(o["rr"] * 1.02, 4.55, f" stream 5 observed {o['rr']:.2f}\n ({o['lo']:.2f}-{o['hi']:.2f})", color=PAL["obs"], fontsize=9, va="top")
    for i, (m, r) in enumerate(out["predicted_rr_from_4_streams"].items()):
        rl = out["rel_likelihood"][m]
        a1.plot([r], [i], "o", color=PAL[m], ms=9)
        a1.text(r, i + 0.22, f"{m}: {r:.2f}  (rel. lik. {rl:.2f})", color=PAL[m], fontsize=9, ha="center")
    a1.axvline(1.0, color=FG, lw=0.8, ls=":", alpha=0.6)
    a1.set_yticks([]); a1.set_ylim(-0.6, 4.7); a1.set_xscale("log")
    a1.set_xticks([0.6, 0.8, 1, 1.25, 1.6, 2, 2.5]); a1.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.6", "2", "2.5"])
    a1.set_xlim(0.5, 3.2); a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_xlabel("crash-rate ratio, friction 0.65 / 0.75 (log)", color=FG)
    c = out["stream5"]["cells"]
    a1.set_title(f"Stream 5 (new): {c['0.65']['crashes']:.0f} vs {c['0.75']['crashes']:.0f} crashes; "
                 "dots = four-stream predictions", color=FG, fontsize=10, loc="left")
    _style(a1)
    f5 = out["five_streams"]
    f4 = json.load(open("results/knee/kneeprofile.json"))
    for f, col, lab, lw, dy in ((f4, PAL["four"], "4 streams (2026-11-16)", 1.6, -46), (f5, PAL["hinge 0.64"], "5 streams (today)", 2.4, 8)):
        g, d = np.array(f["grid"]), np.array(f["drops"])
        lo, hi = f["likelihood_set_3.84"]
        a2.plot(g, d, color=col, lw=lw)
        a2.axvspan(lo, hi, color=col, alpha=0.10, lw=0)
        k = int(np.argmax(d))
        a2.annotate(f"{lab}: best {g[k]:g}, set {lo:g}-{hi:g}", (g[k], d[k]), xytext=(0, dy), textcoords="offset points",
                    color=col, fontsize=9, ha="center")
    for x in (0.65, 0.75):
        a2.axvline(x, color=PAL["obs"], lw=1, ls=":")
    a2.text(0.655, 0.6, "new cells", color=PAL["obs"], fontsize=8.5)
    a2.set_xlim(0.42, 0.96); a2.set_ylim(0, max(max(f4["drops"]), max(f5["drops"])) * 1.3)
    a2.set_xlabel("hinge position (friction)", color=FG); a2.set_ylabel("deviance drop over the power law", color=FG)
    a2.set_title(f"Where is the knee? kink vs smooth curve p {f5['p_knee_vs_curve']:.3f}; "
                 f"bootstrap 95 % {f5['boot_hinge']['q025']:g}-{f5['boot_hinge']['q975']:g}", color=FG, fontsize=10, loc="left")
    _style(a2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    if gif_path:
        from avlab.knee5 import load_all5
        from avlab.kneeprofile_plot import gif
        gif(f5, gif_path, load=load_all5)
