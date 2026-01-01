"""Figure and GIF for avlab.frictionaware (reads results/frictionaware/frictionaware.json)."""
import numpy as np

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
PAL = {"blind": "#e8684a", "headway": "#4c9be8", "slow": "#3fb68b"}
LABEL = {"blind": "blind (T 1.5 s)", "headway": "headway (T 1.5 s / mu)", "slow": "slow (T / mu, v0 x sqrt mu)"}


def _style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(color=GRID, lw=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def plot(res, path="docs/img/frictionaware.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from avlab.whocrashes import rate_ratio
    S = res["summary"]["cells"]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), facecolor=BG)
    mus = sorted({c["mu"] for c in res["cells"].values()}, reverse=True)
    w = 0.26
    for j, pol in enumerate(("blind", "headway", "slow")):
        xs, ys, lo, hi, sp = [], [], [], [], []
        for i, mu in enumerate(mus):
            tag = f"{pol}@{mu:g}"
            if tag not in S:
                continue
            c = res["cells"][tag]
            k, h = float(np.sum(c["av_crashes"])), float(np.sum(c["av_hours"]))
            from scipy import stats
            l = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
            u = stats.chi2.ppf(0.975, 2 * k + 2) / 2
            xs.append(i + (j - 1) * w); ys.append(1000 * k / h); lo.append(1000 * l / h); hi.append(1000 * u / h)
            sp.append(S[tag]["av_speed_kmh"])
        ys, lo, hi = map(np.array, (ys, lo, hi))
        axs[0].bar(xs, ys, w * 0.92, color=PAL[pol], label=LABEL[pol])
        axs[0].errorbar(xs, ys, [ys - lo, hi - ys], fmt="none", ecolor=FG, lw=1, capsize=2)
        axs[1].bar(xs, sp, w * 0.92, color=PAL[pol])
        for x, v in zip(xs, sp):
            axs[1].text(x, v + 0.3, f"{v:.1f}", ha="center", color=FG, fontsize=8)
    for ax, yl in zip(axs, ("AV-involved crashes per 1000 AV-hours (95% CI)", "AV mean speed (km/h)")):
        _style(ax); ax.set_xticks(range(len(mus))); ax.set_xticklabels([f"friction {m:g}" for m in mus], color=FG)
        ax.set_ylabel(yl, color=FG)
    axs[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="upper left")
    T = res["summary"]["tests"]
    t4 = T.get("headway@0.4"); s4 = T.get("slow@0.4")
    sub = ""
    if t4 and s4:
        sub = (f"friction 0.4, rate ratio vs blind: headway {t4['av_rr']['rr']:.2f} ({t4['av_rr']['lo']:.2f}-{t4['av_rr']['hi']:.2f}), "
               f"slow {s4['av_rr']['rr']:.2f} ({s4['av_rr']['lo']:.2f}-{s4['av_rr']['hi']:.2f})")
    fig.suptitle(f"An AV that knows the road's friction, half-AV Oakland ({res['copies']} copies per bar, {res['seconds']:.0f} s)\n{sub}",
                 color=FG, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def gif(res, path="docs/img/frictionaware.gif"):
    """Cumulative AV-involved crashes per 1000 AV-hours as the 480 s unfold, one panel per slick friction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    mus = [m for m in sorted({c["mu"] for c in res["cells"].values()}, reverse=True) if m < 1.0]
    n = len(next(iter(res["cells"].values()))["series_av_crashes"])
    t = res["chunk_s"] * np.arange(1, n + 1)
    fig, axs = plt.subplots(1, len(mus), figsize=(8.4, 3.8), facecolor=BG, dpi=80)
    ymax = 0
    for c in res["cells"].values():
        r = 1000 * np.array(c["series_av_crashes"]) / np.maximum(np.array(c["series_av_hours"]), 1e-9)
        ymax = max(ymax, r[n // 4:].max())
    lines = {}
    for ax, mu in zip(np.atleast_1d(axs), mus):
        _style(ax); ax.set_xlim(0, t[-1] * 1.22); ax.set_ylim(0, ymax * 1.15)
        ax.set_title(f"friction {mu:g}", color=FG, fontsize=10); ax.set_xlabel("seconds scored", color=FG)
        for pol in ("blind", "headway", "slow"):
            ln, = ax.plot([], [], color=PAL[pol], lw=2.2)
            tx = ax.text(0, 0, "", color=PAL[pol], fontsize=8, va="center")
            lines[(mu, pol)] = (ln, tx)
    np.atleast_1d(axs)[0].set_ylabel("AV crashes per 1000 AV-h", color=FG)
    fig.suptitle("Cumulative AV-involved crash rate: blind vs friction-aware AVs", color=FG, fontsize=10.5)
    fig.tight_layout()

    def draw(i):
        i = min(i, n - 1)  # hold the last frame
        for (mu, pol), (ln, tx) in lines.items():
            c = res["cells"][f"{pol}@{mu:g}"]
            r = 1000 * np.array(c["series_av_crashes"][:i + 1]) / np.maximum(np.array(c["series_av_hours"][:i + 1]), 1e-9)
            ln.set_data(t[:i + 1], r)
            tx.set_position((t[i] + 6, r[-1])); tx.set_text(pol)
        return []
    anim = FuncAnimation(fig, draw, frames=n + 15, blit=False)
    anim.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)
