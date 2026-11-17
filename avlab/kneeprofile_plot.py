"""Figure and GIF for avlab.kneeprofile: pooled crash rates with three fits, and the profile deviance of the hinge position."""
import numpy as np

from avlab import knee as K

BG, FG, GRID_C = "#0d1117", "#e6edf3", "#30363d"
PAL = {"data": "#e6edf3", "power": "#8b949e", "quad": "#f2a541", "hinge": "#4c9be8", "boot": "#3fb68b"}


def _data(load=None):
    from avlab.knee4 import load_all4
    from avlab.kneeprofile import aggregate
    yc, tc, muc, runc, _ = (load or load_all4)()
    return aggregate(yc, tc, muc, runc)


def _curve(y, t, mu, run, X_of, xs):
    """Hours-weighted (over runs) model rate per 1000 vehicle-h on a friction grid."""
    b = K.fit(y, t, X_of(mu, run))["beta"]
    runs = np.unique(run)
    w = np.array([t[run == r].sum() for r in runs]); w = w / w.sum()
    out = np.zeros(len(xs))
    for r, wr in zip(runs, w):
        X = X_of(np.r_[np.full(len(runs), 0.7), xs], np.r_[runs, np.full(len(xs), r)])  # prepend all runs so dummies line up
        out += wr * np.exp(X[len(runs):] @ b)
    return 1000 * out


def _style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=FG, which="both"); ax.grid(color=GRID_C, lw=0.6, which="both")
    for sp in ax.spines.values():
        sp.set_color(GRID_C)


def _rates_axes(ax, y, t, mu):
    import matplotlib.ticker
    m = np.unique(mu)
    r = np.array([1000 * y[mu == v].sum() / t[mu == v].sum() for v in m])
    n = np.array([y[mu == v].sum() for v in m]); h = np.array([t[mu == v].sum() for v in m])
    lo = 1000 * np.maximum(n - 1.96 * np.sqrt(n), 0.5) / h; hi = 1000 * (n + 1.96 * np.sqrt(n)) / h
    ax.errorbar(m, r, yerr=[r - lo, hi - r], fmt="o", color=PAL["data"], ms=4, lw=1, capsize=2, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]); ax.set_xticklabels(["0.3", "0.4", "0.5", "0.6", "0.7", "0.85", "1.0"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.27, 1.1); ax.set_ylim(4, 300)
    ax.set_xlabel("road friction (log)", color=FG); ax.set_ylabel("crashes per 1000 vehicle-h (log)", color=FG)
    _style(ax)


def plot(out, path="docs/img/kneeprofile.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from avlab.kneeprofile import design_quad
    y, t, mu, run = _data()
    xs = np.exp(np.linspace(np.log(0.28), np.log(1.05), 200))
    best = out["best_hinge"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=BG, gridspec_kw={"width_ratios": [1.1, 1]})
    _rates_axes(a1, y, t, mu)
    for key, X_of, lab in (("power", K.design, "power law"), ("quad", design_quad, "smooth curve (quadratic in log)"),
                           ("hinge", lambda m, r: K.design(m, r, best), f"knee at {best:g}")):
        c = _curve(y, t, mu, run, X_of, xs)
        a1.plot(xs, c, color=PAL[key], lw=2)
        a1.text(0.97, {"power": 0.93, "quad": 0.87, "hinge": 0.81}[key], lab, color=PAL[key], fontsize=9, ha="right",
                transform=a1.transAxes)
    a1.set_title(f"Pooled cells: {out['copies']} copies, {out['crashes']:.0f} crashes, 4 streams", color=FG, fontsize=10, loc="left")
    g = np.array(out["grid"]); d = np.array(out["drops"])
    lo, hi = out["likelihood_set_3.84"]
    a2.axvspan(lo, hi, color=PAL["hinge"], alpha=0.12, lw=0)
    a2.plot(g, d, color=PAL["hinge"], lw=2.2)
    a2.axhline(out["quad_drop_vs_power"], color=PAL["quad"], lw=1.4, ls="--")
    a2.text(0.43, out["quad_drop_vs_power"] - 1.6, f"smooth curve: {out['quad_drop_vs_power']:.1f}", color=PAL["quad"], fontsize=8.5)
    a2.text(lo + 0.005, 1.0, f"within 3.84 of best:\n{lo:g}-{hi:g}", color=PAL["hinge"], fontsize=8.5)
    hist = out["boot_hinge"]["hist"]
    hx = np.array([float(k) for k in hist]); hc = np.array(list(hist.values()), float)
    ax2 = a2.twinx()
    ax2.bar(hx, hc, width=0.017, color=PAL["boot"], alpha=0.55)
    ax2.set_ylim(0, hc.max() * 3.2); ax2.set_yticks([]); ax2.spines[:].set_color(GRID_C)
    bq = out["boot_hinge"]
    ax2.text(0.955, hc.max() * 1.05, f"bootstrap best hinge\n95 % {bq['q025']:g}-{bq['q975']:g}", color=PAL["boot"], fontsize=8.5, ha="right")
    a2.set_xlabel("hinge position (friction)", color=FG); a2.set_ylabel("deviance drop over the power law", color=FG)
    a2.set_xlim(0.42, 0.96); a2.set_ylim(0, max(d) * 1.25)
    a2.set_title(f"Knee vs smooth curve: T = {out['T_hinge_minus_quad']:.1f}, bootstrap p {out['p_knee_vs_curve']:.3f}",
                 color=FG, fontsize=10, loc="left")
    _style(a2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def gif(out, path="docs/img/kneeprofile.gif", load=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    y, t, mu, run = _data(load)
    xs = np.exp(np.linspace(np.log(0.28), np.log(1.05), 120))
    g = np.array(out["grid"]); d = np.array(out["drops"])
    frames = list(range(0, len(g), 2)) + [int(np.argmax(d))] * 10
    curves = {i: _curve(y, t, mu, run, lambda m, r, h=g[i]: K.design(m, r, h), xs) for i in set(frames)}
    power = _curve(y, t, mu, run, K.design, xs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.8), facecolor=BG, dpi=72)
    _rates_axes(a1, y, t, mu)
    a1.plot(xs, power, color=PAL["power"], lw=1.4, ls="--")
    a1.text(0.29, 5, "grey: power law", color=PAL["power"], fontsize=8)
    a1.text(0.29, 6.3, "blue: power law + knee", color=PAL["hinge"], fontsize=8)
    (ln,) = a1.plot([], [], color=PAL["hinge"], lw=2.2)
    vl = a1.axvline(0.7, color=PAL["hinge"], lw=1, ls=":")
    _style(a2)
    a2.set_xlim(0.42, 0.96); a2.set_ylim(0, d.max() * 1.2)
    a2.set_xlabel("hinge position (friction)", color=FG); a2.set_ylabel("deviance drop", color=FG)
    a2.axhline(out["quad_drop_vs_power"], color=PAL["quad"], lw=1.2, ls="--")
    a2.text(0.43, out["quad_drop_vs_power"] + 0.6, "smooth curve", color=PAL["quad"], fontsize=8)
    (pl,) = a2.plot([], [], color=PAL["hinge"], lw=2)
    (pt,) = a2.plot([], [], "o", color=PAL["hinge"])
    ttl = fig.suptitle("", color=FG, fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def draw(k):
        i = frames[k]
        ln.set_data(xs, curves[i]); vl.set_xdata([g[i], g[i]])
        j = max(frames[:k + 1])  # everything traced so far stays drawn
        pl.set_data(g[:j + 1], d[:j + 1]); pt.set_data([g[i]], [d[i]])
        ttl.set_text(f"Sliding the friction knee: hinge {g[i]:.2f}, deviance drop {d[i]:.1f} (best {g[int(np.argmax(d))]:.2f})")
        return ln, vl, pl, pt, ttl

    FuncAnimation(fig, draw, frames=len(frames), blit=False).save(path, writer=PillowWriter(fps=10))
    plt.close(fig)
