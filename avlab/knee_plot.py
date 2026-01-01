"""Figure and GIF for avlab.knee (dark theme, Okabe-Ito palette)."""
import numpy as np

BG, FG, GR, MUTED = "#0d1117", "#e6edf3", "#30363d", "#8b949e"
BLUE, ORANGE, GREEN, PINK = "#56b4e9", "#e69f00", "#009e73", "#cc79a7"


def _ax(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=FG, labelsize=8); [s.set_color(GR) for s in ax.spines.values()]


def _cells(ax, cells, col, label, dx=1.0):
    from scipy import stats
    for k, c in cells.items():
        x = float(k) * dx
        lo = stats.gamma.ppf(0.025, c["crashes"]) / c["hours"] * 1000 if c["crashes"] > 0 else 0.3
        hi = stats.gamma.ppf(0.975, c["crashes"] + 1) / c["hours"] * 1000
        ax.plot([x, x], [max(lo, 0.3), hi], color=col, lw=2, alpha=0.8)
        ax.plot(x, max(c["rate"], 0.3), "o", color=col, ms=6)
    ax.plot([], [], "o", color=col, label=label)


def _hinge_curve(ax, f, xs, x_ref, rate_ref, side, col, label):
    """Draw the fixed-hinge fit's shape, anchored at a reference rate (the run effects are nuisance)."""
    lh = np.log(f["hinge"]); lx = np.log(xs)
    b = f["slope_above"] if side == "below" else f["slope_above"] - f["extra"]
    extra = f["extra"] * (np.maximum(lh - lx, 0) if side == "below" else np.maximum(lx - lh, 0))
    g = b * lx + extra
    g0 = b * np.log(x_ref) + f["extra"] * (max(lh - np.log(x_ref), 0) if side == "below" else max(np.log(x_ref) - lh, 0))
    ax.plot(xs, rate_ref * np.exp(g - g0), color=col, lw=1.8, ls="--", label=label)


def plot(out, path="docs/img/knee.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=BG)
    for ax in axs:
        _ax(ax)
    # (a) friction: pooled old runs vs the new replication
    ax = axs[0]
    fp = out["friction_pool"]
    _cells(ax, fp["cells"], BLUE, f"pooled 11-06 + 11-08 ({fp['copies']} copies)")
    if "replication" in out:
        rp = out["replication"]
        _cells(ax, {k: v for k, v in rp["cells"].items()}, ORANGE, f"NEW run, seed 1 ({rp['copies']} copies)", dx=1.012)
        xs = np.linspace(0.48, 1.0, 80)
        ref = rp["cells"]["1"]["rate"] if "1" in rp["cells"] else list(rp["cells"].values())[-1]["rate"]
        _hinge_curve(ax, rp["fixed_0.7"], xs, 1.0, ref, "below", ORANGE, "hinge fixed at 0.7, fit to NEW run")
    ax.axvline(0.7, color=MUTED, lw=0.8, ls=":")
    ax.text(0.7, 0.03, " pre-registered\n knee 0.7", color=MUTED, fontsize=7.5, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]); ax.set_xticklabels(["0.3", "0.4", "0.5", "0.6", "0.7", "0.85", "1.0"])
    ax.minorticks_off()
    ax.set_xlabel("road friction (slicker to the right)", color=FG, fontsize=9); ax.set_ylabel("crashes per 1000 vehicle-h (log)", color=FG, fontsize=9)
    ax.set_title("(a) does the knee at 0.7 replicate?", color=FG, fontsize=10, loc="left")
    ax.legend(facecolor=BG, edgecolor=GR, labelcolor=FG, fontsize=7, loc="upper left")
    # (b) speed limit pool
    ax = axs[1]
    lp = out["limit_pool"]
    _cells(ax, lp["cells"], GREEN, f"pooled speed-limit runs ({lp['copies']} copies)")
    xs = np.linspace(0.7, 1.3, 60)
    ref = lp["cells"]["1"]["rate"]
    ax.plot(xs, ref * xs ** lp["power_b"], color=GREEN, lw=1.6, ls="--", label=f"power law, b = {lp['power_b']:.1f}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_xticks([0.7, 0.85, 1.0, 1.15, 1.3]); ax.set_xticklabels(["x0.7", "x0.85", "x1.0", "x1.15", "x1.3"])
    ax.set_xlabel("speed-limit scale", color=FG, fontsize=9)
    ax.set_title("(b) a knee in the speed-limit curve?", color=FG, fontsize=10, loc="left")
    ax.legend(facecolor=BG, edgecolor=GR, labelcolor=FG, fontsize=7, loc="upper left")
    # (c) search-corrected p-values
    ax = axs[2]
    rows = [("friction pool\n(hinge searched)", fp["sup"]["p_boot"], BLUE), ("speed-limit pool\n(knee searched)", lp["sup"]["p_boot"], GREEN)]
    if "replication" in out:
        rows.append(("NEW run\n(hinge fixed 0.7)", out["replication"]["fixed_0.7"]["p_one_sided"], ORANGE))
    for i, (lab, p, col) in enumerate(rows):
        ph = out["holm"][["friction_pool", "limit_pool", "replication"][i]]
        ax.barh(i, -np.log10(max(p, 1e-6)), color=col, height=0.55)
        ax.text(-np.log10(max(p, 1e-6)) + 0.05, i, f"p = {p:.3g}\nHolm {ph:.3g}", color=FG, fontsize=8, va="center")
    ax.axvline(-np.log10(0.05), color=MUTED, ls="--", lw=1)
    ax.text(-np.log10(0.05), len(rows) - 0.45, " p = 0.05", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], color=FG, fontsize=8)
    ax.set_xlim(0, max(3.2, max(-np.log10(max(r[1], 1e-6)) for r in rows) + 1.3)); ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel("-log10 p (longer bar = stronger evidence of a knee)", color=FG, fontsize=9)
    ax.set_title("(c) evidence for a knee", color=FG, fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


def gif(out, path="docs/img/knee.gif", frames=12):
    """Evidence accumulating: the fixed-hinge test on the NEW run as copies are added (copies 1..k of every friction)."""
    if "replication" not in out:
        return
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from avlab.knee import load_replication, fixed_hinge, cells
    y, t, mu, run = load_replication()
    E = int(len(y) / len(np.unique(mu)))
    idx = np.concatenate([np.arange(E)] * len(np.unique(mu)))  # copy index inside each friction (copies are stored friction-major)
    ks = np.unique(np.linspace(4, E, frames).astype(int))
    hist = []
    imgs = []
    for k in ks:
        s = idx < k
        f = fixed_hinge(y[s], t[s], mu[s], run[s], 0.7)
        hist.append((k, f["p_one_sided"], f["extra"], f["extra_se"]))
        fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.6), facecolor=BG)
        for ax in axs:
            _ax(ax)
        ax = axs[0]
        _cells(ax, cells(y[s], t[s], mu[s]), ORANGE, f"NEW run, {k} copies per friction")
        xs = np.linspace(0.48, 1.0, 60)
        c1 = cells(y[s], t[s], mu[s])
        _hinge_curve(ax, f, xs, 1.0, max(c1["1"]["rate"], 0.5), "below", ORANGE, "hinge fixed at 0.7")
        ax.axvline(0.7, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis(); ax.minorticks_off()
        ax.set_xticks([0.5, 0.6, 0.7, 0.85, 1.0]); ax.set_xticklabels(["0.5", "0.6", "0.7", "0.85", "1.0"])
        ax.set_ylim(0.3, 300)
        ax.set_xlabel("road friction", color=FG, fontsize=9); ax.set_ylabel("crashes / 1000 veh-h", color=FG, fontsize=9)
        ax.legend(facecolor=BG, edgecolor=GR, labelcolor=FG, fontsize=7, loc="upper left")
        ax = axs[1]
        h = np.array(hist)
        ax.plot(h[:, 0], h[:, 1], "-o", color=ORANGE, ms=4)
        ax.axhline(0.05, color=MUTED, ls="--", lw=1); ax.text(E, 0.05, "0.05 ", color=MUTED, fontsize=8, ha="right", va="bottom")
        ax.set_yscale("log"); ax.set_ylim(1e-5, 1.2); ax.set_xlim(0, E + 1)
        ax.set_xlabel("copies per friction", color=FG, fontsize=9); ax.set_ylabel("one-sided p, knee at 0.7", color=FG, fontsize=9)
        ax.text(0.03, 0.06, f"extra slope {f['extra']:.2f} (SE {f['extra_se']:.2f})", transform=ax.transAxes, color=FG, fontsize=8)
        fig.suptitle("Pre-registered replication of the friction knee, fresh random stream", color=FG, fontsize=10)
        fig.tight_layout()
        b = io.BytesIO(); fig.savefig(b, format="png", dpi=100, facecolor=BG); plt.close(fig)
        imgs.append(Image.open(b).convert("P", palette=Image.ADAPTIVE, colors=64))
    imgs += [imgs[-1]] * 8  # hold the last frame
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=400, loop=0, optimize=True)
