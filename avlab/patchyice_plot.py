"""Figure and GIF for avlab.patchyice.

    python -m avlab.patchyice_plot --gif   # docs/img/patchyice.gif: U0.7 and P side by side, same random stream
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG, FG, MUTED, GRID = "#0d1117", "#e6edf3", "#8b949e", "#30363d"
# Okabe-Ito (colour-blind safe)
COL = {"U1.0": "#E69F00", "U0.7": "#009E73", "U0.4": "#56B4E9", "P": "#CC79A7"}
ICE, DRY, CRASH = "#56B4E9", "#3a414a", "#D55E00"
LABEL = {"U1.0": "dry everywhere\n(friction 1.0)", "U0.7": "friction 0.7\neverywhere", "U0.4": "ice everywhere\n(friction 0.4)",
         "P": "ice on half the streets\n(0.4 / 1.0, mean 0.7)"}


def poisson_ci(k, t):
    from scipy import stats
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * k + 2) / 2
    return 1000 * lo / t, 1000 * hi / t


def plot(out, path="docs/img/patchyice.png"):
    cells = ["U1.0", "U0.7", "P", "U0.4"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=BG)
    ax.set_facecolor(BG)
    for i, c in enumerate(cells):
        k, t = out["cells"][c]["crashes"], out["cells"][c]["hours"]
        r = 1000 * k / t
        lo, hi = poisson_ci(k, t)
        ax.plot([i, i], [lo, hi], color=COL[c], lw=3, solid_capstyle="round")
        ax.plot(i, r, "o", color=COL[c], ms=10)
        ax.text(i + 0.12, r, f"{r:.1f}\n({int(k)} crashes)", color=COL[c], va="center", fontsize=9)
    mx = out["S1_P_vs_mixture"]
    e = 1000 * mx["expected"] / out["cells"]["P"]["hours"]
    ax.plot(2, e, marker="D", mfc="none", mec=FG, ms=10, mew=1.5)
    ax.text(1.88, e, "mixture of the\nuniform cells", color=MUTED, fontsize=8, ha="right", va="center")
    h1 = out["H1_P_vs_U0.7"]
    ax.set_title(f"Patchy ice vs the same mean friction: rate ratio {h1['rr']:.2f} (95 % {h1['lo']:.2f} to {h1['hi']:.2f}), p {h1['p']:.1g}",
                 color=FG, fontsize=11, loc="left")
    ax.set_xticks(range(len(cells)), [LABEL[c] for c in cells], color=FG, fontsize=9)
    ax.set_xlim(-0.4, len(cells) - 0.2)
    ax.set_ylim(0, None)
    ax.set_ylabel("crashes per 1000 vehicle-hours (95 %)", color=MUTED)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.text(0.01, 0.01, f"humans-only Oakland, 40 copies x 600 s per cell, random stream {out['run']['seed']}; "
             f"P: ice on {out['run']['patched_links']} of {out['run']['links']} links ({100 * out['run']['patched_length_share']:.0f} % of length)",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    print("wrote", path)


def gif(path="docs/img/patchyice.gif", seconds=480.0, every_s=6.0, warm_s=60.0, seed=9, fps=10):
    import jax
    import jax.numpy as jnp
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from avlab.patchyice import ICE as MU_ICE, MASK_SEED, SHARE, cell_behaviour
    from xnet import network as NW
    from xnet import sim as X
    from xnet.render import Tracks
    net, geo, info = NW.build("oakland")
    mask = X.patch_links(net, SHARE, seed=MASK_SEED)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    J["link_patch"] = jnp.asarray(mask)
    behs = [cell_behaviour("U0.7"), cell_behaviour("P")]
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    key = jax.random.PRNGKey(seed)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jnp.stack([key, key]), J, 1024)
    n = int(round(every_s / X.DT))

    from functools import partial

    @partial(jax.jit, static_argnums=1)
    def adv(st, k):
        return jax.lax.scan(lambda s, _: (jax.vmap(lambda si, b: X.step(si, J, b, nh))(s, B), None), st, None, length=k)[0]

    st = adv(st, int(round(warm_s / X.DT)))
    m0 = np.asarray(st.m)
    tracks = Tracks(net, geo)
    ci = [X.METRICS.index(c) for c in X.CRASH_NAMES]
    frames = []
    for _ in range(int(seconds / every_s)):
        st = adv(st, n)
        alive, trk, s, ln = (np.asarray(a) for a in (st.alive, st.trk, st.s, st.length))
        wreck = np.asarray(st.wreck_t) > 0
        f = []
        for e in range(2):
            a = alive[e]
            x, y, _ = tracks.pose(trk[e][a], s[e][a], ln[e][a])
            f.append((x, y, wreck[e][a], float(np.asarray(st.m)[e, ci].sum() - m0[e, ci].sum())))
        frames.append((float(st.t[0]) - warm_s, f))
    segs = [np.asarray(p) for p in geo.link_pts]
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 5.0), facecolor=BG)
    fig.subplots_adjust(0.01, 0.07, 0.99, 0.86, wspace=0.02)
    titles = ["friction 0.7 on every street", "ice (0.4) on the blue streets, dry (1.0) elsewhere"]
    huds, dots, wrecks, marks = [], [], [], [[], []]
    for e, ax in enumerate(axs):
        ax.set_facecolor(BG)
        cols = [ICE if (e == 1 and mask[i]) else DRY for i in range(len(segs))]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.6 if e else 1.2))
        allp = np.concatenate(segs)
        ax.set_xlim(allp[:, 0].min(), allp[:, 0].max()); ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(titles[e], color=COL["U0.7"] if e == 0 else COL["P"], fontsize=10)
        dots.append(ax.scatter([], [], s=4, c=FG, lw=0))
        wrecks.append(ax.scatter([], [], s=140, marker="X", c=CRASH, edgecolors=BG, lw=0.8, zorder=5))
        huds.append(ax.text(0.02, 0.02, "", transform=ax.transAxes, color=FG, fontsize=9))
    fig.text(0.01, 0.95, "Patchy ice: same mean friction, one random stream", color=FG, fontsize=12, weight="bold")
    fig.text(0.01, 0.905, "white: vehicles   orange x: a wrecked vehicle (crash), kept on the map   blue: icy street (right only)",
             color=MUTED, fontsize=8)
    fig.text(0.01, 0.015, "humans-only Oakland, calibrated drivers; both panels use the same random stream; one copy each (illustration, "
             "not the test)", color=MUTED, fontsize=7)

    def draw(k):
        t, f = frames[k]
        for e in range(2):
            x, y, w, nc = f[e]
            dots[e].set_offsets(np.c_[x, y])
            marks[e] += [p for p in zip(x[w], y[w]) if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 400 for q in marks[e])]
            wrecks[e].set_offsets(np.array(marks[e]) if marks[e] else np.empty((0, 2)))
            huds[e].set_text(f"t = {t / 60:4.1f} min   crashes {int(nc)}")
        return []

    FuncAnimation(fig, draw, frames=len(frames)).save(path, writer=PillowWriter(fps=fps), dpi=72, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    shrink(path, fps)
    print("wrote", path)


def shrink(path, fps=10, colors=96):
    """Re-save a GIF with one shared 96-colour palette (keeps it under ~1 MB)."""
    from PIL import Image, ImageSequence
    fr = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(path))]
    pal = fr[-1].quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in fr]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()
    if a.gif:
        gif()
    else:
        plot(json.load(open("results/patchyice/patchyice.json")))
