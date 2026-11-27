"""GIF and figure for dwm.simagents (learned sim agents for a whole city).

    python -m dwm.simagents_plot          # docs/img/simagents_rollout.gif: one shared start, three futures
    python -m dwm.simagents_plot --bench  # docs/img/simagents.png from results/simagents/realism.json

GIF: one copy of Oakland is warmed up for 400 s with the calibrated human drivers. From that same state the city is run
20 s three times: with the real rule-based drivers, with every vehicle driven by the learned model trained by imitation
only, and by the model after one DAgger round (both sample tokens). A 200 m window; vehicles are coloured by speed.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG, FG, MUTED, GRID = "#0d1117", "#e6edf3", "#8b949e", "#30363d"
CELLS = {"gt2": ("real drivers (rerun)", "#8b949e"), "sample": ("learned, sampled", "#56B4E9"),
         "argmax": ("learned, most likely", "#CC79A7"), "idm": ("IDM rule", "#E69F00"),
         "dagger": ("learned + DAgger*", "#009E73")}
FNAME = {"speed": "speed", "accel": "acceleration", "jerk": "jerk", "gap": "gap to leader", "headway": "time headway"}


def rollouts(seconds=20.0, warm_s=400.0, seed=21, area="oakland"):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    from dwm import simagents as SA
    X.SHIELD = "off"
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    hum, agents = C.calibrated(p_av=0.0), C.calibrated(p_av=1.0)
    st = X.reset(jax.random.PRNGKey(seed), J, 1024)
    step_h = jax.jit(lambda s: X.step(s, J, hum, nh))
    n_w = int(warm_s / X.DT)
    st = jax.lax.fori_loop(0, n_w, lambda i, s: X.step(s, J, hum, nh), st)
    p = SA.load_params("results/simagents/model.npz")
    p2 = SA.load_params("results/simagents/model_dagger.npz")
    out = {}
    for name, pol in (("real drivers", None), ("learned (imitation only)", SA.policy(p, "sample")),
                      ("learned + 1 DAgger round", SA.policy(p2, "sample"))):
        s = st if pol is None else st._replace(av=st.alive)
        f = step_h if pol is None else jax.jit(lambda s, k, pol=pol: X.step(s, J, agents, nh, rl=(pol, k))[0])
        key = jax.random.PRNGKey(seed + 5)
        fr = []
        for i in range(int(seconds / X.DT)):
            s = f(s) if pol is None else f(s, jax.random.fold_in(key, i))
            fr.append(jax.tree.map(np.asarray, s))
        out[name] = fr
    return net, geo, info, out


def rollout_gif(path="docs/img/simagents_rollout.gif", fps=10, half=100.0):
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import PolyCollection
    from xnet import render as R
    from xnet import sim as X
    net, geo, info, out = rollouts()
    names = list(out)
    tracks = R.Tracks(net, geo)
    # window centre: the densest 360 m square at the start
    s0 = out[names[0]][0]
    x, y, _ = tracks.pose(s0.trk[s0.alive], s0.s[s0.alive], s0.length[s0.alive])
    best, cx, cy = -1, 0.0, 0.0
    for px, py in zip(x[::5], y[::5]):
        n = np.sum((np.abs(x - px) < half) & (np.abs(y - py) < half))
        if n > best:
            best, cx, cy = n, px, py
    fig = plt.figure(figsize=(9.6, 4.3), facecolor=BG)
    axs = [fig.add_axes([0.01 + i * 0.33, 0.04, 0.32, 0.74]) for i in range(3)]
    fig.text(0.01, 0.94, "Three futures from one shared start: real drivers vs two learned sim-agent models", color=FG,
             fontsize=12, weight="bold")
    fig.text(0.01, 0.86, "Oakland after 400 s of real traffic, a 200 m window; then 20 s with every vehicle driven by each model. Colour = speed "
             "(purple stopped, yellow 15 m/s). Red ring = crash.", color=MUTED, fontsize=8)
    maps = []
    for ax in axs:
        ax.set_facecolor(R.GROUND)
        R.Map(net, geo, info).draw(ax, detail=False, labels=False)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_aspect("equal"); ax.axis("off")
        maps.append([])
    cmap = plt.get_cmap("plasma")
    tit = [ax.set_title(n, color=FG, fontsize=9) for ax, n in zip(axs, names)]
    tt = fig.text(0.98, 0.94, "", color=FG, fontsize=10, ha="right", weight="bold")
    del tit

    def draw(k):
        for i, (ax, n) in enumerate(zip(axs, names)):
            for a in maps[i]:
                a.remove()
            maps[i] = []
            s = out[n][k]
            a = s.alive
            px, py, th = tracks.pose(s.trk[a], s.s[a], s.length[a])
            wd = np.where(s.kind[a] == X.BUS, 2.5, np.where(s.kind[a] == X.BIKE, 0.8, 1.9))
            polys = R.car_polys(px, py, th, s.length[a] - 0.3, wd * 1.6)
            col = cmap(0.3 + 0.7 * np.clip(s.v[a] / 15.0, 0, 1))
            pc = PolyCollection(polys, facecolors=col, edgecolors="none", zorder=7)
            ax.add_collection(pc)
            maps[i].append(pc)
            w = s.wreck_t[a] > 0
            if w.any():
                maps[i].append(ax.scatter(px[w], py[w], s=70, facecolors="none", edgecolors="#ff4d4d", lw=1.4, zorder=9))
        tt.set_text(f"t = {(k + 1) * 0.2:4.1f} s")
    n = len(out[names[0]])
    frames = list(range(0, n)) + [n - 1] * 10
    FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames)).save(path, writer=PillowWriter(fps=fps), dpi=72,
                                                                           savefig_kwargs={"facecolor": BG})
    return path


def bench_fig(src="results/simagents/realism.json", path="docs/img/simagents.png"):
    res = json.load(open(src))
    cells = [c for c in ("gt2", "idm", "sample", "argmax", "dagger") if c in res["cells"]]
    ceil = res["cells"]["gt2"]["features"]
    feats = list(FNAME)
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 4.3), facecolor=BG, gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    ax, bx, cx = axs
    ends = []
    for c in cells:
        lab, col = CELLS[c]
        vals = [res["cells"][c]["features"][f] / ceil[f] for f in feats]
        ax.plot(range(len(feats)), vals, "o-", color=col, lw=1.6, ms=5)
        ends.append([vals[-1], lab, col])
    ends.sort(key=lambda e: -e[0])
    y = 10.0
    for e in ends:  # spread the direct labels so they do not overlap
        y = min(e[0], y - 0.045)
        ax.text(len(feats) - 1 + 0.15, y, e[1], color=e[2], fontsize=7.5, va="center")
    ax.set_xticks(range(len(feats))); ax.set_xticklabels([FNAME[f].replace(" ", "\n", 1) for f in feats], color=MUTED, fontsize=7.5)
    ax.set_xlim(-0.3, len(feats) + 1.3)
    ax.set_ylabel("likelihood of real data / real-driver rerun", color=MUTED, fontsize=8)
    ax.set_title("realism per feature", color=MUTED, fontsize=9)
    for j, c in enumerate(cells):
        lab, col = CELLS[c]
        pc = np.asarray(res["cells"][c]["per_copy_meta"])
        m = res["cells"][c]["meta"]
        bx.plot(pc, np.full(len(pc), j) + np.linspace(-0.15, 0.15, len(pc)), "o", color=col, ms=3, alpha=0.7)
        bx.plot([m, m], [j - 0.3, j + 0.3], color=col, lw=2.5)
        bx.text(m, j + 0.45, f"{m:.3f}", color=col, fontsize=7.5, ha="center")
        k = res["cells"][c]["crashes"]
        r = max(res["cells"][c]["crashes_per_1000h"], 0.5)
        cx.barh(j, r, color=col, height=0.55)
        cx.text(r * 1.15, j, f"{k:.0f}", color=col, fontsize=7.5, va="center")
    g = res["gt"]["crashes_per_1000h"]
    cx.axvline(g, color=FG, lw=0.8, ls="--")
    cx.set_xscale("log"); cx.set_xlim(0.5, 6e4)
    for z in (bx, cx):
        z.set_yticks(range(len(cells))); z.invert_yaxis()
    bx.set_yticklabels([CELLS[c][0] for c in cells], color=MUTED, fontsize=7.5)
    cx.set_yticklabels([])
    bx.set_title("realism meta-metric (dots = copies)", color=MUTED, fontsize=9)
    cx.set_title("crashes per 1000 veh-h\n(numbers = crashes; dashed = held-out real)", color=MUTED, fontsize=8.5)
    for z in axs:
        z.set_facecolor(BG); z.tick_params(colors=MUTED, labelsize=7.5)
        for sp in z.spines.values():
            sp.set_color(GRID)
    t = res["tests"].get("H1", {})
    fig.text(0.01, 0.955, "How real does a city driven by learned sim agents look?", color=FG, fontsize=11.5, weight="bold")
    fig.text(0.01, 0.905, f"Oakland, 8 copies x 600 s per cell, closed loop, every vehicle driven by the model. Pre-registered H1 "
             f"(sampled > IDM) failed: {t['diff']:+.4f} (95 % {t['lo']:+.4f} to {t['hi']:+.4f}). * DAgger added after the "
             "pre-registered runs.", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.8, bottom=0.14, wspace=0.42)
    fig.savefig(path, dpi=110, facecolor=BG)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_fig()
    else:
        print(rollout_gif())
