"""Figure and GIF for avlab.frictionmerge.

plot(res): AV-involved crashes per 1000 AV-hours by policy at friction 0.4, split into merges and all other types, and AV speed.
gif():     a counterfactual black box. A half-AV Oakland at friction 0.4 with blind AVs runs until an AV is in a crash; the city's
           state at the start of the 20 s chunk holding that crash is then forked and replayed twice, once with the blind AV and once with headway+gap (the same cars,
           signals and random numbers; only the AV policy differs from the fork on). The two zoomed replays are shown side by side.
"""
import os

import numpy as np

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
PAL = {"blind": "#e8684a", "headway": "#4c9be8", "gap": "#f2a541", "headway+gap": "#3fb68b"}
LABEL = {"blind": "blind", "headway": "headway\nT / mu", "gap": "gap\ncaution", "headway+gap": "headway\n+ gap"}


def _style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(color=GRID, lw=0.6, axis="y")
    for sp in ax.spines.values():
        sp.set_color(GRID)


def plot(res, path="docs/img/frictionmerge.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    S, T = res["summary"]["cells"], res["summary"]["tests"]
    pols = [p for p in ("blind", "headway", "gap", "headway+gap") if p in S]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8), facecolor=BG, gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    ymax = 0.0
    for i, p in enumerate(pols):
        c = S[p]; h = c["av_hours"]
        merge, other = 1000 * c["av_merges"] / h, 1000 * (c["av_crashes"] - c["av_merges"]) / h
        ax.bar(i, merge, 0.62, color=PAL[p])
        ax.bar(i, other, 0.62, bottom=merge, color=PAL[p], alpha=0.45, hatch="//", edgecolor=BG)
        k = c["av_crashes"]
        lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
        hi = stats.chi2.ppf(0.975, 2 * k + 2) / 2
        tot = 1000 * k / h
        ax.errorbar(i, tot, [[tot - 1000 * lo / h], [1000 * hi / h - tot]], fmt="none", ecolor=FG, lw=1, capsize=3)
        ax.text(i, 1000 * hi / h + 0.8, f"{int(k)} crashes\n({int(c['av_merges'])} merges)", color=FG, fontsize=8, ha="center", va="bottom")
        ymax = max(ymax, 1000 * hi / h)
        if p != "blind":
            rr = T[f"{p} vs blind"]["av_rr"]
            ax.text(i, -0.08, f"x{rr['rr']:.2f}\n({rr['lo']:.2f}-{rr['hi']:.2f})", transform=ax.get_xaxis_transform(),
                    color=PAL[p], fontsize=8, ha="center", va="top")
    ax.set_xticks(range(len(pols))); ax.set_xticklabels([LABEL[p] for p in pols], color=FG, fontsize=9)
    ax.tick_params(axis="x", pad=34)
    ax.set_ylabel("AV-involved crashes per 1000 AV-hours", color=FG)
    ax.text(0.98, 0.97, "solid: merges   hatched: other types\nwhiskers: 95% CI of the total\nbelow: rate ratio vs blind (95% CI)",
            transform=ax.transAxes, color=FG, fontsize=8, va="top", ha="right")
    ax.set_xlim(-0.5, len(pols) - 0.5); ax.set_ylim(0, ymax * 1.18)
    _style(ax)
    ax = axs[1]
    for i, p in enumerate(pols):
        v = S[p]["av_speed_kmh"]
        ax.bar(i, v, 0.62, color=PAL[p]); ax.text(i, v + 0.2, f"{v:.1f}", ha="center", color=FG, fontsize=9)
    ax.set_xticks(range(len(pols))); ax.set_xticklabels([LABEL[p] for p in pols], color=FG, fontsize=9)
    ax.set_ylabel("AV mean speed (km/h)", color=FG)
    _style(ax)
    fig.suptitle(f"Friction 0.4, half-AV Oakland: car-following headway vs gap acceptance "
                 f"({res['copies']} copies per bar, {res['seconds']:.0f} s, seed {res['seed']})", color=FG, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def _fork_pair(mu=0.4, seed=11, copies=8, warm=90.0, chunk=20.0, budget=600.0):
    """Run blind AVs until an AV crash in some chunk; replay that chunk from its start with blind and with headway+gap."""
    from avlab.frictionmerge import merge_idm, scales
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info = NW.build("oakland")
    beh = C.calibrated(friction=mu, p_av=0.5)
    pols = {}
    for p in ("blind", "headway+gap"):
        ts, u1 = scales(p, mu)
        pols[p] = (lambda obs, key, ts=ts, u1=u1: merge_idm(obs, ts, u1))
    A = X.Sim(net, n_envs=copies, beh=beh, seed=seed, reroute_s=0, policy=pols["blind"])
    Bs = X.Sim(net, n_envs=copies, beh=beh, seed=seed, reroute_s=0, policy=pols["headway+gap"])
    idx = [X.METRICS.index("av_crash")]
    A.run(warm)
    for _ in range(int(budget / chunk)):
        snap = A.st
        A.run(chunk)
        hit = np.nonzero(np.asarray(A.st.m)[:, idx].sum(1) > np.asarray(snap.m)[:, idx].sum(1))[0]
        for e in hit:
            after = A.st
            A.st = snap
            fa = A.run(chunk + 6.0, record=(int(e), 1))
            m = np.array([f.m[idx].sum() for f in fa])
            if not (np.diff(m) > 0).any():
                A.st = after
                continue
            Bs.st = snap
            fb = Bs.run(chunk + 6.0, record=(int(e), 1))
            mb = np.array([f.m[idx].sum() for f in fb])
            return net, geo, info, fa, fb, int(e), bool((np.diff(mb) > 0).any())
        A.st = after if len(hit) else A.st
    return None


def gif(res=None, path="docs/img/frictionmerge.gif", seed=11):
    from PIL import Image

    from avlab.blackbox import investigate
    from xnet import render as R
    got = _fork_pair(seed=seed)
    if got is None:
        print("no AV crash found for the GIF")
        return None
    net, geo, info, fa, fb, e, b_crash = got
    rep = investigate(net, geo, fa)
    k = rep["frame"]
    lo, hi = max(0, k - 50), k + 20
    tmp = []
    for name, fr, ttl in (("a", fa, f"blind AV: {rep['types'][0]} crash"),
                          ("b", fb, "same moment, headway + gap AV" + (" (also crashes)" if b_crash else " (no AV crash)"))):
        p = f"{os.path.dirname(path) or '.'}/_fm_{name}.gif"
        R.animate(net, geo, info, fr[lo:hi], p, zoom=(tuple(rep["centre_xy"]), 45.0), title=ttl, fps=10, dpi=60)
        tmp.append(p)
    ga, gb = Image.open(tmp[0]), Image.open(tmp[1])
    frames = []
    try:
        while True:
            a = ga.convert("RGB"); b = gb.convert("RGB")
            w, h = a.width + b.width, max(a.height, b.height)
            im = Image.new("RGB", (w, h), BG)
            im.paste(a, (0, 0)); im.paste(b, (a.width, 0))
            frames.append(im.quantize(colors=96))
            ga.seek(ga.tell() + 1); gb.seek(gb.tell() + 1)
    except EOFError:
        pass
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    for p in tmp:
        os.remove(p)
    meta = {"seed": seed, "copy": e, "types": rep["types"], "frame": k, "twin_av_crash": b_crash, "centre_xy": rep["centre_xy"]}
    print(meta, os.path.getsize(path) // 1024, "KB")
    return meta
