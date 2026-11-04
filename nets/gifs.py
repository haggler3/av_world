"""GIFs for docs/NEURAL_NETS.md.    av-world nets gifs [forks blame occlusion mindreader]   -> docs/img/nets/*.gif"""
import os
import pickle
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
sys.path.insert(0, os.getcwd())

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from avrl import xppo as P  # noqa: E402
from nets import blame as BL  # noqa: E402
from nets import collect as C  # noqa: E402
from nets import common as CM  # noqa: E402
from nets import forks as FK  # noqa: E402
from nets import occlusion as OC  # noqa: E402
from xnet import render as R  # noqa: E402
from xnet import sim as X  # noqa: E402

OUT = "docs/img/nets"
BG, PANEL, INK, MUTED = "#0d1117", "#141a1f", "#e6edf3", "#8b949e"
BLUE, ORANGE, AQUA, RED, GRAY = "#3987e5", "#eb6834", "#1baf7a", "#e66767", "#8b949e"


def save(fig, draw, n, name, fps=6, dpi=72):
    FuncAnimation(fig, draw, frames=n).save(f"{OUT}/{name}", writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", name, flush=True)


def car(ax, x, y, h, L, color, ec=None, lw=0.0, z=5):
    c, s = np.cos(h), np.sin(h)
    pts = np.array([[-L / 2, -1.0], [L / 2, -1.0], [L / 2, 1.0], [-L / 2, 1.0]])
    p = np.stack([x + pts[:, 0] * c - pts[:, 1] * s, y + pts[:, 0] * s + pts[:, 1] * c], 1)
    return ax.add_patch(Polygon(p, closed=True, fc=color, ec=ec or color, lw=lw, zorder=z))


def panel_axes(fig, axs, mp, centre, r):
    for ax in np.ravel(axs):
        ax.set_facecolor(PANEL)
        mp.draw(ax, detail=True, labels=False)
        ax.set_xlim(centre[0] - r, centre[0] + r), ax.set_ylim(centre[1] - r, centre[1] + r)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)


def pos_feat(s, tr, pc):
    return {"trk": s.trk, "s": s.s, "length": s.length, "alive": s.alive, "av": s.av, "uid": s.uid, "wreck": s.wreck_t > 0, "v": s.v}


# ------------------------------------------------------------------ one moment, six futures
def gif_forks(seed=11):
    X.SHIELD = "off"
    w = C.World()
    mp, tracks = R.Map(w.net_np, w.geo, w.info), R.Tracks(w.net_np, w.geo)
    beh = FK.AD.world(5.0)._replace(crn=1.0)
    prior = P.load(FK.PRIOR)
    pol = FK.make_policy(prior)
    st = C.run(w, w.reset(8, seed), beh, 120.0)
    K, B = FK.K, FK.K + 1
    pick = None
    for attempt in range(30):
        N = st.v.shape[1]
        d = np.zeros((8 * B, N, 2), np.float32)
        for b in range(K):
            d[np.arange(8) * B + b] = FK.DELTAS[b]                    # here every AV of a branch takes the same action: easy to watch
        stf = C.fork(st, B)
        pc = {"d": jnp.asarray(d), "pert": jnp.full((8 * B,), FK.PERT, jnp.int32)}
        _, f = C.record(w, stf, beh, 4.0, 1, lambda s, tr, pc: {**pos_feat(s, tr, pc), "bad": tr["crash"] | tr["near"] | ((s.v > 2) & (tr["bar_h"] < FK.BAD_MARGIN))},
                        policy=pol, per_copy=pc)
        f = C.to_host(f, np.float32)
        bad = f["bad"].reshape(20, 8, B, N).astype(bool).any(0)          # [E, B, N]
        av = np.asarray(st.av) & np.asarray(st.alive)
        cand = np.argwhere(av & bad[:, 2] & ~bad[:, K] & ~bad[:, 1])
        if len(cand):
            e, i = cand[0]
            pick = (e, i, f)
            break
        st = jax.tree.map(lambda a: a.reshape((8, B) + a.shape[1:])[:, K], C.record(w, stf, beh, 4.0, 20, lambda s, tr, pc: {"v": s.v}, policy=pol, per_copy=pc)[0])
    e, i, f = pick
    T = f["trk"].shape[0]
    g = lambda k, b: {n: f[n][:, e * B + b] for n in f}
    x0, y0, _ = tracks.pose(g(0, K)["trk"][:1, i], g(0, K)["s"][:1, i], g(0, K)["length"][:1, i])
    fig, axs = plt.subplots(2, 3, figsize=(10.2, 7.4), facecolor=BG)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.86, wspace=0.03, hspace=0.1)
    panel_axes(fig, axs, mp, (x0[0], y0[0]), 45)
    order = [K, 1, 2, 3, 4, 0][:6]
    names = ["what the policy did"] + [FK.NAMES[k] for k in (1, 2, 3, 4)] + ["as is, other AVs vary"]
    names = ["what the policy did", "brake harder", "accelerate", "bigger gaps", "smaller gaps", "(unused)"]
    order = [K, 1, 2, 3, 4]
    axs.ravel()[5].set_visible(False)
    fig.text(0.02, 0.955, "One moment, five futures: the same dice, a different action", color=INK, fontsize=13, weight="bold")
    fig.text(0.02, 0.92, "Every AV in a branch takes that branch's action for 2 s (white = the AV we follow, blue = other AVs, grey = humans). "
             "Red frame: a bad outcome for our AV.", color=MUTED, fontsize=8.5)
    for ax, nm in zip(axs.ravel(), names):
        ax.set_title(nm, color=INK, fontsize=10, loc="left")
    arts = []

    def draw(t):
        nonlocal arts
        for a_ in arts:
            a_.remove()
        arts = []
        for ax, b in zip(axs.ravel(), order):
            gg = g(t, b)
            al = gg["alive"][t].astype(bool)
            idx = np.nonzero(al)[0]
            x, y, h = tracks.pose(gg["trk"][t, idx], gg["s"][t, idx], gg["length"][t, idx])
            for j, k in enumerate(idx):
                col = "#ffffff" if k == i else (BLUE if gg["av"][t, k] else ("#ffa198" if gg["wreck"][t, k] else "#6e7681"))
                arts.append(car(ax, x[j], y[j], h[j], gg["length"][t, k], col))
            badnow = f["bad"][:t + 1, e * B + b, i].astype(bool).any()
            for sp in ax.spines.values():
                sp.set_visible(bool(badnow)), sp.set_color(RED), sp.set_linewidth(3)
        return []

    save(fig, draw, T, "forks.gif", fps=5)
    X.SHIELD = "idm"


# ------------------------------------------------------------------ but-for: the crash, and the world without one driver's mistake
def gif_blame(seed=3):
    w = C.World()
    mp, tracks = R.Map(w.net_np, w.geo, w.info), R.Tracks(w.net_np, w.geo)
    beh = X.Behaviour(**BL.STRESS, p_av=0.25, crn=1.0)
    st = C.run(w, w.reset(8, seed), beh, 120.0)
    snaps = []
    for c in range(200):
        snaps = (snaps + [jax.tree.map(np.asarray, st)])[-BL.REWIND:]
        st, f = C.record(w, st, beh, BL.CHUNK, 1, BL.feat_detect)
        new = np.asarray(f["new"])
        if len(snaps) < BL.REWIND:
            continue
        for e in range(8):
            ts = np.unique(np.nonzero(new[:, e])[0])
            for t in ts:
                slots = np.nonzero(new[t, e])[0]
                if len(slots) < 2:
                    continue
                snap = BL.take(snaps[0], e)
                uid_c = snap.uid[slots[:2]] if False else np.asarray(jax.tree.map(np.asarray, st).uid)[e][slots[:2]]
                inv = [int(np.nonzero((snap.uid == u) & snap.alive)[0][0]) for u in uid_c if ((snap.uid == u) & snap.alive).any()]
                if len(inv) < 2:
                    continue
                culprits = [k for k in inv if snap.runner[k] or snap.nolook[k] or snap.aggr[k] or snap.dist_t[k] > 0]
                for k in culprits:
                    pair = [snap, BL.calm(snap, k)]
                    stb = jax.tree.map(lambda *a: jnp.asarray(np.stack([np.asarray(x) for x in a] * 2)), *pair)   # 4 copies (2 x 2 devices)
                    _, g = C.record(w, stb, beh, BL.WIN, 1, pos_feat)
                    g = C.to_host(g, np.float32)
                    hit = lambda b: (g["wreck"][:, b] & np.isin(g["uid"][:, b], uid_c)).any()
                    if hit(0) and not hit(1):
                        return render_blame(w, mp, tracks, g, uid_c, k, snap)
    print("no clear but-for crash found")


def render_blame(w, mp, tracks, g, uid_c, k, snap):
    T = g["trk"].shape[0]
    tc = int(np.argmax((g["wreck"][:, 0] & np.isin(g["uid"][:, 0], uid_c)).any(1)))
    al = g["alive"][tc, 0].astype(bool)
    sel = np.isin(g["uid"][tc, 0], uid_c) & al
    x, y, _ = tracks.pose(g["trk"][tc, 0][sel], g["s"][tc, 0][sel], g["length"][tc, 0][sel])
    ctr = (x.mean(), y.mean())
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 5.6), facecolor=BG)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.82, wspace=0.03)
    panel_axes(fig, axs, mp, ctr, 40)
    why = "ran the red" if snap.runner[k] else ("was distracted" if snap.dist_t[k] > 0 else ("did not look" if snap.nolook[k] else "accepted a tight gap"))
    fig.text(0.02, 0.94, "But-for blame: replay the crash without one driver's mistake", color=INK, fontsize=13, weight="bold")
    fig.text(0.02, 0.895, f"Left: what happened. Right: the same city and dice, but the driver outlined in orange (who {why}) drives carefully from 6-8 s before.",
             color=MUTED, fontsize=8.5)
    axs[0].set_title("what happened", color=INK, fontsize=10, loc="left")
    axs[1].set_title("without that driver's mistake", color=INK, fontsize=10, loc="left")
    uk = snap.uid[k]
    arts = []

    def draw(t):
        nonlocal arts
        for a_ in arts:
            a_.remove()
        arts = []
        for b, ax in enumerate(axs):
            al = g["alive"][t, b].astype(bool)
            idx = np.nonzero(al)[0]
            xx, yy, hh = tracks.pose(g["trk"][t, b, idx], g["s"][t, b, idx], g["length"][t, b, idx])
            for j, i in enumerate(idx):
                u = g["uid"][t, b, i]
                wr = g["wreck"][t, b, i] and u in uid_c
                col = RED if wr else (BLUE if g["av"][t, b, i] else "#6e7681")
                arts.append(car(ax, xx[j], yy[j], hh[j], g["length"][t, b, i], col, ec=ORANGE if u == uk else None, lw=2.2 if u == uk else 0))
        return []

    save(fig, draw, T, "blame.gif", fps=6)


# ------------------------------------------------------------------ the occlusion belief from one AV's seat
def gif_occlusion(seed=8, seconds=40.0):
    w = C.World()
    tracks = R.Tracks(w.net_np, w.geo)
    p, _ = CM.load_params("results/nets/occlusion_params.pkl")
    beh = X.Behaviour(crn=1.0, p_av=0.25)
    st = C.run(w, w.reset(4, seed), beh, 120.0)
    _, f = C.record(w, st, beh, seconds, 2, OC.feat)
    f = C.to_host(f, np.float32)
    e = 0
    av = np.nonzero(f["av"][0, e] & f["alive"][0, e])[0]
    uid = None
    # follow the AV that stays longest with the most hidden traffic
    best = (-1, None)
    for a in av:
        life = (f["alive"][:, e, a].astype(bool)).sum()
        if life > best[0]:
            best = (life, a)
    a = best[1]
    # follow the AV that has the most hidden traffic around it over its life
    def hidden_count(a_, t):
        al = np.nonzero(f["alive"][t, e])[0]
        if a_ not in al:
            return -1
        x, y, h = tracks.pose(f["trk"][t, e, al], f["s"][t, e, al], f["length"][t, e, al].astype(float))
        ai = int(np.nonzero(al == a_)[0][0])
        d = np.hypot(x - x[ai], y - y[ai])
        near = np.nonzero((d < 40) & (np.arange(len(al)) != ai))[0]
        return int((~OC.visibility(x[ai], y[ai], x[near], y[near], h[near], f["length"][t, e, al][near].astype(float))).sum()) if len(near) else 0
    scores = {int(a_): np.mean([max(hidden_count(a_, t), 0) for t in range(0, f["s"].shape[0], 10)]) * (f["alive"][:, e, a_].mean() > 0.8) for a_ in av}
    a = max(scores, key=scores.get)
    frames = []
    for t in range(f["s"].shape[0]):
        if not f["alive"][t, e, a]:
            break
        al = np.nonzero(f["alive"][t, e])[0]
        x, y, h = tracks.pose(f["trk"][t, e, al], f["s"][t, e, al], f["length"][t, e, al].astype(float))
        L = f["length"][t, e, al].astype(float)
        ai = int(np.nonzero(al == a)[0][0])
        c, s = np.cos(h[ai]), np.sin(h[ai])
        fx, fy = (x - x[ai]) * c + (y - y[ai]) * s, -(x - x[ai]) * s + (y - y[ai]) * c
        near = np.nonzero((np.abs(fx) < 40) & (np.abs(fy) < 40) & (np.arange(len(al)) != ai))[0]
        vis = OC.visibility(x[ai], y[ai], x[near], y[near], h[near], L[near])
        v_i = near[vis][:OC.KMAX]
        kind = f["kind"][t, e, al][v_i]
        dh = h[v_i] - h[ai]
        tok = np.zeros((OC.KMAX, 10), np.float32)
        tok[:len(v_i)] = np.stack([fx[v_i] / 40, fy[v_i] / 40, np.cos(dh), np.sin(dh), f["v"][t, e, al][v_i] / 15.0, L[v_i] / 12.0, kind == 0, kind == 1, kind == 2, kind == 3], 1)
        out = np.asarray(OC.apply(p, jnp.asarray(tok[None])))[0]
        frames.append({"fx": fx[near], "fy": fy[near], "dh": h[near] - h[ai], "L": L[near], "vis": vis, "grid": 1 / (1 + np.exp(-out[:-1])), "ahead": 1 / (1 + np.exp(-out[-1]))})
    fig, ax = plt.subplots(figsize=(6.4, 6.8), facecolor=BG)
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.8)
    ax.set_facecolor(PANEL)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("belief", [PANEL, "#1f4f8f", BLUE, "#9cc7ff"])
    fig.text(0.03, 0.955, "What the AV cannot see", color=INK, fontsize=13, weight="bold")
    fig.text(0.03, 0.925, "The AV's own frame, facing right. Orange: visible to it. Dashed: hidden (ground truth).\nBlue: the network's belief about hidden road users, from the visible ones only.", color=MUTED, fontsize=8.5, va="top")
    hud = fig.text(0.03, 0.83, "", color=INK, fontsize=9)
    arts = []

    def draw(k):
        nonlocal arts
        for a_ in arts:
            a_.remove()
        fr = frames[k]
        arts = [ax.imshow(fr["grid"].reshape(8, 8).T, origin="lower", extent=(-40, 40, -40, 40), cmap=cmap, vmin=0, vmax=0.8, zorder=1)]
        for j in range(len(fr["fx"])):
            if fr["vis"][j]:
                arts.append(car(ax, fr["fx"][j], fr["fy"][j], fr["dh"][j], fr["L"][j], ORANGE, z=4))
            else:
                p_ = car(ax, fr["fx"][j], fr["fy"][j], fr["dh"][j], fr["L"][j], "none", ec="#e6edf3", lw=1.3, z=4)
                p_.set_linestyle("--")
                arts.append(p_)
        arts.append(car(ax, 0, 0, 0, 4.6, "#ffffff", z=6))
        hud.set_text(f"t = {k * 0.4:4.1f} s   hidden now: {int((~fr['vis']).sum())}   P(someone hidden 25 m ahead) = {fr['ahead']:.2f}")
        return []

    ax.set_xlim(-40, 40), ax.set_ylim(-40, 40), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    save(fig, draw, len(frames), "occlusion.gif", fps=5, dpi=80)


# ------------------------------------------------------------------ the mind reader, live
def gif_mindreader(seed=21, seconds=70.0):
    from avlab import adversary as AD
    from nets import mindreader as MR
    w = C.World()
    mp, tracks = R.Map(w.net_np, w.geo, w.info), R.Tracks(w.net_np, w.geo)
    p, _ = CM.load_params("results/nets/mindreader_params.pkl")
    z = np.load("results/nets/mindreader_windows.npz")
    yt = z["y"][z["env"] < 12][:, 2]
    mu, sd = float(yt.mean()), float(yt.std())
    beh = AD.world(5.0)._replace(crn=1.0)
    st = C.run(w, w.reset(4, seed), beh, 120.0)
    _, f = C.record(w, st, beh, seconds, MR.EVERY, MR.make_feat(w.J))
    f = C.to_host(f, np.float32)
    e = 0
    T = f["v"].shape[0]
    # a busy 140 m square
    frames = []
    for t in range(MR.W - 1, T):
        hum = f["human"][t, e].astype(bool)
        idx = np.nonzero(hum)[0]
        ok = []
        for n in idx:
            u = f["uid"][t - MR.W + 1:t + 1, e, n]
            h = f["human"][t - MR.W + 1:t + 1, e, n]
            if (u == u[-1]).all() and h.all():
                ok.append(n)
        ok = np.array(ok, int)
        xs = []
        for n in ok:
            x = np.stack([f[c][t - MR.W + 1:t + 1, e, n] for c in MR.CH], 1)
            trk = f["trk"][t - MR.W + 1:t + 1, e, n]
            lc = np.concatenate([[0], (trk[1:] != trk[:-1]) & (trk[1:] // 2 == trk[:-1] // 2)]).astype(np.float32)
            xs.append(np.concatenate([x, lc[:, None]], 1))
        pred = np.asarray(MR.apply(p, jnp.asarray(MR.normalise(np.stack(xs))))[:, 3]) * sd + mu if len(xs) else np.zeros(0)
        frames.append({"t": t, "slots": ok, "pred": pred, "true": f["T"][t, e, ok]})
    # zoom: the 150 m square holding the most judged cars, mid-run
    fr = frames[len(frames) // 2]
    # positions need trk, s, length: the recorder kept trk and the gap/dist features; pose from trk + dist to the line
    def pose(t, n):
        trk = f["trk"][t, e, n]
        L = w.net_np["link_len"]
        s = np.where(trk < 2 * len(L), L[np.minimum(trk // 2, len(L) - 1)] - f["dist"][t, e, n], 0.0)
        return tracks.pose(trk, s, np.full(len(n), 4.6))
    x, y, _ = pose(fr["t"], fr["slots"])
    best, ctr = -1, (x.mean(), y.mean())
    for cx, cy in zip(x, y):
        c_ = ((np.abs(x - cx) < 55) & (np.abs(y - cy) < 55)).sum()
        if c_ > best:
            best, ctr = c_, (cx, cy)
    cmap = matplotlib.colormaps["RdYlBu"]
    norm = lambda v: np.clip((v - 1.0) / 0.8, 0, 1)
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 6.0), facecolor=BG)
    fig.subplots_adjust(0.01, 0.06, 0.99, 0.8, wspace=0.03)
    panel_axes(fig, axs, mp, ctr, 55)
    fig.text(0.02, 0.945, "Reading driving style from 10 s of driving", color=INK, fontsize=13, weight="bold")
    fig.text(0.02, 0.915, "Each human car coloured by its following headway T: red = tailgater (1.0 s), blue = cautious (1.8 s).\n"
             "Left: the truth the simulator drew. Right: the mind reader's guess from the car's last 10 s.", color=MUTED, fontsize=8.5, va="top")
    axs[0].set_title("hidden truth", color=INK, fontsize=10, loc="left")
    axs[1].set_title("network's estimate", color=INK, fontsize=10, loc="left")
    hud = fig.text(0.02, 0.02, "", color=MUTED, fontsize=9)
    arts = []

    def draw(k):
        nonlocal arts
        for a_ in arts:
            a_.remove()
        arts = []
        fr = frames[k]
        if len(fr["slots"]) == 0:
            return []
        xx, yy, hh = pose(fr["t"], fr["slots"])
        for ax, vals in zip(axs, (fr["true"], fr["pred"])):
            for j in range(len(xx)):
                if abs(xx[j] - ctr[0]) < 80 and abs(yy[j] - ctr[1]) < 80:
                    arts.append(car(ax, xx[j], yy[j], hh[j], 4.6, cmap(norm(vals[j]))))
        r_ = np.corrcoef(fr["true"], fr["pred"])[0, 1] if len(fr["true"]) > 2 else 0.0
        hud.set_text(f"t = {fr['t'] * 0.4:4.1f} s   cars judged: {len(xx)}   correlation of guess and truth in this frame: {r_:.2f}")
        return []

    save(fig, draw, len(frames), "mindreader.gif", fps=6, dpi=72)


def main(argv=None):
    os.makedirs(OUT, exist_ok=True)
    for w_ in (argv or ["forks", "blame", "occlusion", "mindreader"]):
        globals()[f"gif_{w_}"]()


if __name__ == "__main__":
    main(sys.argv[1:])
