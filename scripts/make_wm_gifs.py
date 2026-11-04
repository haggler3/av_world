"""GIFs for docs/WORLD_MODEL_UPGRADES.md.

    python scripts/make_wm_gifs.py [crn|wm2|plan|all]      # -> docs/img/wmup/*.gif
"""
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, os.getcwd())
OUT = "docs/img/wmup"
BG, PANEL, INK, MUTED = "#0d1117", "#141a1f", "#e6edf3", "#8b949e"
BLUE, ORANGE, AQUA, GRAY, RED = "#3987e5", "#eb6834", "#1baf7a", "#9a9a94", "#e66767"
ROOT = "/tmp/claude-0/-home-user/55bcdaf2-8ef3-51cf-9658-ec3ffd940c9e/scratchpad/av2"


def save(fig, draw, n, name, fps=8, dpi=75):
    FuncAnimation(fig, draw, frames=n).save(f"{OUT}/{name}", writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", name, flush=True)


# ------------------------------------------------------------------------------------------ coupled worlds
def gif_crn(seconds=360, every=4.0):
    from xnet import network as NW
    from xnet import render as R
    from xnet import sim as X

    net, geo, info = NW.build("oakland")
    mp, tracks = R.Map(net, geo, info), R.Tracks(net, geo)
    cfg = [(crn, rate) for crn in (0.0, 1.0) for rate in (1 / 60, 1 / 20)]
    sims = [X.Sim(net, n_slots=768, n_envs=1, seed=5, reroute_s=0, beh=X.Behaviour(crn=c, distract_rate=r)) for c, r in cfg]
    frames, win = [], [[set(), set()], [set(), set()]]
    for k in range(int(seconds / every)):
        snap = []
        for sim in sims:
            sim.run(every)
            snap.append(jax_host(sim.st))
        frames.append(snap)
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 9.4), facecolor=BG)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.86, wspace=0.02, hspace=0.02)
    x0, y0 = NW.to_m(geo.bbox[0], geo.bbox[1])
    x1, y1 = NW.to_m(geo.bbox[2], geo.bbox[3])
    for ax in axs.ravel():
        ax.set_facecolor(PANEL)
        mp.draw(ax, detail=False, labels=False)
        ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.02, 0.965, "Same city, same seed, two driver-distraction rates", color=INK, fontsize=13, weight="bold")
    fig.text(0.02, 0.935, "orange = distracted here; white ring = distracted at the other rate too (left: 1/60 s, right: 1/20 s)", color=MUTED, fontsize=9)
    heads = ["", "", "", ""]
    titles = ["slot-indexed dice, low rate", "slot-indexed dice, high rate", "identity-keyed dice (CRN), low rate", "identity-keyed dice (CRN), high rate"]
    for ax, t in zip(axs.ravel(), titles):
        ax.set_title(t, color=INK, fontsize=9, pad=-12, loc="left", x=0.02)
    hud = fig.text(0.02, 0.905, "", color=MUTED, fontsize=9)
    hud2 = fig.text(0.02, 0.88, "", color=MUTED, fontsize=9)
    arts, hist = [], {0: [[], []], 1: [[], []]}

    def draw(k):
        nonlocal arts
        for a_ in arts:
            a_.remove()
        arts = []
        txt = []
        for row in range(2):
            lo, hi = frames[k][2 * row], frames[k][2 * row + 1]
            d_lo = set(lo.uid[lo.alive & (lo.dist_t > 0)].tolist())
            d_hi = set(hi.uid[hi.alive & (hi.dist_t > 0)].tolist())
            for col, (st, other) in enumerate(((lo, d_hi), (hi, d_lo))):
                ax = axs[row, col]
                m = st.alive & (st.kind == 0)
                x, y, _ = tracks.pose(st.trk[m], st.s[m], st.length[m])
                dist = st.dist_t[m] > 0
                shared = np.array([u in other for u in st.uid[m]])
                arts.append(ax.scatter(x[~dist], y[~dist], s=7, c="#8b949e", zorder=5, linewidths=0))
                arts.append(ax.scatter(x[dist], y[dist], s=34, c=ORANGE, zorder=6, linewidths=1.4, edgecolors=np.where(shared[dist], "white", ORANGE).tolist()))
            # nested share over the last ~40 s of events
            for col, d in enumerate((d_lo, d_hi)):
                hist[row][col].append(d)
            a = set().union(*hist[row][0][-10:])
            b = set().union(*hist[row][1][-10:])
            txt.append(f"{'with CRN' if row else 'before  '}: {len(a & b):3d} of {len(a):3d} drivers distracted at the low rate (last 40 s) are also distracted at the high rate")
        hud.set_text(f"t = {k * every / 60:4.1f} min   " + txt[0])
        hud2.set_text("              " + txt[1])
        return []

    save(fig, draw, len(frames), "crn_coupling.gif", fps=6, dpi=70)


def jax_host(st):
    import jax
    return jax.tree.map(lambda a: np.asarray(a[0]), st)


# ------------------------------------------------------------------------------------------ the token model driving real scenes
def gif_wm2():
    import jax

    from wm2 import data as D
    from wm2 import model as M
    from wm2 import run as RUN

    p = jax.tree.map(jax.numpy.asarray, pickle.load(open("results/wm2/params_leader.pkl", "rb")))
    files = sorted(__import__("glob").glob(f"{ROOT}/*/scenario_*.parquet"))
    rng = np.random.default_rng(0)
    test = set(rng.choice(len(files), len(files) // 5, replace=False).tolist())
    model = lambda S, NB, m, l: RUN._model(p, "leader", S, NB, m, l)
    cv = lambda S, NB, m, l: (np.zeros(len(S)), np.zeros(len(S)))
    best, pick = -1, None
    for i in sorted(test)[:40]:
        sc = D.load_scene(files[i])
        tr = []
        e = RUN.rollout_scene(sc, model, trace=tr)
        if e is None:
            continue
        sc_ = len(e)
        if sc_ >= 5 and e[:, -1].mean() < 3.0 and sc_ > best:
            best, pick = sc_, (sc, tr, e)
    sc, tr_m, e_m = pick
    tr_c = []
    RUN.rollout_scene(sc, cv, trace=tr_c)
    ag = np.where(sc["ok"][:, 48:80].all(1) & (sc["kind"] <= 2) & (D.speed(sc, 49) > 1.0))[0]
    log = sc["pos"][ag, 49:80].transpose(1, 0, 2)
    M_, C_ = np.stack(tr_m), np.stack(tr_c)
    ctr = log[0].mean(0)
    allp = np.concatenate([log.reshape(-1, 2), M_.reshape(-1, 2), C_.reshape(-1, 2)])
    r = max(np.abs(allp - ctr).max() * 1.1, 25)
    fig, ax = plt.subplots(figsize=(6.4, 6.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlim(ctr[0] - r, ctr[0] + r), ax.set_ylim(ctr[1] - r, ctr[1] + r), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    others = sc["pos"][:, 49:80]
    for a in range(len(sc["kind"])):                                         # everyone else: replayed from the log
        if a not in ag and sc["ok"][a, 49:80].any():
            ax.plot(others[a, :, 0], others[a, :, 1], color="#3a424b", lw=1.2, zorder=1)
    for i in range(len(ag)):
        ax.plot(log[:, i, 0], log[:, i, 1], color=GRAY, lw=0.8, alpha=0.5, zorder=2)
    fig.text(0.03, 0.955, "A 3 s closed-loop rollout on a held-out Argoverse 2 scene", color=INK, fontsize=11, weight="bold")
    fig.text(0.03, 0.92, "grey: what happened   blue: token model drives   orange: constant velocity", color=MUTED, fontsize=8.5)
    hud = fig.text(0.03, 0.04, "", color=MUTED, fontsize=9)
    arts = []

    def draw(k):
        nonlocal arts
        for a_ in arts:
            a_.remove()
        arts = [ax.scatter(log[k, :, 0], log[k, :, 1], s=46, c=GRAY, zorder=4),
                ax.scatter(C_[k, :, 0], C_[k, :, 1], s=30, c=ORANGE, zorder=5),
                ax.scatter(M_[k, :, 0], M_[k, :, 1], s=30, c=BLUE, zorder=6)]
        em = np.hypot(*(M_[k] - log[k]).T).mean()
        ec = np.hypot(*(C_[k] - log[k]).T).mean()
        hud.set_text(f"t = {k * 0.1:3.1f} s   mean position error:  token model {em:4.2f} m   constant velocity {ec:4.2f} m")
        return []

    save(fig, draw, len(log), "wm2_rollout.gif", fps=8, dpi=80)


# ------------------------------------------------------------------------------------------ planning in a learned model
def gif_plan(n_show=1):
    import jax
    import jax.numpy as jnp

    from avrl import env as E
    from avrl import worldmodel as W
    from wmplan import conservative as C

    rows, st, obs = C.collect()
    data = tuple(x[:C.N_TRAIN] for x in W.transitions(rows))
    model, norm, _ = W.fit(data, jax.random.PRNGKey(0), epochs=15)
    support = C.make_support(data)
    n = 64
    o0, st0 = obs[:n], jax.tree.map(lambda a: a[:n], st)
    keys = jax.random.split(jax.random.PRNGKey(1), n)
    out = {}
    for name, obj, kw, sig in (("naive", C.j_naive, {}, "ensemble"), ("support-penalised", C.j_mopo, {"lam": 0.25}, "support")):
        plans, _ = jax.jit(jax.vmap(lambda o, k: C.cem(model, norm, o, k, obj, kw, support=support, signal=sig)))(o0, keys)
        cfg = E.EnvConfig(shield=False)
        s_, ob = st0, o0
        real_v, real_g, real_r = [], [], []
        for t in range(C.H):
            sp = jax.vmap(E.make_spare, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(t), n), cfg)
            s_, ob, _, _, _ = jax.vmap(E.step, in_axes=(0, 0, None, 0))(s_, plans[:, t], cfg, sp)
            real_v.append(np.asarray(ob[:, 0]) * E.V_SCALE), real_g.append(np.asarray(ob[:, 11]) * 50), real_r.append(np.asarray(C.rhat(ob)))
        im = jax.vmap(lambda o, u: C.imagine(model, norm, o, u))(o0, plans)
        out[name] = {"rv": np.stack(real_v, 1), "rg": np.stack(real_g, 1), "rr": np.stack(real_r, 1), "iv": np.asarray(im[2][..., 0]) * E.V_SCALE,
                     "ig": np.asarray(im[2][..., 11]) * 50, "ir": np.asarray(im[0])}
    gamma = C.GAMMA ** np.arange(C.H)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=BG)
    for a in ax:
        a.set_facecolor(PANEL)
        a.tick_params(colors=MUTED)
        for sp in a.spines.values():
            sp.set_color("#30363d")
    ax[0].set_xlim(0, C.H * E.DT), ax[0].set_ylim(0, 12), ax[0].set_xlabel("seconds into the plan", color=MUTED), ax[0].set_ylabel("speed, m/s", color=MUTED)
    ax[1].set_xlim(0, C.H * E.DT), ax[1].set_ylim(-0.5, 2.2), ax[1].set_xlabel("seconds into the plan", color=MUTED), ax[1].set_ylabel("cumulative discounted return", color=MUTED)
    ax[0].set_title("mean speed", color=INK, fontsize=10, loc="left")
    ax[1].set_title("what the model promises (dashed) vs what happens (solid)", color=INK, fontsize=10, loc="left")
    fig.suptitle("A planner inside a learned model: 64 start states, plans run for real with the safety shield off", color=INK, fontsize=11, x=0.01, ha="left", fontweight="bold")
    t = np.arange(1, C.H + 1) * E.DT
    cols = {"naive": ORANGE, "support-penalised": BLUE}
    lines = []

    def draw(k):
        for l_ in lines:
            l_.remove()
        lines.clear()
        kk = k + 1
        for name, d in out.items():
            c = cols[name]
            lines.append(ax[0].plot(t[:kk], d["rv"].mean(0)[:kk], color=c, lw=2.2, label=f"{name}: real")[0])
            lines.append(ax[0].plot(t[:kk], d["iv"].mean(0)[:kk], color=c, lw=1.6, ls="--")[0])
            lines.append(ax[1].plot(t[:kk], np.cumsum(gamma * d["rr"].mean(0))[:kk], color=c, lw=2.2)[0])
            lines.append(ax[1].plot(t[:kk], np.cumsum(gamma * d["ir"].mean(0))[:kk], color=c, lw=1.6, ls="--")[0])
        if k == 0:
            ax[0].legend(facecolor=PANEL, labelcolor=INK, edgecolor="#30363d", fontsize=8, loc="lower right")
        return []

    save(fig, draw, C.H, "planning_exploitation.gif", fps=5, dpi=80)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, f in (("wm2", gif_wm2), ("plan", gif_plan), ("crn", gif_crn)):
        if which in (name, "all"):
            f()
