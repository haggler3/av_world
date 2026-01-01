"""See the planner think: an AV at a junction or a merge, with every candidate plan unfolded in latent space and decoded.

Left: the street, with the AV (blue, white outline) and, ahead of it along its route, where each of the 32 candidate
plans of the last search round puts it over the next 2.4 s (thin lines, coloured by the plan's score), the plan it
executes (cyan) and where it really went (dashed white). Right: the same plans decoded into speed, the junction
time-to-conflict margin and the leader margin, and the predicted crash risk.

    python -m wmplan.viz            # ~3 min -> docs/img/wm_planner_junction.gif, docs/img/wm_planner_merge.gif
"""
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from avrl import xppo as P  # noqa: E402
from xnet import network as NW  # noqa: E402
from xnet import render as R  # noqa: E402
from xnet import sim as X  # noqa: E402

from . import evaluate as E  # noqa: E402
from . import model as M  # noqa: E402
from . import planner as PL  # noqa: E402

BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"
STEPS, CLIP = 450, 60


def record(seed=21):
    net, geo, info = NW.build("oakland")
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = E.world()
    sim = X.Sim(net, n_envs=1, beh=beh, seed=seed, reroute_s=0)
    sim.run(E.WARM)
    st = jax.tree.map(lambda a: a[0], sim.st)
    ppo, wm = P.load("results/xnet3/policy.npz"), M.load()
    pol = PL.make(wm, ppo)
    X.SHIELD = "off"
    f = jax.jit(lambda s, k: X.step(s, J, beh, nh, rl=(pol, k, True)))
    frames, trs = [], []
    for i in range(STEPS):
        frames.append(jax.tree.map(np.asarray, st))
        st, tr = f(st, jax.random.PRNGKey(1000 + i))
        trs.append({k: np.asarray(tr[k]) for k in ("obs", "margins", "u", "valid")})
    X.SHIELD = "idm"
    return net, geo, info, frames, trs, wm, ppo


def pick(frames, trs, kind, L):
    """An AV slot that stays one vehicle for a whole clip, moves, and enters a junction with a tight time-to-conflict
    margin (or changes lanes into a tight gap) in the middle of it."""
    T = len(trs)
    trk = np.stack([f.trk for f in frames])
    valid = np.stack([t["valid"] for t in trs])
    marg = np.stack([t["margins"] for t in trs])
    obs = np.stack([t["obs"] for t in trs])
    t0s = np.stack([f.t0 for f in frames])
    best, best_s = None, -1e9
    for slot in np.nonzero(valid.any(0))[0]:
        for c in range(CLIP // 2, T - CLIP // 2, 5):
            a, b = c - CLIP // 2, c + CLIP // 2
            if not valid[a:b, slot].all() or (t0s[a:b, slot] != t0s[a, slot]).any() or obs[a:b, slot, 0].mean() * 15 < 3.0:
                continue
            tr_ = trk[c - 5:c + 15, slot]
            if kind == "junction":
                if not ((tr_[:5] < 2 * L).all() and (tr_ >= 2 * L).any()):  # on the approach, enters the box soon after
                    continue
                s_ = -marg[c - 10:c, slot, 2].min() + 0.5 * (obs[c, slot, 15] > 0)  # tightest time-to-conflict before entering
            else:
                link_ = tr_ < 2 * L
                if not (link_.all() and (tr_ // 2 == tr_[0] // 2).all() and (tr_ % 2 != tr_[0] % 2).any()):  # changes lane here
                    continue
                s_ = -marg[c - 5:c + 10, slot, 3].min() + obs[c, slot, 21]  # tight gap behind, mandatory merge
            if s_ > best_s:
                best, best_s = (slot, a, b), s_
    return best


def gif(net, geo, info, frames, trs, wm, ppo, slot, a, b, out, title, kind="junction"):
    mi, mscale, mlab = (2, 5.0, "junction margin: time to conflict - time to clear (s)") if kind == "junction" else \
        (3, 20.0, "merge margin: gap behind - what the car behind needs (m)")
    mp, tracks, heads = R.Map(net, geo, info), R.Tracks(net, geo), R.signal_heads(net, geo)
    # the route the AV really drove (its positions over the clip and 3 s beyond), to place imagined progress on
    end = min(b + 15, len(frames))
    px, py, _ = tracks.pose(np.array([f.trk[slot] for f in frames[a:end]]), np.array([f.s[slot] for f in frames[a:end]]),
                            np.full(end - a, frames[a].length[slot]))
    path = np.stack([px, py], 1)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]

    def along(i0, d):
        """Position d metres beyond the AV's position at clip index i0 along the driven route (extrapolated at the end)."""
        g = seg[i0] + d
        x, y = np.interp(g, seg, path[:, 0]), np.interp(g, seg, path[:, 1])
        over = np.maximum(g - seg[-1], 0)
        dvec = path[-1] - path[-2] if len(path) > 1 else np.array([1.0, 0.0])
        dvec = dvec / max(np.linalg.norm(dvec), 1e-6)
        return x + over * dvec[0], y + over * dvec[1]

    fig = plt.figure(figsize=(12.8, 6.4), facecolor=BG)
    ax = fig.add_axes([0.0, 0.04, 0.5, 0.84])
    axs = [fig.add_axes([0.56, 0.68 - 0.29 * i, 0.41, 0.2]) for i in range(3)]
    ax.set_facecolor("#1a1f24")
    mp.draw(ax, detail=True, labels=True)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.012, 0.955, title, color=INK, fontsize=13, weight="bold")
    hud = fig.text(0.012, 0.915, "", color=INK2, fontsize=9)
    fig.text(0.012, 0.012, "thin: the 32 candidate plans of the last search round, coloured by score (red worst, green best)   cyan: the plan "
             "executed   dashed: what really happened", color=INK2, fontsize=8)
    cmap = plt.get_cmap("RdYlGn")
    arts = []
    key = jax.random.PRNGKey(3)
    fb = P.mean(ppo, jnp.asarray(np.stack([t["obs"][slot] for t in trs[a:b]])))
    z0s = M.encode(wm, jnp.asarray(np.stack([t["obs"][slot] for t in trs[a:b]])))
    plans = []
    for i in range(b - a):
        mu, us, sc = PL.plan(wm, z0s[i:i + 1], jnp.repeat(fb[i:i + 1, None], PL.HP, 1), jax.random.fold_in(key, i))
        _, (zs, trs_h, marg) = PL.score(wm, z0s[i:i + 1], jnp.concatenate([us, mu[:, None]], 1))
        o, _ = M.decode(wm, zs)
        plans.append({k: np.asarray(v[0]) for k, v in dict(us=us, sc=sc, v=o[..., 0] * 15.0, marg=marg,
                                                             crash=jax.nn.sigmoid(trs_h[..., 1])).items()})
    tt = np.arange(PL.HP + 1) * X.DT

    def draw(i):
        nonlocal arts
        for x in arts:
            x.remove()
        arts = []
        k = a + i
        st = frames[k]
        arts += R.snapshot_artists(ax, net, geo, tracks, st, True, heads)
        x, y, th = tracks.pose(st.trk[slot:slot + 1], st.s[slot:slot + 1], st.length[slot:slot + 1])
        ax.set_xlim(x[0] - 42, x[0] + 42)
        ax.set_ylim(y[0] - 42, y[0] + 42)
        arts.append(ax.add_collection(matplotlib.collections.PolyCollection(
            R.car_polys(x, y, th, st.length[slot:slot + 1] + 0.6, np.array([2.5])), facecolors="none", edgecolors="white", lw=1.6, zorder=12)))
        p = plans[i]
        sc = p["sc"]
        cn = (sc - sc.min()) / max(np.ptp(sc), 1e-6)
        n_s = len(sc)
        for j in range(n_s + 1):
            v = p["v"][j]
            d = np.r_[0, np.cumsum(0.5 * (v[1:] + v[:-1]) * X.DT)]
            xs, ys = along(i, d)
            best = j == n_s
            arts += ax.plot(xs, ys, color="#39d0ff" if best else cmap(cn[j]), lw=3.0 if best else 0.9, alpha=1 if best else 0.55,
                            zorder=13 if best else 11)
        # the truth over the same horizon
        h = min(PL.HP, len(frames) - 1 - k)
        real_v = np.array([trs[k + q]["obs"][slot][0] * 15.0 for q in range(h + 1)]) if k + h < len(trs) else None
        if real_v is not None:
            d = np.r_[0, np.cumsum(0.5 * (real_v[1:] + real_v[:-1]) * X.DT)]
            xs, ys = along(i, d)
            arts += ax.plot(xs, ys, color="white", lw=1.6, ls=(0, (3, 2)), zorder=14)
        for q, (axq, (lab, fn, ref)) in enumerate(zip(axs, (
                ("speed (m/s)", lambda j: p["v"][j], lambda q_: trs[q_]["obs"][slot][0] * 15.0),
                (mlab, lambda j: np.r_[np.nan, p["marg"][j][:, mi] * mscale], lambda q_: trs[q_]["margins"][slot][mi] * mscale),
                ("predicted crash probability per step", lambda j: np.r_[np.nan, p["crash"][j]], None)))):
            axq.cla()
            axq.set_facecolor("#161b22")
            for s_ in axq.spines.values():
                s_.set_color("#30363d")
            axq.tick_params(colors=INK2, labelsize=7.5)
            axq.set_title(lab, color=INK2, fontsize=8.5, loc="left")
            for j in range(n_s + 1):
                best = j == n_s
                axq.plot(tt, fn(j), color="#39d0ff" if best else cmap(cn[j]), lw=2.4 if best else 0.8, alpha=1 if best else 0.5)
            if ref is not None and k + PL.HP < len(trs):
                axq.plot(tt, [ref(k + q_) for q_ in range(PL.HP + 1)], color="white", lw=1.4, ls=(0, (3, 2)))
            if q == 1:
                axq.axhline(0, color="#ff6b6b", lw=0.8)
                axq.set_ylim(-5.5, 5.5) if kind == "junction" else axq.set_ylim(-20, 40)
            if q == 2:
                axq.set_yscale("log")
                axq.set_ylim(1e-5, 1)
                axq.set_xlabel("seconds ahead", color=INK2, fontsize=8)
        gsc = 1.15 + 0.35 * np.tanh(trs[k]["u"][slot][1])
        hud.set_text(f"t = {float(st.t) - float(frames[a].t):4.1f} s   AV speed {st.v[slot] * 3.6:4.1f} km/h   "
                     f"chosen accel {float(np.clip(trs[k]['u'][slot][0], -1, 1)) * (3 if trs[k]['u'][slot][0] >= 0 else 8):+4.1f} m/s²   "
                     f"gap acceptance x{gsc:.2f} (rule-based AV: x1.25)")
        return []

    FuncAnimation(fig, draw, frames=list(range(b - a))).save(out, writer=PillowWriter(fps=6), dpi=72, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", out, flush=True)


def main():
    net, geo, info, frames, trs, wm, ppo = record()
    for kind, title in (("junction", "Latent planning at a junction: every candidate future, unfolded and decoded"),
                        ("merge", "Latent planning in a merge: every candidate future, unfolded and decoded")):
        sel = pick(frames, trs, kind, len(net["link_len"]))
        if sel is None:
            print("no clip for", kind)
            continue
        gif(net, geo, info, frames, trs, wm, ppo, *sel, f"docs/img/wm_planner_{kind}.gif", title, kind)


if __name__ == "__main__":
    main()
