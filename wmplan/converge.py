"""Where the world model earns its keep: the moments the policy crashes.

Run the barrier-teacher phase 3 policy (the strongest learned driver) through AVBench scenarios, and at every AV crash fork
the simulator 2-4 s before it (the state is one pytree: a fork is a copy) and replay the same seconds, with the same random
numbers, three ways:

  policy     the phase 3 policy alone (reproduces the crash: a check that the replay is exact)
  brake      a naive fix: the crashing AV brakes harder (u0 - 0.4) for up to 4 s
  planner    the v3 world model plans for the crashing AV every 0.2 s for up to 4 s: 256 imagined futures, refined over 5
             cross-entropy rounds around the policy's own imagined plan, scored by reward, crash / near-miss / red-run risk,
             barrier margins and ensemble disagreement (wmplan.v2.plan / score with the v3 model)

The metric: the share of the policy's crashes each fix averts, and the crashes it causes elsewhere in the same copy.
One averted crash becomes the GIF: the 256 futures narrowing over the 5 rounds, then the two real replays side by side.

    python -m wmplan.converge        # -> results/wmplan/v3/avert.json, docs/img/wm_converge.gif (~15 min on an L4)
"""
import json
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from avrl import xppo as P
from xnet import network as NW
from xnet import sim as X

from . import v2 as M

OUT = "results/wmplan/v3"
SCENARIOS = ("jaywalker_blind", "glare", "dartout", "red_runners", "cut_in", "jaywalkers", "inattentive", "nominal")
N_ENVS, CHUNK, OVERRIDE, REPLAY = 16, 10, 20, 30
W_PLAN = M.PLAN | dict(S=256, ITERS=5, ELITE=24, hard=0.0, gate=0.0, beta=1.0)
BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"


def plan_rounds(wm, prior, obs_row):
    """The cross-entropy search for one AV, every round kept: samples [R, S, HP, 2], scores [R, S], decoded speed
    [R, S, HP+1] (m/s) and crash probability [R, S, HP] (ensemble means), and the policy's own imagined plan."""
    w = W_PLAN
    o = jnp.asarray(obs_row)[None]
    z0 = M.encode(wm, o)
    fb = P.mean(prior, o)
    mu0 = M.prior_plan(wm, prior, z0, fb, w["HP"])
    kn, hp = w["KNOTS"], w["HP"]
    Mi = M._interp(hp, kn)
    lo, hi = wm["stats"]["lo"], wm["stats"]["hi"]
    mu = jnp.clip(mu0[:, np.round(np.linspace(0, hp - 1, kn)).astype(int)], lo, hi)
    sd = jnp.full_like(mu, w["sd0"])
    key = jax.random.PRNGKey(3)
    rounds = []
    sh = jnp.log(M.POS_W)
    for _ in range(w["ITERS"]):
        key, k = jax.random.split(key)
        eps = jax.random.normal(k, (1, w["S"]) + mu.shape[1:]).at[:, 0].set(0.0)
        knots = jnp.clip(mu[:, None] + sd[:, None] * eps, lo, hi)
        us = jnp.einsum("hk,bskd->bshd", Mi, knots)
        sc, (zs, trs, _) = M.score(wm, z0, us, w, o[:, 1] * 3.0, mu0)
        v = M.decode(wm, zs)[0][..., 0] * 15.0
        rounds.append({"us": np.asarray(us[0]), "sc": np.asarray(sc[0]), "v": np.asarray(v[0]),
                       "crash": np.asarray(jax.nn.sigmoid(trs[0, ..., 1] - sh))})
        el = jnp.argsort(-sc, axis=1)[:, :w["ELITE"]]
        e = jnp.take_along_axis(knots, el[:, :, None, None], 1)
        mu, sd = e.mean(1), e.std(1) + 0.05
    _, (zp, trp, _) = M.score(wm, z0, mu0[:, None], w, o[:, 1] * 3.0, mu0)
    return rounds, {"v": np.asarray(M.decode(wm, zp)[0][0, 0, :, 0] * 15.0), "crash": np.asarray(jax.nn.sigmoid(trp[0, 0, :, 1] - sh))}


def run(max_events=60, seconds=300.0, gif_out="docs/img/wm_converge.gif"):
    from avrl import bench as B
    wm, prior = M.load(f"{OUT}/model.pkl"), P.load(M.PRIOR)
    X.SHIELD = "off"
    events, gif_ev, t0 = [], None, time.time()
    n_chunks = int(seconds / X.DT / CHUNK)
    for si, name in enumerate(SCENARIOS):
        area, over = B.SCENARIOS[name]
        net, geo, info = NW.build(area)
        J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
        beh = B.behaviour(over)
        sim = X.Sim(net, n_envs=N_ENVS, beh=beh, seed=101 + si, reroute_s=0)
        sim.run(120.0)
        base = jax.random.PRNGKey(5000 + si)

        def step_keys(c):  # the random numbers of chunk c, the same in every replay
            return jax.random.split(jax.random.fold_in(base, c), CHUNK)

        def make_step(mode):
            """One step for all copies; mode 0 policy, 1 brake, 2 planner, applied to rows where target [E, N] is set."""
            def one(s, k2, tgt, on):
                def pol(obs, key):
                    u = P.mean(prior, obs)
                    if mode == 1:
                        u = jnp.where((tgt & on)[:, None], u + jnp.asarray([-0.4, 0.0]), u)
                    elif mode == 2:
                        i = jnp.argmax(tgt)
                        z0 = M.encode(wm, obs[i][None])
                        mu0 = M.prior_plan(wm, prior, z0, u[i][None], W_PLAN["HP"])
                        mu, _, _ = M.plan(wm, z0, mu0, key, W_PLAN, a0=obs[i, 1][None] * 3.0)
                        u = jnp.where((tgt & on)[:, None], mu[0, 0][None], u)
                    return u, jnp.zeros((obs.shape[0], 2))
                return X.step(s, J, beh, nh, rl=(pol, k2))
            return jax.jit(lambda st, ks, tgt, on: jax.vmap(one, in_axes=(0, 0, 0, None))(st, ks, tgt, on))

        steps = {m: make_step(m) for m in (0, 1, 2)}
        st, snaps, found = sim.st, {}, 0
        no_tgt = jnp.zeros((N_ENVS, st.trk.shape[1]), bool)
        for c in range(n_chunks):
            snaps[c] = st
            snaps.pop(c - 3, None)
            crashed = []
            for j, ks in enumerate(step_keys(c)):
                st, tr = steps[0](st, jax.random.split(ks, N_ENVS), no_tgt, False)
                hit = np.asarray(tr["crash"] & tr["valid"])
                for e, s in zip(*np.nonzero(hit)):
                    crashed.append((j, int(e), int(s)))
            for j, e, s in crashed:
                if c < 2 or len(events) >= max_events:
                    continue
                t_crash = CHUNK + j  # steps after the fork (the snapshot one chunk back)
                tgt = no_tgt.at[e, s].set(True)
                res, frames = {}, {}
                for mode, label in ((0, "policy"), (1, "brake"), (2, "planner")):
                    s_, hit_t, other, fr, tr_rows = snaps[c - 1], None, 0, [], []
                    keys = jnp.concatenate([step_keys(c - 1), step_keys(c), step_keys(c + 1)])[:REPLAY]
                    t00 = np.asarray(s_.t0[e, s])
                    for k in range(REPLAY):
                        if gif_ev is None:
                            fr.append(jax.tree.map(lambda a: np.asarray(a[e]), s_))
                        s_, tr = steps[mode](s_, jax.random.split(keys[k], N_ENVS), tgt, k < OVERRIDE)
                        if gif_ev is None:
                            tr_rows.append({q: np.asarray(tr[q][e]) for q in ("obs", "margins", "u", "valid", "crash")})
                        cr = np.asarray(tr["crash"][e] & tr["valid"][e])
                        same = np.asarray(s_.t0[e, s]) == t00
                        if hit_t is None and cr[s] and same:
                            hit_t = k
                        other += int(cr.sum() - (cr[s] if same else 0))
                    res[label] = {"crash": hit_t is not None, "t": hit_t, "other_crashes": other}
                    frames[label] = (fr, tr_rows)
                ev = {"scenario": name, "env": e, "slot": s, "t_crash": t_crash} | res
                events.append(ev)
                found += 1
                if gif_ev is None and res["policy"]["crash"] and not res["planner"]["crash"] and res["planner"]["other_crashes"] == 0:
                    obs_row = frames["planner"][1][0]["obs"][s]
                    rounds, prior_im = plan_rounds(wm, prior, obs_row)
                    gif_ev = {"ev": ev, "area": area, "frames": frames, "rounds": rounds, "prior_im": prior_im}
                    with open(f"{OUT}/converge_event.pkl", "wb") as f:
                        pickle.dump(gif_ev, f)
        print(f"{name}: {found} crash moments replayed ({time.time() - t0:.0f} s)", flush=True)
        if len(events) >= max_events:
            break
    X.SHIELD = "idm"
    rep = [e for e in events if e["policy"]["crash"]]  # replays that reproduced the crash
    summ = {"events": len(events), "reproduced": len(rep)}
    for lab in ("brake", "planner"):
        summ[lab] = {"averted": int(sum(not e[lab]["crash"] for e in rep)), "averted_share": float(np.mean([not e[lab]["crash"] for e in rep])) if rep else float("nan"),
                     "other_crashes": int(sum(e[lab]["other_crashes"] for e in rep)), "policy_other_crashes": int(sum(e["policy"]["other_crashes"] for e in rep))}
    summ["by_scenario"] = {n: {"crashes": sum(e["scenario"] == n for e in rep),
                               "planner_averted": sum(e["scenario"] == n and not e["planner"]["crash"] for e in rep),
                               "brake_averted": sum(e["scenario"] == n and not e["brake"]["crash"] for e in rep)} for n in SCENARIOS}
    with open(f"{OUT}/avert.json", "w") as f:
        json.dump({"summary": summ, "events": events, "planner": {k: v for k, v in W_PLAN.items()}}, f, indent=1)
    print(json.dumps(summ, indent=1), flush=True)
    if gif_ev is not None:
        gif(gif_ev, gif_out)


def gif(g, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from xnet import render as R
    net, geo, info = NW.build(g["area"])
    mp, tracks, heads = R.Map(net, geo, info), R.Tracks(net, geo), R.signal_heads(net, geo)
    ev, s = g["ev"], g["ev"]["slot"]
    frP, frW = g["frames"]["policy"][0], g["frames"]["planner"][0]
    rounds, pim = g["rounds"], g["prior_im"]
    n_r, hp = len(rounds), rounds[0]["v"].shape[1] - 1
    # the route the AV drove under the plan, to lay imagined distances along
    px, py, _ = tracks.pose(np.array([f.trk[s] for f in frW]), np.array([f.s[s] for f in frW]), np.array([f.length[s] for f in frW]))
    path = np.stack([px, py], 1)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]

    def along(dist):
        x, y = np.interp(dist, seg, path[:, 0]), np.interp(dist, seg, path[:, 1])
        over = np.maximum(dist - seg[-1], 0)
        dv = path[-1] - path[-2] if len(path) > 1 else np.array([1.0, 0.0])
        dv = dv / max(np.linalg.norm(dv), 1e-6)
        return x + over * dv[0], y + over * dv[1]

    allsc = np.concatenate([r["sc"] for r in rounds])
    lo_, hi_ = np.percentile(allsc, 5), allsc.max()
    cmap = plt.get_cmap("RdYlGn")
    tt = np.arange(hp + 1) * X.DT
    fig = plt.figure(figsize=(12.8, 6.4), facecolor=BG)
    axL, axR = fig.add_axes([0.0, 0.05, 0.5, 0.82]), fig.add_axes([0.5, 0.05, 0.5, 0.82])
    axS, axC = fig.add_axes([0.57, 0.52, 0.4, 0.3]), fig.add_axes([0.57, 0.11, 0.4, 0.3])
    for ax in (axL, axR):
        ax.set_facecolor("#1a1f24")
        mp.draw(ax, detail=True, labels=False)
        ax.set_aspect("equal")
        ax.axis("off")
    title = fig.text(0.012, 0.955, "", color=INK, fontsize=13, weight="bold")
    sub = fig.text(0.012, 0.915, "", color=INK2, fontsize=9.5)
    x0, y0, _ = tracks.pose(frW[0].trk[s:s + 1], frW[0].s[s:s + 1], frW[0].length[s:s + 1])
    for ax in (axL, axR):
        ax.set_xlim(x0[0] - 30, x0[0] + 30)
        ax.set_ylim(y0[0] - 30, y0[0] + 30)
    hold = 5
    n_plan, n_play = n_r * hold + 3, len(frP)
    arts = []

    def style(ax, lab):
        ax.cla()
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.set_title(lab, color=INK2, fontsize=9, loc="left")

    def av_box(ax, st, color):
        x, y, th = tracks.pose(st.trk[s:s + 1], st.s[s:s + 1], st.length[s:s + 1])
        return [ax.add_collection(matplotlib.collections.PolyCollection(R.car_polys(x, y, th, st.length[s:s + 1] + 0.6, np.array([2.5])),
                                                                        facecolors="none", edgecolors=color, lw=2.0, zorder=12))]

    def draw(i):
        nonlocal arts
        for a in arts:
            a.remove()
        arts = []
        if i < n_plan:  # the search: 256 imagined futures per round, narrowing
            r = min(i // hold, n_r - 1)
            R_ = rounds[r]
            axR.set_visible(False)
            axS.set_visible(True)
            axC.set_visible(True)
            arts += R.snapshot_artists(axL, net, geo, tracks, frW[0], True, heads)
            arts += av_box(axL, frW[0], "white")
            order = np.argsort(R_["sc"])
            for j in order:
                d = np.r_[0, np.cumsum(0.5 * (R_["v"][j][1:] + R_["v"][j][:-1]) * X.DT)]
                xs, ys = along(d)
                arts += axL.plot(xs, ys, color=cmap(np.clip((R_["sc"][j] - lo_) / (hi_ - lo_ + 1e-9), 0, 1)), lw=1.1, alpha=0.5, zorder=11)
            d = np.r_[0, np.cumsum(0.5 * (pim["v"][1:] + pim["v"][:-1]) * X.DT)]
            xs, ys = along(d)
            arts += axL.plot(xs, ys, color="#ff6bd5", lw=2.2, ls=(0, (4, 2)), zorder=13)
            style(axS, "imagined speed (m/s): each line one future")
            style(axC, "imagined crash probability per step (log)")
            for q in range(r + 1):
                Rq = rounds[q]
                a_ = 0.12 if q < r else 0.5
                for j in np.argsort(Rq["sc"]):
                    c_ = cmap(np.clip((Rq["sc"][j] - lo_) / (hi_ - lo_ + 1e-9), 0, 1))
                    axS.plot(tt, Rq["v"][j], color=c_, lw=0.6, alpha=a_)
                    axC.plot(tt[1:], np.maximum(Rq["crash"][j], 1e-6), color=c_, lw=0.6, alpha=a_)
            axS.plot(tt, pim["v"], color="#ff6bd5", lw=2.0, ls=(0, (4, 2)))
            axC.plot(tt[1:], np.maximum(pim["crash"], 1e-6), color="#ff6bd5", lw=2.0, ls=(0, (4, 2)))
            axC.set_yscale("log")
            axC.set_ylim(1e-5, 1)
            axC.set_xlabel("seconds ahead", color=INK2, fontsize=8)
            spread = float(R_["v"][:, -1].std())
            top = np.argsort(-R_["sc"])[:W_PLAN["ELITE"]]
            title.set_text(f"Round {r + 1} of {n_r}: 256 imagined futures for one AV, {ev['scenario'].replace('_', ' ')}")
            sub.set_text(f"spread of final speed {spread:.2f} m/s · best plans' crash risk {float(R_['crash'][top].max(1).mean()):.4f} · "
                         f"pink dashed: the policy's own plan · green: better, red: worse")
        else:  # what really happened, side by side
            k = min(i - n_plan, n_play - 1)
            axS.set_visible(False)
            axC.set_visible(False)
            axR.set_visible(True)
            for ax, fr, lab, crash in ((axL, frP, "policy alone (barrier-teacher PPO)", ev["policy"]), (axR, frW, "with the world model's plan", ev["planner"])):
                arts += R.snapshot_artists(ax, net, geo, tracks, fr[k], True, heads)
                hit = crash["crash"] and crash["t"] is not None and k >= crash["t"]
                arts += av_box(ax, fr[k], "#ff4d4d" if hit else "#39d0ff")
                arts.append(ax.text(0.02, 0.97, lab, transform=ax.transAxes, color=INK, fontsize=10, va="top", weight="bold", zorder=20))
                if hit:
                    arts.append(ax.text(0.02, 0.91, "CRASH", transform=ax.transAxes, color="#ff4d4d", fontsize=14, va="top", weight="bold", zorder=20))
                elif crash["t"] is None and k > (ev["policy"]["t"] or 0):
                    arts.append(ax.text(0.02, 0.91, "no crash", transform=ax.transAxes, color="#3fb950", fontsize=12, va="top", weight="bold", zorder=20))
            title.set_text(f"The same {REPLAY * X.DT:.0f} seconds, same random numbers: the policy crashes, the converged plan does not")
            sub.set_text(f"t = {k * X.DT:4.1f} s after the fork · the plan is re-searched every 0.2 s for {OVERRIDE * X.DT:.0f} s, then the policy drives")
        return []

    FuncAnimation(fig, draw, frames=list(range(n_plan + n_play))).save(out, writer=PillowWriter(fps=5), dpi=72,
                                                                       savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gif":  # gif [event.pkl out.gif]
        src, dst = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else (f"{OUT}/converge_event.pkl", "docs/img/wm_converge.gif")
        with open(src, "rb") as f:
            gif(pickle.load(f), dst)
    else:
        run()
