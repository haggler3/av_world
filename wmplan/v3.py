"""World model v3: the world model trains the policy, instead of second-guessing it at every step.

v2 (wmplan/v2.py) fixed the latent planner but could not beat the policy it planned around: its counterfactual "go" was
unreliable (the logs hold few matched alternatives), planning at test time was slow, and 3-minute runs could not tell policies
apart. v3, after Think2Drive / Dreamer (learn the policy inside the latent model), SafeDreamer (costs as a Lagrangian
constraint aiming near zero) and MOPO (ensemble pessimism):

  S1 counterfactuals  fork the simulator at risky moments (the state is one pytree: a fork is a copy, as in avrl/fork.py)
                      and replay 6 s under the same random numbers with the risky AVs' first 2 s of action perturbed (brake
                      harder / less, demand a bigger / smaller gap): the same scene, different choices, true outcomes
  S4 latent barrier   fine-tune the v2 model on them, with a discrete control-barrier condition on the decoded margins
                      (h(z') >= (1 - a) h(z) wherever the real data keeps it)
  S2 safe actor       start from the barrier-teacher phase 3 policy and improve it on 15-step imagined rollouts: lambda-
                      returns of reward minus ensemble spread, a cost of crash / red / near-miss probabilities and barrier
                      violations held below half the prior's own imagined cost by a Lagrange multiplier, and a trust
                      region to the prior on real observations
  S6 screening        32 copies x 60 s on three hard scenarios with paired seeds before anything runs the full bench

The result is an ordinary xppo policy: it drives at PPO speed and runs every AVBench scenario.

    python -m wmplan.v3 collect          # counterfactual branches -> results/wmplan/v3/data_cf.npz (~12 min on an L4)
    python -m wmplan.v3 finetune         # -> results/wmplan/v3/model.pkl, model_eval.json
    python -m wmplan.v3 actor 1.0 a      # trust-region weight, tag -> results/wmplan/v3/policy_a.npz, actor_a.json
    python -m wmplan.v3 screen           # the candidates against phase 3 -> results/wmplan/v3/screen.json
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from avrl import xppo as P
from avrl.ppo import mlp
from xnet import network as NW
from xnet import sim as X

from . import v2 as M

OUT = "results/wmplan/v3"
CF_SCENARIOS = ("nominal", "red_runners", "inattentive", "jaywalkers", "cut_in", "dartout", "jaywalker_blind", "glare")
DELTAS = ((0.0, 0.0), (-0.4, 0.0), (0.3, 0.0), (0.0, 0.8), (0.0, -0.8))  # branch 0: the prior itself
CF_ENVS, BRANCH, PERTURB, CYCLES, CF_ENV0 = 16, 30, 10, 10, 100
KEEP = ("obs", "u", "reward", "margins", "crash", "near", "red", "done", "valid")
CF_DATA = f"{OUT}/data_cf.npz"
GAMMA, LAM, H_IMAG = 0.97, 0.95, 15


# ---------------------------------------------------------------- S1: counterfactual branches

def collect_cf(cycles=CYCLES, seed=0):
    from avrl import bench as B
    os.makedirs(OUT, exist_ok=True)
    wm, prior = M.load(), P.load(M.PRIOR)
    sh = jnp.log(M.POS_W)
    rows, eid, t0 = [], CF_ENV0, time.time()
    X.SHIELD = "off"
    for si, name in enumerate(CF_SCENARIOS):
        area, over = B.SCENARIOS[name]
        net, geo, info = NW.build(area)
        J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
        beh = B.behaviour(over)
        sim = X.Sim(net, n_envs=CF_ENVS, beh=beh, seed=seed + si, reroute_s=0)
        sim.run(120.0)
        st = sim.st

        @jax.jit
        def branch(st, key, d):
            def body(st, xs):
                k, kk = xs

                def pol(obs, pkey):
                    u = P.mean(prior, obs)
                    x = jnp.concatenate([M.encode(wm, obs), M.act_in(u)], -1)
                    tr = jax.vmap(lambda m: M.run(m["trans"], x))(wm["ens"]).mean(0)
                    risk = jax.nn.sigmoid(tr[:, 4] - sh) + jax.nn.sigmoid(tr[:, 5] - sh)
                    risky = (risk > 0.02) | (obs[:, 24] > 0.4) | (obs[:, 25] > 0.4)
                    return jnp.where((risky & (k < PERTURB))[:, None], u + d, u), jnp.zeros((obs.shape[0], 2))
                st, tr = jax.vmap(lambda s, k2: X.step(s, J, beh, nh, rl=(pol, k2)))(st, jax.random.split(kk, CF_ENVS))
                keep = {k_: tr[k_] for k_ in KEEP}
                return st, jax.tree.map(lambda x: x.astype(jnp.float16) if x.dtype == jnp.float32 else x, keep)
            return jax.lax.scan(body, st, (jnp.arange(BRANCH), jax.random.split(key, BRANCH)))

        ev = np.zeros((len(DELTAS), 3))
        for c in range(cycles):
            kc = jax.random.PRNGKey(10_000 * (si + 1) + c)  # the same random numbers in every branch: paired futures
            nxt = None
            for b, d in enumerate(DELTAS):
                st_b, tr = branch(st, kc, jnp.asarray(d))
                tr = jax.tree.map(np.asarray, tr)
                t_idx, e_idx, s_idx = np.nonzero(tr["valid"])
                rows.append({k: v[t_idx, e_idx, s_idx] for k, v in tr.items() if k != "valid"} |
                            {"t": t_idx.astype(np.int32), "env": (eid + e_idx).astype(np.int32), "slot": s_idx.astype(np.int16),
                             "branch": np.full(len(t_idx), b, np.int8)})
                ev[b] += [tr["crash"].sum(), tr["near"].sum(), tr["red"].sum()]
                eid += CF_ENVS
                if b == 0:
                    nxt = st_b
            st = nxt
        print(f"{name}: crashes / near misses / red runs per branch {ev.astype(int).tolist()} ({time.time() - t0:.0f} s)", flush=True)
    X.SHIELD = "idm"
    d = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    o = np.lexsort((d["t"], d["slot"], d["env"]))
    np.savez_compressed(CF_DATA, **{k: v[o] for k, v in d.items()})
    print(f"{len(d['t'])} counterfactual decisions in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- S4: fine-tune with the counterfactuals and a latent barrier

def _data():
    """v2's logs and the counterfactual branches (copies numbered from CF_ENV0 + 16; one in ten held out)."""
    d = M.load_data(M.DATA + (CF_DATA,))
    d["cf"] = d["env"] >= CF_ENV0 + 16
    return d


def finetune(steps=5000, batch=1024, seed=0, a_cbf=0.2, w_cbf=1.0):
    t0 = time.time()
    d = _data()
    H = M.H
    starts = np.nonzero(d["cont"][: d["n"] - H])[0]
    held = np.isin(d["env"][starts], M.HOLDOUT) | (d["cf"][starts] & (d["env"][starts] % 10 == 3))
    tr_idx, te_idx = starts[~held], starts[held]
    ev = np.convolve((d["crash"] + d["near"] + d["red"]).astype(float), np.ones(H), "full")[H - 1:H - 1 + d["n"]]
    hot, cf = tr_idx[ev[tr_idx] > 0], tr_idx[d["cf"][tr_idx]]
    print(f"{d['n']} decisions ({int(d['cf'].sum())} counterfactual), {len(tr_idx)} train windows, {len(hot)} with an event, "
          f"{len(cf)} counterfactual ({time.time() - t0:.0f} s)", flush=True)
    D = M.device_data(d)
    p = M.load()

    def loss(p, b):
        l, parts = M.loss_fn(p, b)
        zs, _ = M.unroll(p, M.encode(p, b["obs"][:, 0]), b["u"])
        mh = M.decode_marg(p, zs)[..., :3]  # [E, B, H+1, 3]: stop, leader, junction barrier values
        real = b["marg"][..., :3]
        keep = (real[:, 1:] - (1 - a_cbf) * real[:, :-1] >= 0) * b["valid"][:, 1:, None]  # where the real data keeps it
        viol = jax.nn.relu(-(mh[:, :, 1:] - (1 - a_cbf) * mh[:, :, :-1])) * keep
        l_cbf = viol.sum() / jnp.maximum(keep.sum() * mh.shape[0], 1.0)
        return l + w_cbf * l_cbf, parts + (l_cbf,)

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(3e-4, steps, 0.1), weight_decay=1e-5))
    st = opt.init(p)

    @jax.jit
    def upd(p, st, idx):
        (l, parts), g = jax.value_and_grad(loss, has_aux=True)(p, M.windows(D, idx))
        g["stats"] = jax.tree.map(jnp.zeros_like, g["stats"])
        u_, st = opt.update(g, st, p)
        return optax.apply_updates(p, u_), st, l, parts

    rng = np.random.default_rng(seed)
    t1 = time.time()
    for s_ in range(steps):
        idx = rng.choice(tr_idx, batch)
        idx[: batch // 4] = rng.choice(hot, batch // 4)
        idx[batch // 4: batch // 2] = rng.choice(cf, batch // 4)  # counterfactual windows at about 4x their share
        p, st, l, parts = upd(p, st, jnp.asarray(idx))
        if s_ % 1000 == 0 or s_ == steps - 1:
            print(f"step {s_}: loss {float(l):.4f} events {float(parts[3]):.3f} latent {float(parts[4]):.4f} "
                  f"barrier {float(parts[6]):.4f} ({time.time() - t1:.0f} s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    M.save(p, f"{OUT}/model.pkl")
    te = np.random.default_rng(1).choice(te_idx, min(30000, len(te_idx)), replace=False)
    te_cf = te_idx[d["cf"][te_idx]]
    res = M.evaluate(p, D, te)
    res_cf = M.evaluate(p, D, np.random.default_rng(2).choice(te_cf, min(20000, len(te_cf)), replace=False))
    res |= {f"counterfactual: {k}": v for k, v in res_cf.items() if "AUROC" in k or "events" in k}
    res |= {"train_s": round(time.time() - t1), "steps": steps}
    with open(f"{OUT}/model_eval.json", "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(k, v)


# ---------------------------------------------------------------- S2: a safe actor-critic in imagination

def _critic(key):
    return M._mlp(key, [M.ZD, 256, 256, 1])


def actor(alpha=1.0, tag="a", steps=3000, batch=4096, beta=0.5, w_bar=0.5, seed=0):
    t0 = time.time()
    wm = M.load(f"{OUT}/model.pkl")
    prior = P.load(M.PRIOR)
    d = _data()
    risky = np.nonzero(d["hcrash"] | d["hred"] | (d["near"] > 0) | (d["cf"] & (d["margins"][:, :3].min(1) < 0.2)))[0]
    every = np.arange(d["n"])
    obs_all = jnp.asarray(d["obs"].astype(np.float32))
    print(f"{d['n']} start states, {len(risky)} risky ({time.time() - t0:.0f} s)", flush=True)
    sh = jnp.log(M.POS_W)
    E = M.E

    def imagine(pi, obs, mem):
        """H_IMAG steps of the policy in the latent: per-step reward, cost, latents (with the final one)."""
        B_ = obs.shape[0]

        def body(z, _):
            u = mlp(pi, M.decode(wm, z)[0])
            x = jnp.concatenate([z, M.act_in(u)], -1)
            z1s, trs = jax.vmap(lambda m: (M.simnorm(M.run(m["dyn"], x)), M.run(m["trans"], x)))(wm["ens"])  # [E, B, .]
            z1 = z1s[mem, jnp.arange(B_)]  # each rollout follows one ensemble member
            pe = jax.nn.sigmoid(trs[..., 1:4] - sh)  # [E, B, 3] crash, near miss, red run
            r = trs[..., 0].mean(0) / M.R_SCALE - beta * trs[..., 0].std(0) / M.R_SCALE
            c = pe[..., 0].mean(0) + 0.3 * pe[..., 1].mean(0) + pe[..., 2].mean(0) + beta * pe.std(0).sum(-1) \
                + w_bar * jax.nn.relu(-M.decode_marg(wm, z)[..., :3]).sum(-1)
            return z1, (z, r, c)
        z0 = M.encode(wm, obs)
        zT, (zs, r, c) = jax.lax.scan(body, z0, None, length=H_IMAG)
        return jnp.concatenate([zs, zT[None]], 0), r, c  # [H+1, B, ZD], [H, B], [H, B]

    def lam_ret(x, v):
        """lambda-returns [H, B] of per-step x [H, B] with bootstrap values v [H+1, B]."""
        def body(nxt, xs):
            x_t, v_t1 = xs
            ret = x_t + GAMMA * ((1 - LAM) * v_t1 + LAM * nxt)
            return ret, ret
        _, rets = jax.lax.scan(body, v[-1], (x, v[1:]), reverse=True)
        return rets

    def vf(c, z):
        return M.run(c, z)[..., 0]

    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    params = {"pi": prior["pi"], "V": _critic(k[0]), "Vc": _critic(k[1])}
    opt_pi = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
    opt_c = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    s_pi, s_c = opt_pi.init(params["pi"]), opt_c.init({"V": params["V"], "Vc": params["Vc"]})

    def losses(pi, crit, obs, mem, lam):
        zs, r, c = imagine(pi, obs, mem)
        zs_sg = jax.lax.stop_gradient(zs)
        v, vc = vf(crit["V"], zs), vf(crit["Vc"], zs)
        R, C = lam_ret(r, jax.lax.stop_gradient(v)), lam_ret(c, jax.lax.stop_gradient(vc))
        bc = ((mlp(pi, obs) - mlp(prior["pi"], obs)) ** 2).sum(-1).mean()
        l_pi = -(R[0] - lam * C[0]).mean() + alpha * bc
        l_c = ((vf(crit["V"], zs_sg[:-1]) - jax.lax.stop_gradient(R)) ** 2).mean() + \
            ((vf(crit["Vc"], zs_sg[:-1]) - jax.lax.stop_gradient(C)) ** 2).mean()
        return l_pi, l_c, (R[0].mean(), C[0].mean(), bc, mlp(pi, obs)[:, 0].mean())

    @jax.jit
    def upd(params, s_pi, s_c, obs, mem, lam, train_pi):
        crit = {"V": params["V"], "Vc": params["Vc"]}
        (l_pi, aux), g_pi = jax.value_and_grad(lambda pi: (lambda a, b, c_: (a, c_))(*losses(pi, crit, obs, mem, lam)), has_aux=True)(params["pi"])
        l_c, g_c = jax.value_and_grad(lambda cr: losses(params["pi"], cr, obs, mem, lam)[1])(crit)
        u_pi, s_pi2 = opt_pi.update(g_pi, s_pi, params["pi"])
        pi = jax.tree.map(lambda a, b: jnp.where(train_pi, a, b), optax.apply_updates(params["pi"], u_pi), params["pi"])
        s_pi = jax.tree.map(lambda a, b: jnp.where(train_pi, a, b), s_pi2, s_pi)
        u_c, s_c = opt_c.update(g_c, s_c, crit)
        crit = optax.apply_updates(crit, u_c)
        return {"pi": pi, **crit}, s_pi, s_c, l_pi, l_c, aux

    rng = np.random.default_rng(seed)

    def batch_():
        idx = np.concatenate([rng.choice(risky, batch // 2), rng.choice(every, batch // 2)])
        return obs_all[idx], jnp.asarray(rng.integers(0, E, batch))

    # the prior's own imagined cost sets the target: half of it
    c_prior = float(np.mean([float(imagine(prior["pi"], *batch_())[2].sum(0).mean()) for _ in range(8)]))
    c_target, lam, log = 0.5 * c_prior, 1.0, []
    print(f"prior's imagined cost per start {c_prior:.4f} -> target {c_target:.4f}", flush=True)
    warm = 300  # critics first
    for s_ in range(steps + warm):
        obs, mem = batch_()
        params, s_pi, s_c, l_pi, l_c, (R0, C0, bc, u0) = upd(params, s_pi, s_c, obs, mem, lam, s_ >= warm)
        if s_ >= warm:
            lam = float(np.clip(lam * np.exp(0.05 * (float(C0) / c_target - 1.0)), 0.1, 300.0))  # dual ascent, in log space
        if s_ % 250 == 0 or s_ == steps + warm - 1:
            row = {"step": s_, "return": float(R0), "cost": float(C0), "lambda": lam, "bc": float(bc), "mean_u0": float(u0),
                   "critic_loss": float(l_c), "s": round(time.time() - t0)}
            log.append(row)
            print(row, flush=True)
    out = dict(prior) | {"pi": params["pi"]}
    P.save(out, f"{OUT}/policy_{tag}.npz")
    with open(f"{OUT}/actor_{tag}.json", "w") as f:
        json.dump({"alpha": alpha, "beta": beta, "c_prior": c_prior, "c_target": c_target, "log": log}, f, indent=1)


# ---------------------------------------------------------------- S2': improvement only where the counterfactuals were measured
# The actor above (pathwise gradients through 15 imagined steps) exploited the model: in the real simulator it crashed 30-40
# times per AV-hour against its prior's 0.04-0.1. So instead: for each real state, score in the model only the five actions the
# counterfactual branches tested (the prior, brake harder / less, a bigger / smaller gap, for 2 s, then the prior), pick the best
# where it beats the prior's own by a margin, and distil those choices into the policy by regression on real observations.

W_COST = dict(crash=30.0, near=3.0, red=40.0, bar=0.5)


def choose(n_states=2_000_000, chunk=16384, beta=1.0, gate=0.5, seed=0):
    """-> (row indices, chosen delta index per row, score gain) over real logged states (v2 logs and counterfactual branch 0)."""
    wm, prior = M.load(f"{OUT}/model.pkl"), P.load(M.PRIOR)
    d = _data()
    real = np.nonzero(~d["cf"] | (d.get("branch", np.zeros(d["n"], np.int8)) == 0))[0] if "branch" in d else np.arange(d["n"])
    rng = np.random.default_rng(seed)
    rows = rng.choice(real, min(n_states, len(real)), replace=False)
    obs = d["obs"]
    deltas = jnp.asarray(DELTAS, jnp.float32)
    sh = jnp.log(M.POS_W)

    @jax.jit
    def scores(o):
        B_ = o.shape[0]
        z0 = jnp.repeat(M.encode(wm, o)[:, None], len(DELTAS), 1)  # [B, 5, ZD]

        def member(m):
            def body(z, k):
                u = P.mean(prior, M.decode(wm, z)[0]) + jnp.where(k < PERTURB, 1.0, 0.0) * deltas
                x = jnp.concatenate([z, M.act_in(u)], -1)
                tr = M.run(m["trans"], x)
                pe = jax.nn.sigmoid(tr[..., 1:4] - sh)
                cost = W_COST["crash"] * pe[..., 0] + W_COST["near"] * pe[..., 1] + W_COST["red"] * pe[..., 2] \
                    + W_COST["bar"] * jax.nn.relu(-M.decode_marg(wm, z)[..., :3]).sum(-1)
                return M.simnorm(M.run(m["dyn"], x)), (tr[..., 0] / M.R_SCALE - cost) * GAMMA ** k
            _, g = jax.lax.scan(body, z0, jnp.arange(H_IMAG))
            return g.sum(0)  # [B, 5]
        s = jax.vmap(member)(wm["ens"])  # [E, B, 5]
        return s.mean(0) - beta * s.std(0)

    best, gain, t0 = [], [], time.time()
    for i in range(0, len(rows), chunk):
        s = np.asarray(scores(jnp.asarray(obs[rows[i:i + chunk]].astype(np.float32))))
        b = s.argmax(1)
        g = s[np.arange(len(b)), b] - s[:, 0]
        b = np.where(g > gate, b, 0)
        best.append(b)
        gain.append(np.where(b > 0, g, 0.0))
    best, gain = np.concatenate(best), np.concatenate(gain)
    share = {str(DELTAS[k]): float((best == k).mean()) for k in range(len(DELTAS))}
    print(f"{len(rows)} states scored in {time.time() - t0:.0f} s; chosen actions {share}", flush=True)
    return rows, best, gain, share


def improve(tag="c", steps=3000, batch=4096, gate=0.5, beta=1.0, seed=0):
    rows, best, gain, share = choose(gate=gate, beta=beta, seed=seed)
    d = _data()
    prior = P.load(M.PRIOR)
    obs = jnp.asarray(d["obs"][rows].astype(np.float32))
    tgt = np.asarray(P.mean(prior, obs)) + np.asarray(DELTAS, np.float32)[best]
    tgt = jnp.asarray(tgt)
    changed = np.nonzero(best > 0)[0]
    pi = prior["pi"]
    opt = optax.adam(3e-4)
    st = opt.init(pi)

    @jax.jit
    def upd(pi, st, idx):
        l, g = jax.value_and_grad(lambda q: ((mlp(q, obs[idx]) - tgt[idx]) ** 2).sum(-1).mean())(pi)
        u_, st = opt.update(g, st, pi)
        return optax.apply_updates(pi, u_), st, l

    rng = np.random.default_rng(seed)
    for s_ in range(steps):
        idx = rng.integers(0, len(rows), batch)
        if len(changed):
            idx[: batch // 4] = rng.choice(changed, batch // 4)  # the states where the model says the prior should act differently
        pi, st, l = upd(pi, st, jnp.asarray(idx))
        if s_ % 1000 == 0 or s_ == steps - 1:
            print(f"distil step {s_}: loss {float(l):.5f}", flush=True)
    P.save(dict(prior) | {"pi": pi}, f"{OUT}/policy_{tag}.npz")
    with open(f"{OUT}/improve_{tag}.json", "w") as f:
        json.dump({"gate": gate, "beta": beta, "states": len(rows), "chosen": share, "mean_gain_where_changed": float(gain[changed].mean()) if len(changed) else 0.0}, f, indent=1)


# ---------------------------------------------------------------- S6: screening

def screen(cands=None, scenarios=("red_runners", "inattentive", "dartout"), seconds=60.0, n_envs=32):
    """Candidates against the prior, 32 copies x 60 s per scenario from the bench's warm state and random numbers (paired)."""
    from avrl import bench as B
    from avrl import scorecard as SC
    cands = cands or {"phase 3 (prior)": M.PRIOR} | {f"v3 {t}": f"{OUT}/policy_{t}.npz" for t in ("a", "b") if os.path.exists(f"{OUT}/policy_{t}.npz")}
    B.N_ENVS, B.SECONDS = n_envs, seconds
    res = {}
    for n in scenarios:
        area, over = B.SCENARIOS[n]
        r = B.runner(area, B.behaviour(over))
        res[n] = {}
        for name, path in cands.items():
            prm = P.load(path)
            raw, sh_ = r["raw"](prm), r["shielded"](prm)
            sc, pr = SC.scorecard(raw, sh_), B.pooled(raw)
            res[n][name] = {"av_score": float(sc["av_score"]), "crash_per_h_unaided": pr["av_crash_per_h"], "av_crash": pr["av_crash"],
                            "red": pr["av_red"], "near_per_h": pr["av_near_per_h"], "speed_kmh": pr["av_speed_kmh"],
                            "hard_per_h": pr["av_hard_per_h"], "av_hours": pr["av_hours"]}
            print(n, name, {k: round(v, 3) for k, v in res[n][name].items()}, flush=True)
    with open(f"{OUT}/screen.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        collect_cf()
    elif cmd == "finetune":
        finetune()
    elif cmd == "actor":
        actor(alpha=float(sys.argv[2]), tag=sys.argv[3])
    elif cmd == "improve":
        improve(tag=sys.argv[2], gate=float(sys.argv[3]))
    elif cmd == "screen":
        screen(cands={"phase 3 (prior)": M.PRIOR} | {f"v3 {t}": f"{OUT}/policy_{t}.npz" for t in sys.argv[2:]} if len(sys.argv) > 2 else None)
