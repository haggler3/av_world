"""Make a learned AV drive like people: an adversarial human-likeness reward (a small GAIL), aimed at what the AV Turing test found.

    python -m avlab.humanlike          # 4 rounds x 10 PPO updates from the safer fine-tune, then the Turing test and AVBench (~10 min)

The Turing test (avlab.turing) tells the bundled PPO policy from humans with AUC 0.88, and re-weighting the existing reward terms
made it *more* recognisable (0.91). So reward human-likeness directly:

    repeat for R rounds:
        record 4 copies x 2 min with the current policy driving the AVs; per step and vehicle: speed v, acceleration a, jerk j
        fit a discriminator D (logistic regression on v, a, j, a^2, j^2, v a; standardised) to tell human steps from AV steps
        PPO for K updates with reward += lam * clip(logit P(human | v, a, j), -3, 3) on every AV step
          (avrl.xppo.make_fns(bonus=...): the simulator and its reward are unchanged)
    then: the Turing test on the result, and AVBench's nominal scenario, against the policy it started from

As the policy gets more human-like the discriminator gets harder to fit; its own AUC per round is logged as the progress signal.
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from avlab import turing as T
from xnet import sim as X

OUT = "results/turing"


def step_features(v, a, j):
    return np.stack([v, a, j, a * a, j * j, v * a], -1)


def samples(rec, n=60000, seed=0):
    """Per-step (v, a, jerk) of humans (cars) and AVs from turing.record arrays; only steps whose previous step is the same trip."""
    v, a, alive, av, car, t0 = (np.asarray(x) for x in rec)
    v, a = v.astype(np.float32), a.astype(np.float32)
    same = alive[1:] & alive[:-1] & (t0[1:] == t0[:-1])
    jerk = (a[1:] - a[:-1]) / X.DT
    F = step_features(v[1:], a[1:], jerk)
    hum = same & car[1:] & ~av[1:]
    avm = same & av[1:]
    rng = np.random.default_rng(seed)
    h, m = F[hum], F[avm]
    h = h[rng.choice(len(h), min(n, len(h)), replace=False)]
    m = m[rng.choice(len(m), min(n, len(m)), replace=False)]
    return h, m


def fit_discriminator(h, m):
    Xf = np.concatenate([h, m])
    y = np.r_[np.ones(len(h), bool), np.zeros(len(m), bool)]
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    w = T.logistic((Xf - mu) / sd, y.astype(float), l2=1.0)
    auc = T.auc(np.c_[np.ones(len(Xf)), (Xf - mu) / sd] @ w, y)
    return {"w": w, "mu": mu, "sd": sd, "auc": auc}


def bonus_fn(D, lam):
    w, mu, sd = (jnp.asarray(D[k], jnp.float32) for k in ("w", "mu", "sd"))

    def bonus(tr):
        o = tr["obs"]
        v, a = 15.0 * o[..., 0], 3.0 * o[..., 1]
        j = jnp.concatenate([jnp.zeros_like(a[:1]), (a[1:] - a[:-1]) / X.DT], 0)
        f = (jnp.stack([v, a, j, a * a, j * j, v * a], -1) - mu) / sd
        logit = w[0] + f @ w[1:]
        return lam * jnp.clip(logit, -3.0, 3.0)
    return bonus


def train(init, reward, rounds=4, updates=10, lam=0.03, n_envs=8, seed=0, out=None):
    out = out or f"{OUT}/humanlike_lam{lam:g}"
    from av_world import data_path
    from avrl import xppo as P
    from xnet import network as NW
    os.makedirs(out, exist_ok=True)
    X.R = X.Reward(**reward)
    cfg = P.Config(n_envs=n_envs, warm=60.0, reward=reward)
    net, _, _ = NW.build(cfg.area)
    beh = P.behaviours(cfg.n_envs, seed, json.load(open(data_path("calibration.json")))["calibrated"])
    params = P.load(init)
    J = X.to_jnp(net)
    key = jax.random.PRNGKey(seed)
    st = jax.vmap(lambda k: X.reset(k, J, cfg.n_slots))(jax.random.split(key, cfg.n_envs))
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad), optax.adam(cfg.lr))
    opt_state = opt.init(params)
    log = []
    t0 = time.time()
    for r in range(rounds):
        rec = T.record(policy=P.deterministic(params), copies=4, warm_s=60.0, seconds=120.0, seed=100 + r)
        h, m = samples(rec, seed=r)
        D = fit_discriminator(h, m)
        warm, _, update = P.make_fns(cfg, net, beh, opt, bonus=bonus_fn(D, lam))
        if r == 0:
            for i in range(int(cfg.warm / (X.DT * (cfg.n_steps + 1)))):
                st = warm(params, st, jax.random.fold_in(key, 10_000 + i))
        for i in range(updates):
            params, opt_state, st, s = update(params, opt_state, st, jax.random.fold_in(key, 1000 * r + i))
        row = {"round": r, "discriminator_auc": D["auc"], "reward_per_step": float(s["reward_per_step"]), **P.summarize(s["m"]),
               "wall_s": round(time.time() - t0)}
        log.append(row)
        print(f"  round {r}: discriminator AUC {D['auc']:.3f} (before this round's updates), AV crashes {row['av_crashes']}, "
              f"AV speed {row['av_speed_kmh']:.1f} km/h ({row['wall_s']} s)", flush=True)
    P.save(params, f"{out}/policy.npz")
    json.dump(log, open(f"{out}/log.json", "w"), indent=1, default=float)
    return f"{out}/policy.npz", log


def evaluate(path, name):
    from avrl import bench
    from avrl import xppo as P
    F, y, g = T.trips(T.record(policy=P.deterministic(P.load(path))))
    t = T.test(F, y, g)
    sc = bench.evaluate(path, ["nominal"])["scenarios"]["nominal"]
    return {"name": name, "policy": path, "auc": t["auc"], "ci": t["ci"], "av_mean": t["av_mean"], "human_mean": t["human_mean"],
            "av_score": sc["av_score"], "crash_per_h_unaided": sc["metrics"]["crash_per_h_unaided"], "comfort": sc["metrics"]["comfort_pass"],
            "progress": sc["metrics"]["progress"], "filter_share": sc["metrics"]["filter_share"]}


def main(start=f"{OUT}/humanize_1/policy.npz", reward=None, lam=0.03):
    reward = reward or {"jerk": 0.0, "comfort": 0.0, "hard": 0.6}  # the safer fine-tune's weights (avlab.turing --humanize)
    path, log = train(start, reward, lam=lam)
    res = {"start": evaluate(start, "start: no smoothness reward, 3x hard-brake penalty"), "humanlike": evaluate(path, f"+ human-likeness reward (lam {lam})"),
           "log": log, "reward": reward, "lam": lam}
    json.dump(res, open(f"{OUT}/humanlike_lam{lam:g}.json", "w"), indent=1, default=float)
    for k in ("start", "humanlike"):
        r = res[k]
        print(f"{r['name']:55s} Turing AUC {r['auc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  av_world score {r['av_score']:.1f}  "
              f"unaided crashes/AV-h {r['crash_per_h_unaided']:.3f}  comfort {r['comfort']:.3f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.03, help="weight of the human-likeness reward; 0 = the control (same updates, no bonus)")
    main(lam=ap.parse_args().lam)
