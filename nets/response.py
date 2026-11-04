"""N7: how the human behind the AV answers its action, learned from paired forks.

For every AV fork with a human driver directly behind (results/nets/forks.npz), the follower's acceleration over the next 4 s is known
in all K + 1 branches. With shared dice the paired difference

  delta_k(t) = a_follower(t | AV takes action k) - a_follower(t | nobody perturbed)

is the causal effect of the action on the follower (plus the small effect of the other AVs' Latin-square actions, which the branch
"as is" measures: there this AV acts as in the log and only the others change).

  interventional   R(o, f, a_k) -> delta_k(t), t = 0.2 .. 3.8 s, trained on the forks (Huber loss)
  observational    F(o, f, u) -> a_follower(t), trained on the log only; its effect estimate is F(o, f, u + d_k) - F(o, f, u)
  zero             delta = 0 (the follower ignores us)

o: the AV's 28 observation features, f: the follower's gap, speed and acceleration and the AV's speed and acceleration, u: the policy's action.
Test on held-out copies: MAE of the 4 s effect, correlation of the predicted and true effect on the follower's hardest braking, and
AUROC for "this action makes the follower brake harder than 3 m/s^2 when it otherwise would not".

    python -m nets.response              # ~3 min -> results/nets/response.json
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import common as CM

OUT = "results/nets"
DATA = f"{OUT}/forks.npz"
HARD = -3.0


def x_of(obs, fs, u):
    f = fs / np.array([20.0, 15.0, 3.0, 1.0, 15.0, 3.0], np.float32)
    return np.concatenate([obs, f, u], 1).astype(np.float32)


def huber_loss(ps, b):
    return sum(optax.huber_loss(CM.mlp(p, b["x"]), b["y"]).mean() for p in ps) / len(ps)


def train(x, y, seed=0, n=3, steps=4000):
    key = jax.random.PRNGKey(seed)
    ps = [CM.init_mlp(jax.random.fold_in(key, i), [x.shape[1], 128, 128, y.shape[1]]) for i in range(n)]
    return CM.fit(huber_loss, ps, {"x": x, "y": y.astype(np.float32)}, steps=steps, seed=seed)


def pred(ps, x):
    return np.asarray(jnp.mean(jnp.stack([CM.mlp(p, jnp.asarray(x)) for p in ps]), 0))


def main():
    d = np.load(DATA)
    deltas = d["deltas"]
    K = len(deltas)
    keep = d["has_fol"].astype(bool) & (d["fol_state"][:, 3] < 0.5)          # a human directly behind
    obs, up, fs, fa, env = d["obs"][keep], d["u_prior"][keep], d["fol_state"][keep], d["fol_acc"][keep].astype(np.float32), d["env"][keep]
    base = fa[:, K]                                                             # [M, 19] the follower in the unperturbed world
    eff = fa[:, :K] - base[:, None]                                             # [M, K, 19] paired effect of each action
    test, tr = env >= 24, env < 24
    xi = np.concatenate([x_of(obs[tr], fs[tr], up[tr] + deltas[k]) for k in range(K)])
    yi = np.concatenate([eff[tr, k] for k in range(K)])
    R, cr = train(xi, yi)
    F, cf = train(x_of(obs[tr], fs[tr], up[tr]), base[tr])
    res = {"n_train": int(tr.sum()), "n_test": int(test.sum()), "actions": deltas.tolist(), "per_action": {}}
    p0 = pred(F, x_of(obs[test], fs[test], up[test]))
    for k in range(K):
        true = eff[test, k]
        p_int = pred(R, x_of(obs[test], fs[test], up[test] + deltas[k]))
        p_obs = pred(F, x_of(obs[test], fs[test], up[test] + deltas[k])) - p0
        hard_new = (fa[test, k].min(1) < HARD) & (base[test].min(1) >= HARD)
        tmin = fa[test, k].min(1) - base[test].min(1)
        row = {}
        for name, p in (("interventional", p_int), ("observational", p_obs), ("zero", np.zeros_like(true))):
            pmin = (base[test] + p).min(1) - base[test].min(1)
            row[name] = {"mae": CM.boot(np.abs(p - true).mean(1), groups=env[test]),
                         "corr_hardest_braking": float(np.corrcoef(pmin, tmin)[0, 1]) if p.std() > 0 else 0.0,
                         "auroc_new_hard_brake": CM.auroc(-(base[test] + p).min(1), hard_new) if hard_new.any() else float("nan")}
        row["true_mean_effect_on_hardest_braking"] = float(tmin.mean())
        row["new_hard_brakes"] = int(hard_new.sum())
        res["per_action"][str(k)] = row
        print(k, {n: {a: (round(b[0], 4) if isinstance(b, tuple) else round(b, 3)) for a, b in v.items()} for n, v in row.items() if isinstance(v, dict)},
              "true effect", round(row["true_mean_effect_on_hardest_braking"], 3), "new hard brakes", row["new_hard_brakes"], flush=True)
    res["curves"] = {"interventional": cr[::40].tolist(), "observational": cf[::40].tolist()}
    with open(f"{OUT}/response.json", "w") as f:
        json.dump(res, f)
    k = 1
    np.savez_compressed(f"{OUT}/response_preds.npz", true=eff[test, k], p_int=pred(R, x_of(obs[test], fs[test], up[test] + deltas[k])),
                        p_obs=pred(F, x_of(obs[test], fs[test], up[test] + deltas[k])) - p0, gap=fs[test, 0])


if __name__ == "__main__":
    main()
