"""N2: an interventional risk critic. Does training on exact forks beat training on logs at choosing actions?

Q(o, a) = P(bad outcome for this AV in the next 4 s | observation o, action a), bad = a crash, a near miss, or the AV moving more than
1 m inside its barrier. Two critics with the same network and the same states (results/nets/forks.npz, nets/forks.py):

  interventional   trained on the forks: for each state, the outcome of each of the K = 5 actions, with every human's dice shared
  observational    trained on the log: for each state, the outcome of the action the policy actually took

The log never shows what would have happened under another action, so the observational critic has to extrapolate from how outcomes
vary across states with the action the policy chose there, and the policy brakes when it is in trouble. Test, on copies held out of
both trainings: for every AV fork choose argmin_a Q over the K actions and read the true outcome of that action from its branch.

    python -m nets.counterfactual            # ~3 min -> results/nets/counterfactual.json
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import common as CM

OUT = "results/nets"


def features(obs, u):
    return np.concatenate([obs, u], 1).astype(np.float32)


def loss(ps, b):
    tot = 0.0
    for p in ps:
        y = CM.mlp(p, b["x"])
        tot = tot + optax.sigmoid_binary_cross_entropy(y[:, 0], b["y"]).mean() + 0.1 * optax.huber_loss(y[:, 1], b["m"]).mean()
    return tot / len(ps)


def train(x, y, m, seed=0, n_models=3, steps=4000):
    key = jax.random.PRNGKey(seed)
    ps = [CM.init_mlp(jax.random.fold_in(key, i), [x.shape[1], 128, 128, 2]) for i in range(n_models)]
    return CM.fit(loss, ps, {"x": x, "y": y.astype(np.float32), "m": m.astype(np.float32)}, steps=steps, seed=seed)


def predict(ps, x):
    return np.asarray(jnp.mean(jnp.stack([jax.nn.sigmoid(CM.mlp(p, jnp.asarray(x))[:, 0]) for p in ps]), 0))


def main(out_dir=OUT, quick=False):
    steps = 1000 if quick else 4000
    d = np.load(f"{out_dir}/forks.npz")
    deltas = d["deltas"]
    K = len(deltas)
    obs, up, bad, env = d["obs"], d["u_prior"], d["bad"].astype(bool), d["env"]
    marg = np.clip(d["margin"], -10, 30) / 10.0
    test = env >= CM.split_env(env)
    tr = ~test
    # interventional rows: every state x every action
    xi = np.concatenate([features(obs[tr], up[tr] + deltas[k]) for k in range(K)])
    yi = np.concatenate([bad[tr, k] for k in range(K)])
    mi = np.concatenate([marg[tr, k] for k in range(K)])
    # observational rows: the state and what the policy did (the unperturbed branch, index K)
    xo, yo, mo = features(obs[tr], up[tr]), bad[tr, K], marg[tr, K]
    # equal compute and equal number of rows: the observational set is repeated K times by the sampler (same steps)
    q_int, c_int = train(xi, yi, mi, steps=steps)
    q_obs, c_obs = train(xo, yo, mo, steps=steps)
    CM.save_params(f"{out_dir}/critic_params.pkl", {"interventional": q_int, "observational": q_obs}, deltas=deltas)
    res = {"n_train_states": int(tr.sum()), "n_test_states": int(test.sum()), "bad_rate_per_action": bad.mean(0).tolist(), "actions": deltas.tolist()}
    pi = np.stack([predict(q_int, features(obs[test], up[test] + deltas[k])) for k in range(K)], 1)
    po = np.stack([predict(q_obs, features(obs[test], up[test] + deltas[k])) for k in range(K)], 1)
    truth = bad[test, :K]
    g = env[test] * 10_000 + d["cycle"][test]
    rng = np.random.default_rng(0)
    choice = {"as is (the policy)": np.zeros(test.sum(), int), "random action": rng.integers(0, K, test.sum()),
              "observational critic": po.argmin(1), "interventional critic": pi.argmin(1), "oracle (best true action)": truth.argmin(1)}
    res["choices"] = {}
    risky = truth.any(1)
    for name, k in choice.items():
        r = truth[np.arange(len(k)), k]
        res["choices"][name] = {"bad_rate": CM.boot(r, groups=env[test]), "bad_rate_risky": CM.boot(r[risky], groups=env[test][risky]),
                                "action_share": np.bincount(k, minlength=K).tolist()}
        print(name, {a: [round(x, 4) for x in b] for a, b in res["choices"][name].items() if a.startswith("bad")}, flush=True)
    res["auroc"] = {"interventional": [CM.auroc(pi[:, k], truth[:, k]) for k in range(K)], "observational": [CM.auroc(po[:, k], truth[:, k]) for k in range(K)]}
    # does the critic know what each action changes? predicted vs true paired effect of action k against acting as is
    cal = {}
    for name, p in (("interventional", pi), ("observational", po)):
        dp = (p[:, 1:] - p[:, :1]).ravel()
        dt = (truth[:, 1:].astype(float) - truth[:, :1]).ravel()
        q = np.quantile(dp, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, dp, side="right") - 1, 0, 9)
        cal[name] = {"pred": [float(dp[b == i].mean()) for i in range(10)], "true": [float(dt[b == i].mean()) for i in range(10)],
                     "corr": float(np.corrcoef(dp, dt)[0, 1]),
                     "mean_pred_effect": [float((p[:, k] - p[:, 0]).mean()) for k in range(K)]}
    cal["true_mean_effect"] = [float((truth[:, k].astype(float) - truth[:, 0]).mean()) for k in range(K)]
    res["effect_calibration"] = cal
    res["curves"] = {"interventional": c_int[::40].tolist(), "observational": c_obs[::40].tolist()}
    print("AUROC", {k: [round(x, 3) for x in v] for k, v in res["auroc"].items()})
    print("mean effect of each action vs as-is: true", np.round(cal["true_mean_effect"], 4), "int", np.round(cal["interventional"]["mean_pred_effect"], 4),
          "obs", np.round(cal["observational"]["mean_pred_effect"], 4))
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/counterfactual.json", "w") as f:
        json.dump(res, f)
    np.savez_compressed(f"{out_dir}/counterfactual_preds.npz", pi=pi, po=po, truth=truth, env=env[test])
    res["budget"] = budget_curves(out_dir=out_dir)
    print("bad-outcome rate at an intervention budget of", res["budget"]["budgets"], {k: np.round(v, 5).tolist() for k, v in res["budget"].items() if k != "budgets"})
    with open(f"{out_dir}/counterfactual.json", "w") as f:
        json.dump(res, f)


def budget_curves(budgets=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0), out_dir=OUT):
    """A fairer decision test than argmin: acting differently has a cost (braking harder everywhere is safe but slow), so give each
    critic a budget. It may change the policy's action at a fraction b of the moments: those where it predicts the largest risk
    reduction, using its best action there; everywhere else the AV acts as the policy did. Random: the same budget spent on random
    moments with the action that is safest on average (brake harder)."""
    z = np.load(f"{out_dir}/counterfactual_preds.npz")
    truth = z["truth"].astype(float)
    n = len(truth)
    out = {"budgets": list(budgets)}
    for name, p in (("interventional", z["pi"]), ("observational", z["po"])):
        gain = p[:, 0] - p.min(1)
        k = p.argmin(1)
        order = np.argsort(-gain)
        rates = []
        for b in budgets:
            act = np.zeros(n, bool)
            act[order[:int(round(b * n))]] = True
            chosen = np.where(act, k, 0)
            rates.append(float(truth[np.arange(n), chosen].mean()))
        out[name] = rates
    rng = np.random.default_rng(0)
    out["random_moments_brake"] = []
    for b in budgets:
        act = rng.uniform(size=n) < b
        out["random_moments_brake"].append(float(np.where(act, truth[:, 1], truth[:, 0]).mean()))
    best = truth.min(1)
    gain_true = truth[:, 0] - best
    order = np.argsort(-gain_true)
    out["oracle"] = []
    for b in budgets:
        act = np.zeros(n, bool)
        act[order[:int(round(b * n))]] = True
        out["oracle"].append(float(np.where(act, best, truth[:, 0]).mean()))
    out["as_is"] = float(truth[:, 0].mean())
    return out


if __name__ == "__main__":
    main()
