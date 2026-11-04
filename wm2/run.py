"""Train and evaluate the agent-token world model on Argoverse 2 (idea 3) and report the learned human driver's accuracy (idea 5).

    python -m wm2.run [scenario_root] [steps]         # ~8 min on 4 CPU cores -> results/wm2/results.json, params_<variant>.pkl

Held-out scenarios (20 %). One-step: MAE of acceleration and yaw rate, and the log-likelihood, for
  zero (constant velocity) / persistence (last step's acceleration) / IDM fitted to the training data / none / leader / all.
Closed loop: from step 49 of each held-out scenario every car, bus and cyclist that is tracked through step 79 is driven by the model
for 3 s (the rest replay the log), and its position error against the log is the ADE / FDE.
"""
import json
import os
import pickle
import sys
import time

import jax
import numpy as np

from . import data as D
from . import model as M

OUT = "results/wm2"
ROOT = "/tmp/claude-0/-home-user/55bcdaf2-8ef3-51cf-9658-ec3ffd940c9e/scratchpad/av2"
SC = np.array([3.0, 0.2])


def gauss_nll(mu, ls, y):
    return float((0.5 * ((y - mu) / np.exp(ls)) ** 2 + ls + 0.5 * np.log(2 * np.pi)).mean())


def one_step(d, tr, te, steps):
    res, params = {}, {}
    y = d["Y"][te]
    lead = d["lead"][te].any(1)
    zero, persist = np.zeros_like(y), np.stack([d["S"][te][:, 5] * 3.0, d["S"][te][:, 6]], 1)
    q, _ = M.fit_idm(d, tr)
    idm_a = M.idm_predict(q, d, te)
    rows = {"zero": zero, "persistence": persist, "idm_fitted": np.stack([idm_a, persist[:, 1]], 1)}
    for name, pred in rows.items():
        res[name] = {"mae_a": float(np.abs(pred[:, 0] - y[:, 0]).mean()), "mae_w": float(np.abs(pred[:, 1] - y[:, 1]).mean()),
                     "mae_a_leader": float(np.abs(pred[lead, 0] - y[lead, 0]).mean())}
    res["idm_fitted"]["params"] = q
    for variant in ("none", "leader", "all"):
        t0 = time.time()
        p, curve = M.fit(d, tr, variant, steps=steps)
        mu, ls = M.predict(p, d, te, variant)
        res[variant] = {"mae_a": float(np.abs(mu[:, 0] - y[:, 0]).mean()), "mae_w": float(np.abs(mu[:, 1] - y[:, 1]).mean()),
                        "mae_a_leader": float(np.abs(mu[lead, 0] - y[lead, 0]).mean()),
                        "nll": gauss_nll(mu / SC, ls - np.log(SC), y / SC), "seconds": round(time.time() - t0), "curve": curve[::50].tolist()}
        params[variant] = jax.tree.map(np.asarray, p)
        print(variant, {k: round(v, 4) for k, v in res[variant].items() if isinstance(v, float)}, flush=True)
    return res, params, q


def rollout_scene(sc, predict, t0=49, H=30, trace=None):
    """Drive the agents tracked over [t0-1, t0+H] with `predict(S, NB, mask, lead) -> (a, w)`; others replay. -> per-agent position error [n, H]."""
    ag = np.where(sc["ok"][:, t0 - 1:t0 + H + 1].all(1) & (sc["kind"] <= 2) & (D.speed(sc, t0) > 1.0))[0]
    if len(ag) == 0:
        return None
    pos, vel, head = sc["pos"][:, t0].copy(), sc["vel"][:, t0].copy(), sc["head"][:, t0].copy()
    spd = np.hypot(*vel.T)
    prev_a = (spd - np.hypot(*sc["vel"][:, t0 - 1].T)) / D.DT
    prev_w = D.wrap(head - sc["head"][:, t0 - 1]) / D.DT
    err = np.zeros((len(ag), H))
    if trace is not None:
        trace.append(pos[ag].copy())
    for h in range(H):
        ok = sc["ok"][:, t0 + h]
        S, NB, mask, lead = D.features(sc, t0 + h, ag, state=(pos, vel, head, ok, prev_a, prev_w))
        a, w = predict(S, NB, mask, lead)
        a, w = np.clip(a, -9, 5), np.clip(w, -1.5, 1.5)
        sp = np.maximum(np.hypot(*vel[ag].T) + a * D.DT, 0.0)
        hd = head[ag] + w * D.DT
        npos, nvel, nhead = sc["pos"][:, t0 + h + 1].copy(), sc["vel"][:, t0 + h + 1].copy(), sc["head"][:, t0 + h + 1].copy()   # replayed agents
        nvel[ag] = np.stack([sp * np.cos(hd), sp * np.sin(hd)], 1)
        npos[ag] = pos[ag] + 0.5 * (vel[ag] + nvel[ag]) * D.DT
        nhead[ag] = hd
        prev_a, prev_w = prev_a.copy(), prev_w.copy()
        prev_a[ag], prev_w[ag] = a, w
        pos, vel, head = npos, nvel, nhead
        err[:, h] = np.hypot(*(pos[ag] - sc["pos"][ag, t0 + h + 1]).T)
        if trace is not None:
            trace.append(pos[ag].copy())
    return err


def closed_loop(d, te_files, params, q):
    fns = {"constant_velocity": lambda S, NB, m, l: (np.zeros(len(S)), np.zeros(len(S))),
           "constant_turn_rate": lambda S, NB, m, l: (np.zeros(len(S)), S[:, 6]),
           "idm_fitted": lambda S, NB, m, l: (_idm_from_tokens(q, S, NB, l), S[:, 6])}
    for variant, p in params.items():
        fns[f"token_{variant}"] = (lambda variant, p: lambda S, NB, m, l: _model(p, variant, S, NB, m, l))(variant, p)
    errs = {k: [] for k in fns}
    for f in te_files:
        sc = D.load_scene(f)
        for k, fn in fns.items():
            e = rollout_scene(sc, fn)
            if e is not None:
                errs[k].append(e)
    out = {}
    for k, v in errs.items():
        e = np.concatenate(v)
        out[k] = {"ade": float(e.mean()), "fde": float(e[:, -1].mean()), "fde_1s": float(e[:, 9].mean()), "n_agents": int(len(e)), "fde_per_agent": e[:, -1].tolist()}
        print(k, {a: round(b, 3) for a, b in out[k].items() if isinstance(b, float)}, out[k]["n_agents"], flush=True)
    return out


def _model(p, variant, S, NB, mask, lead):
    mu, _ = M.apply(p, S, NB, M.visible(mask, lead, variant))
    mu = np.asarray(mu) * SC
    return mu[:, 0], mu[:, 1]


def _idm_from_tokens(q, S, NB, lead):
    has = lead.any(1)
    j = lead.argmax(1)
    nb = NB[np.arange(len(S)), j]
    v = S[:, 4] * 10.0
    gap = np.hypot(nb[:, 0] * 20.0, nb[:, 1] * 20.0) - 4.5
    vl = v + nb[:, 2] * 10.0
    return np.clip(np.where(has, M.idm(v, gap, vl, **q), q["a_max"] * (1 - (v / q["v0"]) ** 4)), -9, 5)


def main(root=ROOT, steps=5000):
    os.makedirs(OUT, exist_ok=True)
    d = D.build(root)
    n_sc = len(d["files"])
    rng = np.random.default_rng(0)
    test_sc = set(rng.choice(n_sc, n_sc // 5, replace=False).tolist())
    te = np.where(np.isin(d["scene"], list(test_sc)))[0]
    tr = np.where(~np.isin(d["scene"], list(test_sc)))[0]
    print(f"{n_sc} scenes, {len(tr)} train / {len(te)} test samples", flush=True)
    res, params, q = one_step(d, tr, te, steps)
    for v, p in params.items():
        with open(os.path.join(OUT, f"params_{v}.pkl"), "wb") as f:
            pickle.dump(p, f)
    res = {"n_scenes": n_sc, "n_train": int(len(tr)), "n_test": int(len(te)), "steps": steps, "one_step": res}
    params = {k: jax.tree.map(jax.numpy.asarray, v) for k, v in params.items()}
    res["closed_loop"] = closed_loop(d, [d["files"][i] for i in sorted(test_sc)], params, q)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or [ROOT]), *(int(x) for x in sys.argv[2:3]))
