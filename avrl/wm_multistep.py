"""Idea 4: a world model that looks back and is trained on its own rollouts.

`avrl/worldmodel.py` fits (obs_t, u_t) -> obs_{t+1} one step at a time and, in docs/WORLD_MODELS.md, its 3 s open-loop gap error
is too large to train a policy inside it: errors compound because the model never sees its own mistakes in training, and from the
AV's 22 features alone the car ahead's next move is not predictable. Two changes, the ones that file lists as next steps:

  history   the input is the last H (obs, action) pairs, so the model can see the leader's trend
  rollout   the loss is summed over K steps of the model's own (soft) predictions fed back as inputs, teacher-forced on the
            logged actions:   L = sum_k gamma^k l(o_hat_{t+k+1}, o_{t+k+1}),   o_hat_{t+k+1} = f(o_hat_{t+k}, ..., u_{t+k})

hist = 1, K = 1 is the one-step model (same Huber loss), so the four corners of {1, 3} x {1, 10} are an ablation on the same data
and the same members. Evaluated on held-out vehicles: open-loop error of speed and gap after 1 s and 3 s, for all windows and for
"following" windows (a leader all along and the speed changes).

    python -m avrl.wm_multistep                 # ~10 min on 4 CPU cores -> results/wm_multistep/results.json, wm_multistep.png
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import env as E
from . import ppo
from . import worldmodel as W

OUT = "results/wm_multistep"
CONT, BIN, LIGHT = jnp.array(W.CONT), jnp.array(W.BINARY), jnp.array(W.LIGHT)
NC, NB = len(W.CONT), len(W.BINARY)
PLACEHOLDER = ((11, 1.2), (13, 0.0), (14, 0.0), (15, 0.0), (16, 1.0), (19, 0.0), (20, 2.0))  # what observe() writes without a leader
D = E.OBS_DIM


# ---------------------------------------------------------------- windows
def sort_rows(rows):
    return rows[np.lexsort((rows[:, 1], rows[:, 0]))]


def chain(rows):
    """c[j]: row j+1 is the same vehicle on the same segment 0.1 s later."""
    return (rows[1:, 0] == rows[:-1, 0]) & (rows[1:, 2] == rows[:-1, 2]) & (np.abs(rows[1:, 1] - rows[:-1, 1] - 0.1) < 0.02)


def window_starts(rows, length):
    """Start rows i with rows i .. i+length-1 an unbroken chain."""
    cs = np.concatenate([[0], np.cumsum(chain(rows))])
    i = np.arange(len(rows) - length + 1)
    return i[cs[i + length - 1] - cs[i] == length - 1]


def split_by_vehicle(rows, frac=0.2, seed=0):
    ids = np.unique(rows[:, 0])
    test = set(np.random.default_rng(seed).choice(ids, int(len(ids) * frac), replace=False).tolist())
    return np.array([r in test for r in rows[:, 0]])


# ---------------------------------------------------------------- model
def init(key, hist, hidden=128, n_models=3):
    return [ppo.init_mlp(k, [hist * (D + 1), hidden, hidden, NC + NB + 3]) for k in jax.random.split(key, n_models)]


def assemble(o, cont, pb, pl, hard):
    """Write predictions into a full observation. hard: thresholded flags, argmax light (evaluation); soft: probabilities
    (training, so that the rollout loss can reach back through the inputs)."""
    o2 = o.at[..., CONT].set(cont)
    if hard:
        o2 = o2.at[..., BIN].set((pb > 0.5).astype(o.dtype)).at[..., LIGHT].set(jax.nn.one_hot(jnp.argmax(pl, -1), 3))
        no = (o2[..., 10] < 0.5).astype(o.dtype)
    else:
        o2 = o2.at[..., BIN].set(pb).at[..., LIGHT].set(pl)
        no = 1.0 - o2[..., 10]
    for i, val in PLACEHOLDER:
        o2 = o2.at[..., i].set(no * val + (1 - no) * o2[..., i])
    return o2.at[..., 12].set(no * o2[..., 0] + (1 - no) * o2[..., 12])


def heads(y):
    return y[..., :NC], y[..., NC:NC + NB], y[..., NC + NB:]


def rollout(models, norm, o_hist, u_seq, hard, mean_over_members):
    """o_hist [B, H, D] true observations before the first prediction, u_seq [B, H + K, ...] actions u_0 .. u_{H+K-1}.
    Returns the predicted observations for steps H .. H+K-1 (K of them) and, per step, the logits for the losses.
    mean_over_members: average the members' predictions at each step (the model as used); otherwise each member feeds back
    its own prediction (the model as trained)."""
    mu, sd = norm
    B, H = o_hist.shape[:2]
    K = u_seq.shape[1] - H
    n = len(models)
    buf = jnp.concatenate([o_hist, u_seq[:, :H, None]], -1)  # [B, H, D + 1], newest last
    bufs = [buf] * (1 if mean_over_members else n)

    preds, logits = [], []
    for k in range(K):
        if mean_over_members:
            ys = [ppo.mlp(m, bufs[0].reshape(B, -1)) for m in models]
            dc = jnp.mean(jnp.stack([heads(y)[0] for y in ys]), 0)
            pb = jnp.mean(jnp.stack([jax.nn.sigmoid(heads(y)[1]) for y in ys]), 0)
            pl = jnp.mean(jnp.stack([jax.nn.softmax(heads(y)[2]) for y in ys]), 0)
            cur = bufs[0][:, -1, :D]
            o2 = assemble(cur, cur[..., CONT] + dc * sd + mu, pb, pl, hard)
            nxt = [o2]
            logits.append(None)
        else:
            nxt, lg = [], []
            for m, b in zip(models, bufs):
                y = ppo.mlp(m, b.reshape(B, -1))
                dc, lb, ll = heads(y)
                cur = b[:, -1, :D]
                nxt.append(assemble(cur, cur[..., CONT] + dc * sd + mu, jax.nn.sigmoid(lb), jax.nn.softmax(ll), hard))
                lg.append((lb, ll))
            logits.append(lg)
        preds.append(nxt)
        u_next = u_seq[:, H + k, None]
        bufs = [jnp.concatenate([b[:, 1:], jnp.concatenate([o, u_next], -1)[:, None]], 1) for b, o in zip(bufs, nxt)]
    return preds, logits


def member_loss(m, norm, o_true, u_seq, hist, K, gamma):
    """One member's loss on a batch of windows [B, hist + K, D]: Huber on the continuous features (in units of one-step delta
    SD), BCE on flags, CE on the light, summed over the K rollout steps with weights gamma^k."""
    mu, sd = norm
    preds, logits = rollout([m], norm, o_true[:, :hist], u_seq, False, False)
    total = 0.0
    for k in range(K):
        o_hat, (lb, ll) = preds[k][0], logits[k][0]
        tgt = o_true[:, hist + k]
        err = (o_hat[:, CONT] - tgt[:, CONT]) / sd
        l_c = optax.huber_loss(err, jnp.zeros_like(err), delta=1.0).mean()
        l_b = optax.sigmoid_binary_cross_entropy(lb, tgt[:, BIN]).mean()
        l_l = optax.softmax_cross_entropy(ll, tgt[:, LIGHT]).mean()
        total = total + gamma ** k * (l_c + l_b + l_l)
    return total / sum(gamma ** k for k in range(K))


def fit(rows, hist, K, key, steps=2500, batch=512, lr=1e-3, n_models=3, gamma=0.9):
    """rows: sorted training rows. Members are trained on their own bootstrap batches, each on its own rollouts."""
    L = hist + K
    starts = jnp.asarray(window_starts(rows, L))
    R = jnp.asarray(rows)
    ci = np.nonzero(chain(rows))[0]
    d1 = rows[ci + 1, 3:3 + D][:, W.CONT] - rows[ci, 3:3 + D][:, W.CONT]
    norm = (jnp.asarray(d1.mean(0)), jnp.asarray(d1.std(0) + 1e-3))
    opt = optax.adam(lr)
    models = init(key, hist, n_models=n_models)

    def train_one(m, k):
        def body(c, k):
            m, st = c
            i = starts[jax.random.randint(k, (batch,), 0, starts.shape[0])]
            w = R[i[:, None] + jnp.arange(L)]
            l, g = jax.value_and_grad(member_loss)(m, norm, w[..., 3:3 + D], w[..., 3 + D], hist, K, gamma)
            upd, st = opt.update(g, st, m)
            return (optax.apply_updates(m, upd), st), l

        return jax.lax.scan(body, (m, opt.init(m)), jax.random.split(k, steps))

    train_one = jax.jit(train_one)
    out, curves = [], []
    for i, m in enumerate(models):
        (m, _), ls = train_one(m, jax.random.fold_in(key, i))
        out.append(m)
        curves.append(np.asarray(ls))
    return out, norm, np.mean(curves, 0)


def evaluate(models, norm, rows, hist, horizon=30, n=600, seed=0):
    """Open-loop errors after 1 s and 3 s on windows from held-out vehicles, the same windows for every model (they are drawn
    with the longest history, so a model with a shorter history just ignores the earlier rows)."""
    HMAX = 3
    L = HMAX + horizon
    starts = window_starts(rows, L)
    rng = np.random.default_rng(seed)
    out = {}
    for subset in ("all", "following"):
        s = starts
        if subset == "following":
            w = rows[s[:, None] + np.arange(L)]
            ok = (w[..., 3 + 10] > 0.5).all(1) & (np.abs(w[:, -1, 3] - w[:, HMAX - 1, 3]) * E.V_SCALE > 1.0)
            s = s[ok]
        s = rng.permutation(s)[:n]
        if len(s) < 2:
            out[f"{subset}/n"] = int(len(s))
            continue
        w = jnp.asarray(rows[s[:, None] + np.arange(L)])
        o_true, u = w[..., 3:3 + D], w[..., 3 + D]
        o_h, u_h = o_true[:, HMAX - hist:HMAX], u[:, HMAX - hist:]
        preds, _ = jax.jit(lambda mm, o_h, u_h: rollout(mm, norm, o_h, u_h, True, True))(models, o_h, u_h)
        traj = jnp.stack([p[0] for p in preds], 1)  # [n, horizon, D]
        tgt = o_true[:, HMAX:]
        for name, i, sc in (("speed_kmh", 0, E.V_SCALE * 3.6), ("gap_m", 11, 50.0)):
            e = np.asarray(jnp.abs(traj[..., i] - tgt[..., i])) * sc
            for sec in (1, 3):
                if sec * 10 > horizon:
                    continue
                x = e[:, sec * 10 - 1]
                out[f"{subset}/{name}@{sec}s"] = {"mae": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x)))}
        out[f"{subset}/n"] = int(len(s))
    return out


CONFIGS = {"one-step": (1, 1), "history": (3, 1), "rollout": (1, 10), "history+rollout": (3, 10)}


def main(steps=2500, n_models=3):
    os.makedirs(OUT, exist_ok=True)
    rows = sort_rows(W.load_rows("data/world.bin"))
    te = split_by_vehicle(rows)
    tr_rows, te_rows = rows[~te], rows[te]
    res = {"train_rows": int(len(tr_rows)), "test_rows": int(len(te_rows)), "steps": steps, "configs": {}}
    for name, (hist, K) in CONFIGS.items():
        t0 = time.time()
        models, norm, curve = fit(tr_rows, hist, K, jax.random.PRNGKey(0), steps=steps, n_models=n_models)
        r = evaluate(models, norm, te_rows, hist)
        res["configs"][name] = {"hist": hist, "K": K, "eval": r, "curve": curve[::25].tolist(), "seconds": round(time.time() - t0)}
        print(name, {k: round(v["mae"], 3) for k, v in r.items() if isinstance(v, dict)}, f"{time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:3]))
