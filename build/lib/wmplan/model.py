"""A latent world model of the AV's surroundings: encode the scene, unroll it in latent space under an action sequence,
decode the futures that matter for planning.

    z_t      = enc(obs_t)                                   48-d latent of the AV's scene (28 features: own motion, the
                                                            leader, the stop target, the signal, right of way, the
                                                            junction's conflicting traffic, the merge situation)
    z_t+1    = z_t + dyn(z_t, u_t)                          the latent unfolds under the action u = (accel, gap acceptance)
    decoders state:      obs_t, the five safety margins (barrier values: stop, leader, junction time-to-conflict,
                         merge gap behind / ahead)
             transition: reward, P(crash), P(near miss), P(red-light run) of taking u_t in z_t

Trained by unrolling 15 steps (3 s) from each encoded start on windows of logged decisions, with every prediction
compared to what really happened, plus a consistency loss pulling the unrolled latent toward the encoding of the real
next scene. Held-out copy of the area for evaluation.

    python -m wmplan.model         # ~6 min -> results/wmplan/model.npz, model_eval.json
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

OUT = "results/wmplan"
H, ZD, HID = 15, 48, 192
OBS, ACT, NMARG = 28, 2, 5
R_SCALE = 10.0  # rewards are ~0.05 per step
HOLDOUT_ENV = 7


def _layer(key, i, o):
    return {"w": jax.random.normal(key, (i, o)) * np.sqrt(2.0 / i), "b": jnp.zeros(o)}


def init(key, stats=None):
    ks = iter(jax.random.split(key, 16))
    mlp = lambda sizes: [_layer(next(ks), a, b) for a, b in zip(sizes[:-1], sizes[1:])]  # noqa: E731
    return {"enc": mlp([OBS, HID, HID, ZD]), "dyn": mlp([ZD + ACT, HID, HID, ZD]), "state": mlp([ZD, HID, OBS + NMARG]),
            "trans": mlp([ZD + ACT, HID, 4]), "norm": [stats or {"w": jnp.zeros(OBS + NMARG), "b": jnp.ones(OBS + NMARG)}]}


def run(layers, x):
    for i, l in enumerate(layers):
        x = x @ l["w"] + l["b"]
        if i < len(layers) - 1:
            x = jax.nn.elu(x)
    return x


def norm(p, x, sl):
    """Standardise features with the training data's mean (norm.w) and std (norm.b)."""
    return (x - p["norm"][0]["w"][sl]) / p["norm"][0]["b"][sl]


def encode(p, obs):
    return run(p["enc"], norm(p, obs, slice(0, OBS)))


def act_in(u):
    return jnp.stack([jnp.clip(u[..., 0], -1.0, 1.0), jnp.tanh(u[..., 1])], -1)  # what the simulator does with u


def step(p, z, u):
    a = act_in(u)
    tr = run(p["trans"], jnp.concatenate([z, a], -1))  # reward, crash / near-miss / red-run logits
    z1 = z + run(p["dyn"], jnp.concatenate([z, a], -1))
    return z1, tr


def decode(p, z):
    """-> (obs, margins) in their original units."""
    out = run(p["state"], z) * p["norm"][0]["b"] + p["norm"][0]["w"]
    obs = out[..., :OBS]
    return obs.at[..., 0].set(jnp.maximum(obs[..., 0], 0.0)), out[..., OBS:]  # speed >= 0


def unroll(p, z0, us):
    """z0 [..., ZD], us [..., T, 2] -> latents [..., T+1, ZD], transition heads [..., T, 3]."""
    def body(z, u):
        z1, tr = step(p, z, u)
        return z1, (z1, tr)
    _, (zs, trs) = jax.lax.scan(body, z0, jnp.moveaxis(us, -2, 0))
    zs = jnp.concatenate([z0[None], zs], 0)
    return jnp.moveaxis(zs, 0, -2), jnp.moveaxis(trs, 0, -2)


def load_data(path=f"{OUT}/data.npz"):
    """Rows sorted by (copy, slot, time). A window starts at any row; step k of it is valid while rows i .. i+k are one
    vehicle's consecutive decisions, so a window may end early (at a crash or the end of a trip) and still teach."""
    d = dict(np.load(path))
    n = len(d["t"])
    same = (d["env"][1:] == d["env"][:-1]) & (d["slot"][1:] == d["slot"][:-1]) & (d["t"][1:] == d["t"][:-1] + 1) & (d["done"][:-1] == 0)
    d["cont"] = np.concatenate([same, [False]])  # row i continues into row i + 1
    d["n"] = n
    starts = np.nonzero(d["cont"][: n - H])[0]  # at least two steps
    return d, starts


def windows(d, idx):
    j = np.minimum(idx[:, None] + np.arange(H + 1)[None], d["n"] - 1)
    valid = np.cumprod(np.concatenate([np.ones((len(idx), 1), bool), d["cont"][j[:, :-1]]], 1), 1).astype(np.float32)
    f = lambda k: jnp.asarray(d[k][j].astype(np.float32))  # noqa: E731
    return {"obs": f("obs"), "u": f("u")[:, :H], "marg": f("margins"), "valid": jnp.asarray(valid),
            "rew": jnp.clip(f("reward")[:, :H], -0.5, 0.5) * R_SCALE,  # the dense part (crash / red penalties: the event heads)
            "crash": f("crash")[:, :H], "near": f("near")[:, :H], "red": f("red")[:, :H]}


def loss_fn(p, b, pos_w=30.0):
    z0 = encode(p, b["obs"][:, 0])
    zs, trs = unroll(p, z0, b["u"])
    o_hat, m_hat = decode(p, zs)
    v = b["valid"]
    vt = v[:, :H]
    wsum = lambda x, w: (x * w).sum() / jnp.maximum(w.sum(), 1.0)  # noqa: E731
    sd = p["norm"][0]["b"]
    w0 = v.at[:, 0].multiply(4.0)  # reconstructing the present scene well is the base of every prediction
    l_obs = wsum((((o_hat - b["obs"]) / sd[:OBS]) ** 2).mean(-1), w0)
    l_m = wsum((((m_hat - b["marg"]) / sd[OBS:]) ** 2).mean(-1), v)
    l_r = wsum((trs[..., 0] - b["rew"]) ** 2, vt)
    bce = lambda logit, y: -(pos_w * y * jax.nn.log_sigmoid(logit) + (1 - y) * jax.nn.log_sigmoid(-logit))  # noqa: E731
    l_c = wsum(bce(trs[..., 1], b["crash"]) + bce(trs[..., 2], b["near"]) + bce(trs[..., 3], b["red"]), vt)
    z_tgt = jax.lax.stop_gradient(encode(p, b["obs"][:, 1:]))
    l_z = wsum(((zs[:, 1:] - z_tgt) ** 2).mean(-1), v[:, 1:])
    return l_obs + 2.0 * l_m + l_r + 0.3 * l_c + 0.5 * l_z, (l_obs, l_m, l_r, l_c, l_z)


def save(p, path):
    flat = {f"{k}{i}_{w}": np.asarray(l[w]) for k, ls in p.items() for i, l in enumerate(ls) for w in ("w", "b")}
    np.savez(path, **flat)


def load(path=f"{OUT}/model.npz"):
    z = np.load(path)
    p = {}
    for k in ("enc", "dyn", "state", "trans", "norm"):
        n = len([f for f in z.files if f.startswith(k) and f.endswith("_w")])
        p[k] = [{"w": jnp.asarray(z[f"{k}{i}_w"]), "b": jnp.asarray(z[f"{k}{i}_b"])} for i in range(n)]
    return p


def evaluate(p, d, idx):
    """Prediction error vs horizon on held-out windows, against holding the current scene constant, and crash ranking."""
    b = windows(d, idx)
    z0 = encode(p, b["obs"][:, 0])
    zs, trs = unroll(p, z0, b["u"])
    o_hat, m_hat = decode(p, zs)
    res = {}
    v = np.asarray(b["valid"])
    rms = lambda e, k: float(np.sqrt((np.asarray(e) ** 2 * v[:, k]).sum() / v[:, k].sum()))  # noqa: E731
    ob = np.asarray(b["obs"])
    for name, j, scale, present in (("speed m/s", 0, 15.0, None), ("leader gap m", 9, 50.0, 8), ("stop gap m", 7, 100.0, 6)):
        for k in (0, 5, 10, 15):
            m = v[:, k] * (1.0 if present is None else ob[:, 0, present] * ob[:, k, present])  # the leader / stop target exists
            r = lambda e: float(np.sqrt((np.asarray(e) ** 2 * m).sum() / max(m.sum(), 1)))  # noqa: E731
            res[f"{name} @{k * 0.2:.0f}s"] = {"model": r(o_hat[:, k, j] - b["obs"][:, k, j]) * scale,
                                              "hold": r(b["obs"][:, 0, j] - b["obs"][:, k, j]) * scale}
    for i, name in enumerate(("stop margin", "leader margin", "junction margin", "merge gap behind", "merge gap ahead")):
        k = 10
        res[f"{name} @2s (RMS)"] = {"model": rms(m_hat[:, k, i] - b["marg"][:, k, i], k), "hold": rms(b["marg"][:, 0, i] - b["marg"][:, k, i], k)}
    vt = v[:, :H].ravel() > 0
    from scipy.stats import rankdata
    for name, plural, i in (("crash", "crashes", 1), ("red-light run", "red-light runs", 3)):
        y = np.asarray(b["crash" if i == 1 else "red"]).ravel()[vt]
        s = np.asarray(trs[..., i]).ravel()[vt]
        if y.sum() > 0:
            r = rankdata(s)
            res[f"{name} AUROC"] = float((r[y > 0].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum())))
            res[f"held-out {plural}"] = int(y.sum())
    return res


def train(steps=4000, batch=256, seed=0):
    d, starts = load_data()
    held = d["env"][starts] == HOLDOUT_ENV
    tr_idx, te_idx = starts[~held], starts[held]
    # oversample windows that contain a crash or near miss (rare, and what the planner most needs to foresee)
    ev = np.convolve(np.maximum(np.maximum(d["crash"], d["near"]), d["red"]).astype(float), np.ones(H), "full")[H - 1:H - 1 + d["n"]]
    hot = tr_idx[ev[tr_idx] > 0]
    print(f"{len(starts)} windows ({len(tr_idx)} train, {len(te_idx)} held out), {len(hot)} with a crash / near miss", flush=True)
    key = jax.random.PRNGKey(seed)
    x = np.concatenate([d["obs"][tr_idx].astype(np.float32), d["margins"][tr_idx].astype(np.float32)], 1)
    p = init(key, {"w": jnp.asarray(x.mean(0)), "b": jnp.asarray(np.maximum(x.std(0), 1e-2))})
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(1e-3, steps, 0.05), weight_decay=1e-5))
    st = opt.init(p)

    @jax.jit
    def upd(p, st, b):
        (l, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(p, b)
        g["norm"] = jax.tree.map(jnp.zeros_like, g["norm"])  # the statistics are fixed
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l, parts

    rng = np.random.default_rng(seed)
    t0 = time.time()
    for s in range(steps):
        idx = rng.choice(tr_idx, batch)
        if len(hot):
            idx[: batch // 8] = rng.choice(hot, batch // 8)
        p, st, l, parts = upd(p, st, windows(d, idx))
        if s % 500 == 0 or s == steps - 1:
            print(f"step {s}: loss {float(l):.4f} (obs {float(parts[0]):.4f} margins {float(parts[1]):.4f} reward {float(parts[2]):.4f} "
                  f"events {float(parts[3]):.4f} latent {float(parts[4]):.4f}) {time.time() - t0:.0f} s", flush=True)
    save(p, f"{OUT}/model.npz")
    te = np.random.default_rng(1).choice(te_idx, min(20000, len(te_idx)), replace=False)
    res = evaluate(p, d, te)
    with open(f"{OUT}/model_eval.json", "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 6000)
