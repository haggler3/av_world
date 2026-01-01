"""A small transformer over agent tokens, and the baselines it is compared with.

tokens = [self] + [K neighbours]; two pre-norm attention blocks (padded neighbours masked); the output of the self token is the mean and
log-SD of the agent's next longitudinal acceleration and yaw rate. Variants differ only in which neighbours the self token may see:
  all     every neighbour within 30 m                    (the agent-token model)
  leader  only the nearest agent ahead in lane           (what a car-following model sees)
  none    nobody                                         (a model of the agent alone)
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import data as D

HEADS, DM, LAYERS = 4, 64, 2


def _lin(key, i, o):
    return {"w": jax.random.normal(key, (i, o)) / np.sqrt(i), "b": jnp.zeros(o)}


def init(key, d=DM, layers=LAYERS):
    ks = iter(jax.random.split(key, 6 + 6 * layers))
    p = {"emb_s": _lin(next(ks), D.SELF_DIM, d), "emb_n": _lin(next(ks), D.NB_DIM, d), "out": _lin(next(ks), d, 4), "blocks": []}
    for _ in range(layers):
        p["blocks"].append({"qkv": _lin(next(ks), d, 3 * d), "proj": _lin(next(ks), d, d), "f1": _lin(next(ks), d, 2 * d), "f2": _lin(next(ks), 2 * d, d),
                            "g1": jnp.ones(d), "g2": jnp.ones(d)})
    p["out"]["w"] = p["out"]["w"] * 0.1
    return p


def _ln(x, g):
    x = x - x.mean(-1, keepdims=True)
    return g * x / jnp.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5)


def _dense(l, x):
    return x @ l["w"] + l["b"]


def apply(p, S, NB, vis):
    """S [B, 7], NB [B, K, 11], vis [B, K] bool (neighbours the self token may attend to). -> mean [B, 2], log-SD [B, 2]."""
    x = jnp.concatenate([_dense(p["emb_s"], S)[:, None], _dense(p["emb_n"], NB)], 1)      # [B, 1+K, d]
    keep = jnp.concatenate([jnp.ones((S.shape[0], 1), bool), vis], 1)                      # [B, 1+K]
    B, T, d = x.shape
    for b in p["blocks"]:
        h = _ln(x, b["g1"])
        q, k, v = jnp.split(_dense(b["qkv"], h), 3, -1)
        sp = lambda z: z.reshape(B, T, HEADS, d // HEADS).transpose(0, 2, 1, 3)
        att = (sp(q) @ sp(k).transpose(0, 1, 3, 2)) / np.sqrt(d // HEADS)
        att = jnp.where(keep[:, None, None, :], att, -1e9)
        h = (jax.nn.softmax(att, -1) @ sp(v)).transpose(0, 2, 1, 3).reshape(B, T, d)
        x = x + _dense(b["proj"], h)
        x = x + _dense(b["f2"], jax.nn.gelu(_dense(b["f1"], _ln(x, b["g2"]))))
    y = _dense(p["out"], x[:, 0])
    return y[:, :2], jnp.clip(y[:, 2:], -3.0, 1.5)


def visible(mask, lead, variant):
    return {"all": mask, "leader": lead, "none": np.zeros_like(mask)}[variant]


def nll(p, S, NB, vis, Y):
    mu, ls = apply(p, S, NB, vis)
    return (0.5 * ((Y - mu) / jnp.exp(ls)) ** 2 + ls).mean()


def fit(d, idx, variant, steps=4000, batch=512, lr=2e-3, seed=0):
    """Train on the samples `idx`. Targets are scaled so both heads are O(1). -> params, curve"""
    key = jax.random.PRNGKey(seed)
    p = init(key)
    sc = jnp.array([3.0, 0.2])
    S, NB, Y = jnp.asarray(d["S"][idx]), jnp.asarray(d["NB"][idx]), jnp.asarray(d["Y"][idx]) / sc
    V = jnp.asarray(visible(d["mask"], d["lead"], variant)[idx])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=1e-4))

    @jax.jit
    def run(p, key):
        def body(c, k):
            p, st = c
            i = jax.random.randint(k, (batch,), 0, S.shape[0])
            l, g = jax.value_and_grad(nll)(p, S[i], NB[i], V[i], Y[i])
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), l

        return jax.lax.scan(body, (p, opt.init(p)), jax.random.split(key, steps))

    (p, _), curve = run(p, jax.random.fold_in(key, 1))
    return p, np.asarray(curve)


def predict(p, d, idx, variant):
    sc = np.array([3.0, 0.2])
    f = jax.jit(apply)
    mu, ls = f(p, jnp.asarray(d["S"][idx]), jnp.asarray(d["NB"][idx]), jnp.asarray(visible(d["mask"], d["lead"], variant)[idx]))
    return np.asarray(mu) * sc, np.asarray(ls) + np.log(sc)


# ---------------------------------------------------------------- the car-following baseline
def idm(v, gap, vl, T, a_max, b, s0, v0):
    s_star = s0 + np.maximum(0.0, v * T + v * (v - vl) / (2 * np.sqrt(a_max * b)))
    return a_max * (1 - (v / v0) ** 4 - (s_star / np.maximum(gap, 0.5)) ** 2)


def leader_inputs(d, idx):
    """own speed, bumper gap (centre distance - 4.5 m), leader speed, has_leader: from the nearest-ahead token."""
    lead = d["lead"][idx]
    has = lead.any(1)
    j = lead.argmax(1)
    nb = d["NB"][idx][np.arange(len(idx)), j]
    v = d["S"][idx][:, 4] * 10.0
    gap = np.hypot(nb[:, 0] * 20.0, nb[:, 1] * 20.0) - 4.5
    vl = v + nb[:, 2] * 10.0                                           # the leader's speed along our heading
    return v, gap, vl, has


def fit_idm(d, idx, n=600, seed=0):
    """Random search over IDM parameters for the lowest acceleration MAE on the training samples; free-road acceleration when no leader."""
    rng = np.random.default_rng(seed)
    v, gap, vl, has = leader_inputs(d, idx)
    a = d["Y"][idx][:, 0]
    best, bp = np.inf, None
    for _ in range(n):
        q = dict(T=rng.uniform(0.3, 2.5), a_max=rng.uniform(0.5, 3.5), b=rng.uniform(0.5, 4.0), s0=rng.uniform(0.5, 4.0), v0=rng.uniform(8, 20))
        pred = np.where(has, idm(v, gap, vl, **q), q["a_max"] * (1 - (v / q["v0"]) ** 4))
        e = np.abs(np.clip(pred, -9, 5) - a).mean()
        if e < best:
            best, bp = e, q
    return bp, best


def idm_predict(q, d, idx):
    v, gap, vl, has = leader_inputs(d, idx)
    return np.clip(np.where(has, idm(v, gap, vl, **q), q["a_max"] * (1 - (v / q["v0"]) ** 4)), -9, 5)
