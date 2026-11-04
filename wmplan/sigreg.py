"""Idea 7: keep a joint-embedding world model from collapsing with a Gaussian regulariser instead of tricks.

A JEPA-style model learns z = f(o) and a predictor g(z, u) ~ f(o'); the loss |g(f(o), u) - f(o')|^2 is minimised by a constant f.
SimNorm (wmplan/v2.py) and stop-gradients avoid that by construction. LeJEPA / LeWorldModel (Balestriero & LeCun 2025; Maes et al.
2026) add a term that pushes the embeddings towards an isotropic Gaussian, tested on random 1-D projections with the Epps-Pulley
statistic:

  EP(x) = N * integral |phi_hat(t) - exp(-t^2 / 2)|^2 exp(-t^2 / 2) dt,   phi_hat(t) = mean_n exp(i t x_n),   x = <z, direction>

SIGReg(z) = mean over M random unit directions of EP(projection). For z ~ N(0, I) its expectation is not zero but the constant
int (1 - exp(-t^2)) exp(-t^2 / 2) dt = sqrt(2 pi) - sqrt(2 pi / 3) = 1.06 (it does not grow with N); for embeddings that are constant, low-rank or
clustered it is hundreds to thousands, so the total loss  L = L_pred + lam * SIGReg(f(o))  has no collapsed minimum.

    python -m wmplan.sigreg            # collapse test on the corridor environment's transitions, ~1 min -> results/extras/sigreg.json
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from avrl import ppo

T_GRID = jnp.linspace(-5.0, 5.0, 41)


def epps_pulley(x):
    """x [N] standardised projections -> the (scaled) Epps-Pulley distance to N(0, 1) by trapezoid quadrature."""
    n = x.shape[0]
    ang = x[:, None] * T_GRID[None]
    re, im = jnp.cos(ang).mean(0), jnp.sin(ang).mean(0)
    w = jnp.exp(-0.5 * T_GRID ** 2)
    integrand = ((re - w) ** 2 + im ** 2) * w
    return n * jnp.trapezoid(integrand, T_GRID)


def sigreg(z, key, n_dirs=64):
    """z [N, d] -> scalar. Random unit directions, one fresh draw per call."""
    d = z.shape[-1]
    dirs = jax.random.normal(key, (d, n_dirs))
    dirs = dirs / jnp.linalg.norm(dirs, axis=0, keepdims=True)
    return jax.vmap(epps_pulley, in_axes=1)(z @ dirs).mean()


def effective_rank(z):
    """exp(entropy of the normalised singular values of the centred embeddings): d for isotropic, 1 for a line, ~0 for a point."""
    s = np.linalg.svd(np.asarray(z) - np.asarray(z).mean(0), compute_uv=False)
    if s.sum() < 1e-8:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def init(key, d_obs, d_z=8, hidden=64):
    k = jax.random.split(key, 2)
    return {"enc": ppo.init_mlp(k[0], [d_obs, hidden, hidden, d_z]), "pred": ppo.init_mlp(k[1], [d_z + 1, hidden, d_z])}


def encode(p, o):
    return ppo.mlp(p["enc"], o)


def predict(p, z, u):
    return z + ppo.mlp(p["pred"], jnp.concatenate([z, u[..., None]], -1))


def loss(p, o, u, o2, key, lam):
    z, z2 = encode(p, o), encode(p, o2)
    l_pred = ((predict(p, z, u) - z2) ** 2).mean()
    reg = sigreg(z, key) if lam > 0 else 0.0
    return l_pred + lam * reg, (l_pred, reg)


def train(data, lam, steps=3000, batch=512, seed=0):
    o, u, o2 = (jnp.asarray(x) for x in data[:3])
    key = jax.random.PRNGKey(seed)
    p = init(key, o.shape[-1])
    opt = optax.adam(1e-3)

    @jax.jit
    def run(p, key):
        def body(c, k):
            p, st = c
            k1, k2 = jax.random.split(k)
            i = jax.random.randint(k1, (batch,), 0, o.shape[0])
            (l, aux), g = jax.value_and_grad(loss, has_aux=True)(p, o[i], u[i], o2[i], k2, lam)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), jnp.stack([aux[0], jnp.asarray(aux[1], jnp.float32)])

        return jax.lax.scan(body, (p, opt.init(p)), jax.random.split(key, steps))

    (p, _), curve = run(p, key)
    return p, np.asarray(curve)


def probe_r2(z, target, n_train):
    """Held-out R^2 of a ridge regression z -> target (is the information still in the embedding?)."""
    z, t = np.asarray(z), np.asarray(target)
    A = np.concatenate([z, np.ones((len(z), 1))], 1)
    w = np.linalg.solve(A[:n_train].T @ A[:n_train] + 1e-3 * np.eye(A.shape[1]), A[:n_train].T @ t[:n_train])
    res = t[n_train:] - A[n_train:] @ w
    return float(1 - (res ** 2).sum() / max(((t[n_train:] - t[n_train:].mean()) ** 2).sum(), 1e-12))


def main(out="results/extras"):
    from avrl import worldmodel as W

    os.makedirs(out, exist_ok=True)
    o, u, o2, _ = W.transitions(W.load_rows("data/env.bin"))
    n_train = int(0.8 * len(o))
    res = {}
    for name, lam in (("no_regulariser", 0.0), ("sigreg", 1.0)):
        p, curve = train((o[:n_train], u[:n_train], o2[:n_train]), lam)
        z = np.asarray(encode(p, jnp.asarray(o[n_train:])))
        zt = np.asarray(encode(p, jnp.asarray(o))); 
        res[name] = {"effective_rank": effective_rank(z), "z_std": float(z.std(0).mean()), "pred_loss": float(curve[-100:, 0].mean()),
                     "probe_r2_speed": probe_r2(zt, o[:, 0], n_train), "probe_r2_gap": probe_r2(zt, o[:, 11], n_train),
                     "curve": curve[::30].tolist()}
        print(name, {k: round(v, 4) for k, v in res[name].items() if isinstance(v, float)}, flush=True)
    with open(os.path.join(out, "sigreg.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
