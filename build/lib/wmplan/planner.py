"""Planning in the latent world model: model-predictive control by the cross-entropy method.

Every 0.2 s, for every AV at once (up to K_PLAN of them; any beyond fall back to the PPO policy):

  1. encode its scene: z0 = enc(obs)
  2. propose M smooth action sequences over the next HP steps (2.4 s; three knots per action dimension, interpolated):
     accel and gap acceptance, around the current plan (initialised at the PPO policy's action held constant);
     candidate 0 is always the current mean
  3. unfold each in latent space, z_k+1 = z_k + dyn(z_k, u_k), and decode what matters at every step: the reward, the
     probability of a crash, of a near miss and of running a red light, and the five safety margins (stop target, leader, junction
     time-to-conflict, merge gap behind / ahead)
  4. score = sum_k gamma^k [ reward - W_crash P(crash) - W_near P(near miss) - W_red P(red run) - W_margin sum relu(-margin)
                             - W_jerk (a_k - a_k-1)^2 - W_prior |u_k - u_ppo|^2 ]
     (the event heads were trained with positives weighted x POS_W, so their logits are shifted back by log POS_W; the
     margins penalised are the stop target, the leader and the junction: the merge gaps act through P(crash))
  5. refit the proposal to the best ELITE sequences, repeat ITERS times; execute the first action of the mean, replan
     next step

This is where merging and junction decisions get made: the second action dimension sets how much gap the AV accepts at
its next junction and in a merge, and the planner sees, in the unrolled latent, whether the junction margin or the merge
gaps go negative (and the crash probability rises) a few seconds after accepting.
"""
import jax
import jax.numpy as jnp
import numpy as np

from avrl import xppo as P

from . import model as M

K_PLAN, N_SAMP, ITERS, HP, ELITE, KNOTS = 48, 32, 2, 12, 6, 3
POS_W = 30.0  # wmplan.model.loss_fn's weight on positive events
W = dict(crash=30.0, near=3.0, red=20.0, margin=0.3, jerk=0.002, prior=0.02, gamma=0.97, sd0=0.5)
_INTERP = jnp.asarray(np.stack([np.interp(np.linspace(0, KNOTS - 1, HP), np.arange(KNOTS), np.eye(KNOTS)[i]) for i in range(KNOTS)], 1))


def accel(u):
    u0 = jnp.clip(u[..., 0], -1.0, 1.0)
    return jnp.where(u0 >= 0, 3.0 * u0, 8.0 * u0)


def score(wm, z0, us, w=W, a0=None, u_ref=None):
    """z0 [..., ZD], us [..., S, HP, 2] -> plan scores [..., S] and the decoded futures. a0: the current acceleration
    (for the jerk of the first step), u_ref [..., HP, 2]: the policy prior."""
    z = jnp.broadcast_to(z0[..., None, :], us.shape[:-2] + z0.shape[-1:])
    zs, trs = M.unroll(wm, z, us)
    _, marg = M.decode(wm, zs[..., 1:, :])
    disc = w["gamma"] ** jnp.arange(us.shape[-2])
    shift = jnp.log(POS_W)
    step = trs[..., 0] / M.R_SCALE - w["crash"] * jax.nn.sigmoid(trs[..., 1] - shift) - w["near"] * jax.nn.sigmoid(trs[..., 2] - shift) \
        - w["red"] * jax.nn.sigmoid(trs[..., 3] - shift) \
        - w["margin"] * jax.nn.relu(-marg[..., :3]).sum(-1)
    a = accel(us)
    prev = jnp.concatenate([(a[..., :1] if a0 is None else jnp.broadcast_to(a0[..., None, None], a[..., :1].shape)), a[..., :-1]], -1)
    step = step - w["jerk"] * ((a - prev) / 0.2) ** 2
    if u_ref is not None:
        step = step - w["prior"] * ((us - u_ref[..., None, :, :]) ** 2).sum(-1)
    return (step * disc).sum(-1), (zs, trs, marg)


def plan(wm, z0, mu0, key, w=W, a0=None):
    """CEM over KNOTS knots per action dimension from the initial mean mu0 [K, HP, 2] (held constant: the policy's
    action) -> (final mean [K, HP, 2], last samples, their scores)."""
    mu = mu0[:, :KNOTS]  # knots
    sd = jnp.full_like(mu, w["sd0"])
    us = sc = None
    for _ in range(ITERS):
        key, k = jax.random.split(key)
        eps = jax.random.normal(k, mu.shape[:1] + (N_SAMP,) + mu.shape[1:]).at[:, 0].set(0.0)
        knots = jnp.clip(mu[:, None] + sd[:, None] * eps, -2.5, 2.5)
        us = jnp.einsum("hk,bskd->bshd", _INTERP, knots)
        sc, _ = score(wm, z0, us, w, a0, mu0)
        el = jnp.argsort(-sc, axis=1)[:, :ELITE]
        e = jnp.take_along_axis(knots, el[:, :, None, None], 1)
        mu, sd = e.mean(1), e.std(1) + 0.05
    return jnp.einsum("hk,bkd->bhd", _INTERP, mu), us, sc


def make(wm, ppo, w=W):
    """A policy for xnet.sim.step(rl=(policy, key, True)): plans for the AV rows of `mask`."""
    def policy(obs, key, mask):
        fb = P.mean(ppo, obs)
        idx = jnp.nonzero(mask, size=K_PLAN, fill_value=0)[0]
        ok = mask[idx]
        z0 = M.encode(wm, obs[idx])
        mu0 = jnp.repeat(fb[idx][:, None], HP, 1)
        mu, _, _ = plan(wm, z0, mu0, key, w, a0=obs[idx, 1] * 3.0)
        u = fb.at[idx].set(jnp.where(ok[:, None], mu[:, 0], fb[idx]))
        return u, jnp.zeros((obs.shape[0], 2))
    return policy
