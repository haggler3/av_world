"""AV sensor noise: what the AV policy perceives about its leader, with range error, speed error and missed detections.

The simulator hands every AV policy the true 28 features (av_world.sil.OBS_NAMES). `noisy(policy, model)` returns a policy with the
same signature that first corrupts the leader features, then calls `policy`. The safety envelope inside xnet.sim.step still sees
the true state, so the wrapper separates two questions: how much a noisy perception hurts the planner, and how much of that the
envelope takes back. xnet is untouched; with `SensorModel()` (all zeros) the wrapper returns the observation unchanged.

Model, per AV and per 0.2 s step, independent (the noise key is `fold_in(step key, SALT)`, so a fixed seed gives a fixed run and
the simulator's own random numbers are not reused):

    detected  = has_leader and (u >= p_miss),                      u ~ U(0, 1)
    gap'      = gap * exp(sigma_range * e1),                       e1 ~ N(0, 1)   (relative range error, as radar/camera)
    v_lead'   = max(v_lead + sigma_speed * e2, 0),                 e2 ~ N(0, 1)
    and the derived features (closing speed, decel needed behind the leader, time headway) are recomputed from gap', v_lead'
    with the simulator's own formulas; a missed leader looks like no leader (has_leader 0, gap 150 m, derived features 0).

    from avlab import sensornoise as SN
    pol = SN.noisy(T.idm_policy_masked, SN.SensorModel(0.1, 0.5, 0.1))    # use in rl=(pol, key, True)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

SALT = 0x5E05  # folded into the step key so the noise stream is separate from the simulator's draws


class SensorModel(NamedTuple):
    sigma_range: float = 0.0  # relative (log) range error of the leader gap
    sigma_speed: float = 0.0  # m/s, error of the leader's speed
    p_miss: float = 0.0       # probability per step that the leader is not detected


CLEAN = SensorModel()
LEVELS = {"clean": CLEAN, "mild": SensorModel(0.05, 0.25, 0.02), "moderate": SensorModel(0.10, 0.5, 0.05),
          "severe": SensorModel(0.20, 1.0, 0.15)}


def perturb(obs, key, m: SensorModel):
    """The corrupted observation [N, 28]. Only the leader features (8-11, 25, 26) change; the clean model returns `obs` itself."""
    if m == CLEAN:
        return obs
    k1, k2, k3 = jax.random.split(jax.random.fold_in(key, SALT), 3)
    n = obs.shape[0]
    v = 15.0 * obs[:, 0]
    has = obs[:, 8] > 0.5
    gap = 50.0 * obs[:, 9]
    det = has & (jax.random.uniform(k3, (n,)) >= m.p_miss)
    gap_n = gap * jnp.exp(m.sigma_range * jax.random.normal(k1, (n,)))
    vl_n = jnp.maximum(15.0 * obs[:, 10] + m.sigma_speed * jax.random.normal(k2, (n,)), 0.0)
    f = det.astype(obs.dtype)
    closing = jnp.clip((v - vl_n) / 10.0, -2.0, 2.0) * f
    dec = jnp.clip(jnp.maximum(v - vl_n, 0.0) ** 2 / (2.0 * jnp.maximum(gap_n - 2.0, 0.5)) / 8.0, 0.0, 2.0) * f
    hw = jnp.clip(gap_n / jnp.maximum(v, 1.0) / 3.0, 0.0, 2.0) * f
    new = {8: f, 9: jnp.where(det, jnp.clip(gap_n / 50.0, -0.1, 3.0), 3.0), 10: jnp.where(det, vl_n / 15.0, obs[:, 10]),
           11: closing, 25: dec, 26: hw}
    out = obs
    for j, col in new.items():
        out = out.at[:, j].set(jnp.where(has, col, obs[:, j]))  # rows with no leader are left as they were
    return out


def noisy(policy, m: SensorModel = CLEAN, masked=True):
    """`policy` behind the sensor model. masked: the policy takes (obs, key, mask), as in rl=(pol, key, True)."""
    if m == CLEAN:
        return policy
    if masked:
        return lambda obs, key, mask: policy(perturb(obs, key, m), key, mask)
    return lambda obs, key: policy(perturb(obs, key, m), key)
