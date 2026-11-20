"""Timed Elastic Band (TEB) as an AV planning rule, in the simulator's policy slot.

TEB (Rösmann et al., ROBOTIK 2012; sparse form ECMR 2013) plans a band of poses with a time interval between each pair, and
minimises the total time plus soft penalties for speed, acceleration and obstacle limits. Here the road is one-dimensional along
the AV's lane, so a band is N intervals with mean speeds v_i >= 0 and durations dT_i in [dT_min, dT_max]; positions are
s_i = sum_{j <= i} v_j dT_j. The planner is stateless: each step it starts from an IDM rollout (examples/my_policy.careful_idm,
the same features) and refines it with K Levenberg-Marquardt steps (as g2o does for TEB) on the sum of squared residuals

    J = w_t sum dT_i^2                                    time (TEB's quadratic time term)
      + w_g (s_N - D)^2                                   reach the goal D: the stop target (minus 1 m) or v0 * H
      + w_v sum relu(v_i - v0)^2                          speed limit (junction cap included)
      + w_a sum relu(a_i - a_max)^2 + relu(-a_i - b)^2    acceleration limits (a_i from consecutive v_i)
      + w_o sum relu(s_i + s0 + T_h v_i - x_lead(t_i))^2  leader at constant speed, with a standstill gap and headway
      + w_s sum relu(s_i - (g_stop - 1))^2 + w_s v_N^2    do not pass an active stop target; arrive at rest
      + w_j sum (a_i - a_{i-1})^2                         smoothness

and returns the first acceleration, 2 (v_1 - v) / dT_1, clipped to [-8, 3] m/s^2 and encoded as u0 (3 u forward, 8 u braking).
u1 (junction gap acceptance) stays at the default, 0. All AV rows are planned in parallel (vmapped), with no Python loop.

    from avlab.teb import teb_policy        # f(obs [N, 28], key) -> (u [N, 2], aux), like examples/my_policy.careful_idm
"""
import jax
import jax.numpy as jnp

N, H, K = 8, 4.0, 10  # intervals, horizon (s), Levenberg-Marquardt iterations
DT_MIN, DT_MAX = 0.1, 1.5
W = dict(t=1.0, g=0.5, v=20.0, a=10.0, o=60.0, s=60.0, j=0.5, vend=5.0)
A_MAX, B_COMF, S0, T_H = 1.5, 3.0, 2.5, 1.2
CFG = {"dt1": None}  # a fixed first interval (s), e.g. the 0.2 s control step; None: free like the others (read at trace time)


def features(obs):
    """Physical quantities from the normalised observation (names in av_world.sil.OBS_NAMES)."""
    v = 15 * obs[:, 0]
    v0 = jnp.maximum(jnp.minimum(15 * obs[:, 3], 15 * obs[:, 16] + 0.5), 0.5)
    return dict(v=v, v0=v0, has_lead=obs[:, 8] > 0.5, gap_lead=50 * obs[:, 9], v_lead=jnp.maximum(15 * obs[:, 10], 0.0),
                has_stop=obs[:, 6] > 0.5, gap_stop=100 * obs[:, 7])


def idm_acc(v, f, T=1.5, s0=2.5, a_max=1.5, b=2.0):
    """careful_idm's acceleration for speed v given the features (used for the warm start)."""
    def term(gap, dv):
        s_star = s0 + jnp.maximum(v * T + v * dv / (2 * jnp.sqrt(a_max * b)), 0.0)
        return (s_star / jnp.maximum(gap, 0.1)) ** 2
    inter = jnp.maximum(jnp.where(f["has_lead"], term(f["gap_lead"], v - f["v_lead"]), 0.0), jnp.where(f["has_stop"], term(f["gap_stop"], v), 0.0))
    return a_max * (1 - (v / f["v0"]) ** 4 - inter)


def _residuals(z, f, w=None):
    """TEB residuals for one vehicle (the cost is their sum of squares, as in g2o's least squares).
    z = (raw speeds [N], raw durations [N]); f: that vehicle's scalar features."""
    w = W if w is None else w
    v = jax.nn.softplus(z[0])
    dT = DT_MIN + (DT_MAX - DT_MIN) * jax.nn.sigmoid(z[1])
    if CFG["dt1"] is not None:
        dT = dT.at[0].set(CFG["dt1"])
    s = jnp.cumsum(v * dT)
    t = jnp.cumsum(dT)
    v_prev = jnp.concatenate([f["v"][None], v[:-1]])
    dT_prev = jnp.concatenate([dT[:1], dT[:-1]])
    a = jnp.where(jnp.arange(N) == 0, 2 * (v - v_prev) / dT, (v - v_prev) / (0.5 * (dT + dT_prev)))
    stop_goal = jnp.maximum(f["gap_stop"] - 1.0, 0.0)
    D = jnp.where(f["has_stop"], jnp.minimum(stop_goal, f["v0"] * H), f["v0"] * H)
    x_lead = f["gap_lead"] + f["v_lead"] * t
    r = lambda x, k: jnp.sqrt(w[k]) * x  # noqa: E731
    return jnp.concatenate([
        r(dT, "t"), r(s[-1:] - D, "g"), r(jax.nn.relu(v - f["v0"]), "v"),
        r(jax.nn.relu(a - A_MAX), "a"), r(jax.nn.relu(-a - B_COMF), "a"),
        r(jnp.where(f["has_lead"], jax.nn.relu(s + S0 + T_H * v - x_lead), 0.0), "o"),
        r(jnp.where(f["has_stop"], jax.nn.relu(s - stop_goal), 0.0), "s"),
        r(jnp.where(f["has_stop"], v[-1:], 0.0), "vend"), r(a[1:] - a[:-1], "j")])


def _cost(z, f, w=None):
    return (_residuals(z, f, w) ** 2).sum()


def _init(f):
    """IDM rollout over the band (dT = H / N) as the starting band, in the raw parameters."""
    dT = H / N

    def body(v, _):
        a = jnp.clip(idm_acc(v, f), -B_COMF, A_MAX)
        v1 = jnp.maximum(v + a * dT, 0.0)
        return v1, 0.5 * (v + v1)
    _, vm = jax.lax.scan(body, f["v"], None, length=N)
    raw_v = jnp.log(jnp.expm1(jnp.maximum(vm, 1e-3)))  # softplus^-1
    p = (dT - DT_MIN) / (DT_MAX - DT_MIN)
    raw_t = jnp.full(N, jnp.log(p / (1 - p)))
    return jnp.stack([raw_v, raw_t])


def plan_one(f, iters=K, lam0=1.0):
    """Optimise one vehicle's band with Levenberg-Marquardt (the solver TEB uses through g2o): a step
    dz = -(J'J + lam I)^-1 J'r is kept only if it lowers the cost; lam shrinks after a kept step and grows otherwise.
    Returns (first acceleration, band speeds, band durations, final cost)."""
    z0 = _init(f).ravel()
    res = lambda zz: _residuals(zz.reshape(2, N), f)  # noqa: E731
    jac = jax.jacfwd(res)

    def body(c, _):
        z, lam, cost = c
        r, Jm = res(z), jac(z)
        A = Jm.T @ Jm + lam * jnp.eye(2 * N)
        dz = -jnp.linalg.solve(A, Jm.T @ r)
        z1 = z + dz
        c1 = (res(z1) ** 2).sum()
        ok = c1 < cost
        return (jnp.where(ok, z1, z), jnp.where(ok, lam * 0.3, lam * 4.0), jnp.where(ok, c1, cost)), None
    (z, _, cost), _ = jax.lax.scan(body, (z0, jnp.asarray(lam0), (res(z0) ** 2).sum()), None, length=iters)
    z = z.reshape(2, N)
    v = jax.nn.softplus(z[0])
    dT = DT_MIN + (DT_MAX - DT_MIN) * jax.nn.sigmoid(z[1])
    if CFG["dt1"] is not None:
        dT = dT.at[0].set(CFG["dt1"])
    a0 = jnp.clip(2 * (v[0] - f["v"]) / dT[0], -8.0, 3.0)
    return a0, v, dT, cost


def teb_accel(obs):
    f = features(obs)
    return jax.vmap(lambda fi: plan_one(fi)[0])(f)


def teb_policy(obs, key=None):
    """Policy-slot function: f(obs [N, 28], key) -> (u [N, 2], aux [N, 2])."""
    a = teb_accel(obs)
    u0 = jnp.clip(jnp.where(a >= 0, a / 3.0, a / 8.0), -1.0, 1.0)
    return jnp.stack([u0, jnp.zeros_like(u0)], 1), jnp.zeros((obs.shape[0], 2))


def teb_policy_masked(obs, key, mask, cap=256):
    """The same rule, planning only the AV rows (the slot's mask), up to `cap` per copy; rows beyond cap use careful_idm.
    Non-AV rows are ignored by the simulator. For xnet.sim.step(rl=(f, key, True))."""
    idx = jnp.argsort(~mask, stable=True)[:cap]  # AV rows first, in slot order
    a_teb = teb_accel(obs[idx])
    a = idm_acc(15 * obs[:, 0], features(obs))
    a = a.at[idx].set(jnp.where(mask[idx], a_teb, a[idx]))
    u0 = jnp.clip(jnp.where(a >= 0, a / 3.0, a / 8.0), -1.0, 1.0)
    return jnp.stack([u0, jnp.zeros_like(u0)], 1), jnp.zeros((obs.shape[0], 2))


def idm_policy_masked(obs, key, mask):
    """examples/my_policy.careful_idm with the masked signature (the baseline in the same slot)."""
    a = idm_acc(15 * obs[:, 0], features(obs))
    u0 = jnp.clip(jnp.where(a >= 0, a / 3.0, a / 8.0), -1.0, 1.0)
    return jnp.stack([u0, jnp.zeros_like(u0)], 1), jnp.zeros((obs.shape[0], 2))
