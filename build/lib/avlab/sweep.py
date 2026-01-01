"""Condition sweeps: hundreds of different worlds at once, one per vmapped copy of the area.

Each copy of the map gets its own condition, drawn by Latin hypercube sampling over nine factors:

  demand            0.6 - 1.6     traffic volume (x the network's inferred demand)
  risk              1 - 30 (log)  human risk x the calibrated rates: red runners, drivers who don't look, distraction,
                                  close passes of cyclists
  ped_scale         0.3 - 3 (log) pedestrian volume
  p_ped_violate     0.03 - 0.5    share of pedestrians crossing against the signal or into a short gap
  p_bike            0 - 0.2       share of trips by bicycle
  p_av              0.1 - 0.8     AV share
  friction          0.3 - 1.0     road friction (snow ~0.3, wet ~0.6, dry 1): the hardest possible braking is 9 x friction
  perception_noise  0 - 0.3       AV sensor noise on gaps (sigma of a multiplicative error)
  breakdowns        1 - 50 (log)  breakdown rate x the default (lanes blocked by broken-down vehicles)

Every condition is warmed up once (rule-based AVs), then the same warm state and the same random numbers are run under
each driver configuration, so every difference between configurations is the driver's:

  rule        the rule-based AV (IDM, gap acceptance)                 rule+cbf  the same under the barrier function
  ppo+idm     the 234k-step PPO policy under the IDM envelope          ppo+cbf   the same under the barrier function
  ppo raw     the PPO policy with no filter

    python -m avlab.sweep warm oakland            # sample + warm 192 conditions (~2 min)
    python -m avlab.sweep run oakland rule        # one configuration (~1.5 min); also rule+cbf, ppo+idm, ppo+cbf, "ppo raw"
    python -m avlab.sweep grid                    # friction x perception-noise grid for the ODD map (warm + 3 configs, ~5 min)
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from avrl import ladder as LD
from avrl import xppo as P
from xnet import network as NW
from xnet import sim as X

OUT = "results/avlab"
POLICY = "results/xnet3/policy.npz"
WARM, SECONDS, N_SLOTS = 240.0, 150.0, 1024
FACTORS = {  # name: (low, high, log scale)
    "demand": (0.6, 1.6, False),
    "risk": (1.0, 30.0, True),
    "ped_scale": (0.3, 3.0, True),
    "p_ped_violate": (0.03, 0.5, True),
    "p_bike": (0.0, 0.2, False),
    "p_av": (0.1, 0.8, False),
    "friction": (0.3, 1.0, False),
    "perception_noise": (0.0, 0.3, False),
    "breakdowns": (1.0, 50.0, True),
}
CONFIGS = {"rule": (None, "idm"), "rule+cbf": (None, "cbf"), "ppo+idm": (POLICY, "idm"), "ppo+cbf": (POLICY, "cbf"),
           "ppo raw": (POLICY, "off")}
N_COND = {"oakland": 192, "squirrel_hill": 64, "downtown": 64}
NOMINAL = dict(demand=1.0, risk=1.0, ped_scale=1.0, p_ped_violate=0.03, p_bike=0.03, p_av=0.25, friction=1.0,
               perception_noise=0.0, breakdowns=1.0)


def lhs(n, seed):
    """Latin hypercube over the factors -> {name: values [n]}."""
    rng = np.random.default_rng(seed)
    u = (np.stack([rng.permutation(n) for _ in FACTORS], 1) + rng.random((n, len(FACTORS)))) / n
    out = {}
    for j, (k, (lo, hi, lg)) in enumerate(FACTORS.items()):
        out[k] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo))) if lg else lo + u[:, j] * (hi - lo)
    return out


def behaviour(c):
    """One condition {factor: value} -> xnet Behaviour."""
    knobs = {k: min(v * c["risk"], 0.95) for k, v in LD.knobs().items()}
    d = X.Behaviour()
    return X.Behaviour(**(knobs | dict(demand=c["demand"], ped_scale=c["ped_scale"], p_ped_violate=c["p_ped_violate"],
                                       p_bike=c["p_bike"], p_av=c["p_av"], friction=c["friction"],
                                       perception_noise=c["perception_noise"], breakdown_rate=d.breakdown_rate * c["breakdowns"],
                                       p_close_pass=min(d.p_close_pass * c["risk"], 0.95))))


def batch(conds):
    n = len(next(iter(conds.values())))
    behs = [behaviour({k: float(v[i]) for k, v in conds.items()}) for i in range(n)]
    return jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)


def rollout(net, beh, seconds, params=None, shield="idm"):
    """A jitted function st -> (st, metrics delta [E, NM]) running every copy under its own condition."""
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    n = int(round(seconds / X.DT))
    pol = None if params is None else P.deterministic(params)

    def stp(s, b, k):
        if pol is None:
            return X.step(s, J, b, nh)
        return X.step(s, J, b, nh, rl=(pol, k))[0]

    @jax.jit
    def go(st, key):
        def body(st, k):
            return jax.vmap(stp)(st, beh, jax.random.split(k, st.t.shape[0])), None
        m0 = st.m
        st, _ = jax.lax.scan(body, st, jax.random.split(key, n))
        return st, st.m - m0

    def run(st, key):
        X.SHIELD = shield  # read when go is traced
        try:
            return go(st, key)
        finally:
            X.SHIELD = "idm"
    return run


def save_state(st, path):
    np.savez_compressed(path, *[np.asarray(x) for x in jax.tree.leaves(st)])


def load_state(path, like):
    z = np.load(path)
    return jax.tree.unflatten(jax.tree.structure(like), [jnp.asarray(z[f"arr_{i}"]) for i in range(len(z.files))])


def warm(area, conds=None, tag=None, seed=0):
    tag = tag or area
    os.makedirs(OUT, exist_ok=True)
    conds = conds if conds is not None else lhs(N_COND[area], seed=list(N_COND).index(area))
    n = len(next(iter(conds.values())))
    net, geo, info = NW.build(area)
    t0 = time.time()
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), n), X.to_jnp(net), N_SLOTS)
    st, _ = rollout(net, batch(conds), WARM)(st, jax.random.PRNGKey(seed + 1))
    save_state(st, f"{OUT}/{tag}_warm.npz")
    with open(f"{OUT}/{tag}_conditions.json", "w") as f:
        json.dump({k: np.asarray(v).tolist() for k, v in conds.items()}, f)
    print(f"{tag}: warmed {n} conditions in {time.time() - t0:.0f} s", flush=True)


def run(area, config, tag=None):
    tag = tag or area
    conds = {k: np.array(v) for k, v in json.load(open(f"{OUT}/{tag}_conditions.json")).items()}
    n = len(next(iter(conds.values())))
    net, geo, info = NW.build(area)
    like = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(0), n), X.to_jnp(net), N_SLOTS)
    st = load_state(f"{OUT}/{tag}_warm.npz", like)
    path, shield = CONFIGS[config]
    t0 = time.time()
    _, m = rollout(net, batch(conds), SECONDS, None if path is None else P.load(path), shield)(st, jax.random.PRNGKey(99))
    m = np.asarray(m)
    np.save(f"{OUT}/{tag}_{config.replace(' ', '_').replace('+', '_')}.npy", m)
    d = dict(zip(X.METRICS, m.sum(0)))
    print(f"{tag} {config:9s} {time.time() - t0:4.0f} s  AV-h {d['av_veh_s'] / 3600:6.1f}  AV crashes {int(d['av_crash']):5d} "
          f"({d['av_crash'] / max(d['av_veh_s'] / 3600, 1e-9):.3f}/h)  AV km/h {3.6 * d['av_veh_m'] / max(d['av_veh_s'], 1):.1f}", flush=True)


GRID_F = np.linspace(0.3, 1.0, 6)
GRID_N = np.linspace(0.0, 0.3, 6)
GRID_SEEDS = 4


def grid_conditions():
    f, nz = np.meshgrid(GRID_F, GRID_N, indexing="ij")
    f, nz = np.repeat(f.ravel(), GRID_SEEDS), np.repeat(nz.ravel(), GRID_SEEDS)
    c = {k: np.full(len(f), v, float) for k, v in NOMINAL.items()}
    c["friction"], c["perception_noise"] = f, nz
    c["risk"] = np.full(len(f), 5.0)  # humans x5 riskier: enough events to separate the drivers
    return c


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "warm":
        warm(sys.argv[2])
    elif cmd == "run":
        for c in sys.argv[3:]:
            run(sys.argv[2], c)
    elif cmd == "grid":
        if not os.path.exists(f"{OUT}/grid_warm.npz"):
            warm("oakland", grid_conditions(), tag="grid")
        for c in sys.argv[2:] or ("rule+cbf", "ppo+idm", "ppo+cbf"):
            run("oakland", c, tag="grid")
