"""Shared by the examples: run many copies of a city at once, each copy with its own Behaviour (AV share, demand, ...)."""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere in the checkout

from xnet import network as NW  # noqa: E402
from xnet import sim as X  # noqa: E402


def calibrated(**over):
    """The calibrated human drivers (as AVBench uses them), with overrides."""
    from avrl import ladder as LD
    return X.Behaviour(**(dict(LD.knobs()) | over))


def run_copies(area, behs, warm_s=120.0, seconds=600.0, seed=0, n_slots=1024, chunk_s=60.0, keep_states=False):
    """One compiled run of len(behs) copies of `area`, copy e driven by behs[e]. Returns per-copy counters measured after the
    warm-up [E, NM] (and, with keep_states, the final state of every copy)."""
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    E = len(behs)
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    keys = jax.random.split(jax.random.PRNGKey(seed), E)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, n_slots)
    n = int(round(chunk_s / X.DT))

    @jax.jit
    def chunk(st):
        def body(s, _):
            return jax.vmap(lambda si, b: X.step(si, J, b, nh))(s, B), None
        return jax.lax.scan(body, st, None, length=n)[0]

    for _ in range(int(round(warm_s / chunk_s))):
        st = chunk(st)
    m0 = np.asarray(st.m)
    for _ in range(int(round(seconds / chunk_s))):
        st = chunk(st)
    m = np.asarray(st.m) - m0
    return (m, st, (net, geo, info)) if keep_states else m


def per_copy(m):
    """Counters [E, NM] -> list of dicts of rates per copy."""
    out = []
    for row in np.asarray(m, float):
        d = dict(zip(X.METRICS, row))
        h = max(d["veh_s"] / 3600, 1e-9)
        av_h = d["av_veh_s"] / 3600
        hum_h = max(h - av_h, 1e-9)
        crashes = sum(d[k] for k in X.CRASH_NAMES)
        out.append({"vehicle_hours": h, "crashes": crashes, "crashes_per_1000_vehicle_h": 1000 * crashes / h,
                    "av_crashes": d["av_crash"], "human_only_crashes": crashes - d["av_crash"],
                    "trips_finished": d["finished"], "mean_speed_kmh": 3.6 * d["veh_m"] / max(d["veh_s"], 1e-9),
                    "human_speed_kmh": 3.6 * (d["veh_m"] - d["av_veh_m"]) / max(d["veh_s"] - d["av_veh_s"], 1e-9),
                    "human_hours": hum_h, "av_hours": av_h, "veh_m": d["veh_m"], "veh_s": d["veh_s"], "teleports": d["teleports"],
                    "blocked": d["blocked"]})
    return out


def poisson_ci(k, t, z=1.96):
    """Approximate 95 % interval of a Poisson rate k / t (Wilson-style, fine for k >= 1)."""
    k = float(k)
    lo = max(0.0, (k + z * z / 2 - z * np.sqrt(k + z * z / 4)) / t)
    hi = (k + z * z / 2 + z * np.sqrt(k + z * z / 4)) / t
    return lo, hi


def save(name, obj):
    os.makedirs("results/examples", exist_ok=True)
    json.dump(obj, open(f"results/examples/{name}.json", "w"), indent=1, default=float)
