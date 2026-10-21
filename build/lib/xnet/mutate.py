"""Mutating cities: a different city around every copy of the area.

A policy trained in one map can memorise it: the speed limits, the signal timing, which streets are busy, which are
closed. Mutations keep the map's geometry (so every array keeps its shape and copies can run in one vmapped batch) and
change what lives on it. The mutated arrays are per-copy inputs to `xnet.sim.step`:

  link_v         speed limits: a factor per district (500 m cells), 12 % of links slowed to 45-70 % (school and work zones)
  node_cycle,
  node_g0,
  node_offset    signal plans: cycle 50-110 s, an uneven green split, a random offset at every signalised junction
  src_cdf        where trips start: lognormal hot spots by district and by link (the total flow is unchanged)
  dest_cdf       where they go: lognormal weights per destination
  leg_rate       where pedestrians cross: lognormal hot spots by district and by crosswalk
  nh             routing: next hops recomputed with 4 % of the links closed (avoided by routing, no trips start there)

`strength` scales every deviation (0 = the real map). Held-out cities use seeds >= HELD_OUT.

    mutate(net, geo, seed)  -> {array: value}, closed links
    worlds(net, geo, seeds) -> {"net": {key: [E, ...]}, "nh": [E, L, D]} for xnet.sim.step under vmap
"""
import jax.numpy as jnp
import numpy as np

from . import network as NW

KEYS = ("link_v", "node_cycle", "node_g0", "node_offset", "src_cdf", "src_rate", "dest_cdf", "leg_rate")
HELD_OUT = 1000
CELL = 500.0  # metres


def mutate(net, geo, seed, strength=1.0):
    rng = np.random.default_rng(seed)
    s = strength
    L = len(net["link_len"])
    xy = np.asarray(geo.node_xy, float)
    cell = np.floor((xy[:, 0] - xy[:, 0].min()) / CELL).astype(int) * 1000 + np.floor((xy[:, 1] - xy[:, 1].min()) / CELL).astype(int)
    _, node_cell = np.unique(cell, return_inverse=True)
    n_cells = node_cell.max() + 1

    def district(sig, idx):  # a lognormal factor per district, read at the nodes `idx`
        return np.exp(rng.normal(0.0, sig * s, n_cells))[node_cell[idx]]

    out = {}
    # ---- speed limits
    end = net["link_end"]
    v = net["link_v"] * np.clip(district(0.25, end), 0.55, 1.3)
    slow = rng.random(L) < 0.12 * s
    out["link_v"] = np.maximum(np.where(slow, v * rng.uniform(0.45, 0.7, L), v), 3.0).astype(np.float32)
    # ---- signal plans
    n_nodes = len(net["node_cycle"])
    sig = np.asarray(net["node_signal"])
    cyc0 = np.asarray(net["node_cycle"], float)
    cyc = np.where(sig, cyc0 + s * (rng.uniform(50.0, 110.0, n_nodes) - cyc0), cyc0)
    ratio = 0.6 + s * (rng.uniform(0.3, 0.7, n_nodes) - 0.6)  # green split of axis 0 (0.6 in the real map)
    out["node_cycle"] = cyc.astype(np.float32)
    out["node_g0"] = np.where(sig, (cyc - 10.0) * ratio, net["node_g0"]).astype(np.float32)
    out["node_offset"] = rng.uniform(0.0, cyc).astype(np.float32)
    # ---- closures: links that are not map-edge entries, exits or destinations
    dest = np.asarray(net["dest_link"])
    cand = np.nonzero(~np.asarray(net["link_edge_in"]) & ~np.asarray(net["link_exit"]) & ~np.isin(np.arange(L), dest)
                      & (net["link_len"] > 30.0))[0]
    closed = rng.choice(cand, size=min(int(round(0.04 * s * L)), len(cand)), replace=False) if s > 0 else np.zeros(0, int)
    # ---- where trips start
    w = np.diff(np.concatenate([[0.0], net["src_cdf"]])) * float(net["src_rate"])
    w = w * district(0.5, end) * np.exp(rng.normal(0.0, 0.3 * s, L))
    w[closed] = 0.0
    w *= float(net["src_rate"]) / w.sum()
    out["src_cdf"] = (np.cumsum(w) / w.sum()).astype(np.float32)
    out["src_rate"] = np.float32(w.sum())
    # ---- where they go
    dw = np.diff(np.concatenate([[0.0], net["dest_cdf"]])) * np.exp(rng.normal(0.0, 0.4 * s, len(dest)))
    dw[np.isin(dest, closed)] = 0.0
    out["dest_cdf"] = (np.cumsum(dw) / dw.sum()).astype(np.float32)
    # ---- where pedestrians cross
    leg = np.asarray(net["leg_node"])
    out["leg_rate"] = (net["leg_rate"] * district(0.5, leg) * np.exp(rng.normal(0.0, 0.3 * s, len(leg)))).astype(np.float32)
    # ---- routing around the closures, at the mutated free-flow speeds
    m = dict(net)
    m.update(out)
    tt = NW.link_tt(m)
    tt[closed] = 1e4
    out["nh"] = NW.next_hops(m, tt)
    return out, closed


def worlds(net, geo, seeds, strength=1.0):
    """Per-copy cities for a vmapped batch: {"net": {key: [E, ...]}, "nh": [E, L, D]}."""
    ms = [mutate(net, geo, int(s), strength)[0] for s in seeds]
    return {"net": {k: jnp.asarray(np.stack([m[k] for m in ms])) for k in KEYS}, "nh": jnp.asarray(np.stack([m["nh"] for m in ms]))}


def pool(net, geo, n, first=0, strength=1.0):
    """n mutated cities, computed once: a list of {array: value} (index it to draw a batch)."""
    return [mutate(net, geo, first + i, strength)[0] for i in range(n)]


def stack(items):
    """A list of mutated cities -> the batch dict of `worlds`."""
    return {"net": {k: jnp.asarray(np.stack([m[k] for m in items])) for k in KEYS}, "nh": jnp.asarray(np.stack([m["nh"] for m in items]))}
