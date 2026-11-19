"""Per-crash event log, with exposure per street (foundations backlog item 1).

`xnet.sim` counts crashes by type per copy but does not say where or when each one happened. This module runs the
unchanged simulator step with `diag=True` (which already exposes the per-vehicle crash masks) inside a scan, and keeps

  * a fixed-capacity log per copy: one row per vehicle that becomes a wreck at that step, with
    time, slot, track, position on the track, its speed just before contact (impact speed), AV flag, kind,
    role (rear-end striker, junction conflict, lane change, pedestrian, other / struck) and the reference link
    (the link, or the approach link of the junction move for a car in a box);
  * the vehicle-seconds spent on each link (cars in a junction box count on their approach link).

The log does not change the dynamics: the state after a logged run equals the state after a plain run (see
tests/test_crashlog.py). Rows are vehicles; `events()` groups vehicles that crashed in the same copy, step and link
into one crash event.

    from avlab import crashlog as CL
    lg = CL.Logger(J, nh, B, cap=512)            # J: jnp network, nh: next hops, B: batched Behaviour
    st, buf = lg.run(st, seconds=600.0)            # returns the final state and a CrashLog
    rows = CL.table(buf)                           # numpy structured rows (valid ones only)
"""
from typing import NamedTuple

import numpy as np

FIELDS = ("t", "copy", "slot", "trk", "s", "v", "av", "kind", "role", "link", "patched")
ROLES = ("other", "rear", "junction", "lane_change", "pedestrian")


class CrashLog(NamedTuple):
    rows: object  # [E, cap, F] float32: one row per new wreck (FIELDS without "copy")
    n: object  # [E] int32: rows written (rows beyond cap are dropped, n keeps counting)
    expo: object  # [E, L] float32: vehicle-seconds on each link


def _ref_link(net, trk):
    import jax.numpy as jnp
    L = net["link_len"].shape[0]
    on_link = trk < 2 * L
    return jnp.where(on_link, trk // 2, net["mov_in"][jnp.where(on_link, 0, trk - 2 * L)])


def log_step(st, net, beh, nh, buf_rows, n, expo):
    """One simulator step for one copy, plus the log update. Returns (new state, rows, n, expo)."""
    import jax.numpy as jnp
    from xnet import sim as X
    out, d = X.step(st, net, beh, nh, diag=True)
    link = _ref_link(net, st.trk)
    new = d["crashed"] & (st.wreck_t <= 0)
    role = jnp.where(d["ped_hit"], 4, jnp.where(d["ss"], 3, jnp.where(d["in_hit"], 2, jnp.where(d["rear"], 1, 0))))
    patched = net["link_patch"][link] if "link_patch" in net else jnp.zeros_like(new)
    vals = jnp.stack([jnp.broadcast_to(st.t, st.v.shape), jnp.arange(st.v.shape[0], dtype=jnp.float32), st.trk, st.s, st.v,
                      st.av, st.kind, role, link, patched], 1).astype(jnp.float32)
    idx = jnp.where(new, n + jnp.cumsum(new) - 1, buf_rows.shape[0])
    buf_rows = buf_rows.at[idx].set(vals, mode="drop")
    expo = expo.at[jnp.where(st.alive, link, expo.shape[0])].add(X.DT, mode="drop")
    return out, buf_rows, n + new.sum(), expo


class Logger:
    """A jitted, vmapped logged step over E copies with per-copy Behaviour B (a Behaviour of [E] arrays)."""

    def __init__(self, J, nh, B, cap=512, chunk_s=60.0):
        import jax
        import jax.numpy as jnp
        from xnet import sim as X
        self.J, self.nh, self.B, self.cap = J, nh, B, cap
        self.n_chunk = int(round(chunk_s / X.DT))
        L = J["link_len"].shape[0]
        self.L = L

        def one(s, r, n, e, b):
            return log_step(s, J, b, nh, r, n, e)

        f = jax.vmap(one)

        @jax.jit
        def chunk(st, buf):
            def body(c, _):
                s, r, n, e = c
                s, r, n, e = f(s, r, n, e, B)
                return (s, r, n, e), None
            (st, r, n, e), _ = jax.lax.scan(body, (st,) + tuple(buf), None, length=self.n_chunk)
            return st, CrashLog(r, n, e)

        self._chunk = chunk
        self._jnp = jnp

    def empty(self, E):
        jnp = self._jnp
        return CrashLog(jnp.zeros((E, self.cap, len(FIELDS) - 1), jnp.float32), jnp.zeros(E, jnp.int32), jnp.zeros((E, self.L), jnp.float32))

    def run(self, st, seconds, buf=None):
        from xnet import sim as X
        if buf is None:
            buf = self.empty(st.v.shape[0])
        k = int(round(seconds / X.DT))
        assert k % self.n_chunk == 0, "seconds must be a multiple of chunk_s"
        for _ in range(k // self.n_chunk):
            st, buf = self._chunk(st, buf)
        return st, buf


def table(buf):
    """Valid rows of a CrashLog as a dict of numpy arrays (FIELDS), copy index included. Raises if a copy overflowed."""
    rows, n = np.asarray(buf.rows), np.asarray(buf.n)
    if (n > rows.shape[1]).any():
        raise ValueError(f"crash log overflow: {int(n.max())} rows in a copy, capacity {rows.shape[1]}")
    keep = np.arange(rows.shape[1])[None, :] < n[:, None]
    e = np.broadcast_to(np.arange(rows.shape[0])[:, None], keep.shape)[keep]
    r = rows[keep]
    cols = {"copy": e.astype(int)}
    for i, f in enumerate([f for f in FIELDS if f != "copy"]):
        cols[f] = r[:, i]
    for f in ("slot", "trk", "av", "kind", "role", "link", "patched"):
        cols[f] = cols[f].astype(int)
    return cols


def events(rows, xy=None, radius=25.0):
    """Group vehicle rows into crash events: rows of the same copy and time step that are linked by a chain of distances
    under `radius` metres (xy = positions(net, geo, rows)); without xy, rows of the same copy, step and reference link.
    Returns a dict of arrays, one entry per event: copy, t, link (of its first row), patched (any row on a patched link),
    size (vehicles), max_v (highest impact speed), av (any AV), role (highest role code)."""
    names = ("copy", "t", "link", "patched", "size", "max_v", "av", "role")
    if len(rows["t"]) == 0:
        return {k: np.zeros(0) for k in names}
    step = np.round(rows["t"] * 10).astype(int)
    lab = -np.ones(len(step), int)
    nxt = 0
    for (c, k) in sorted(set(zip(rows["copy"].tolist(), step.tolist()))):
        idx = np.where((rows["copy"] == c) & (step == k))[0]
        par = {i: i for i in idx}

        def find(i):
            while par[i] != i:
                i = par[i]
            return i
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                near = (np.hypot(xy[0][i] - xy[0][j], xy[1][i] - xy[1][j]) < radius) if xy is not None else rows["link"][i] == rows["link"][j]
                if near:
                    par[find(i)] = find(j)
        roots = {}
        for i in idx:
            r = find(i)
            if r not in roots:
                roots[r] = nxt; nxt += 1
            lab[i] = roots[r]
    first = np.array([np.where(lab == g)[0][0] for g in range(nxt)])
    agg = lambda x, fn: np.array([fn(x[lab == g]) for g in range(nxt)])  # noqa: E731
    return {"copy": rows["copy"][first], "t": rows["t"][first], "link": rows["link"][first], "patched": agg(rows["patched"], np.max),
            "size": np.bincount(lab, minlength=nxt), "max_v": agg(rows["v"], np.max), "av": agg(rows["av"], np.max),
            "role": agg(rows["role"], np.max)}


def positions(net, geo, rows):
    """World x, y of each logged row (uses xnet.render.Tracks)."""
    from xnet.render import Tracks
    tr = Tracks(net, geo)
    x, y, _ = tr.pose(rows["trk"], rows["s"], np.full(len(rows["s"]), 4.5))
    return np.asarray(x), np.asarray(y)
