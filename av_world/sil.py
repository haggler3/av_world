"""Software-in-the-loop and hardware-in-the-loop: drive the simulator's AV with your own controller, in another process or on another machine.

The simulator stays as it is. This plugin uses the policy hook every AV already has (`xnet.sim.step(..., rl=...)`): for the `egos` AVs it asks
an external controller for the command, over a TCP connection carrying one JSON object per line; every other AV keeps the bundled policy.
Nothing here is imported by the core, and the core runs at full speed when the plugin is not used.

    av-world sil --area chicago --egos 1 --mode lockstep --port 5555      # the simulator side: waits for a controller
    python -m av_world.sil_client --port 5555                            # an example controller (or your own: see docs/SIL_HIL.md)

Modes
  lockstep   SIL. The simulator waits for every command (up to `timeout_s`); simulated time is independent of the controller's speed.
  realtime   HIL. The simulator runs at wall-clock pace (x `speed`); a command that is not back within `deadline_ms` is a miss, the AV gets the
             fallback (`policy`: the bundled policy, `brake`: a firm brake, `hold`: the last command), and a late reply is discarded.

Protocol (newline-delimited JSON over TCP; the simulator is the server)
  sim -> ctl   {"type": "hello", "dt": 0.2, "mode": ..., "deadline_ms": ..., "obs_names": [28 names], "command": "u0 in [-1, 1] -> accel ..."}
  sim -> ctl   {"type": "step", "step": k, "t": seconds, "egos": [{"id": slot, "obs": [28 floats], "speed_mps", "lead_gap_m", "lead_speed_mps",
                                                                  "stop_gap_m", "speed_limit_mps", "signal": "red" | "yellow" | "none"}]}
  ctl -> sim   {"type": "cmd", "step": k, "u": {"<slot>": [u0, u1]}}            u0: acceleration command in [-1, 1] (-1 = -8 m/s^2, 1 = +3 m/s^2)
                                                                                u1: gap acceptance at the next junction (0 = default)
  sim -> ctl   {"type": "bye", "summary": {...}}
The command passes through the simulator's safety filter (`shield`: "idm" envelope, "cbf" barrier, or "off" for the raw command).
"""
import json
import select
import socket
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

OBS_NAMES = ["speed/15", "accel/3", "speed limit/15", "desired speed/15", "distance to end of lane/100", "on a link (0: in a junction)",
             "has a stop target", "stop target gap/100", "has a leader", "leader gap/50", "leader speed/15", "closing speed/10", "red ahead",
             "yellow ahead", "stop sign ahead", "must yield", "junction speed cap/15", "gap acceptable", "room after the junction", "crosswalk busy",
             "time to conflict/10", "must change lane", "waiting time/10", "may enter the junction", "decel needed for the stop target/8",
             "decel needed behind the leader/8", "time headway/3", "gap-acceptance scale (normalised)"]
BRAKE = (-0.4, 0.0)  # -3.2 m/s^2


def decode(o):
    """Physical quantities from one observation row (for controllers that do not want the normalised features)."""
    o = [float(x) for x in o]
    r = lambda x: round(x, 3)  # noqa: E731
    return {"speed_mps": r(15.0 * o[0]), "lead_gap_m": r(50.0 * o[9]) if o[8] > 0.5 else None, "lead_speed_mps": r(15.0 * o[10]) if o[8] > 0.5 else None,
            "stop_gap_m": r(100.0 * o[7]) if o[6] > 0.5 else None, "speed_limit_mps": r(15.0 * o[2]),
            "signal": "red" if o[12] > 0.5 else ("yellow" if o[13] > 0.5 else "none")}


class Link:
    """One controller connection, newline-delimited JSON."""

    def __init__(self, host="127.0.0.1", port=0):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.conn, self.buf = None, b""

    def accept(self, timeout=None):
        self.srv.settimeout(timeout)
        self.conn, _ = self.srv.accept()
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(self, obj):
        self.conn.sendall((json.dumps(obj) + "\n").encode())

    def recv(self, timeout):
        """The next message, or None if none arrives within `timeout` seconds."""
        end = time.perf_counter() + timeout
        while b"\n" not in self.buf:
            left = end - time.perf_counter()
            if left <= 0:
                return None
            r, _, _ = select.select([self.conn], [], [], left)
            if not r:
                return None
            chunk = self.conn.recv(65536)
            if not chunk:
                raise ConnectionError("controller disconnected")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)

    def close(self):
        for s in (self.conn, self.srv):
            try:
                s.close()
            except Exception:
                pass


class Session:
    """The simulator side. `area`: anything av_world.make accepts. `egos`: how many AVs the controller drives (the first AVs that appear,
    replaced when one leaves). The other AVs use the bundled policy."""

    def __init__(self, area="oakland", egos=1, mode="lockstep", deadline_ms=50.0, speed=1.0, fallback="policy", beh=None, shield="idm",
                 host="127.0.0.1", port=0, seed=0, timeout_s=30.0, record=None):
        from av_world import env as E
        from xnet import sim as X
        self.X = X
        self.env = E.Env(area, n_envs=1, beh=beh or dict(p_av=0.25), shield=shield, seed=seed)
        self.k, self.mode, self.deadline, self.speed, self.fallback = egos, mode, deadline_ms / 1000.0, speed, fallback
        self.timeout, self.seed, self.record = timeout_s, seed, record
        self.link = Link(host, port)
        self.port = self.link.port
        self.ego, self.last, self.step_k = [], {}, 0
        self.stats = {"steps": 0, "replies": 0, "misses": 0, "late_discarded": 0, "overruns": 0, "latency_ms": []}
        default = E.load_policy()
        net, nh, behv = self.env.net, self.env.nh, self.env.beh

        def hybrid(obs, key, mask):
            u_def, aux = default(obs, key)
            u_ext = io_callback(self._exchange, jax.ShapeDtypeStruct(u_def.shape, jnp.float32), obs, mask, ordered=True)
            return jnp.where(jnp.isnan(u_ext[:, :1]), u_def, u_ext), aux

        self._step = jax.jit(lambda s: X.step(s, net, behv, nh, rl=(hybrid, s.key, True))[0])
        self._plain = jax.jit(lambda s: X.step(s, net, behv, nh, rl=(default, s.key))[0])

    # ---- called from inside the jitted step, once per simulator step
    def _exchange(self, obs, mask):
        obs, mask = np.asarray(obs), np.asarray(mask)
        self.ego = [e for e in self.ego if mask[e]]
        for e in np.nonzero(mask)[0]:
            if len(self.ego) >= self.k:
                break
            if e not in self.ego:
                self.ego.append(int(e))
        k = self.step_k
        msg = {"type": "step", "step": k, "t": round(k * self.X.DT, 3), "egos": [{"id": int(e), "obs": [round(float(x), 5) for x in obs[e]], **decode(obs[e])} for e in self.ego]}
        t0 = time.perf_counter()
        self.link.send(msg)
        wait = self.timeout if self.mode == "lockstep" else self.deadline
        reply = None
        while True:
            left = wait - (time.perf_counter() - t0)
            r = self.link.recv(max(left, 0.0))
            if r is None:
                break
            if r.get("type") == "cmd" and r.get("step") == k:
                reply = r
                break
            self.stats["late_discarded"] += 1  # an answer to an earlier step
        out = np.full((obs.shape[0], 2), np.nan, np.float32)
        if reply is not None:
            self.stats["replies"] += 1
            self.stats["latency_ms"].append(1000 * (time.perf_counter() - t0))
        for e in self.ego:
            u = None if reply is None else reply.get("u", {}).get(str(e))
            if u is None:
                self.stats["misses"] += 1
                u = {"policy": None, "brake": BRAKE, "hold": self.last.get(e)}[self.fallback]
            if u is not None:
                out[e] = np.clip(np.asarray(u, np.float32), -1.0, 1.0)
                self.last[e] = list(map(float, out[e]))
        self.stats["steps"] += 1
        if self.record is not None:
            self.record.write(json.dumps({"sent": msg, "reply": reply}) + "\n")
        if self.mode == "realtime":  # wall-clock pacing
            target = self.t_start + (k + 1) * self.X.DT / self.speed
            late = target - time.perf_counter()
            if late < 0:
                self.stats["overruns"] += 1  # this step took longer than its share of wall-clock time
            time.sleep(max(0.0, late))
        self.step_k += 1
        return out

    def run(self, seconds=60.0, warm_s=60.0, accept_timeout=None):
        """Warm up without the controller, wait for it to connect, then drive `seconds`. -> summary dict."""
        X = self.X
        X.SHIELD = self.env.shield
        st = X.reset(jax.random.PRNGKey(self.seed), self.env.net, 1024)
        for _ in range(int(warm_s / X.DT)):
            st = self._plain(st)
        step = self._step.lower(st).compile()  # compile before the clock starts (lowering does not call the controller)
        self.link.accept(accept_timeout)
        self.link.send({"type": "hello", "dt": X.DT, "mode": self.mode, "deadline_ms": self.deadline * 1000, "obs_names": OBS_NAMES,
                        "command": "u0 in [-1, 1] -> accel 3*u0 (u0 >= 0) or 8*u0 m/s^2; u1: gap acceptance, 0 = default"})
        m0 = np.asarray(st.m)
        crashed, dist = set(), 0.0
        self.t_start = time.perf_counter()
        for _ in range(int(seconds / X.DT)):
            prev = list(self.ego)
            st = step(st)
            v, w = np.asarray(st.v), np.asarray(st.wreck_t)
            for e in prev:
                dist += float(v[e]) * X.DT
                if w[e] > 0:
                    crashed.add(e)
        wall = time.perf_counter() - self.t_start
        d = dict(zip(X.METRICS, np.asarray(st.m) - m0))
        lat = np.array(self.stats["latency_ms"]) if self.stats["latency_ms"] else None
        pct = (lambda q: float(np.percentile(lat, q))) if lat is not None else (lambda q: None)  # noqa: E731
        summary = {"steps": self.stats["steps"], "replies": self.stats["replies"], "misses": self.stats["misses"],
                   "late_discarded": self.stats["late_discarded"], "realtime_overruns": self.stats["overruns"], "latency_ms_p50": pct(50),
                   "latency_ms_p99": pct(99), "latency_ms_max": pct(100),
                   "simulated_s": seconds, "wall_s": wall, "real_time_factor": seconds / wall, "ego_crashes": len(crashed), "ego_distance_m": dist,
                   "area_crashes": float(sum(d[c] for c in X.CRASH_NAMES)), "area_av_crashes": float(d["av_crash"])}
        try:
            self.link.send({"type": "bye", "summary": summary})
        finally:
            self.link.close()
        self.state = st
        return summary
