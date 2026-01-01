"""One scene format for real logs and simulated runs, so every tagger runs on both.

A scene is a dict of dense arrays on a 0.2 s grid:
  id, source ("av2" | "sim"), dt, T
  x, y, yaw, v [V, T]   NaN outside a vehicle's life      alive [V, T]
  kind [V] ("car", "bus", "bike", "van", "av"), av [V], length [V], ego (index of the vehicle the scene is centred on)
  px, py [P, T]         pedestrians (NaN when absent)
  truth (sim only)      per-vehicle hidden state the simulator knows: distracted [V, T], runner [V], T_head [V], crash [V, T] (crash onset),
                        lane_change [V, T], turning [V, T]
Real scenes come from xnet.avdata (Argoverse 2 at every second frame); simulated scenes are 10 s windows cut around AVs from one
recording of the city (nets.collect.record), with only the road users within 80 m of the ego, as in a real log.
"""
import glob
import os

import numpy as np

DT = 0.2
WINDOW = 50            # 10 s
RADIUS = 80.0


def from_av2(path):
    from xnet import avdata as AV
    sc = AV.read_av2(path)
    s = AV.av2_to_scene(sc, 2)
    T, V = s["frames"], len(s["vehicles"])
    x, y, yaw, v = (np.full((V, T), np.nan) for _ in range(4))
    kind, av, length = [], np.zeros(V, bool), np.zeros(V)
    for k, veh in enumerate(s["vehicles"]):
        a = veh["start"]
        n = min(len(veh["x"]), T - a)
        for arr, f in ((x, "x"), (y, "y"), (yaw, "yaw"), (v, "v")):
            arr[k, a:a + n] = np.asarray(veh[f])[:n]
        kind.append(veh["kind"]), length.__setitem__(k, veh.get("length", 4.5))
        av[k] = veh["av"]
    P = len(s["pedestrians"])
    px, py = np.full((P, T), np.nan), np.full((P, T), np.nan)
    for k, p in enumerate(s["pedestrians"]):
        a = p["start"]
        n = min(len(p["x"]), T - a)
        px[k, a:a + n], py[k, a:a + n] = np.asarray(p["x"])[:n], np.asarray(p["y"])[:n]
    ego = int(np.argmax(av)) if av.any() else 0
    return {"id": f"av2:{sc['id']}", "source": "av2", "dt": DT, "T": T, "x": x, "y": y, "yaw": yaw, "v": v, "alive": ~np.isnan(x),
            "kind": np.array(kind), "av": av, "length": length, "ego": ego, "px": px, "py": py, "truth": None}


def av2_scenes(root=None, limit=None):
    root = root or os.environ.get("AV2_ROOT", "data/av2")
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*")))[:limit]:
        try:
            out.append(from_av2(d))
        except Exception as e:                                     # a malformed log must not stop a 10,000-scene job
            print(f"skipped {d}: {e}")
    return out


# ---------------------------------------------------------------- simulated scenes
def _sim_feat(J):
    import jax
    import jax.numpy as jnp
    L = J["link_len"].shape[0]

    def one(s):
        on = s.trk < 2 * L
        mv = jnp.where(on, 0, s.trk - 2 * L)
        turning = ~on & s.alive & (J["mov_turn"][jnp.maximum(mv, 0)] != 1)
        return {"trk": s.trk, "s": s.s, "length": s.length.astype(jnp.float16), "alive": s.alive, "kind": s.kind.astype(jnp.int8), "av": s.av,
                "v": s.v.astype(jnp.float16), "uid": s.uid, "dist": s.dist_t > 0, "runner": s.runner, "T": s.T.astype(jnp.float16),
                "wreck": s.wreck_t > 0, "turning": turning, "ped_t": s.ped_t.astype(jnp.float16)}
    return lambda s, tr, pc: jax.vmap(one)(s)


def record_sim(minutes=6, n_envs=8, seed=0, beh=None):
    """One recording of n_envs copies of Oakland; returns the host slabs [T, E, N] and the world."""
    from nets import collect as C
    from xnet import sim as X
    w = C.World()
    if beh is None:
        from avlab.paired import STRESS                    # mistakes and crashes common enough that every hidden tag has positives
        beh = X.Behaviour(**STRESS, p_av=0.25, crn=1.0)
    st = C.run(w, w.reset(n_envs, seed), beh, 120.0)
    chunks = []
    for _ in range(int(minutes)):
        st, f = C.record(w, st, beh, 60.0, 1, _sim_feat(w.J))
        chunks.append(C.to_host(f, np.float32))
    rec = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    return rec, w


def sim_scenes(rec, w, per_window=4, stride=WINDOW, seed=0):
    """Cut 10 s windows around AVs: per copy and window, up to `per_window` AVs alive throughout, each with its 80 m neighbourhood."""
    from xnet import render as R
    tracks = R.Tracks(w.net_np, w.geo)
    rng = np.random.default_rng(seed)
    kinds = {0: "car", 1: "bus", 2: "van", 3: "bike"}
    leg_xy = np.asarray(w.geo.leg_xy)
    T_all, E, N = rec["v"].shape
    out = []
    for e in range(E):
        for t0 in range(0, T_all - WINDOW, stride):
            sl = slice(t0, t0 + WINDOW)
            uid = rec["uid"][sl, e]
            alive = rec["alive"][sl, e].astype(bool)
            same = (uid == uid[0][None]).all(0) & alive.all(0)
            avs = np.nonzero(same & rec["av"][t0, e].astype(bool))[0]
            if len(avs) == 0:
                continue
            # positions of every slot over the window (NaN when dead or when the slot changes vehicle)
            X_, Y_, H_ = (np.full((N, WINDOW), np.nan) for _ in range(3))
            for k in range(WINDOW):
                a = np.nonzero(alive[k] & (uid[k] == uid[0]))[0]
                if len(a):
                    xx, yy, hh = tracks.pose(rec["trk"][t0 + k, e, a], rec["s"][t0 + k, e, a], rec["length"][t0 + k, e, a].astype(float))
                    X_[a, k], Y_[a, k], H_[a, k] = xx, yy, hh
            ped_x, ped_y = _peds(rec["ped_t"][sl, e], leg_xy)
            for ego in rng.permutation(avs)[:per_window]:
                d = np.hypot(X_[:, WINDOW // 2] - X_[ego, WINDOW // 2], Y_[:, WINDOW // 2] - Y_[ego, WINDOW // 2])
                keep = np.nonzero(np.isfinite(d) & (d < RADIUS))[0]
                keep = np.r_[ego, keep[keep != ego]]
                lane = rec["trk"][sl, e][:, keep].T
                lc = np.zeros_like(lane, bool)
                lc[:, 1:] = (lane[:, 1:] != lane[:, :-1]) & (lane[:, 1:] // 2 == lane[:, :-1] // 2) & (lane[:, 1:] < 2 * len(w.net_np["link_len"]))
                wr = rec["wreck"][sl, e][:, keep].T.astype(bool)
                onset = np.zeros_like(wr)
                onset[:, 1:] = wr[:, 1:] & ~wr[:, :-1]
                pd = np.hypot(ped_x - X_[ego, WINDOW // 2], ped_y - Y_[ego, WINDOW // 2])
                pk = np.nonzero(np.nanmin(np.where(np.isfinite(pd), pd, np.inf), 1) < RADIUS)[0] if len(ped_x) else np.zeros(0, int)
                v = rec["v"][sl, e][:, keep].T.astype(float)
                out.append({"id": f"sim:{seed}:{e}:{t0}:{int(uid[0][ego])}", "source": "sim", "dt": DT, "T": WINDOW,
                            "x": X_[keep], "y": Y_[keep], "yaw": H_[keep], "v": np.where(np.isfinite(X_[keep]), v, np.nan),
                            "alive": np.isfinite(X_[keep]), "kind": np.array(["av" if rec["av"][t0, e, i] else kinds[int(rec["kind"][t0, e, i])] for i in keep]),
                            "av": rec["av"][t0, e, keep].astype(bool), "length": rec["length"][t0, e, keep].astype(float), "ego": 0,
                            "px": ped_x[pk], "py": ped_y[pk],
                            "truth": {"distracted": rec["dist"][sl, e][:, keep].T.astype(bool), "runner": rec["runner"][t0, e, keep].astype(bool),
                                      "T_head": rec["T"][t0, e, keep].astype(float), "crash": onset, "lane_change": lc,
                                      "turning": rec["turning"][sl, e][:, keep].T.astype(bool), "env": e}})
    return out


def _peds(ped_t, leg_xy):
    """Crossing legs -> pedestrian positions: a pedestrian walks from one end of the crosswalk to the other while the leg is busy."""
    T, G = ped_t.shape
    busy = ped_t > 0
    px, py = np.full((G, T), np.nan), np.full((G, T), np.nan)
    for g in np.nonzero(busy.any(0))[0]:
        a, b = leg_xy[g]
        start = None
        for t in range(T):
            if busy[t, g] and start is None:
                start, dur = t, float(ped_t[t, g]) / DT
            if busy[t, g]:
                f = min((t - start) / max(dur, 1.0), 1.0)
                px[g, t], py[g, t] = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            else:
                start = None
    return px, py
