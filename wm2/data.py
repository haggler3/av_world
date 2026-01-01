"""Argoverse 2 scenarios -> per-agent samples for an agent-token model.

A scenario is 11 s at 10 Hz. For every agent i and step t where i is seen at t-1, t and t+1 we build

  self token       kind (car, bus, bike, ped) one-hot, speed, previous longitudinal acceleration, previous yaw rate
  neighbour tokens the K = 8 nearest agents within 30 m, in agent i's frame at t (x ahead, y left): position, velocity relative
                   to i, heading difference, kind (static objects included), and a flag for "the leader" (nearest ahead in lane)
  target           longitudinal acceleration a = (speed_{t+1} - speed_t) / dt and yaw rate w = wrap(h_{t+1} - h_t) / dt
"""
import glob
import os

import numpy as np

DT = 0.1
K = 8
RADIUS = 30.0
KIND = {"vehicle": 0, "bus": 1, "cyclist": 2, "motorcyclist": 2, "riderless_bicycle": 2, "pedestrian": 3}
STATIC = 4  # everything else (parked, construction, background): a neighbour, never a target
SELF_DIM, NB_DIM = 7, 11
LANE_HALF_WIDTH = 2.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_scene(path):
    """-> dict of dense arrays: pos [A, T, 2], vel [A, T, 2], head [A, T], ok [A, T], kind [A]."""
    import pandas as pd

    df = pd.read_parquet(path)
    ids = {t: i for i, t in enumerate(df.track_id.unique())}
    T = int(df.timestep.max()) + 1
    A = len(ids)
    pos, vel, head, ok = np.zeros((A, T, 2)), np.zeros((A, T, 2)), np.zeros((A, T)), np.zeros((A, T), bool)
    kind = np.full(A, STATIC)
    a = df.track_id.map(ids).to_numpy()
    t = df.timestep.to_numpy()
    pos[a, t] = df[["position_x", "position_y"]].to_numpy()
    vel[a, t] = df[["velocity_x", "velocity_y"]].to_numpy()
    head[a, t] = df.heading.to_numpy()
    ok[a, t] = True
    for tid, g in df.groupby("track_id"):
        kind[ids[tid]] = KIND.get(g.object_type.iloc[0], STATIC)
    return {"pos": pos, "vel": vel, "head": head, "ok": ok, "kind": kind, "id": os.path.basename(os.path.dirname(path))}


def speed(sc, t):
    return np.hypot(sc["vel"][:, t, 0], sc["vel"][:, t, 1])


def features(sc, t, agents, state=None):
    """Self and neighbour tokens for `agents` (indices) at step t. `state` = (pos, vel, head, ok, prev_a, prev_w) overrides the log
    (a closed-loop rollout). -> S [n, SELF_DIM], NB [n, K, NB_DIM], mask [n, K] (True = a real neighbour), lead [n, K]."""
    if state is None:
        pos, vel, head, ok = sc["pos"][:, t], sc["vel"][:, t], sc["head"][:, t], sc["ok"][:, t]
        pos1, vel1, head1, ok1 = sc["pos"][:, t - 1], sc["vel"][:, t - 1], sc["head"][:, t - 1], sc["ok"][:, t - 1]
        prev_a = (np.hypot(*vel.T) - np.hypot(*vel1.T)) / DT
        prev_w = wrap(head - head1) / DT
    else:
        pos, vel, head, ok, prev_a, prev_w = state
    kind = sc["kind"]
    spd = np.hypot(vel[:, 0], vel[:, 1])
    n = len(agents)
    S = np.zeros((n, SELF_DIM), np.float32)
    S[np.arange(n), kind[agents].clip(max=3)] = 1.0
    S[:, 4] = spd[agents] / 10.0
    S[:, 5] = np.clip(prev_a[agents], -9, 5) / 3.0
    S[:, 6] = np.clip(prev_w[agents], -1.5, 1.5)
    d = pos[None, :, :] - pos[agents][:, None, :]                      # [n, A, 2] world offsets
    c, s = np.cos(head[agents]), np.sin(head[agents])
    fx = d[..., 0] * c[:, None] + d[..., 1] * s[:, None]               # ahead
    fy = -d[..., 0] * s[:, None] + d[..., 1] * c[:, None]              # left
    dist = np.hypot(fx, fy)
    dist[~ok[None, :].repeat(n, 0)] = np.inf
    dist[np.arange(n), agents] = np.inf
    order = np.argsort(dist, 1)[:, :K]
    dsel = np.take_along_axis(dist, order, 1)
    if order.shape[1] < K:                                              # a scene with fewer than K other agents: pad with empty slots
        pad = K - order.shape[1]
        order = np.concatenate([order, np.zeros((n, pad), order.dtype)], 1)
        dsel = np.concatenate([dsel, np.full((n, pad), np.inf)], 1)
    mask = dsel < RADIUS
    rv = vel[order] - vel[agents][:, None, :]
    rvx = rv[..., 0] * c[:, None] + rv[..., 1] * s[:, None]
    rvy = -rv[..., 0] * s[:, None] + rv[..., 1] * c[:, None]
    dh = head[order] - head[agents][:, None]
    NB = np.zeros((n, K, NB_DIM), np.float32)
    NB[..., 0] = np.take_along_axis(fx, order, 1) / 20.0
    NB[..., 1] = np.take_along_axis(fy, order, 1) / 20.0
    NB[..., 2], NB[..., 3] = rvx / 10.0, rvy / 10.0
    NB[..., 4], NB[..., 5] = np.cos(dh), np.sin(dh)
    NB[np.arange(n)[:, None], np.arange(K)[None], 6 + kind[order]] = 1.0
    NB *= mask[..., None]
    ahead = mask & (np.take_along_axis(fx, order, 1) > 0) & (np.abs(np.take_along_axis(fy, order, 1)) < LANE_HALF_WIDTH) & (kind[order] <= 2)
    first = np.cumsum(ahead, 1) == 1
    lead = ahead & first                                               # nearest ahead in lane (neighbours are sorted by distance)
    return S, NB, mask, lead


def build(root, limit=None, step=2, v_min=0.5):
    """All training samples from the scenarios under root/*/scenario_*.parquet. -> dict of arrays, with 'scene' = scenario index."""
    files = sorted(glob.glob(os.path.join(root, "*", "scenario_*.parquet")))[:limit]
    S, NB, M, L, Y, SC, V = [], [], [], [], [], [], []
    for si, f in enumerate(files):
        sc = load_scene(f)
        T = sc["ok"].shape[1]
        for t in range(1, T - 1, step):
            ag = np.where(sc["ok"][:, t - 1] & sc["ok"][:, t] & sc["ok"][:, t + 1] & (sc["kind"] <= 2))[0]
            sp0, sp1 = speed(sc, t), speed(sc, t + 1)
            ag = ag[(np.maximum(sp0[ag], sp1[ag]) > v_min)]
            if len(ag) == 0:
                continue
            s, nb, m, l = features(sc, t, ag)
            a = (sp1[ag] - sp0[ag]) / DT
            w = wrap(sc["head"][ag, t + 1] - sc["head"][ag, t]) / DT
            S.append(s), NB.append(nb), M.append(m), L.append(l)
            Y.append(np.stack([a, w], 1).astype(np.float32)), SC.append(np.full(len(ag), si)), V.append(sp0[ag])
    cat = lambda x: np.concatenate(x)
    d = {"S": cat(S), "NB": cat(NB), "mask": cat(M), "lead": cat(L), "Y": cat(Y), "scene": cat(SC), "speed": cat(V), "files": files}
    ok = (np.abs(d["Y"][:, 0]) < 12) & (np.abs(d["Y"][:, 1]) < 3)       # tracking glitches
    for k in ("S", "NB", "mask", "lead", "Y", "scene", "speed"):
        d[k] = d[k][ok]
    return d
