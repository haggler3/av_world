"""A batched, pure-JAX microsimulation of a signalised 4-way intersection with turn lanes and risky human behaviour.

Per intersection: up to M vehicles on 16 paths (geometry.py). Each step (DT = 0.2 s):
  signals (4-phase, protected-then-permissive lefts, yellow + all-red, right turn on red)
  -> leader search (approach lane, path through the box, exit lane; merges included)
  -> stop line and gap acceptance (time-to-arrival at every conflict point vs a critical gap)
  -> human behaviour: distraction (no reaction for 1.5-3 s), red-light running, not looking before turning or changing
     lanes, not signalling (the driver cut in front of is surprised and reacts late), aggressive gap acceptance
  -> IDM car following -> lane changes to reach the turn lane -> collision detection by type -> exits -> arrivals.

Everything has a fixed shape, so `jax.vmap` runs thousands of intersections at once (`run_batch`). Vehicles can be AVs:
they always look and signal, never run reds, are never distracted and accept larger gaps.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import geometry as G

DT = 0.2
M = 64  # vehicle slots per intersection (the busiest of 1024 intersections peaked at 52)
VLIM = 13.9  # 50 km/h
LEN_CLEAR = 3.2  # conflict-zone half length: half a car + half a car width


class Behaviour(NamedTuple):
    """Human behaviour knobs (per driver probabilities, per-second hazard for distraction)."""
    p_red: float = 0.01  # will run a red / late yellow when arriving at speed
    red_window: float = 0.3  # how late into the red a runner still enters [s] (later entries are far more dangerous)
    p_nolook: float = 0.01  # does not check before turning across traffic or changing lanes
    p_nosignal: float = 0.25  # does not signal lane changes (the driver cut in front of reacts late)
    p_aggressive: float = 0.2  # accepts 35 % shorter gaps
    distract_rate: float = 1 / 600  # distraction episodes per second of driving (1.5-3 s without reacting)
    p_wrong_lane: float = 0.15  # arrives in a lane that does not allow its turn and must change lanes
    p_av: float = 0.0  # share of vehicles that are AVs
    demand: float = 700.0  # vehicles per hour per approach
    p_left: float = 0.2
    p_right: float = 0.15


class State(NamedTuple):
    key: jnp.ndarray
    t: jnp.ndarray
    offset: jnp.ndarray
    alive: jnp.ndarray
    app: jnp.ndarray
    lane: jnp.ndarray
    mov: jnp.ndarray
    path: jnp.ndarray  # fixed when the stop line is crossed (-1 before)
    s: jnp.ndarray
    v: jnp.ndarray
    v0: jnp.ndarray
    T: jnp.ndarray
    amax: jnp.ndarray
    b: jnp.ndarray
    s0: jnp.ndarray
    length: jnp.ndarray
    runner: jnp.ndarray
    nolook: jnp.ndarray
    nosignal: jnp.ndarray
    aggr: jnp.ndarray
    av: jnp.ndarray
    dist_t: jnp.ndarray  # seconds of distraction / surprise left
    ran_red: jnp.ndarray
    stopped: jnp.ndarray  # has come to a stop at the line (right turn on red)
    ydec: jnp.ndarray  # yellow decision taken at onset: 0 none, 1 stop, 2 go
    react: jnp.ndarray  # reaction time to a car crossing its path [s] (humans 0.7-1.5, AVs 0.3)
    alarm: jnp.ndarray  # how long a crossing threat has been visible [s]
    red_t: jnp.ndarray  # how long my light has been red [s] (runners only enter in the first 1 s: "late" red running)
    green_t: jnp.ndarray  # how long my light has been green [s] (a stopped driver starts after its reaction time)
    wait_t: jnp.ndarray  # how long I have been stopped at a stop sign [s] (first stopped goes first)
    m: jnp.ndarray  # metrics: [entered, exited, rear, angle, left, sideswipe, other, rlr crashes, red runs, lane changes, veh-s, blocked spawns]


N_METRICS = 12
KMAX = 64  # conflict points per layout (padded)
S_MAX = 12  # signal segments per layout (padded)


class Layout(NamedTuple):
    """One intersection's tables (geometry.py + a control plan). Stack layouts to vmap over different intersections."""
    path_table: jnp.ndarray  # [4, 3, 3] path for (approach, lane, move), -1 invalid
    default_move: jnp.ndarray  # [4, 3]
    lc_target: jnp.ndarray  # [4, 3, 3] lane that allows the move
    spawn_lane: jnp.ndarray  # [4, 3, 2] the lanes a vehicle making a move may arrive in
    move_ok: jnp.ndarray  # [4, 3]
    nin: jnp.ndarray  # [4] inbound lanes
    path_box: jnp.ndarray
    path_len: jnp.ndarray
    path_exit: jnp.ndarray
    path_cap: jnp.ndarray  # [16] speed cap through the box (turns)
    cp: jnp.ndarray  # [KMAX] conflict points (padded with path -1)
    cq: jnp.ndarray
    sp: jnp.ndarray
    sq: jnp.ndarray
    ctype: jnp.ndarray
    pc_idx: jnp.ndarray  # [16, kp] per path: conflict index, other side, own position
    pc_side: jnp.ndarray
    pc_own: jnp.ndarray
    conf_pp: jnp.ndarray  # [16, 16] paths that conflict
    sig_end: jnp.ndarray  # [S_MAX] segment end times (padded with 1e9)
    sig_state: jnp.ndarray  # [S_MAX, 16] 0 red, 1 yellow, 2 green
    sig_perm: jnp.ndarray  # [S_MAX, 16] permissive (yield) movement
    cycle: jnp.ndarray
    stop_sign: jnp.ndarray  # [16] stop-controlled path: go after a full stop when clear
    rtor: jnp.ndarray  # [16] right turn on red allowed
    demand: jnp.ndarray  # [4] demand multiplier per approach (x Behaviour.demand)
    vlim: jnp.ndarray  # [4] speed limit per approach [m/s]


PATH_MOVE = jnp.asarray(G.CANON.path_move)  # the path slots are the same in every layout: L, T, T, R per approach
PATH_APP = jnp.asarray(G.CANON.path_a)


def _signal_plan(geo, control, major=(0, 2), green=None):
    """Signal plan: per axis (slots 0+2, 1+3) a protected left phase if the axis has left-turn bays, then through with
    permissive lefts; 3 s yellow, 1-2 s all-red. Two-way stop: the major road is always green (lefts yield), the minor
    road stops. Returns ends, state, perm, stop_sign, rtor."""
    segs = []
    valid = geo.path_valid

    def st(paths, lvl, perm=()):
        s = np.zeros(G.N_PATHS, np.int32)
        pm = np.zeros(G.N_PATHS, bool)
        for p in paths:
            s[p] = lvl
        for p in perm:
            s[p] = lvl
            pm[p] = True
        return s, pm

    rtor = np.array([valid[p] and G.MOVES[p % 4] == "R" for p in range(G.N_PATHS)])
    stop = np.zeros(G.N_PATHS, bool)
    if control == "signal":
        green = green or {(0, 2): 30.0, (1, 3): 30.0}
        for axis in ((0, 2), (1, 3)):
            ps = [a * 4 + k for a in axis for k in range(4) if valid[a * 4 + k]]
            if not ps:
                continue
            lefts = [a * 4 for a in axis if valid[a * 4]]
            bays = lefts and all(geo.nin[a] == 3 for a in axis if geo.present[a])
            if bays:
                segs += [(12, *st(lefts, 2)), (3, *st(lefts, 1)), (1, *st([], 0))]
                others = [p for p in ps if p not in lefts]
            else:
                others = [p for p in ps if p not in lefts]
            segs += [(green[axis], *st(others, 2, lefts)), (3, *st(others, 1, lefts)), (2, *st([], 0))]
    else:  # two-way stop
        majors = [a * 4 + k for a in major for k in range(4) if valid[a * 4 + k]]
        lefts = [p for p in majors if p % 4 == 0]
        segs = [(60.0, *st([p for p in majors if p % 4 != 0], 2, lefts))]
        stop = np.array([valid[p] and (p // 4) not in major for p in range(G.N_PATHS)])
        rtor[:] = False
    # safety net (skewed layouts): two conflicting movements green together without either yielding -> the turn yields
    conflicts = [(p, q) for p, q, *_ in geo.conflicts]
    for _, s_, pm in segs:
        for p, q in conflicts:
            if s_[p] >= 1 and s_[q] >= 1 and not pm[p] and not pm[q]:
                order = sorted((p, q), key=lambda x: {"L": 0, "R": 1, "T": 2}[G.MOVES[x % 4]])
                pm[order[0]] = True
    n = len(segs)
    assert n <= S_MAX
    ends = np.full(S_MAX, 1e9, np.float32)
    ends[:n] = np.cumsum([x[0] for x in segs])
    state = np.zeros((S_MAX, G.N_PATHS), np.int32)
    perm = np.zeros((S_MAX, G.N_PATHS), bool)
    state[:n] = np.stack([x[1] for x in segs])
    perm[:n] = np.stack([x[2] for x in segs])
    return ends, state, perm, stop, rtor


def make_layout(geo=None, control="signal", major=(0, 2), green=None, demand=None, vlim=None):
    geo = geo or G.CANON
    c = geo.conflicts
    k = len(c)
    assert k <= KMAX, k
    cp = -np.ones(KMAX, np.int32)
    cq = -np.ones(KMAX, np.int32)
    sp = np.zeros(KMAX, np.float32)
    sq = np.zeros(KMAX, np.float32)
    ct = np.full(KMAX, G.OTHER, np.int32)
    for i, (p, q, s_p, s_q, t) in enumerate(c):
        cp[i], cq[i], sp[i], sq[i], ct[i] = p, q, s_p, s_q, t
    rows = [[] for _ in range(G.N_PATHS)]
    for i, (p, q, s_p, s_q, _t) in enumerate(c):
        rows[p].append((i, 1, s_p))  # the other side is q; own conflict position s_p
        rows[q].append((i, 0, s_q))
    kp = 16
    assert max(len(r) for r in rows) <= kp
    idx = -np.ones((G.N_PATHS, kp), np.int32)
    side = np.zeros((G.N_PATHS, kp), np.int32)
    own = np.zeros((G.N_PATHS, kp), np.float32)
    for p, r in enumerate(rows):
        for j, (i, sd, so) in enumerate(r):
            idx[p, j], side[p, j], own[p, j] = i, sd, so
    ends, state, perm, stop, rtor = _signal_plan(geo, control, major, green)
    cyc = float(ends[ends < 1e8][-1])
    demand = np.ones(4) if demand is None else np.asarray(demand, float)
    demand = demand * (geo.nin > 0)
    vlim = np.full(4, VLIM) if vlim is None else np.asarray(vlim, float)
    f = jnp.asarray
    return Layout(f(geo.path_table), f(geo.default_move), f(geo.lc_target), f(geo.spawn_lane), f(geo.move_ok), f(geo.nin),
                  f(geo.path_box), f(geo.path_len), f(geo.path_exit), f(geo.path_cap), f(cp), f(cq), f(sp), f(sq), f(ct), f(idx), f(side), f(own),
                  f(geo.conf_pp), f(ends), f(state), f(perm), f(np.float32(cyc)), f(stop), f(rtor),
                  f(demand.astype(np.float32)), f(vlim.astype(np.float32)))


CANON = make_layout()
CYCLE = float(CANON.cycle)
SIG_END, SIG_STATE, SIG_PERM = CANON.sig_end, CANON.sig_state, CANON.sig_perm


def stack(layouts):
    return jax.tree.map(lambda *x: jnp.stack(x), *layouts)


def signals(t, offset, lay: Layout = CANON):
    tau = jnp.mod(t + offset, lay.cycle)
    seg = jnp.searchsorted(lay.sig_end, tau, side="right")
    seg = jnp.minimum(seg, S_MAX - 1)
    return lay.sig_state[seg], lay.sig_perm[seg]


def idm(v, v0, gap, vl, amax, b, T, s0):
    s_star = s0 + jnp.maximum(0.0, v * T + v * (v - vl) / (2.0 * jnp.sqrt(amax * b)))
    return amax * (1.0 - (v / v0) ** 4 - (s_star / jnp.maximum(gap, 0.1)) ** 2)


# ---------------------------------------------------------------- reset / spawn
def reset(key, beh: Behaviour = Behaviour(), lay: Layout = CANON):
    k1, k2 = jax.random.split(key)
    z = jnp.zeros(M)
    zi = jnp.zeros(M, jnp.int32)
    zb = jnp.zeros(M, bool)
    return State(key=k1, t=jnp.zeros(()), offset=jax.random.uniform(k2, maxval=lay.cycle), alive=zb, app=zi, lane=zi, mov=zi,
                 path=-jnp.ones(M, jnp.int32), s=z, v=z, v0=jnp.full(M, VLIM), T=z + 1.4, amax=z + 1.5, b=z + 2.0, s0=z + 2.0,
                 length=z + 4.6, runner=zb, nolook=zb, nosignal=zb, aggr=zb, av=zb, dist_t=z, ran_red=zb, stopped=zb, ydec=zi, react=z + 1.0, alarm=z, red_t=z, green_t=z, wait_t=z,
                 m=jnp.zeros(N_METRICS))


def intended_path(st: State, lay: Layout = CANON):
    p = lay.path_table[st.app, st.lane, st.mov]
    p_def = lay.path_table[st.app, st.lane, lay.default_move[st.app, st.lane]]
    return jnp.where(st.path >= 0, st.path, jnp.where(p >= 0, p, p_def))


def step(st: State, beh: Behaviour = Behaviour(), lay: Layout = CANON):
    key, k_u = jax.random.split(st.key)
    u = jax.random.uniform(k_u, (3 * M + 4 * 16,))
    u_dist, u_dur, u_lc, u_sp = u[:M], u[M:2 * M], u[2 * M:3 * M], u[3 * M:].reshape(4, 16)
    alive = st.alive
    path = intended_path(st, lay)
    box = lay.path_box[path]
    in_app = st.s < G.APP
    in_box = (~in_app) & (st.s < G.APP + box)
    sig, perm = signals(st.t, st.offset, lay)

    # ---- leader search over two (track key, coordinate) views per vehicle
    k1 = jnp.where(in_app, st.app * 3 + st.lane, jnp.where(in_box, 12 + path, 28 + lay.path_exit[path]))
    c1 = jnp.where(in_box | in_app, st.s, st.s - G.APP - box)
    k2 = jnp.where(in_app & (st.s > G.APP - 40), 12 + path, jnp.where(in_box, 28 + lay.path_exit[path], -1))
    c2 = jnp.where(in_app, st.s, st.s - G.APP - box)
    k1 = jnp.where(alive, k1, -2)
    k2 = jnp.where(alive, k2, -3)
    big = 1e9

    def pair(ki, ci, kj, cj):
        d = cj[None, :] - ci[:, None]
        ok = (ki[:, None] == kj[None, :]) & (ki[:, None] >= 0) & (d > 0)
        return jnp.where(ok, d, big)

    d_phys = jnp.minimum(jnp.minimum(pair(k1, c1, k1, c1), pair(k1, c1, k2, c2)), pair(k2, c2, k1, c1))
    d = jnp.minimum(d_phys, pair(k2, c2, k2, c2))  # both projected onto the exit lane they merge into
    d = jnp.where(jnp.eye(M, dtype=bool), big, d)
    gaps = d - st.length[None, :]
    lead = jnp.argmin(gaps, axis=1)
    gap_lead = jnp.take_along_axis(gaps, lead[:, None], 1)[:, 0]
    has_lead = gap_lead < 1e8
    v_lead = st.v[lead]

    # ---- gap acceptance: time for other vehicles to reach each conflict point
    # yellow: decided once at onset. Go only if a comfortable stop is impossible and the line is reached in time (turners
    # slow down for the turn); otherwise stop, braking hard if needed. Red runners always go.
    dist = G.APP - st.s
    s_state = sig[path]
    cap0 = lay.path_cap[path]
    t_reach = dist / jnp.maximum(jnp.minimum(st.v, cap0 + 3.0), 0.5)
    red_t = jnp.where(in_app & (s_state == 0), st.red_t + DT, 0.0)
    green_t = jnp.where(in_app & (s_state == 2), st.green_t + DT, 0.0)
    late_runner = st.runner & (st.v > 4.0) & (red_t < beh.red_window)  # red running is mostly in the first second of red
    go_y = ((st.v ** 2 / 6.0 >= dist - 0.5) & (t_reach < 2.6)) | (st.runner & (st.v > 4.0))
    ydec = jnp.where(in_app & (s_state == 1), jnp.where(st.ydec == 0, jnp.where(go_y, 2, 1), st.ydec), 0)
    # a car facing red (or that decided to stop on yellow) before its line will not come; others may accelerate
    # cars waiting at the line to yield (stop sign, or a permissive turn): the first to stop goes first. They are not
    # "coming" for each other, so two of them facing each other cannot wait for each other forever.
    s_perm = perm[path]
    waiting = alive & in_app & (dist < 3) & (st.v < 0.5) & s_perm & (s_state >= 1)
    at_sign = lay.stop_sign[path] & alive & in_app & st.stopped & (dist < 3)
    prio = waiting | at_sign
    wait_t = jnp.where(prio, st.wait_t + DT, 0.0)
    first = (wait_t[None, :] > wait_t[:, None]) | ((wait_t[None, :] == wait_t[:, None]) & (jnp.arange(M)[None, :] < jnp.arange(M)[:, None]))
    yield_to = (prio[None, :] & lay.conf_pp[path[:, None], path[None, :]] & first).any(1)
    queued = in_app & has_lead & (gap_lead < 8.0) & (v_lead < 0.5) & (st.v < 0.5)  # standing behind a stopped car
    coming = (alive & ~(in_app & (((s_state == 0) & ~late_runner) | (ydec == 1))) & ~waiting & ~queued) | (at_sign & (st.v >= 0.5))

    def tta(cpath, cs):
        rem = cs[:, None] - st.s[None, :]  # [K, M]
        on = (path[None, :] == cpath[:, None]) & coming[None, :] & (rem > -LEN_CLEAR - st.length[None, :])
        r = jnp.maximum(rem, 0.0)
        t_ = (jnp.sqrt(st.v[None, :] ** 2 + 3.0 * r) - st.v[None, :]) / 1.5
        t_ = jnp.minimum(t_, r / jnp.maximum(st.v[None, :], 0.1))  # at speed, it may simply keep going
        return jnp.where(on, t_, big).min(1)  # [K]

    t_side = jnp.stack([tta(lay.cp, lay.sp), tta(lay.cq, lay.sq)])  # [2, K]: arrival time of the nearest vehicle on side p / side q
    ci, cs_ = lay.pc_idx[path], lay.pc_side[path]  # [M, kmax]
    t_other = jnp.where(ci >= 0, t_side[cs_, jnp.maximum(ci, 0)], big)
    tcrit = jnp.where(PATH_MOVE[path] == 0, 4.5, 5.5) * jnp.where(st.aggr, 0.65, 1.0) * jnp.where(st.av, 1.25, 1.0)
    # own time to clear each conflict point, accelerating at 1.5 m/s^2 from the current speed
    d_clear = jnp.maximum(lay.pc_own[path] - st.s[:, None], 0.0) + LEN_CLEAR + st.length[:, None]
    capm = jnp.minimum(lay.path_cap[path], 30.0)[:, None]
    v_ = jnp.minimum(st.v[:, None], capm)
    t_acc = (capm - v_) / 1.5  # accelerate to the turn speed, then hold it
    d_acc = 0.5 * (v_ + capm) * t_acc
    t_self = jnp.where(d_clear <= d_acc, (jnp.sqrt(v_ ** 2 + 3.0 * d_clear) - v_) / 1.5, t_acc + (d_clear - d_acc) / capm)
    margin = jnp.where(st.aggr, 0.3, 1.0)[:, None] * jnp.where(st.av, 1.5, 1.0)[:, None]
    gap_ok = ((t_other > tcrit[:, None]) & (t_other > t_self + margin)).all(1) | st.nolook

    # ---- evasive braking: a car moving through the box will reach a conflict point on my path before I could clear it.
    # Seen after my reaction time (if I am not distracted), I brake to stop short of that point.
    def tta_box(cpath, cs):
        rem = cs[:, None] - st.s[None, :]
        on = (path[None, :] == cpath[:, None]) & alive[None, :] & (st.s[None, :] > G.APP - 2.0) & (st.v[None, :] > 1.0) \
            & (rem > -LEN_CLEAR - st.length[None, :])
        return jnp.where(on, jnp.maximum(rem, 0.0) / jnp.maximum(st.v[None, :], 1.0), big).min(1)

    t_box = jnp.stack([tta_box(lay.cp, lay.sp), tta_box(lay.cq, lay.sq)])
    t_thr = jnp.where(ci >= 0, t_box[cs_, jnp.maximum(ci, 0)], big)  # [M, kmax]
    to_zone = lay.pc_own[path] - LEN_CLEAR - st.s[:, None]  # my front to the conflict zone
    t_me = jnp.maximum(to_zone, 0.0) / jnp.maximum(st.v[:, None], 0.5)
    threat = (ci >= 0) & (to_zone > 0.0) & (to_zone < 40.0) & (t_thr < t_me + 2.0)
    alarm = jnp.where(threat.any(1) & (st.dist_t <= 0), st.alarm + DT, 0.0)
    evade = threat & (alarm >= st.react)[:, None]
    evade_gap = jnp.where(evade, to_zone - 0.5, big).min(1)

    # ---- stop line
    box_busy = ((ci >= 0) & (t_thr < 6.0)).any(1)  # a car is moving through the box on a path that crosses mine
    near = in_app & (dist < 60)
    moving_runner = late_runner
    rtor = lay.rtor[path] & st.stopped & gap_ok
    # stop sign: after a full stop, go when the gaps are clear and nobody is crossing; of two stopped cars on crossing
    # paths, the one that stopped first goes first
    go_sign = at_sign & ((gap_ok & ~box_busy & ~yield_to & (wait_t >= st.react)) | (st.nolook & (wait_t > 0.5)))
    must = near & (((s_state == 0) & ~moving_runner & ~rtor & ~go_sign) | ((s_state == 1) & (ydec == 1))
                   | ((s_state >= 1) & s_perm & (~gap_ok | (waiting & yield_to)) & (dist < 20))  # permissive: yield on green and yellow
                   | ((s_state == 2) & st.stopped & (dist < 3) & (green_t < st.react))  # start-up lost time
                   | ((s_state == 2) & st.stopped & (dist < 3) & box_busy & ~st.nolook))  # fresh green: look before going
    stopped = st.stopped | (near & (dist < 3) & (st.v < 0.5))

    # ---- distraction (and the surprise of being cut in front of without a signal)
    start = (u_dist < beh.distract_rate * DT) & (st.dist_t <= 0) & ~st.av & alive
    dist_t = jnp.where(start, 1.5 + 1.5 * u_dur, jnp.maximum(st.dist_t - DT, 0.0))
    distracted = dist_t > 0

    # ---- longitudinal control
    cap = lay.path_cap[path]
    v0e = jnp.minimum(st.v0, jnp.where(in_app | in_box, cap + 0.3 * jnp.maximum(dist, 0.0), 99.0))
    stop_gap = jnp.minimum(jnp.where(must, dist - 0.5, big), evade_gap)
    g_eff = jnp.minimum(jnp.where(has_lead, gap_lead, big), stop_gap)
    vl_eff = jnp.where(stop_gap < gap_lead, 0.0, v_lead)
    acc = jnp.where(g_eff < 1e8, idm(st.v, v0e, g_eff, vl_eff, st.amax, st.b, st.T, st.s0), st.amax * (1 - (st.v / v0e) ** 4))
    acc = jnp.clip(acc, -9.0, st.amax)
    acc = jnp.where(distracted, jnp.minimum(0.0, acc) * 0.0, acc)  # distracted: no reaction, holds its speed
    v_new = jnp.maximum(st.v + acc * DT, 0.0)
    s_new = st.s + 0.5 * (st.v + v_new) * DT
    v_new, s_new = jnp.where(alive, v_new, 0.0), jnp.where(alive, s_new, 0.0)

    # ---- lane changes on the approach (to reach a lane that allows the turn)
    need = in_app & (lay.path_table[st.app, st.lane, st.mov] < 0) & (st.s > 10) & (dist > 15) & alive
    goal = lay.lc_target[st.app, st.lane, st.mov]
    target = st.lane + jnp.sign(goal - st.lane)
    same_t = alive[None, :] & in_app[None, :] & (st.app[None, :] == st.app[:, None]) & (st.lane[None, :] == target[:, None])
    ds = s_new[None, :] - s_new[:, None]
    gf = jnp.where(same_t & (ds > 0), ds - st.length[None, :], big).min(1)
    gb_all = jnp.where(same_t & (ds <= 0), -ds - st.length[:, None], big)
    follower = jnp.argmin(gb_all, axis=1)
    gb = gb_all.min(1)
    fac = jnp.where(st.nosignal, 0.5, 1.0)
    v_ahead = v_new[jnp.argmin(jnp.where(same_t & (ds > 0), ds, big), axis=1)]
    need_f = 2.0 + 0.3 * v_new + jnp.maximum(v_new - v_ahead, 0.0) ** 2 / 6.0
    need_b = 2.0 + 0.5 * v_new[follower] + jnp.maximum(v_new[follower] - v_new, 0.0) ** 2 / 6.0
    accept = ((gf > need_f * fac) & (gb > need_b * fac)) | st.nolook
    # into the middle lane only from one side per step (two cars merging side by side into the same spot)
    middle = (target > 0) & (target < lay.nin[st.app] - 1)
    side_ok = jnp.where(middle, (st.lane < target) == (jnp.round(st.t / DT).astype(jnp.int32) % 2 == 0), True)
    do_lc = need & (u_lc < 0.3) & accept & side_ok
    lane = jnp.where(do_lc, target, st.lane)
    ss_crash = do_lc & ((gf < 0) | (gb < 0))  # merged into a car that was there (nobody looked)
    ss_victim = jnp.zeros(M, bool).at[follower].max(ss_crash & (gb < 0))
    lead_t = jnp.argmin(jnp.where(same_t & (ds > 0), ds, big), axis=1)
    ss_victim = ss_victim.at[lead_t].max(ss_crash & (gf < 0))
    surprise = do_lc & st.nosignal & (gb < 12)
    dist_t = dist_t.at[follower].max(jnp.where(surprise, 1.0, 0.0))

    # ---- crossing the stop line: the path is fixed, red runs counted
    cross = alive & in_app & (s_new >= G.APP)
    new_path = jnp.where(cross, path, st.path)
    red_run = cross & (s_state == 0) & ~((lay.rtor[path] | lay.stop_sign[path]) & st.stopped)  # legal after a full stop
    ran_red = st.ran_red | red_run

    # ---- collisions
    disp = s_new - st.s
    # a rear-end needs the two cars on one physical lane: a leader seen only through the merge projection counts once
    # both are at the merge (their side-by-side contact before it is a sideswipe at the merge conflict point)
    phys = (jnp.take_along_axis(d_phys, lead[:, None], 1)[:, 0] < 1e8) | (c2 > -st.length - 1.0)
    rear = alive & has_lead & alive[lead] & phys & (gap_lead + disp[lead] - disp < 0)
    path_now = jnp.where(new_path >= 0, new_path, path)

    def occ(cpath, cs):  # the car's body (rear bumper .. front bumper), widened by half a car width
        body = (s_new[None, :] + 1.0 > cs[:, None]) & (s_new[None, :] - st.length[None, :] - 1.0 < cs[:, None])
        return (path_now[None, :] == cpath[:, None]) & alive[None, :] & body

    oP, oQ = occ(lay.cp, lay.sp), occ(lay.cq, lay.sq)  # [K, M]
    hit = oP.any(1) & oQ.any(1)  # [K]
    in_conf = ((oP | oQ) & hit[:, None]).any(0)
    rlr_conf = (((oP | oQ) & hit[:, None]) & ran_red[None, :]).any(1)
    crashed = rear | jnp.zeros(M, bool).at[lead].max(rear) | in_conf | ss_crash | ss_victim
    ctype_counts = jnp.zeros(5).at[lay.ctype].add(hit.astype(jnp.float32))
    ctype_counts = ctype_counts.at[G.REAR].add(rear.sum()).at[G.SIDESWIPE].add(ss_crash.sum())
    rlr = rlr_conf.sum() + (rear & (ran_red | ran_red[lead])).sum()

    # ---- exits
    done = alive & (s_new > lay.path_len[path_now])
    alive = alive & ~crashed & ~done

    # ---- arrivals: one Bernoulli per approach per step
    want = u_sp[:, 0] < beh.demand * lay.demand / 3600.0 * DT
    pw = jnp.array([beh.p_left, 1.0 - beh.p_left - beh.p_right, beh.p_right])[None, :] * lay.move_ok  # [4, 3]
    cw = jnp.cumsum(pw, 1) / jnp.maximum(pw.sum(1, keepdims=True), 1e-9)
    mov = (u_sp[:, 1:2] >= cw[:, :2]).sum(1).astype(jnp.int32)  # 0 L, 1 T, 2 R
    good_lane = lay.spawn_lane[jnp.arange(4), mov, (u_sp[:, 2] >= 0.5).astype(jnp.int32)]
    nl = jnp.maximum(lay.nin, 1)
    wrong = (u_sp[:, 3] < beh.p_wrong_lane) & (nl > 1)
    lane_sp = jnp.where(wrong, (good_lane + 1 + ((u_sp[:, 2] < 0.5) & (nl == 3)).astype(jnp.int32)) % nl, good_lane)
    in_lane = alive[None, :] & (st.app[None, :] == jnp.arange(4)[:, None]) & (lane[None, :] == lane_sp[:, None]) & (s_new[None, :] < G.APP)
    e_gap_all = jnp.where(in_lane, s_new[None, :] - st.length[None, :], big)
    e_gap = e_gap_all.min(1)
    e_v = v_new[jnp.argmin(e_gap_all, axis=1)]
    v_spawn = jnp.where(e_gap < 1e8, jnp.minimum(8.0 + 4.0 * u_sp[:, 5], e_v + 2.0), 8.0 + 4.0 * u_sp[:, 5])
    entry_clear = e_gap > 5.0 + 1.2 * v_spawn
    order = jnp.argsort(alive)  # free slots first
    slots = order[:4]
    ok = want & entry_clear & ~alive[slots]
    blocked = (want & ~ok).sum()

    def put(arr, val):
        return arr.at[slots].set(jnp.where(ok, val, arr[slots]))

    is_av = u_sp[:, 4] < beh.p_av
    human = ~is_av
    vals = dict(
        app=jnp.arange(4), lane=lane_sp, mov=mov, path=-jnp.ones(4, jnp.int32), s=jnp.zeros(4), v=v_spawn,
        v0=lay.vlim * (0.9 + 0.25 * u_sp[:, 6]), T=jnp.where(is_av, 1.3, 1.0 + 0.8 * u_sp[:, 7]),
        amax=jnp.where(is_av, 1.6, 1.2 + 0.8 * u_sp[:, 8]), b=jnp.where(is_av, 2.0, 1.5 + 1.0 * u_sp[:, 9]),
        s0=1.5 + 1.0 * u_sp[:, 10], length=4.2 + 0.8 * u_sp[:, 11],
        runner=human & (u_sp[:, 12] < beh.p_red), nolook=human & (u_sp[:, 13] < beh.p_nolook),
        nosignal=human & (u_sp[:, 14] < beh.p_nosignal), aggr=human & (u_sp[:, 15] < beh.p_aggressive), av=is_av,
        dist_t=jnp.zeros(4), ran_red=jnp.zeros(4, bool), stopped=jnp.zeros(4, bool), ydec=jnp.zeros(4, jnp.int32),
        react=jnp.where(is_av, 0.3, 0.7 + 0.8 * u_sp[:, 2]), alarm=jnp.zeros(4), red_t=jnp.zeros(4), green_t=jnp.zeros(4), wait_t=jnp.zeros(4),
        alive=jnp.ones(4, bool))
    cur = dict(app=st.app, lane=lane, mov=st.mov, path=new_path, s=s_new, v=v_new, v0=st.v0, T=st.T, amax=st.amax, b=st.b,
               s0=st.s0, length=st.length, runner=st.runner, nolook=st.nolook, nosignal=st.nosignal, aggr=st.aggr, av=st.av,
               dist_t=dist_t, ran_red=ran_red, stopped=stopped & alive, ydec=ydec, react=st.react, alarm=alarm, red_t=red_t, green_t=green_t, wait_t=wait_t, alive=alive)
    new = {k: put(cur[k], vals[k]) for k in cur}

    m = st.m + jnp.concatenate([jnp.stack([cross.sum(), done.sum()]), ctype_counts,
                                jnp.stack([rlr, red_run.sum(), do_lc.sum(), st.alive.sum() * DT, blocked])])
    return State(key=key, t=st.t + DT, offset=st.offset, m=m, **new)


def run_batch(key, n_envs, seconds, beh: Behaviour = Behaviour(), warmup=120.0, lay: Layout = CANON, per_env=False):
    """Simulate `n_envs` independent intersections for `seconds` (after a warm-up); returns summed metrics (or [n_envs,
    N_METRICS] with per_env). `lay` is one Layout for all, or a stacked Layout with a leading n_envs axis."""
    ax = 0 if lay.cycle.ndim == 1 else None
    st = jax.vmap(reset, in_axes=(0, None, ax))(jax.random.split(key, n_envs), beh, lay)
    vstep = jax.vmap(step, in_axes=(0, None, ax))

    @jax.jit
    def chunk(st, lay):
        return jax.lax.scan(lambda s, _: (vstep(s, beh, lay), None), st, None, length=250)[0]  # 50 s

    for _ in range(int(warmup / (250 * DT))):
        st = chunk(st, lay)
    st = st._replace(m=jnp.zeros_like(st.m))
    for _ in range(int(np.ceil(seconds / (250 * DT)))):
        st = chunk(st, lay)
    return (np.asarray(st.m) if per_env else np.asarray(st.m.sum(0))), st


def summary(m):
    entered = max(m[0], 1.0)
    mev = entered / 1e6
    crashes = m[2:7]
    tot = crashes.sum()
    out = {"entered_vehicles": int(m[0]), "crashes": int(tot), "crashes_per_MEV": tot / mev,
           "share": {n: (c / tot if tot else 0.0) for n, c in zip(G.CRASH_NAMES, crashes)},
           "red_light_running_share": (m[7] / tot if tot else 0.0), "red_runs_per_1000_entering": 1000 * m[8] / entered,
           "lane_changes_per_vehicle": m[9] / entered, "blocked_arrivals": int(m[11])}
    return out
