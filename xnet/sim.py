"""A batched, pure-JAX microsimulation of a whole road network (xnet.network): car following on every lane, lane changes
and zipper merges, junctions with signals / two-way stops / all-way stops, gap acceptance and right-of-way at every
conflict point inside a junction, pedestrians on crosswalks, buses at stops, double-parked deliveries, breakdowns, crash
wrecks that block a lane, congestion-aware routing, and the risky human behaviour of xworld.

Vehicles live on "tracks": a lane of a link (2 * link + lane) or a path through a junction box (2 * L + move). Each step
(DT = 0.2 s), for every vehicle slot at once:

  sort all vehicles by (track, position)  -> leader on the same track, gaps on the other lane, the last car on the next track
  signals and stop signs, right of way (priority by approach and turn; first to stop goes first among equals),
  gap acceptance against cars in the box and cars arriving, room after the box ("don't block the box"), crosswalks
  -> IDM car following toward the leader / stop line / a stopped bus, delivery van, breakdown or wreck
  -> lane changes: mandatory (turn lane, lane ends: zipper merge) and discretionary (pass a slow or stopped car);
     passing a blocked single lane on the opposing lane when it is clear
  -> crashes: rear-end, angle, left-turn, sideswipe, pedestrian; crashed cars stay as wrecks for a few minutes
  -> moves between tracks, trips that finish, gridlock teleports, new trips from the map edge and parking

Everything has a fixed shape, so `jax.vmap` runs many copies (seeds, behaviour mixes) of the area at once.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

DT = 0.2
OBS_DIM = 28  # AV policy observation (see step)
SCALE = 32768  # sort key = track * SCALE + position in 1/8 m
CRASH_NAMES = ("rear-end", "angle", "left-turn", "sideswipe", "pedestrian")
METRICS = ("entered", "finished", "trip_s", "spawned", "blocked", "teleports", "rear-end", "angle", "left-turn",
           "sideswipe", "pedestrian", "red_runs", "lane_changes", "veh_s", "veh_m", "breakdowns", "stops", "passes",
           "peds", "rlr_crashes", "av_finished", "av_trip_s", "av_veh_s", "av_veh_m", "av_hard", "av_jerk2", "av_crash",
           "av_red", "av_near", "av_shield")
NM = len(METRICS)
CAR, BUS, VAN = 0, 1, 2


class Behaviour(NamedTuple):
    """Per-driver probabilities (per-second hazards for distraction and breakdowns); defaults = xworld's calibration."""
    p_red: float = 0.0514
    red_window: float = 0.3
    p_nolook: float = 1.13e-5
    p_nosignal: float = 0.0
    p_aggressive: float = 0.2
    distract_rate: float = 1.93e-7
    p_av: float = 0.0
    p_bus: float = 0.03
    p_van: float = 0.05
    p_park: float = 0.3  # a delivery van double-parks on a link of tertiary class or above
    breakdown_rate: float = 2e-6  # per second of driving
    p_ped_violate: float = 0.03  # a pedestrian crosses against the signal / without a gap
    demand: float = 1.0  # multiplier on the network's inferred demand


class Reward(NamedTuple):
    """RL reward weights per 0.2 s step (the corridor env's, rescaled from 0.1 s steps)."""
    progress: float = 0.02  # per metre
    idle: float = 0.05  # standing still with nothing to wait for (no stop target within 10 m, no car within 8 m)
    team: float = 0.01  # per m/s of the mean speed of every vehicle in the area (credit for keeping traffic moving)
    comfort: float = 0.001  # per (m/s^2)^2
    hard: float = 0.2  # per m/s^2 of braking beyond 3
    jerk: float = 0.00004  # per (m/s^3)^2
    headway: float = 0.1  # per second of time headway below 0.8 s
    speed: float = 0.1  # per m/s over the limit
    shield: float = 0.04  # per m/s^2 the safety envelope took off the command
    crash: float = -30.0
    red: float = -20.0
    near: float = -3.0  # entering a junction less than 2 s ahead of a car with priority


R = Reward()


class State(NamedTuple):
    key: jnp.ndarray
    t: jnp.ndarray
    alive: jnp.ndarray
    trk: jnp.ndarray
    s: jnp.ndarray
    v: jnp.ndarray
    dest: jnp.ndarray
    nxt: jnp.ndarray  # slot at the end of the current link (-1: the trip ends on this link)
    kind: jnp.ndarray
    v0f: jnp.ndarray
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
    react: jnp.ndarray
    dist_t: jnp.ndarray
    stop_t: jnp.ndarray  # stopped for a bus stop / delivery / breakdown [s left]
    park_s: jnp.ndarray  # where on this link the next planned stop is (-1 none)
    broken: jnp.ndarray
    wreck_t: jnp.ndarray  # crashed: a wreck for this many more seconds
    stopped: jnp.ndarray
    wait_t: jnp.ndarray
    red_t: jnp.ndarray
    ydec: jnp.ndarray
    ran_red: jnp.ndarray
    stuck_t: jnp.ndarray
    behind_t: jnp.ndarray  # how long stuck behind a stopped vehicle (then pass on the opposing lane)
    lc_t: jnp.ndarray  # lane-change cooldown
    t0: jnp.ndarray  # trip start
    acc: jnp.ndarray  # acceleration applied last step
    gsc: jnp.ndarray  # gap-acceptance scale (AVs: chosen by the policy; 1.25 = the rule-based AV)
    ped_t: jnp.ndarray  # [legs] crossing time left
    link_vs: jnp.ndarray  # [L] sum of speeds (routing)
    link_n: jnp.ndarray
    m: jnp.ndarray


SS_METHOD = "scan"


def _ss(a, v):
    return jnp.searchsorted(a, v, method=SS_METHOD)


def to_jnp(net):
    return {k: jnp.asarray(v) for k, v in net.items()}


def reset(key, net, n_slots):
    z = jnp.zeros(n_slots)
    zi = jnp.zeros(n_slots, jnp.int32)
    zb = jnp.zeros(n_slots, bool)
    L = net["link_len"].shape[0]
    return State(key=key, t=jnp.zeros(()), alive=zb, trk=zi, s=z, v=z, dest=zi, nxt=zi, kind=zi, v0f=z + 1, T=z + 1.4,
                 amax=z + 1.5, b=z + 2.0, s0=z + 2.0, length=z + 4.6, runner=zb, nolook=zb, nosignal=zb, aggr=zb, av=zb,
                 react=z + 1.0, dist_t=z, stop_t=z, park_s=z - 1, broken=zb, wreck_t=z, stopped=zb, wait_t=z, red_t=z, ydec=zi,
                 ran_red=zb, stuck_t=z, behind_t=z, lc_t=z, t0=z, acc=z, gsc=z + 1.25, ped_t=jnp.zeros(net["leg_node"].shape[0]),
                 link_vs=jnp.zeros(L), link_n=jnp.zeros(L), m=jnp.zeros(NM))


def idm(v, v0, gap, vl, amax, b, T, s0):
    s_star = s0 + jnp.maximum(0.0, v * T + v * (v - vl) / (2.0 * jnp.sqrt(amax * b)))
    return amax * (1.0 - (v / jnp.maximum(v0, 0.1)) ** 4 - (s_star / jnp.maximum(gap, 0.1)) ** 2)


def signal_state(net, t):
    """Per link: 0 red, 1 yellow, 2 green at its end (unsignalised: 2, stop sign: 0)."""
    nd = net["link_end"]
    cyc, g0 = net["node_cycle"][nd], net["node_g0"][nd]
    tau = jnp.mod(t + net["node_offset"][nd], cyc)
    s0 = jnp.where(tau < g0, 2, jnp.where(tau < g0 + 3, 1, 0))
    s1 = jnp.where((tau >= g0 + 5) & (tau < cyc - 5), 2, jnp.where((tau >= cyc - 5) & (tau < cyc - 2), 1, 0))
    ax = net["link_axis"]
    return jnp.where(ax == 0, s0, jnp.where(ax == 1, s1, jnp.where(net["link_stop"], 0, 2)))


def leg_walk(net, t):
    """Per crosswalk leg: pedestrians may start crossing in the 10 s WALK interval after their road's axis turns red;
    unsignalised legs: always."""
    nd = net["leg_node"]
    cyc, g0 = net["node_cycle"][nd], net["node_g0"][nd]
    tau = jnp.mod(t + net["node_offset"][nd], cyc)
    ax = net["leg_axis"]
    return jnp.where(ax == 0, (tau >= g0 + 3) & (tau < g0 + 13), jnp.where(ax == 1, (tau >= cyc - 2) | (tau < 8), True))


def step(st: State, net, beh: Behaviour, nh, diag=False, rl=None):
    """One step. With `rl = (policy_fn, key)` every AV is driven by `policy_fn(obs [N, OBS_DIM], key) -> (u [N, 2], aux)`
    (u[:, 0]: acceleration command in [-1, 1] -> [-8, 3] m/s^2, capped by the safety envelope; u[:, 1]: gap-acceptance
    scale for its next junction) and the step also returns the transition (obs, u, aux, reward, valid, done)."""
    N = st.alive.shape[0]
    L = net["link_len"].shape[0]
    Mv = net["mov_len"].shape[0]
    C = net["c_type"].shape[0]
    G = net["leg_node"].shape[0]
    big = 1e9
    key, k_u = jax.random.split(st.key)
    u = jax.random.uniform(k_u, (8 * N + 3 * G + 16 * 24,))
    u_lc, u_dist, u_dur, u_brk, u_misc, u_park, u_park2, u_pass = (u[i * N:(i + 1) * N] for i in range(8))
    u_ped = u[8 * N:8 * N + 3 * G].reshape(3, G)
    u_sp = u[8 * N + 3 * G:].reshape(16, 24)
    alive = st.alive
    wreck = st.wreck_t > 0
    on_link = st.trk < 2 * L
    link = jnp.where(on_link, st.trk // 2, 0)
    lane = jnp.where(on_link, st.trk % 2, 0)
    mbox = jnp.where(on_link, 0, st.trk - 2 * L)
    nlanes = net["link_lanes"][link]
    slot_ok = (net["succ_mov"][link, 0, jnp.maximum(st.nxt, 0)] >= 0) | (net["succ_mov"][link, 1, jnp.maximum(st.nxt, 0)] >= 0)
    nxt_ok = (st.nxt >= 0) & slot_ok  # otherwise the trip ends on this link (destination, map edge, dead end)
    mv = jnp.where(on_link & nxt_ok, net["succ_mov"][link, lane, jnp.maximum(st.nxt, 0)], -1)
    mv_other = jnp.where(on_link & nxt_ok & (nlanes == 2), net["succ_mov"][link, 1 - lane, jnp.maximum(st.nxt, 0)], -1)
    finishing = on_link & ~nxt_ok
    in_box = alive & ~on_link
    tlen = jnp.where(on_link, net["link_len"][link], net["mov_len"][mbox])
    dist = tlen - st.s
    sig = signal_state(net, st.t)
    s_state = jnp.where(on_link, sig[link], 2)
    is_signal = net["link_axis"][link] >= 0
    stop_sign = net["link_stop"][link] & on_link
    turn = jnp.where(mv >= 0, net["mov_turn"][jnp.maximum(mv, 0)], 1)

    # ---- sort by (track, position): leaders, tails of tracks
    key_ = jnp.where(alive, st.trk * SCALE + jnp.clip((st.s * 8).astype(jnp.int32), 0, SCALE - 1), jnp.iinfo(jnp.int32).max)
    order = jnp.argsort(key_)
    sk = key_[order]
    rank = jnp.zeros(N, jnp.int32).at[order].set(jnp.arange(N))
    nxt_idx = order[jnp.minimum(rank + 1, N - 1)]
    same_lead = alive & (rank < N - 1) & alive[nxt_idx] & (st.trk[nxt_idx] == st.trk)
    lead = jnp.where(same_lead, nxt_idx, -1)

    T = 2 * L + Mv
    first_rank = jnp.full(T + 1, N, jnp.int32).at[jnp.where(alive, st.trk, T + 1)].min(rank, mode="drop")  # one scatter per step

    def tail_of(trk):
        """The rearmost vehicle on a track (index, or -1)."""
        r = first_rank[jnp.where(trk >= 0, trk, T)]
        return jnp.where(r < N, order[jnp.minimum(r, N - 1)], -1)

    def at_pos(trk, s):
        """(leader, follower) on track trk around position s."""
        i = _ss(sk, trk * SCALE + jnp.clip((s * 8).astype(jnp.int32), 0, SCALE - 1))
        jl = order[jnp.minimum(i, N - 1)]
        jf = order[jnp.maximum(i - 1, 0)]
        okl = (i < N) & alive[jl] & (st.trk[jl] == trk)
        okf = (i > 0) & alive[jf] & (st.trk[jf] == trk)
        return jnp.where(okl, jl, -1), jnp.where(okf, jf, -1)

    gap_same = jnp.where(lead >= 0, st.s[jnp.maximum(lead, 0)] - st.length[jnp.maximum(lead, 0)] - st.s, big)

    # ---- right of way: priority per move = 4 * approach + turn (through 2, right 1, left 0)
    sig_m = sig[net["mov_in"]]
    app_m = jnp.where(net["link_axis"][net["mov_in"]] >= 0, jnp.where(sig_m >= 1, 2, jnp.where(net["mov_turn"] == 2, 1, 0)),
                      jnp.where(net["link_stop"][net["mov_in"]], 1, 2))
    # stop-sign approaches are all equal (first to stop goes first); everything else is strictly ordered, so of two
    # conflicting movements one always yields (approach, then turn, then road class, then a fixed tie-break)
    tb = 0.05 * net["link_rank"][net["mov_in"]] + 1e-5 * (net["mov_in"] % 1000)
    pr_m = jnp.where(net["link_stop"][net["mov_in"]], 4.0, 4.0 * app_m + jnp.array([0.0, 2.0, 1.0])[net["mov_turn"]] + tb)
    pr_i = jnp.where(mv >= 0, pr_m[jnp.maximum(mv, 0)], 0)
    # yellow decided once at onset; late red runners
    red_t = jnp.where(on_link & (s_state == 0), st.red_t + DT, 0.0)
    late_runner = st.runner & (st.v > 4.0) & (red_t < beh.red_window) & is_signal
    go_y = ((st.v ** 2 / 6.0 >= dist - 0.5) & (dist / jnp.maximum(st.v, 0.5) < 2.6)) | (st.runner & (st.v > 4.0))
    ydec = jnp.where(on_link & (s_state == 1), jnp.where(st.ydec == 0, jnp.where(go_y, 2, 1), st.ydec), 0)
    # room after the box (don't block the box): the out-lane's last car, minus cars already in the box heading there
    out_trk = jnp.where(mv >= 0, 2 * net["mov_out"][jnp.maximum(mv, 0)] + net["mov_out_lane"][jnp.maximum(mv, 0)], -1)
    box_out = jnp.where(in_box, 2 * net["mov_out"][mbox] + net["mov_out_lane"][mbox], 2 * L)
    pending = jnp.zeros(2 * L).at[box_out].add(jnp.where(in_box, st.length + 2.0, 0.0), mode="drop")
    t_out = tail_of(out_trk)
    room = jnp.where(t_out >= 0, st.s[jnp.maximum(t_out, 0)] - st.length[jnp.maximum(t_out, 0)], net["link_len"][jnp.maximum(out_trk // 2, 0)])
    room = room - pending[jnp.maximum(out_trk, 0)]
    out_len = net["link_len"][jnp.maximum(out_trk // 2, 0)]
    room_ok = room > jnp.minimum(st.length + 2.0, out_len - 0.5)  # a very short link only has to be empty
    queued = on_link & (lead >= 0) & (gap_same < 8.0) & (st.v < 0.5) & (st.v[jnp.maximum(lead, 0)] < 0.5)
    waiting = on_link & (mv >= 0) & (dist < 3) & (st.v < 0.5)
    yielder = stop_sign | (turn == 0) | ((turn == 2) & (s_state == 0))
    wait_t = jnp.where(waiting & yielder & st.stopped, st.wait_t + DT, jnp.where(waiting, st.wait_t, 0.0))
    sneaking = is_signal & (turn == 0) & (s_state == 0) & (red_t < 1.6) & (dist < 3.0) & st.stopped & (st.v >= 0.5)
    coming = alive & on_link & (mv >= 0) & ~queued & ~(waiting & yielder) & (dist < 160) & ~((st.v < 0.5) & ~room_ok) & \
        (((s_state >= 1) & ~stop_sign & (ydec != 1)) | late_runner | sneaking)
    def t_at(d):
        return jnp.minimum((jnp.sqrt(st.v ** 2 + 3.0 * jnp.maximum(d, 0)) - st.v) / 1.5, d / jnp.maximum(st.v, 0.1))

    tarr_m = jnp.full(Mv, big).at[jnp.where(coming, mv, Mv)].min(t_at(dist), mode="drop")
    # look further upstream: cars in the box or on the link before, heading for a junction one link ahead
    up_m = jnp.where(on_link, jnp.maximum(mv, 0), mbox)
    up_ok = alive & (in_box | (on_link & (mv >= 0) & (s_state >= 1) & ~stop_sign & (st.v > 1.0)))
    B = net["mov_out"][up_m]
    lB = net["mov_out_lane"][up_m]
    dB = net["dest_link"][st.dest]
    slotB = nh[B, st.dest]
    m2 = jnp.where((B != dB) & (slotB >= 0), net["succ_mov"][B, jnp.minimum(lB, net["link_lanes"][B] - 1), jnp.maximum(slotB, 0)], -1)
    d2 = jnp.where(on_link, dist + net["mov_len"][up_m], dist) + net["link_len"][B]
    coming2 = up_ok & (m2 >= 0) & (d2 < 200) & (sig[B] >= 1) & ~net["link_stop"][B]
    tarr_m = tarr_m.at[jnp.where(coming2, m2, Mv)].min(t_at(d2), mode="drop")
    # and one junction further (fast traffic through two closely spaced junctions)
    m2c = jnp.maximum(m2, 0)
    B2 = net["mov_out"][m2c]
    slot2 = nh[B2, st.dest]
    m3 = jnp.where((m2 >= 0) & (B2 != dB) & (slot2 >= 0),
                   net["succ_mov"][B2, jnp.minimum(net["mov_out_lane"][m2c], net["link_lanes"][B2] - 1), jnp.maximum(slot2, 0)], -1)
    d3 = d2 + net["mov_len"][m2c] + net["link_len"][B2]
    coming3 = coming2 & (m3 >= 0) & (d3 < 250) & (sig[B2] >= 1) & ~net["link_stop"][B2]
    tarr_m = tarr_m.at[jnp.where(coming3, m3, Mv)].min(t_at(d3), mode="drop")
    wfirst_m = jnp.full(Mv, -1.0).at[jnp.where(waiting & yielder, mv, Mv)].max(wait_t, mode="drop")
    # box occupancy before / at each conflict point
    ci = net["mov_c"][mbox]  # [N, KC]
    cside = net["mov_side"][mbox]
    cs = net["mov_cs"][mbox]
    vb = in_box[:, None] & (ci >= 0)
    before = vb & (st.s[:, None] < cs + st.length[:, None] + 1.0)
    cidx = jnp.where(vb, ci * 2 + cside, 2 * C)
    occ_before = jnp.zeros(2 * C, bool).at[jnp.where(before, cidx, 2 * C)].max(True, mode="drop")
    # gap acceptance for the move I intend
    cj = net["mov_c"][jnp.maximum(mv, 0)]
    sj = net["mov_side"][jnp.maximum(mv, 0)]
    valid = (mv >= 0)[:, None] & (cj >= 0)
    other_m = net["c_mov"][jnp.maximum(cj, 0), 1 - sj]
    occ = occ_before[jnp.maximum(cj, 0) * 2 + 1 - sj]
    tcrit = (jnp.where(turn == 0, 4.5, 4.0) + jnp.where(stop_sign, 1.0, 0.0)) * jnp.where(st.aggr, 0.65, 1.0) * jnp.where(st.av, st.gsc, 1.0)
    higher = pr_m[other_m] > pr_i[:, None]
    equal = pr_m[other_m] == pr_i[:, None]
    arriving = (higher | (equal & (other_m < mv[:, None]))) & (tarr_m[other_m] < tcrit[:, None] + jnp.where(st.v[:, None] < 1.0, 1.0, 0.0))
    w_other = wfirst_m[other_m]
    first_other = (equal & (w_other >= 0) & ((w_other > wait_t[:, None]) | ((w_other == wait_t[:, None]) & (other_m < mv[:, None])))) | \
        (higher & (w_other >= 0))  # a higher-priority driver waiting at the line goes first
    gap_ok = ~(valid & (occ | arriving | first_other)).any(1)
    # crosswalks
    leg_in = net["link_leg_in"][link]
    leg_out = jnp.where(mv >= 0, net["link_leg_out"][net["mov_out"][jnp.maximum(mv, 0)]], -1)
    ped_on = st.ped_t > 0
    cw_busy = (on_link & (leg_in >= 0) & ped_on[jnp.maximum(leg_in, 0)]) | ((leg_out >= 0) & ped_on[jnp.maximum(leg_out, 0)])
    # entering the box
    stopped = st.stopped | (on_link & (dist < 3) & (st.v < 0.5))
    sneak = is_signal & (turn == 0) & (s_state == 0) & (red_t < 1.6) & (dist < 3.0) & stopped  # left turner clears at the change
    sig_ok = (s_state == 2) | ((s_state == 1) & (ydec == 2)) | late_runner | sneak | \
        ((turn == 2) & is_signal & (s_state == 0) & stopped) | (stop_sign & stopped & (wait_t >= st.react))
    sig_ok = jnp.where(stop_sign | is_signal, sig_ok, True)
    look_ok = (gap_ok & ~cw_busy) | st.nolook
    can_stop = (st.v ** 2 / (2 * 4.0) < dist) | (st.v < 2.0)  # committed only when too fast to stop before the line
    enter = on_link & (mv >= 0) & sig_ok & room_ok & (look_ok | ~can_stop)
    line_stop = on_link & ~finishing & ~enter & (dist < 80)

    # ---- distraction
    start = (u_dist < beh.distract_rate * DT) & (st.dist_t <= 0) & ~st.av & alive
    dist_t = jnp.where(start, 1.5 + 1.5 * u_dur, jnp.maximum(st.dist_t - DT, 0.0))
    distracted = dist_t > 0

    # ---- zipper merge: the driver in the other lane next to a car that must merge holds back to let it in
    must_lc0 = on_link & nxt_ok & (mv < 0) & (mv_other >= 0) & (dist < 60) & (st.s > 12.0)
    _, f_m = at_pos(2 * link + 1 - lane, st.s)
    polite = must_lc0 & (f_m >= 0) & ~st.aggr[jnp.maximum(f_m, 0)] & (st.v < 3.0)
    cg = st.s - st.length - st.s[jnp.maximum(f_m, 0)]
    court_gap = jnp.full(N, big).at[jnp.where(polite & (cg > 0.5), f_m, N)].min(cg, mode="drop")
    court_v = jnp.zeros(N).at[jnp.where(polite & (cg > 0.5), f_m, N)].set(st.v, mode="drop")

    # ---- longitudinal control
    nxt_trk = jnp.where(on_link, jnp.where(enter, 2 * L + mv, -1), box_out)
    nxt_trk = jnp.where(in_box | enter, nxt_trk, -1)
    t_n = tail_of(nxt_trk)
    gap_next = jnp.where(t_n >= 0, dist + st.s[jnp.maximum(t_n, 0)] - st.length[jnp.maximum(t_n, 0)], big)
    # the track after that (across a short link or box): link -> box -> out-link lane; box -> link -> its next box
    nt_len = jnp.where(on_link, net["mov_len"][jnp.maximum(mv, 0)], net["link_len"][B])
    nxt2 = jnp.where(on_link, 2 * net["mov_out"][jnp.maximum(mv, 0)] + net["mov_out_lane"][jnp.maximum(mv, 0)], jnp.where(m2 >= 0, 2 * L + m2, -1))
    nxt2 = jnp.where((nxt_trk >= 0) & (t_n < 0) & (dist + nt_len < 60.0), nxt2, -1)
    t_n2 = tail_of(nxt2)
    gap_next = jnp.where(t_n2 >= 0, dist + nt_len + st.s[jnp.maximum(t_n2, 0)] - st.length[jnp.maximum(t_n2, 0)], gap_next)
    t_n = jnp.where(t_n >= 0, t_n, t_n2)
    lead_any = jnp.where(lead >= 0, lead, t_n)
    gap_lead = jnp.where(lead >= 0, gap_same, gap_next)
    v_lead = jnp.where(lead_any >= 0, st.v[jnp.maximum(lead_any, 0)], 0.0)
    out_cw = in_box & ped_on[jnp.maximum(net["link_leg_out"][net["mov_out"][mbox]], 0)] & (net["link_leg_out"][net["mov_out"][mbox]] >= 0)
    stop_gap = jnp.where(line_stop | (out_cw & ~st.nolook), dist - 0.5 + st.s0, big)  # IDM keeps s0: stop 0.5 m short
    park_gap = jnp.where(on_link & (st.park_s >= 0) & (st.park_s > st.s - 1.0), st.park_s - st.s + st.s0, big)
    # a crosswalk in use at the end of the next link (seen across a box or a short link)
    legB = net["link_leg_in"][B]
    ped2 = alive & (in_box | (on_link & (mv >= 0))) & (legB >= 0) & ped_on[jnp.maximum(legB, 0)] & ~st.nolook & (d2 < 80)
    ped2_gap = jnp.where(ped2, d2 - 0.5 + st.s0, big)
    stop_gap = jnp.minimum(stop_gap, ped2_gap)
    gap_lead_c = jnp.minimum(gap_lead, court_gap)
    v_lead = jnp.where(court_gap < gap_lead, court_v, v_lead)
    g_eff = jnp.minimum(jnp.minimum(gap_lead_c, stop_gap), park_gap)
    vl_eff = jnp.where(jnp.minimum(stop_gap, park_gap) < gap_lead_c, 0.0, v_lead)
    vlim = jnp.where(on_link, net["link_v"][link], net["mov_cap"][mbox])
    cap_next = jnp.where(mv >= 0, net["mov_cap"][jnp.maximum(mv, 0)], 99.0)
    slow_in = jnp.where(yielder, 0.2, 0.35)  # drivers who will have to yield approach slowly
    v0 = jnp.minimum(vlim * st.v0f, jnp.where(on_link, cap_next + slow_in * jnp.maximum(dist, 0.0), 99.0))
    v0 = jnp.where(in_box, jnp.minimum(net["mov_cap"][mbox], vlim * st.v0f), v0)
    acc = jnp.where(g_eff < 1e8, idm(st.v, v0, g_eff, vl_eff, st.amax, st.b, st.T, st.s0), st.amax * (1 - (st.v / jnp.maximum(v0, 0.1)) ** 4))
    acc = jnp.clip(acc, -9.0, st.amax)
    acc = jnp.where(distracted, 0.0, acc)  # no reaction
    halted = (st.stop_t > 0) | wreck
    # ---- the AV policy (optional): acceleration under the safety envelope, and the gap acceptance for the next junction
    g_stop = jnp.minimum(stop_gap, park_gap)
    shield = jnp.minimum(jnp.minimum(jnp.where(g_stop < 1e8, idm(st.v, jnp.maximum(v0, 1.0), g_stop, 0.0, 2.0, 3.0, 0.5, st.s0), 99.0),
                                     jnp.where(gap_lead_c < 1e8, idm(st.v, jnp.maximum(v0, 1.0), gap_lead_c, jnp.maximum(v_lead, 0.0), 2.0, 3.0, 0.6, 2.0), 99.0)),
                         jnp.maximum(1.5 * (v0 - st.v), -2.0))
    av_ctl = alive & st.av & ~wreck
    gsc = st.gsc
    shield_cut = jnp.zeros(N)
    if rl is not None:
        policy_fn, pkey = rl
        tmin_hi = jnp.where(valid & higher, tarr_m[other_m], big).min(1)
        obs = jnp.stack([
            st.v / 15.0, st.acc / 3.0, vlim / 15.0, v0 / 15.0, jnp.clip(dist / 100.0, -0.1, 3.0), on_link.astype(jnp.float32),
            (g_stop < 1e8).astype(jnp.float32), jnp.clip(jnp.where(g_stop < 1e8, g_stop, 300.0) / 100.0, 0.0, 3.0),
            (gap_lead_c < 1e8).astype(jnp.float32), jnp.clip(jnp.where(gap_lead_c < 1e8, gap_lead_c, 150.0) / 50.0, -0.1, 3.0),
            v_lead / 15.0, jnp.clip((st.v - v_lead) / 10.0, -2.0, 2.0) * (gap_lead_c < 1e8),
            ((s_state == 0) & is_signal).astype(jnp.float32), ((s_state == 1) & is_signal).astype(jnp.float32),
            stop_sign.astype(jnp.float32), (yielder & on_link).astype(jnp.float32), jnp.clip(cap_next, 0.0, 15.0) / 15.0,
            (gap_ok & (mv >= 0)).astype(jnp.float32), (room_ok & (mv >= 0)).astype(jnp.float32), cw_busy.astype(jnp.float32),
            jnp.clip(tmin_hi / 10.0, 0.0, 1.0), must_lc0.astype(jnp.float32), jnp.clip(wait_t / 10.0, 0.0, 2.0),
            enter.astype(jnp.float32),
            jnp.clip(st.v ** 2 / (2.0 * jnp.maximum(jnp.where(g_stop < 1e8, g_stop - st.s0, 300.0), 0.5)) / 8.0, 0.0, 2.0),
            jnp.clip(jnp.maximum(st.v - v_lead, 0.0) ** 2 / (2.0 * jnp.maximum(gap_lead_c - 2.0, 0.5)) / 8.0, 0.0, 2.0) * (gap_lead_c < 1e8),
            jnp.clip(gap_lead_c / jnp.maximum(st.v, 1.0) / 3.0, 0.0, 2.0) * (gap_lead_c < 1e8),
            (st.gsc - 1.15) / 0.35,
        ], axis=1).astype(jnp.float32)
        a_rule = acc  # what the rule-based driver would do (for behaviour cloning)
        u_pol, aux = policy_fn(obs, pkey)
        u0 = jnp.clip(u_pol[:, 0], -1.0, 1.0)
        a_pol = jnp.where(u0 >= 0.0, 3.0 * u0, 8.0 * u0)
        a_av = jnp.maximum(jnp.minimum(a_pol, shield), -9.0)
        shield_cut = jnp.where(av_ctl, a_pol - a_av, 0.0)
        acc = jnp.where(av_ctl & ~halted, a_av, acc)
        gsc = jnp.where(st.av, 1.15 + 0.35 * jnp.tanh(u_pol[:, 1]), st.gsc)
    acc = jnp.where(halted, -9.0, acc)
    v_new = jnp.maximum(st.v + acc * DT, 0.0)
    s_new = st.s + 0.5 * (st.v + v_new) * DT
    v_new, s_new = jnp.where(alive, v_new, 0.0), jnp.where(alive, s_new, 0.0)
    # a car that may not enter and is not crossing stays before the line
    hold = on_link & ~finishing & ((~enter & ~distracted) | (mv < 0))
    s_new = jnp.where(hold & (s_new > tlen - 0.05), tlen - 0.05, s_new)
    v_new = jnp.where(hold & (s_new >= tlen - 0.05), 0.0, v_new)

    # ---- planned stops (bus stop, delivery double-parking) and breakdowns
    arrive_park = on_link & (st.park_s >= 0) & (s_new >= st.park_s - 1.5) & (v_new < 1.0)
    stop_t = jnp.where(arrive_park, jnp.where(st.kind == BUS, 15 + 20 * u_park, 60 + 120 * u_park), jnp.maximum(st.stop_t - DT, 0.0))
    park_s = jnp.where(arrive_park, -1.0, st.park_s)
    brk = alive & on_link & (st.v > 1) & (u_brk < beh.breakdown_rate * DT) & (st.kind != BUS)
    stop_t = jnp.where(brk, 300 + 600 * u_park2, stop_t)
    broken = (st.broken | brk) & (stop_t > 0)

    # ---- lane changes (two-lane links)
    must_lc = on_link & nxt_ok & (mv < 0) & (mv_other >= 0)
    lead_c = jnp.maximum(lead, 0)
    blocked_ahead = (lead >= 0) & ((st.stop_t[lead_c] > 0) | (st.wreck_t[lead_c] > 0)) & (gap_same < 30)
    slow_ahead = (lead >= 0) & (gap_same < 25) & (st.v[lead_c] < 0.6 * vlim * st.v0f) & (vlim * st.v0f - st.v[lead_c] > 3)
    other_fine = (mv_other >= 0) | finishing | (dist > 80)
    early_ok = (st.s > 12.0) | (must_lc & (pending[jnp.minimum(2 * link + 1 - lane, 2 * L - 1)] == 0) & (st.s > 1.0))
    want = on_link & (nlanes == 2) & alive & ~halted & (st.lc_t <= 0) & early_ok & \
        (must_lc | ((blocked_ahead | (slow_ahead & ~queued)) & other_fine & (dist > 15)))
    tgt = 2 * link + 1 - lane
    l_t, f_t = at_pos(tgt, s_new)
    lt_, ft_ = jnp.maximum(l_t, 0), jnp.maximum(f_t, 0)
    src_trk = jnp.where(in_box, 2 * net["mov_in"][mbox] + net["mov_lane"][mbox], 2 * L)
    end_rear = jnp.full(2 * L, big).at[src_trk].min(st.s - st.length, mode="drop")  # < 0: a car still overhangs the lane end
    gf = jnp.where(l_t >= 0, s_new[lt_] - st.length[lt_] - s_new, jnp.minimum(big, tlen - s_new + end_rear[jnp.minimum(tgt, 2 * L - 1)]))
    gb = jnp.where(f_t >= 0, s_new - st.length - s_new[ft_], big)
    v_ahead = jnp.where(l_t >= 0, v_new[lt_], v_new)
    v_back = jnp.where(f_t >= 0, v_new[ft_], 0.0)
    zipper = jnp.where(must_lc & (dist < 40), 0.5, 1.0)  # near the end of a lane, drivers let one in
    fac = jnp.where(st.nosignal, 0.5, 1.0) * zipper * jnp.where(st.aggr, 0.8, 1.0)
    need_f = 2.0 + 0.3 * v_new + jnp.maximum(v_new - v_ahead, 0.0) ** 2 / 6.0
    need_b = 2.0 + 0.5 * v_back + jnp.maximum(v_back - v_new, 0.0) ** 2 / 6.0
    accept = ((gf > need_f * fac) & (gb > need_b * fac)) | (st.nolook & (gf > -1) )
    parity = (jnp.round(st.t / DT).astype(jnp.int32) % 2) == lane
    do_lc = want & accept & parity & (u_lc < jnp.where(must_lc, 0.4, 0.08)) & ~distracted
    ss_crash = do_lc & ((gf < 0) | (gb < 0))
    trk_new = jnp.where(do_lc, tgt, st.trk)
    lc_t = jnp.where(do_lc, 3.0, jnp.maximum(st.lc_t - DT, 0.0))
    surprise = do_lc & st.nosignal & (gb < 12) & (f_t >= 0)
    dist_t = dist_t.at[jnp.where(surprise, f_t, N)].max(1.0, mode="drop")

    # ---- passing a stopped vehicle on a single-lane two-way street, on the opposing lane when it is clear
    behind_t = jnp.where(on_link & blocked_ahead & (gap_same < 12) & (st.v < 0.5) & (nlanes == 1), st.behind_t + DT, 0.0)
    obst = lead_c
    gap_beyond = jnp.where(lead[obst] >= 0, st.s[jnp.maximum(lead[obst], 0)] - st.length[jnp.maximum(lead[obst], 0)], tlen + 50.0)
    new_s_pass = st.s[obst] + 3.0 + st.length
    do_pass = (behind_t > 4.0) & (gap_beyond > new_s_pass + 3.0) & (new_s_pass < tlen - 5.0) & (u_pass < 0.2) & ~distracted
    s_new = jnp.where(do_pass, new_s_pass, s_new)
    behind_t = jnp.where(do_pass, 0.0, behind_t)

    # ---- collisions
    disp = s_new - st.s
    la = jnp.maximum(lead_any, 0)
    rear = alive & ~wreck & (lead_any >= 0) & (disp > 0.02) & (gap_lead + disp[la] - disp < 0) & ~do_lc & ~do_pass & ~do_lc[la] & ~do_pass[la]
    cross = on_link & alive & (s_new >= tlen) & ~finishing & (mv >= 0)
    # box conflicts: both sides inside the conflict zone (car bodies)
    s_body = s_new
    near = vb & (s_body[:, None] + 1.0 > cs) & (s_body[:, None] - st.length[:, None] - 1.0 < cs)
    near_any = jnp.zeros(2 * C, bool).at[jnp.where(near, cidx, 2 * C)].max(True, mode="drop")
    near_new = jnp.zeros(2 * C, bool).at[jnp.where(near & ~wreck[:, None], cidx, 2 * C)].max(True, mode="drop")
    na, nn = near_any.reshape(C, 2), near_new.reshape(C, 2)
    hit = (na[:, 0] & nn[:, 1]) | (nn[:, 0] & na[:, 1])
    in_hit = (near & hit[jnp.maximum(ci, 0)]).any(1)
    rlr_c = jnp.zeros(C, bool).at[jnp.where(near & st.ran_red[:, None], ci, C)].max(True, mode="drop") & hit
    ped_hit_in = cross & (leg_in >= 0) & ped_on[jnp.maximum(leg_in, 0)]
    exit_box = in_box & (s_new >= tlen)
    leg_ex = net["link_leg_out"][net["mov_out"][mbox]]
    ped_hit_out = exit_box & (leg_ex >= 0) & ped_on[jnp.maximum(leg_ex, 0)]
    ped_hit = (ped_hit_in | ped_hit_out) & ~wreck
    crashed = (rear | (in_hit & ~wreck) | ss_crash | ped_hit) & alive
    crashed = crashed | jnp.zeros(N, bool).at[jnp.where(rear, la, N)].max(True, mode="drop")
    crashed = crashed | jnp.zeros(N, bool).at[jnp.where(ss_crash & (gf < 0), lt_, N)].max(True, mode="drop")
    crashed = crashed | jnp.zeros(N, bool).at[jnp.where(ss_crash & (gb < 0) & (gf >= 0), ft_, N)].max(True, mode="drop")
    ctype = jnp.zeros(5).at[jnp.maximum(net["c_type"], 0)].add(hit.astype(jnp.float32))  # 1 angle, 2 left, 3 merge
    counts = jnp.stack([rear.sum(), ctype[1], ctype[2], ctype[3] + ss_crash.sum(), ped_hit.sum()]).astype(jnp.float32)
    rlr = rlr_c.sum() + (rear & (st.ran_red | st.ran_red[la])).sum()
    new_wreck = crashed & ~wreck
    wreck_t = jnp.where(new_wreck, jnp.where(on_link, 300.0, 120.0), jnp.maximum(st.wreck_t - DT, 0.0))
    v_new = jnp.where(crashed, 0.0, v_new)
    clear_wreck = wreck & (wreck_t <= 0)
    legs_hit = jnp.where(ped_hit_in, leg_in, jnp.where(ped_hit_out, leg_ex, G))
    ped_t = st.ped_t.at[legs_hit].set(0.0, mode="drop")

    # ---- moves between tracks
    red_run = cross & (s_state == 0) & is_signal & ~((turn == 2) & stopped) & ~sneak  # clearing a left at the change is legal
    ran_red = jnp.where(cross, red_run, st.ran_red)
    to_box = cross & ~crashed
    trk_new = jnp.where(to_box, 2 * L + mv, trk_new)
    s_new = jnp.where(to_box, s_new - tlen, s_new)
    out_l = net["mov_out"][mbox]
    to_link = exit_box & ~crashed
    dest_l = net["dest_link"][st.dest]
    nxt_new = jnp.where(out_l == dest_l, -1, nh[out_l, st.dest])
    trk_new = jnp.where(to_link, box_out, trk_new)
    s_new = jnp.where(to_link, s_new - tlen, s_new)
    nxt = jnp.where(to_link, nxt_new, st.nxt)
    stopped = jnp.where(to_box | to_link | do_lc, False, stopped)
    wait_t = jnp.where(to_box | to_link, 0.0, wait_t)
    rk = net["link_rank"][out_l]
    park_s = jnp.where(to_link, jnp.where((st.kind == BUS) & (net["link_bus"][out_l] >= 0), net["link_bus"][out_l],
                                         jnp.where((st.kind == VAN) & (rk >= 2) & (u_park2 < beh.p_park) & (net["link_len"][out_l] > 50),
                                                   (0.3 + 0.4 * u_misc) * net["link_len"][out_l], -1.0)), park_s)
    done = finishing & alive & (s_new >= tlen - 1.0) & ~crashed
    # gridlock: a car that has not moved for 4 minutes (not a planned stop, not a wreck) is removed
    stuck_t = jnp.where(alive & (v_new < 0.1) & ~halted, st.stuck_t + DT, 0.0)
    teleport = alive & (stuck_t > 240.0)
    alive = alive & ~done & ~clear_wreck & ~teleport

    # ---- pedestrians
    walk = leg_walk(net, st.t)
    leg_box = jnp.zeros(G + 1).at[jnp.where(in_box, leg_ex, G)].add(1.0, mode="drop")[:G]
    near_line = on_link & alive & (st.v > 0.5) & (dist < 5.0 + 3.0 * st.v + st.v ** 2 / 5.0)  # could not stop comfortably
    leg_app = jnp.zeros(G + 1).at[jnp.where(near_line, leg_in, G)].add(1.0, mode="drop")[:G]
    soon = on_link & alive & (mv >= 0) & (enter | (st.v > 0.5)) & (dist < 5.0 + 3.0 * st.v + st.v ** 2 / 5.0)  # crosses the out-leg
    near2 = alive & (st.v > 0.5) & (d2 < 5.0 + 3.0 * st.v + st.v ** 2 / 5.0) & (in_box | (on_link & (mv >= 0)))
    leg_app = leg_app + jnp.zeros(G + 1).at[jnp.where(near2, net["link_leg_in"][B], G)].add(1.0, mode="drop")[:G]
    leg_m2 = jnp.where(m2 >= 0, net["link_leg_out"][net["mov_out"][jnp.maximum(m2, 0)]], G)
    leg_app = leg_app + jnp.zeros(G + 1).at[jnp.where(near2 & (m2 >= 0), leg_m2, G)].add(1.0, mode="drop")[:G]
    leg_app = leg_app + jnp.zeros(G + 1).at[jnp.where(soon, leg_out, G)].add(1.0, mode="drop")[:G]
    want_ped = (st.ped_t <= 0) & (u_ped[0] < net["leg_rate"] * DT)
    lawful = walk & (leg_box == 0) & (leg_app == 0)
    # a violating pedestrian crosses against the signal or into a short gap, but does not step in front of a close car
    close = on_link & alive & (st.v > 0.5) & (dist < 4.0 + 1.5 * st.v)
    leg_close = jnp.zeros(G + 1).at[jnp.where(close, leg_in, G)].add(1.0, mode="drop")[:G]
    leg_close = leg_close + jnp.zeros(G + 1).at[jnp.where(close & (mv >= 0), leg_out, G)].add(1.0, mode="drop")[:G]
    reckless = ~lawful & (u_ped[1] < beh.p_ped_violate) & (leg_box == 0) & (leg_close == 0)
    ped_start = want_ped & (lawful | reckless)
    ped_t = jnp.where(ped_start, 8.0 + 6.0 * u_ped[2], jnp.maximum(ped_t - DT, 0.0))

    # ---- arrivals: 24 attempts per step, sampled from the sources by rate
    S = u_sp.shape[1]
    p_go = net["src_rate"] * beh.demand * DT / S
    src = jnp.minimum(_ss(net["src_cdf"], u_sp[0]), L - 1)
    dst = jnp.minimum(_ss(net["dest_cdf"], u_sp[1]), net["dest_link"].shape[0] - 1)
    ln = jnp.where(net["link_lanes"][src] == 2, (u_sp[2] < 0.5).astype(jnp.int32), 0)
    strk = 2 * src + ln
    is_bus = u_sp[3] < beh.p_bus
    is_van = ~is_bus & (u_sp[3] < beh.p_bus + beh.p_van)
    is_av = ~is_bus & ~is_van & (u_sp[4] < beh.p_av)
    human = ~is_av
    length = jnp.where(is_bus, 12.0, jnp.where(is_van, 7.0, 4.2 + 0.8 * u_sp[5]))
    # from the map edge: at the link start; from parking: somewhere along the block, between two cars
    s_sp = jnp.where(net["link_edge_in"][src], 0.0, (0.2 + 0.6 * u_sp[14]) * net["link_len"][src] + length)  # front bumper
    l_sp, f_sp = at_pos(strk, s_sp)
    e_gap = jnp.where(l_sp >= 0, s_new[jnp.maximum(l_sp, 0)] - st.length[jnp.maximum(l_sp, 0)] - s_sp, big)
    b_gap = jnp.where(f_sp >= 0, s_sp - length - s_new[jnp.maximum(f_sp, 0)], big)
    e_v = jnp.where(l_sp >= 0, v_new[jnp.maximum(l_sp, 0)], 99.0)
    v_sp = jnp.where(net["link_edge_in"][src], jnp.minimum(0.5 * net["link_v"][src], e_v), 0.0)
    b_v = jnp.where(f_sp >= 0, v_new[jnp.maximum(f_sp, 0)], 0.0)
    ok_sp = (u_sp[6] < p_go) & (e_gap > 4.0 + v_sp) & (b_gap > 4.0 + 1.5 * b_v + b_v ** 2 / 6.0) & \
        (s_sp < net["link_len"][src]) & (pending[strk] == 0)
    # one arrival per source track per step
    first = jnp.ones(S, bool)
    for a_ in range(1, S):
        first = first.at[a_].set(~jnp.any((strk[:a_] == strk[a_]) & ok_sp[:a_]))
    ok_sp = ok_sp & first
    slots = jnp.nonzero(~alive, size=S, fill_value=0)[0]  # the first S free slots (no full sort)
    ok_sp = ok_sp & ~alive[slots]
    blocked_sp = ((u_sp[6] < p_go) & ~ok_sp).sum()
    rk0 = net["link_rank"][src]
    vals = dict(
        alive=jnp.ones(S, bool), trk=strk, s=s_sp, v=v_sp, dest=dst,
        nxt=jnp.where(src == net["dest_link"][dst], -1, nh[src, dst]), kind=jnp.where(is_bus, BUS, jnp.where(is_van, VAN, CAR)),
        v0f=jnp.where(is_bus, 0.85, jnp.where(is_av, 1.0, 0.9 + 0.3 * u_sp[7])), T=jnp.where(is_av, 1.3, jnp.where(is_bus, 1.6, 1.0 + 0.8 * u_sp[5])),
        amax=jnp.where(is_bus | is_van, 1.0, jnp.where(is_av, 1.6, 1.2 + 0.8 * u_sp[2])), b=jnp.where(is_av, 2.0, 1.5 + 1.0 * u_sp[4]),
        s0=1.5 + 1.0 * u_sp[8], length=length,
        runner=human & (u_sp[9] < beh.p_red), nolook=human & (u_sp[10] < beh.p_nolook),
        nosignal=human & (u_sp[11] < beh.p_nosignal), aggr=human & (u_sp[12] < beh.p_aggressive), av=is_av,
        react=jnp.where(is_av, 0.3, 0.7 + 0.8 * u_sp[13]), dist_t=jnp.zeros(S), stop_t=jnp.zeros(S),
        park_s=jnp.where(is_bus & (net["link_bus"][src] >= 0), net["link_bus"][src],
                         jnp.where(is_van & (rk0 >= 2) & (u_sp[14] < beh.p_park), (0.3 + 0.4 * u_sp[15]) * net["link_len"][src], -1.0)),
        broken=jnp.zeros(S, bool), wreck_t=jnp.zeros(S), stopped=jnp.zeros(S, bool), wait_t=jnp.zeros(S), red_t=jnp.zeros(S),
        ydec=jnp.zeros(S, jnp.int32), ran_red=jnp.zeros(S, bool), stuck_t=jnp.zeros(S), behind_t=jnp.zeros(S), lc_t=jnp.zeros(S),
        t0=jnp.full(S, st.t), acc=jnp.zeros(S), gsc=jnp.full(S, 1.25))
    cur = dict(alive=alive, trk=trk_new, s=s_new, v=v_new, dest=st.dest, nxt=nxt, kind=st.kind, v0f=st.v0f, T=st.T, amax=st.amax,
               b=st.b, s0=st.s0, length=st.length, runner=st.runner, nolook=st.nolook, nosignal=st.nosignal, aggr=st.aggr, av=st.av,
               react=st.react, dist_t=dist_t, stop_t=stop_t, park_s=park_s, broken=broken, wreck_t=wreck_t, stopped=stopped & alive,
               wait_t=wait_t, red_t=red_t, ydec=ydec, ran_red=ran_red, stuck_t=stuck_t, behind_t=behind_t, lc_t=lc_t, t0=st.t0,
               acc=jnp.where(alive, acc, 0.0), gsc=gsc)
    new = {k: cur[k].at[slots].set(jnp.where(ok_sp, vals[k], cur[k][slots])) for k in cur}

    lk = jnp.where(st.alive & on_link, link, L)
    link_vs = st.link_vs.at[lk].add(st.v, mode="drop")
    link_n = st.link_n.at[lk].add(1.0, mode="drop")
    m = st.m + jnp.concatenate([
        jnp.stack([to_box.sum(), done.sum(), jnp.where(done, st.t - st.t0, 0.0).sum(), ok_sp.sum(), blocked_sp, teleport.sum()]).astype(jnp.float32),
        counts,
        jnp.stack([red_run.sum(), do_lc.sum(), st.alive.sum() * DT, (st.v * st.alive).sum() * DT, brk.sum(), arrive_park.sum(),
                   do_pass.sum(), ped_start.sum(), rlr]).astype(jnp.float32), jnp.zeros(10)])
    # ---- AV statistics (and the RL reward)
    av0 = st.alive & st.av & ~wreck
    hard = av0 & (acc < -4.0) & (st.acc >= -4.0) & (st.v > 3.0)
    jerk = (acc - st.acc) / DT
    crash_av = av0 & crashed
    near = av0 & to_box & (jnp.where(valid & higher, tarr_m[other_m], big).min(1) < 2.0)  # entered just ahead of a car with priority
    red_av = av0 & red_run
    m = m + jnp.zeros(NM).at[NM - 10:].set(jnp.stack([(done & st.av).sum(), jnp.where(done & st.av, st.t - st.t0, 0.0).sum(), av0.sum() * DT,
                       jnp.where(av0, disp, 0.0).sum(), hard.sum(), jnp.where(av0, jerk ** 2, 0.0).sum() * DT, crash_av.sum(),
                       red_av.sum(), near.sum(), (av0 & (shield_cut > 0.5)).sum()]).astype(jnp.float32))
    new["acc"] = new["acc"].at[slots].set(jnp.where(ok_sp, 0.0, new["acc"][slots]))
    new["gsc"] = new["gsc"].at[slots].set(jnp.where(ok_sp, 1.25, new["gsc"][slots]))
    out = State(key=key, t=st.t + DT, ped_t=ped_t, link_vs=link_vs, link_n=link_n, m=m, **new)
    if rl is not None:
        thw = gap_lead_c / jnp.maximum(st.v, 1.0)
        idle = (st.v < 0.5) & (g_stop > 10.0) & (gap_lead_c > 8.0) & ~halted
        team = (st.v * st.alive).sum() / jnp.maximum(st.alive.sum(), 1)
        reward = (R.progress * disp + R.team * team - R.idle * idle - R.comfort * acc ** 2 - R.hard * jnp.maximum(-acc - 3.0, 0.0) * (st.v > 2.0)
                  - R.jerk * jerk ** 2 - R.headway * jnp.maximum(0.8 - thw, 0.0) * (st.v > 1.0)
                  - R.speed * jnp.maximum(st.v - vlim, 0.0) - R.shield * shield_cut
                  + R.crash * crash_av + R.red * red_av + R.near * near)
        ended = av0 & (done | crashed | teleport | ~alive)
        return out, dict(obs=obs, u=u_pol, aux=aux, reward=jnp.where(av0, reward, 0.0), valid=av0, done=ended, a_rule=a_rule)
    if diag:
        return out, dict(mv=mv, enter=enter, sig_ok=sig_ok, gap_ok=gap_ok, room_ok=room_ok, room=room, cw_busy=cw_busy, dist=dist,
                         lead=lead_any, gap_lead=gap_lead, s_state=s_state, must_lc=must_lc, crashed=crashed, rear=rear,
                         ped_hit=ped_hit, in_hit=in_hit, ss=ss_crash, occ=(valid & occ).any(1), arriving=(valid & arriving).any(1),
                         first_other=(valid & first_other).any(1), teleport=teleport, line_stop=line_stop, halted=halted)
    return out


class Sim:
    """Runs E copies of one area in a vmapped batch, with congestion-aware rerouting every `reroute_s` seconds."""

    def __init__(self, net_np, n_slots=1024, n_envs=8, beh=Behaviour(), seed=0, reroute_s=300.0, chunk=250, policy=None):
        """`policy(obs, key) -> (u, aux)` drives the AVs (deterministically, for evaluation and GIFs)."""
        from . import network as NW

        self.net_np, self.NW = net_np, NW
        self.net = to_jnp(net_np)
        self.beh, self.chunk, self.reroute_s = beh, chunk, reroute_s
        self.nh = jnp.broadcast_to(jnp.asarray(net_np["nh"]), (n_envs,) + net_np["nh"].shape)
        keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
        self.st = jax.vmap(reset, in_axes=(0, None, None))(keys, self.net, n_slots)
        net, beh_ = self.net, beh

        def stp(s, h):
            if policy is None:
                return step(s, net, beh_, h)
            return step(s, net, beh_, h, rl=(policy, s.key))[0]

        @jax.jit
        def run_chunk(st, nh):
            f = jax.vmap(stp)
            return jax.lax.scan(lambda s, _: (f(s, nh), None), st, None, length=chunk)[0]

        @jax.jit
        def one(st, nh):
            return jax.vmap(stp)(st, nh)

        self.run_chunk, self.one = run_chunk, one
        self.since = 0.0

    def reroute(self):
        """Next-hop tables from each env's measured mean link speeds (free flow where no car was seen)."""
        vs, n = np.asarray(self.st.link_vs), np.asarray(self.st.link_n)
        free = self.net_np["link_v"] * 0.85
        tabs = []
        for e in range(vs.shape[0]):
            sp = np.where(n[e] > 20, vs[e] / np.maximum(n[e], 1), free)
            tabs.append(self.NW.next_hops(self.net_np, self.NW.link_tt(self.net_np, np.maximum(sp, 0.5))))
        self.nh = jnp.asarray(np.stack(tabs))
        self.st = self.st._replace(link_vs=jnp.zeros_like(self.st.link_vs), link_n=jnp.zeros_like(self.st.link_n))

    def run(self, seconds, record=None):
        """Advance `seconds`; with record=(env, every_k) also return a list of per-step snapshots of that env."""
        frames = []
        n = int(round(seconds / DT))
        while n > 0:
            if self.reroute_s and self.since >= self.reroute_s:
                self.reroute()
                self.since = 0.0
            if record is None and n >= self.chunk:
                self.st = self.run_chunk(self.st, self.nh)
                n -= self.chunk
                self.since += self.chunk * DT
            else:
                self.st = self.one(self.st, self.nh)
                n -= 1
                self.since += DT
                if record is not None and (n % record[1] == 0):
                    frames.append(jax.tree.map(lambda a: np.asarray(a[record[0]]), self.st))
        return frames


def summary(m):
    m = np.asarray(m, float)
    if m.ndim > 1:
        m = m.sum(0)
    d = dict(zip(METRICS, m))
    crashes = sum(d[k] for k in CRASH_NAMES)
    mev = max(d["entered"], 1) / 1e6
    return {"junction_entries": int(d["entered"]), "trips_started": int(d["spawned"]), "trips_finished": int(d["finished"]),
            "mean_trip_min": d["trip_s"] / max(d["finished"], 1) / 60, "vehicle_km": d["veh_m"] / 1000,
            "mean_speed_kmh": 3.6 * d["veh_m"] / max(d["veh_s"], 1), "teleports": int(d["teleports"]),
            "blocked_arrivals": int(d["blocked"]), "crashes": int(crashes),
            "crashes_per_M_junction_entries": crashes / mev, "by_type": {k: int(d[k]) for k in CRASH_NAMES},
            "red_runs_per_1000_entries": 1000 * d["red_runs"] / max(d["entered"], 1), "lane_changes": int(d["lane_changes"]),
            "breakdowns": int(d["breakdowns"]), "bus_and_delivery_stops": int(d["stops"]), "passes_on_opposing_lane": int(d["passes"]),
            "pedestrian_crossings": int(d["peds"])}
