"""Geometry of a 3- or 4-leg intersection (right-hand traffic), from a layout spec.

Legs sit in four approach slots a = 0..3 whose outward directions are roughly -90, 0, 90, 180 degrees (south, east, north,
west), but each leg keeps its real bearing, lane counts and one-way direction. Slot a's vehicles can make up to 4 paths
k = 0..3: L, T (first through lane), T (second through lane), R, so a layout has at most 16 paths; paths a layout does not
have are marked invalid. Each path = approach straight (APP m) + a curve through the box + EXIT m out. The curve is a
cubic Bezier with circular-arc handles, so a 90-degree turn is a quarter circle and a through move is a straight line.

`Geometry(spec)` computes, once and in NumPy: path polylines (for drawing), lengths, exit lanes, the lane/move tables, and
the conflict points where two paths cross or merge (arc-length position on each path and a crash type for the pair).
`CANON` is the default intersection: 4 legs at right angles, 3 lanes in and out (left-turn bay, through, through/right).
"""
import numpy as np

LANE_W = 3.6
LANES = 3  # max lanes per direction
APP = 120.0  # approach length (spawn point to stop line)
EXIT = 50.0  # exit length after the box
MOVES = ("L", "T", "T", "R")  # per path index within an approach
TURN = {"L": 3, "T": 2, "R": 1}  # exit slot = (a + TURN[move]) % 4
N_PATHS = 16
# crash types
REAR, ANGLE, LEFT, SIDESWIPE, OTHER = range(5)
CRASH_NAMES = ("rear-end", "angle", "left-turn", "sideswipe", "other")


def lane_x(lane):
    return LANE_W * (lane + 0.5)  # lanes counted from the centre line outwards


def canonical_spec():
    return {"legs": [{"slot": a, "bearing": -90.0 + 90 * a, "lanes_in": 3, "lanes_out": 3} for a in range(4)],
            "control": "signal"}


def _unit(deg):
    t = np.radians(deg)
    return np.array([np.cos(t), np.sin(t)])


def _bezier(p0, h0, p3, h3, n=60):
    c = np.linalg.norm(p3 - p0)
    th = np.arccos(np.clip(h0 @ h3, -1, 1))
    k = c / 3 if th < 1e-3 else c * (4 / 3) * np.tan(th / 4) / (2 * np.sin(th / 2))
    p1, p2 = p0 + h0 * k, p3 - h3 * k
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def lane_plan(nin, valid, nout_t):
    """Which lanes make which moves for one approach. valid = (L, T, R) exit exists; nout_t = lanes on the through exit.
    Returns (lanes, {k: lane}) with k the path index within the approach (0 L, 1 T, 2 T, 3 R)."""
    vl, vt, vr = valid
    if not (vl or vt or vr) or nin == 0:
        return 0, {}
    if not vt:
        nin = min(nin, int(vl) + int(vr))
    moves = [set() for _ in range(nin)]
    if vl:
        moves[0].add("L")
    if vr:
        moves[nin - 1].add("R")
    if vt:
        cand = list(range(1, nin)) if (nin == 3 and vl) else list(range(nin))
        cand.sort(key=lambda ln: (len(moves[ln]), -ln))
        for ln in sorted(cand[:min(2, max(nout_t, 1))]):
            moves[ln].add("T")
    keep = [ln for ln in range(nin) if moves[ln]]  # a lane with no move is dropped
    moves = [moves[ln] for ln in keep]
    t_lanes = [i for i, m in enumerate(moves) if "T" in m]
    plan = {}
    if vl:
        plan[0] = 0
    for j, ln in enumerate(t_lanes):
        plan[1 + j] = ln
    if vr:
        plan[3] = len(moves) - 1
    return len(moves), plan


class Geometry:
    def __init__(self, spec=None):
        spec = spec or canonical_spec()
        self.spec = spec
        legs = {l["slot"]: l for l in spec["legs"]}
        self.legs = legs
        self.present = np.array([a in legs for a in range(4)])
        nin = np.array([legs[a]["lanes_in"] if a in legs else 0 for a in range(4)])
        self.spec_nin = nin
        nout = np.array([legs[a]["lanes_out"] if a in legs else 0 for a in range(4)])
        self.bearing = np.array([legs[a]["bearing"] if a in legs else -90.0 + 90 * a for a in range(4)])
        self.u = np.stack([_unit(b) for b in self.bearing])  # outward unit vector per slot
        self.nout = nout
        self.Hs = self._stop_lines(nin, nout)
        self.H = float(self.Hs[self.present].max())
        # per-approach lanes and lane plans
        self.nin = np.zeros(4, int)
        self.plan = {}
        for a in range(4):
            valid = tuple(bool(self.present[(a + TURN[m]) % 4] and nout[(a + TURN[m]) % 4] > 0) for m in "LTR")
            n, plan = lane_plan(int(nin[a]), valid, int(nout[(a + 2) % 4]))
            self.nin[a], self.plan[a] = n, plan
        self.paths = self._build()
        self.conflicts = self._conflicts()
        self._tables()

    def _stop_lines(self, nin, nout, margin=1.0, cap=25.0):
        """Per leg: how far from the centre the leg's road strip stops overlapping the other legs' strips (+ margin),
        so the stop line clears the crossing roads. Very skewed legs are capped (their strips then overlap a little)."""
        import shapely
        from shapely.geometry import Polygon as SPoly

        strips = {}
        for a in range(4):
            if not self.present[a]:
                continue
            u = self.u[a]
            r = np.array([-u[1], u[0]])
            lo, hi = self.lateral(a, nin, nout)
            strips[a] = (u, SPoly([r * lo, u * 200 + r * lo, u * 200 + r * hi, r * hi]))
        Hs = np.zeros(4)
        # floor: the widest road's half-width, so the box is big enough for opposing left turns to pass each other
        floor = max(max(abs(x) for x in self.lateral(a, nin, nout)) for a in strips) + margin
        for a, (u, sa) in strips.items():
            ext = 0.0
            for b, (_, sb) in strips.items():
                if b != a:
                    ov = sa.intersection(sb)
                    if not ov.is_empty:
                        ext = max(ext, float((shapely.get_coordinates(ov) @ u).max()))
            Hs[a] = min(max(ext + margin, floor), cap)
        return Hs

    def lateral(self, a, nin=None, nout=None):
        """Lateral extent (lo, hi) of leg a's road, measured to the right of the inbound heading."""
        nin = self.spec_nin if nin is None else nin
        nout = self.nout if nout is None else nout
        ni, no = nin[a], nout[a]
        off_in = 0.0 if no else ni * LANE_W / 2
        off_out = 0.0 if ni else no * LANE_W / 2
        lo = -(no * LANE_W - off_out) if no else -off_in
        hi = ni * LANE_W - off_in if ni else off_out
        return lo, hi

    # ---------------------------------------------------------------- lanes in world coordinates
    def _in_pt(self, a, lane, d):
        """Point on inbound lane `lane` of slot a, at distance d from the centre along the leg."""
        u = self.u[a]
        r = np.array([-u[1], u[0]])  # right of the inbound heading -u
        off = 0.0 if self.nout[a] else self.nin[a] * LANE_W / 2  # one-way: lanes straddle the centre line
        return u * d + r * (lane_x(lane) - off)

    def _out_pt(self, b, lane, d):
        u = self.u[b]
        r = np.array([u[1], -u[0]])  # right of the outbound heading u
        off = 0.0 if self.nin[b] else self.nout[b] * LANE_W / 2
        return u * d + r * (lane_x(lane) - off)

    def _build(self):
        Hs = self.Hs
        paths = []
        for a in range(4):
            for k in range(4):
                move = MOVES[k]
                b = (a + TURN[move]) % 4
                if k not in self.plan[a]:
                    paths.append(dict(a=a, k=k, move=move, valid=False, lane=0, exit_road=b, exit_lane=0, pts=np.zeros((2, 2)),
                                      s=np.array([0.0, 1.0]), box_len=1.0, length=APP + 1.0 + EXIT, turn=0.0))
                    continue
                lane = self.plan[a][k]
                no = self.nout[b]
                if move == "L":
                    xl = 0
                elif move == "R":
                    xl = no - 1
                else:
                    t_paths = sorted(kk for kk in (1, 2) if kk in self.plan[a])
                    xl = no - len(t_paths) + t_paths.index(k) if no >= len(t_paths) else min(lane, no - 1)
                    xl = max(0, min(xl, no - 1))
                app = np.stack([self._in_pt(a, lane, d) for d in np.linspace(Hs[a] + APP, Hs[a], 40)])
                box = _bezier(self._in_pt(a, lane, Hs[a]), -self.u[a], self._out_pt(b, xl, Hs[b]), self.u[b])
                ex = np.stack([self._out_pt(b, xl, d) for d in np.linspace(Hs[b], Hs[b] + EXIT, 20)])
                turn = float(np.degrees(np.arccos(np.clip(-self.u[a] @ self.u[b], -1, 1))))
                pts = np.concatenate([app, box[1:], ex[1:]])
                s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
                box_len = float(np.sum(np.linalg.norm(np.diff(box, axis=0), axis=1)))
                paths.append(dict(a=a, k=k, move=move, valid=True, lane=lane, pts=pts, s=s, box_len=box_len, length=float(s[-1]),
                                  exit_road=b, exit_lane=int(xl), turn=turn))
        return paths

    def _conflicts(self, tol=1.2):
        """(p, q, s_p, s_q, type) for every pair of paths from different approaches whose box pieces cross, plus merges
        (two paths entering the same exit lane)."""
        out = []
        P_ = self.paths
        for i in range(N_PATHS):
            for j in range(i + 1, N_PATHS):
                P, Q = P_[i], P_[j]
                if P["a"] == Q["a"] or not (P["valid"] and Q["valid"]):
                    continue
                same_exit = (P["exit_road"], P["exit_lane"]) == (Q["exit_road"], Q["exit_lane"])
                mp = (P["s"] >= APP) & (P["s"] <= APP + P["box_len"])
                mq = (Q["s"] >= APP) & (Q["s"] <= APP + Q["box_len"])
                A, B = P["pts"][mp], Q["pts"][mq]
                dist = np.linalg.norm(A[:, None] - B[None], axis=2)
                ii, jj = np.unravel_index(np.argmin(dist), dist.shape)
                rel = (Q["a"] - P["a"]) % 4
                if same_exit:
                    out.append((i, j, APP + P["box_len"], APP + Q["box_len"], SIDESWIPE))
                elif dist[ii, jj] < tol:
                    sp, sq = P["s"][mp][ii], Q["s"][mq][jj]
                    moves = {P["move"], Q["move"]}
                    kind = LEFT if (rel == 2 and "L" in moves) else (ANGLE if rel in (1, 3) else OTHER)
                    out.append((i, j, sp, sq, kind))
        return out

    def _tables(self):
        P = self.paths
        self.path_valid = np.array([p["valid"] for p in P])
        self.path_a = np.array([p["a"] for p in P])
        self.path_lane = np.array([p["lane"] for p in P])
        self.path_move = np.array(["LTR".index(p["move"]) for p in P])  # 0 L, 1 T, 2 R
        self.path_box = np.array([p["box_len"] for p in P], np.float32)
        self.path_len = np.array([p["length"] for p in P], np.float32)
        self.path_exit = np.array([p["exit_road"] * LANES + p["exit_lane"] for p in P])
        # speed cap through the box [m/s]: by move, but a shallow turn (< 45 degrees) is driven like a through move and
        # a near-U-turn (> 135 degrees) at walking-pace speed
        cap = np.array([{"L": 8.0, "T": 99.0, "R": 5.0}[p["move"]] for p in P], np.float32)
        turn = np.array([p["turn"] for p in P])
        self.path_cap = np.where(turn < 45, 99.0, np.where(turn > 135, 5.0, cap)).astype(np.float32)
        t = -np.ones((4, LANES, 3), np.int32)  # path for (approach, lane, move), -1 if that lane cannot make it
        for p_i, p in enumerate(P):
            if p["valid"]:
                m = "LTR".index(p["move"])
                if t[p["a"], p["lane"], m] < 0:
                    t[p["a"], p["lane"], m] = p_i
        self.path_table = t
        # default move per lane (prefer through), lane-change target lane, spawn lanes per move
        self.default_move = np.ones((4, LANES), np.int32)
        self.lc_target = np.zeros((4, LANES, 3), np.int32)
        self.spawn_lane = np.zeros((4, 3, 2), np.int32)
        self.move_ok = np.zeros((4, 3), bool)
        for a in range(4):
            for ln in range(LANES):
                ok = [m for m in (1, 0, 2) if t[a, ln, m] >= 0]
                self.default_move[a, ln] = ok[0] if ok else 1
                for m in range(3):
                    lanes = [x for x in range(self.nin[a]) if t[a, x, m] >= 0]
                    self.lc_target[a, ln, m] = min(lanes, key=lambda x: abs(x - ln)) if lanes else ln
            for m in range(3):
                lanes = [x for x in range(self.nin[a]) if t[a, x, m] >= 0]
                self.move_ok[a, m] = bool(lanes)
                if lanes:
                    self.spawn_lane[a, m] = (lanes[0], lanes[-1])
        conf = np.zeros((N_PATHS, N_PATHS), bool)
        for p, q, *_ in self.conflicts:
            conf[p, q] = conf[q, p] = True
        self.conf_pp = conf

    def xy(self, path, s):
        """World position and heading of a point at arc length s on a path (for drawing)."""
        P = self.paths[path]
        s = np.clip(s, 0, P["length"] - 1e-3)
        i = np.clip(np.searchsorted(P["s"], s) - 1, 0, len(P["s"]) - 2)
        f = (s - P["s"][i]) / (P["s"][i + 1] - P["s"][i])
        pt = P["pts"][i] * (1 - f)[..., None] + P["pts"][i + 1] * f[..., None]
        d = P["pts"][i + 1] - P["pts"][i]
        return pt, np.arctan2(d[..., 1], d[..., 0])


CANON = Geometry()
H = CANON.H
PATHS = CANON.paths
CONFLICTS = CANON.conflicts
PATH_LEN = CANON.path_len
xy = CANON.xy
