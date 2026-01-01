"""Geometry of a signalised 4-way intersection with 3 inbound and 3 outbound lanes per road (right-hand traffic).

Approach a = 0..3 enters heading 90, 180, 270, 0 degrees (from the south, east, north, west). Inbound lanes per approach:
lane 0 = left-turn bay, lane 1 = through, lane 2 = through or right. That gives 16 paths (approach x {L from lane 0,
T from lane 1, T from lane 2, R from lane 2}), each = approach straight (APP m) + a piece through the box + EXIT m out.

Everything here is NumPy, computed once: path polylines (for drawing), lengths, exit lanes, and the conflict points where
two paths cross or merge (with the arc-length position on each path and a crash type for that pair).
"""
import numpy as np

LANE_W = 3.6
H = 3 * LANE_W + 1.0  # half-size of the box: the stop line is H from the centre
APP = 120.0  # approach length (spawn point to stop line)
EXIT = 50.0  # exit length after the box
LANES = 3
MOVES = ("L", "T", "T", "R")  # per path index within an approach
PATH_LANE = (0, 1, 2, 2)
N_PATHS = 16
# crash types
REAR, ANGLE, LEFT, SIDESWIPE, OTHER = range(5)
CRASH_NAMES = ("rear-end", "angle", "left-turn", "sideswipe", "other")


def lane_x(lane):
    return LANE_W * (lane + 0.5)  # inbound lanes on the right of the heading (approach-0 frame: +x)


def _rot(a):
    t = np.pi / 2 * a
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _box_piece(move, lane, n=60):
    """The path through the box in the approach-0 frame (heading +y), from the stop line (y = -H)."""
    x = lane_x(lane)
    if move == "T":
        ys = np.linspace(-H, H, n)
        return np.stack([np.full(n, x), ys], 1), (0, x)  # exits heading +y in the same lane offset
    if move == "R":  # quarter circle to heading +x, into the outermost outbound lane of the east road (y = -lane_x(2))
        yo = -lane_x(2)
        cx, cy = H, -H
        r = H - x
        th = np.linspace(np.pi, np.pi / 2, n)
        pts = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1)
        pts[-1] = [H, yo] if abs(cy + r - yo) < 1e-6 else pts[-1]
        return pts, (3, -yo)  # exits heading +x (approach-3 frame lane offset)
    # left: quarter circle to heading -x, into the innermost outbound lane of the west road (y = +lane_x(0))
    cx, cy = -H, -H
    r = x + H
    th = np.linspace(0, np.pi / 2, n)
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1), (1, lane_x(0))


def build():
    paths = []
    for a in range(4):
        R = _rot(a)
        for k in range(4):
            move, lane = MOVES[k], PATH_LANE[k]
            x = lane_x(lane)
            app = np.stack([np.full(40, x), np.linspace(-H - APP, -H, 40)], 1)
            box, (turn, _) = _box_piece(move, lane)
            # exit straight in the exit heading
            d = box[-1] - box[-2]
            d = d / np.linalg.norm(d)
            ex = box[-1] + np.outer(np.linspace(0, EXIT, 20), d)
            pts = np.concatenate([app, box[1:], ex[1:]]) @ R.T
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seg)])
            box_len = float(np.sum(np.linalg.norm(np.diff(box, axis=0), axis=1)))
            exit_road = (a + turn) % 4  # the approach index whose heading the vehicle now has
            paths.append(dict(a=a, k=k, move=move, lane=lane, pts=pts, s=s, box_len=box_len, length=float(s[-1]),
                              exit_road=exit_road))
    # exit lane identity: the exit point's lateral offset in the exit road's frame
    for p in paths:
        q = p["pts"][np.searchsorted(p["s"], APP + p["box_len"])]
        local = q @ _rot(p["exit_road"])  # rotate back into the exit road's approach-0 frame
        p["exit_lane"] = int(np.argmin([abs(local[0] - lane_x(l)) for l in range(LANES)]))
    return paths


def conflicts(paths, tol=1.2):
    """(p, q, s_p, s_q, type) for every pair of paths from different approaches whose box pieces cross, plus merges
    (two paths entering the same exit lane)."""
    out = []
    for i in range(N_PATHS):
        for j in range(i + 1, N_PATHS):
            P, Q = paths[i], paths[j]
            if P["a"] == Q["a"]:
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
                if rel == 2 and "L" in moves:
                    kind = LEFT
                elif rel in (1, 3):
                    kind = ANGLE
                else:
                    kind = OTHER
                out.append((i, j, sp, sq, kind))
    return out


PATHS = build()
CONFLICTS = conflicts(PATHS)
PATH_A = np.array([p["a"] for p in PATHS])
PATH_LANE_ARR = np.array([p["lane"] for p in PATHS])
PATH_MOVE = np.array(["LTR".index(p["move"]) for p in PATHS])  # 0 L, 1 T, 2 R
PATH_BOX = np.array([p["box_len"] for p in PATHS], np.float32)
PATH_LEN = np.array([p["length"] for p in PATHS], np.float32)
PATH_EXIT = np.array([p["exit_road"] * LANES + p["exit_lane"] for p in PATHS])


def path_index(a, lane, move):
    """Path for (approach, lane, move 0 L / 1 T / 2 R), or -1 if that lane cannot make that move."""
    table = {(0, 0): 0, (1, 1): 1, (2, 1): 2, (2, 2): 3}
    k = table.get((lane, move), -1)
    return -1 if k < 0 else a * 4 + k


def path_table():
    """[4, LANES, 3] path index for (approach, lane, move), -1 if invalid; and the default move per lane."""
    t = -np.ones((4, LANES, 3), np.int32)
    for a in range(4):
        for lane in range(LANES):
            for m in range(3):
                t[a, lane, m] = path_index(a, lane, m)
    return t, np.array([0, 1, 1])  # lane 0 must turn left, lanes 1-2 default to through


def conflict_arrays():
    c = np.array(CONFLICTS, dtype=np.float64)
    return (c[:, 0].astype(np.int32), c[:, 1].astype(np.int32), c[:, 2].astype(np.float32), c[:, 3].astype(np.float32),
            c[:, 4].astype(np.int32))


def xy(path, s):
    """World position and heading of a point at arc length s on a path (for drawing)."""
    P = PATHS[path]
    s = np.clip(s, 0, P["length"] - 1e-3)
    i = np.clip(np.searchsorted(P["s"], s) - 1, 0, len(P["s"]) - 2)
    f = (s - P["s"][i]) / (P["s"][i + 1] - P["s"][i])
    pt = P["pts"][i] * (1 - f)[..., None] + P["pts"][i + 1] * f[..., None]
    d = P["pts"][i + 1] - P["pts"][i]
    return pt, np.arctan2(d[..., 1], d[..., 0])
