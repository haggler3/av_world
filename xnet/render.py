"""GIFs of a whole simulated area: an overview (every vehicle, coloured by speed) and a zoomed street view (car sprites,
signals, stop signs, pedestrians on crosswalks, buses at stops, double-parked vans, breakdowns, crashes).

    python -m xnet.render                    # -> docs/xnet_oakland_overview.gif, docs/xnet_oakland_zoom.gif
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Polygon, RegularPolygon  # noqa: E402
from shapely.geometry import LineString, MultiPoint  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from . import network as NW  # noqa: E402
from . import sim as X  # noqa: E402

BG, ASPHALT, GROUND, LINE = "#0d1117", "#30353d", "#141a1f", "#c9d1d9"
LIGHT = np.array(["#ff4d4d", "#ffb020", "#3ddc84"])
CARS = ["#8b949e", "#a5d6ff", "#d2a8ff", "#ffa198", "#7ee787", "#6e7681", "#b392f0"]
BUS, VAN, AVC, WRECK, BIKE = "#f2cc60", "#e6edf3", "#2f81f7", "#f0883e", "#3ddc84"


class Tracks:
    """Every track's polyline (link lanes, moves through boxes) concatenated for vectorised position lookups."""

    def __init__(self, net, geo):
        L = len(net["link_len"])
        polys = []
        for li in range(L):
            for lane in range(2):
                polys.append(NW._offset(geo.link_pts[li], geo.link_off[li] + NW.W * (lane + 0.5)))
        polys += list(geo.mov_pts)
        self.off = np.zeros(len(polys))
        xs, ys, ss = [], [], []
        base = 0.0
        for t, p in enumerate(polys):
            s = NW._poly_len(p)
            nominal = net["link_len"][t // 2] if t < 2 * L else net["mov_len"][t - 2 * L]
            s = s * (nominal / max(s[-1], 1e-6))  # map the simulator's arc length onto the drawn polyline
            self.off[t] = base
            xs.append(p[:, 0])
            ys.append(p[:, 1])
            ss.append(base + s)
            base += s[-1] + 1000.0
        self.x, self.y, self.s = np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)

    def pose(self, trk, s, length):
        """Centre position and heading of vehicles (their front is at arc length s)."""
        g = self.off[trk] + np.maximum(s - length / 2, 0.0)
        g2 = g + 0.8
        x, y = np.interp(g, self.s, self.x), np.interp(g, self.s, self.y)
        x2, y2 = np.interp(g2, self.s, self.x), np.interp(g2, self.s, self.y)
        return x, y, np.arctan2(y2 - y, x2 - x)


def car_polys(x, y, th, L, Wd):
    c, s = np.cos(th), np.sin(th)
    loc = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    px = loc[None, :, 0] * L[:, None]
    py = loc[None, :, 1] * Wd[:, None]
    return np.stack([x[:, None] + c[:, None] * px - s[:, None] * py, y[:, None] + s[:, None] * px + c[:, None] * py], -1)


class Map:
    """The static street map, drawn once."""

    def __init__(self, net, geo, info):
        self.net, self.geo, self.info = net, geo, info
        shapes = []
        for pts, fwd, bwd, rank, name in geo.chains:
            lanes = 2 if (rank >= 4 or (not (fwd and bwd) and rank >= 3)) else 1
            half = lanes * NW.W * (1.0 if (fwd and bwd) else 0.5)
            shapes.append(LineString(pts).buffer(half, cap_style=2, join_style=2))
        ends = {}
        for li, p in enumerate(geo.link_pts):  # junction boxes: the hull of the road ends around each junction
            lanes = net["link_lanes"][li]
            for nd, end in ((net["link_end"][li], True), (info["links"][li]["a"], False)):
                d = NW._dir(p, end)
                r = np.array([d[1], -d[0]])
                q = p[-1] if end else p[0]
                lo, hi = geo.link_off[li] - (lanes * NW.W if geo.link_off[li] == 0 else 0), geo.link_off[li] + lanes * NW.W
                ends.setdefault(nd, []).extend([q + r * lo, q + r * hi])
        for nd, pts in ends.items():
            if geo.node_H[nd] > 1.5 and len(pts) >= 3:
                shapes.append(MultiPoint(pts).convex_hull.buffer(1.0))
        self.road = unary_union(shapes)

    def draw(self, ax, detail=True, labels=True):
        net, geo = self.net, self.geo
        for g in getattr(self.road, "geoms", [self.road]):
            ax.add_patch(Polygon(np.asarray(g.exterior.coords), fc=ASPHALT, ec="none", zorder=1))
            for hole in g.interiors:
                ax.add_patch(Polygon(np.asarray(hole.coords), fc=GROUND, ec="none", zorder=1.1))
        if not detail:
            return
        segs_y, segs_w = [], []
        for pts, fwd, bwd, rank, name in geo.chains:
            if fwd and bwd:
                segs_y += [NW._offset(pts, -0.15), NW._offset(pts, 0.15)]
        for li, p in enumerate(geo.link_pts):
            if net["link_lanes"][li] == 2:
                q = NW._offset(p, geo.link_off[li] + NW.W)
                s = NW._poly_len(q)
                for a in np.arange(2.0, s[-1] - 3, 6.0):
                    segs_w.append(NW._cut(q, a, a + 3))
            if net["link_stop"][li] or net["link_axis"][li] >= 0:  # stop line
                d = NW._dir(p, True)
                r = np.array([d[1], -d[0]])
                a = p[-1] + r * (geo.link_off[li] + 0.2)
                b = p[-1] + r * (geo.link_off[li] + NW.W * net["link_lanes"][li] - 0.2)
                segs_w.append(np.stack([a, b]))
        ax.add_collection(LineCollection(segs_y, colors="#e3b341", linewidths=0.6, zorder=2))
        ax.add_collection(LineCollection(segs_w, colors=LINE, linewidths=0.6, alpha=0.7, zorder=2))
        for k, (a, b) in enumerate(geo.leg_xy):  # crosswalks
            if not np.any(a) and not np.any(b):
                continue
            v = b - a
            n = max(int(np.linalg.norm(v) / 1.3), 2)
            d = np.array([-v[1], v[0]]) / max(np.linalg.norm(v), 1e-6)
            stripes = [np.stack([a + v * (j + 0.5) / n - d * 1.3, a + v * (j + 0.5) / n + d * 1.3]) for j in range(n)]
            ax.add_collection(LineCollection(stripes, colors=LINE, linewidths=1.6, alpha=0.35, zorder=2))
        for li in np.nonzero(net["link_stop"])[0]:
            p = geo.link_pts[li]
            d = NW._dir(p, True)
            r = np.array([d[1], -d[0]])
            xy = p[-1] + r * (geo.link_off[li] + NW.W * net["link_lanes"][li] + 1.6)
            ax.add_patch(RegularPolygon(xy, 8, radius=1.2, orientation=np.pi / 8, fc="#d1242f", ec="white", lw=0.4, zorder=6))
        if labels:
            done = set()
            order = sorted(range(len(geo.chains)), key=lambda i: -NW._poly_len(geo.chains[i][0])[-1])
            for i in order:
                pts, fwd, bwd, rank, name = geo.chains[i]
                if not name or name in done or rank < 2 or NW._poly_len(pts)[-1] < 80:
                    continue
                done.add(name)
                s = NW._poly_len(pts)
                m = np.searchsorted(s, s[-1] / 2)
                d = pts[min(m + 1, len(pts) - 1)] - pts[max(m - 1, 0)]
                ang = (np.degrees(np.arctan2(d[1], d[0])) + 90) % 180 - 90
                ax.text(*pts[m], name, color="#e6edf3", fontsize=6, rotation=ang, ha="center", va="center", zorder=12,
                        rotation_mode="anchor", alpha=0.85,
                        bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.6))


def snapshot_artists(ax, net, geo, tracks, st, zoom, sig_heads):
    """Dynamic layer for one frame; returns the artists to remove next frame."""
    arts = []
    a = st.alive
    trk, s, length = st.trk[a], st.s[a], st.length[a]
    x, y, th = tracks.pose(trk, s, length)
    v = st.v[a]
    kind = st.kind[a]
    wreck = st.wreck_t[a] > 0
    if zoom:
        wd = np.where(kind == X.BUS, 2.5, np.where(kind == X.VAN, 2.1, np.where(kind == X.BIKE, 0.7, 1.85)))
        polys = car_polys(x, y, th, length - 0.3, wd)
        idx = np.nonzero(a)[0]
        cols = np.array(CARS)[idx % len(CARS)]
        cols = np.where(kind == X.BUS, BUS, np.where(kind == X.VAN, VAN, np.where(kind == X.BIKE, BIKE, cols)))
        cols = np.where(st.av[a], AVC, cols)
        cols = np.where(wreck, WRECK, cols)
        ec = np.where(st.broken[a] | wreck, "#ff4d4d", np.where(st.stop_t[a] > 0, "#ffb020", np.where(st.dist_t[a] > 0, "#ffb020", "none")))
        pc = PolyCollection(polys, facecolors=cols, edgecolors=ec, linewidths=1.2, zorder=7)
        ax.add_collection(pc)
        arts.append(pc)
        # windshields
        ws = car_polys(x + np.cos(th) * length * 0.18, y + np.sin(th) * length * 0.18, th, length * 0.12, wd * 0.8)
        pw = PolyCollection(ws, facecolors="#0d1b2a", edgecolors="none", alpha=0.8, zorder=8)
        ax.add_collection(pw)
        arts.append(pw)
    else:
        vlim = net["link_v"][np.minimum(trk // 2, len(net["link_v"]) - 1)]
        rel = np.clip(v / np.maximum(vlim, 1), 0, 1)
        cmap = plt.get_cmap("turbo")
        sc = ax.scatter(x, y, s=np.where(kind == X.BUS, 7, 3.5), c=cmap(0.15 + 0.7 * (1 - rel)), lw=0, zorder=7)
        arts.append(sc)
        if wreck.any():
            arts.append(ax.scatter(x[wreck], y[wreck], s=60, facecolors="none", edgecolors=WRECK, lw=1.2, zorder=9))
    # signal heads
    sig = np.asarray(X.signal_state({k: np.asarray(v_) for k, v_ in net.items()}, float(st.t)))
    hx, hy, li = sig_heads
    arts.append(ax.scatter(hx, hy, s=9 if zoom else 2.5, c=LIGHT[sig[li]], lw=0, zorder=9))
    # pedestrians
    on = st.ped_t > 0
    if zoom and on.any():
        k = np.nonzero(on)[0]
        prog = np.clip(1 - st.ped_t[k] / 11.0, 0.05, 0.95)
        pa, pb = geo.leg_xy[k, 0], geo.leg_xy[k, 1]
        flip = (k % 2 == 1)[:, None]
        p0, p1 = np.where(flip, pb, pa), np.where(flip, pa, pb)
        pp = p0 + (p1 - p0) * prog[:, None]
        arts.append(ax.scatter(pp[:, 0], pp[:, 1], s=14, c="#f778ba", edgecolors="white", lw=0.5, zorder=10))
    return arts


def signal_heads(net, geo):
    hx, hy, li = [], [], []
    for l in np.nonzero(net["link_axis"] >= 0)[0]:
        p = geo.link_pts[l]
        d = NW._dir(p, True)
        r = np.array([d[1], -d[0]])
        xy = p[-1] + r * (geo.link_off[l] + NW.W * net["link_lanes"][l] + 1.4) - d * 1.0
        hx.append(xy[0])
        hy.append(xy[1])
        li.append(l)
    return np.array(hx), np.array(hy), np.array(li)


def animate(net, geo, info, frames, out, zoom=None, title="", fps=8, dpi=80, every=1):
    mp = Map(net, geo, info)
    tracks = Tracks(net, geo)
    heads = signal_heads(net, geo)
    W_, H_ = (7.2, 7.6) if zoom else (9.6, 7.6)
    fig, ax = plt.subplots(figsize=(W_, H_), facecolor=BG)
    fig.subplots_adjust(0.01, 0.06, 0.99, 0.9)
    ax.set_facecolor(GROUND)
    mp.draw(ax, detail=zoom is not None, labels=True)
    if zoom:
        (cx, cy), r = zoom
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
    else:
        x0, y0 = NW.to_m(geo.bbox[0], geo.bbox[1])
        x1, y1 = NW.to_m(geo.bbox[2], geo.bbox[3])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.015, 0.955, title, color="#e6edf3", fontsize=12, weight="bold")
    hud = fig.text(0.015, 0.918, "", color="#8b949e", fontsize=8.5)
    if zoom:
        foot = ("grey/pastel: drivers   yellow: bus   white: delivery van   blue: AV   green: cyclist   pink dots: pedestrians\n"
                "amber outline: stopped at a bus stop / double-parked / distracted   red outline + orange: breakdown or crash")
    else:
        foot = "every vehicle, coloured by speed vs the limit (blue fast ... red stopped); dots at junctions: signal heads"
    fig.text(0.015, 0.012, foot + "\nmap: Overture Maps / OpenStreetMap; lanes, signals, stop signs and demand inferred",
             color="#8b949e", fontsize=6.8)
    arts = []
    m0 = frames[0].m

    def draw(k):
        nonlocal arts
        for a_ in arts:
            a_.remove()
        st = frames[k]
        arts = snapshot_artists(ax, net, geo, tracks, st, zoom is not None, heads)
        d = dict(zip(X.METRICS, st.m - m0))
        nv = int(st.alive.sum())
        sp = 3.6 * float((st.v * st.alive).sum() / max(nv, 1))
        crashes = int(sum(d[c] for c in X.CRASH_NAMES))
        hud.set_text(f"t = {float(st.t) / 60:5.1f} min   vehicles {nv}   mean speed {sp:4.1f} km/h   trips done {int(d['finished'])}   "
                     f"pedestrians crossing {int((st.ped_t > 0).sum())}   crashes {crashes}")
        return []

    FuncAnimation(fig, draw, frames=list(range(0, len(frames), every))).save(
        out, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="oakland")
    ap.add_argument("--demand", type=float, default=1.3)
    ap.add_argument("--warm", type=float, default=600)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--amplify", type=float, default=0.0, help="risky behaviour x this (0 = calibrated rates)")
    a = ap.parse_args()
    net, geo, info = NW.build(a.area)
    knobs = json.load(open("results/calibration.json"))["calibrated"]
    amp = max(a.amplify, 1.0)
    beh = X.Behaviour(**{k: min(v * amp, 1.0) for k, v in knobs.items()}, demand=a.demand, p_av=0.1)
    sim = X.Sim(net, n_slots=2048, n_envs=1, beh=beh, seed=3)
    sim.run(a.warm)
    frames = sim.run(a.seconds, record=(0, 2))  # every 0.4 s
    name = a.area.replace("_", " ").title()
    animate(net, geo, info, frames[::5], f"docs/xnet_{a.area}_overview.gif",
            title=f"{name}, Pittsburgh: the whole area simulated (one of a vmapped batch)", fps=6, dpi=72)
    # zoom: a named junction if the area has it, else the busiest signalised one
    want = {"oakland": ("Forbes Avenue", "South Craig Street")}.get(a.area)
    counts = np.bincount(frames[-1].trk[frames[-1].alive] // 2, minlength=2 * len(net["link_len"]))
    sig_nodes = np.nonzero(net["node_signal"])[0]
    named = [nd for nd in sig_nodes if want and set(want) <= {geo.link_name[l] for l in np.nonzero(net["link_end"] == nd)[0]}]
    busy = named[0] if named else max(sig_nodes, key=lambda nd: sum(counts[l] for l in np.nonzero(net["link_end"] == nd)[0]))
    label = " & ".join(want) if named else "busiest signal"
    animate(net, geo, info, frames, f"docs/xnet_{a.area}_zoom.gif", zoom=(geo.node_xy[busy], 110.0),
            title=f"{name}: street view around {label} (same run)", fps=8, dpi=80, every=1)


if __name__ == "__main__":
    main()
