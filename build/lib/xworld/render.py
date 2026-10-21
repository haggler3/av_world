"""Top-down 2D GIFs of simulated intersections: real leg bearings and lane counts, signals or stop signs, turning cars,
and crashes, with labels.

    python -m xworld.render --out docs/intersection.gif        # the default 4-way, 3-lane intersection
    python -m pgh.gifs                                         # mined Pittsburgh intersections (uses this module)
"""
import argparse

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, RegularPolygon, Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from shapely.geometry import Polygon as SPoly  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from . import geometry as G  # noqa: E402
from . import sim as S  # noqa: E402

BG, ASPHALT, WALK, GRASS = "#0d1117", "#2b2f36", "#3a3f47", "#16241a"
LIGHT = {0: "#ff4d4d", 1: "#ffb020", 2: "#3ddc84"}
HUMAN = ["#8b949e", "#6e7681", "#a5d6ff", "#d2a8ff", "#ffa198", "#7ee787", "#e3b341"]
AV = "#2f81f7"
W = G.LANE_W


def car(ax, x, y, th, color, alpha=1.0, outline=None, L=4.6, Wd=1.9, z=5):
    """Car sprite centred at (x, y) heading th (rad)."""
    tr = Affine2D().rotate(th).translate(x, y) + ax.transData
    ax.add_patch(FancyBboxPatch((-L / 2, -Wd / 2), L, Wd, boxstyle="round,pad=0,rounding_size=0.5", fc=color,
                                ec=outline or "none", lw=1.5 if outline else 0, alpha=alpha, transform=tr, zorder=z))
    ax.add_patch(Polygon([(L * 0.12, -Wd * 0.38), (L * 0.28, -Wd * 0.42), (L * 0.28, Wd * 0.42), (L * 0.12, Wd * 0.38)],
                         fc="#0d1b2a", alpha=0.85 * alpha, transform=tr, zorder=z + 1))  # windshield
    ax.add_patch(Polygon([(-L * 0.36, -Wd * 0.40), (-L * 0.25, -Wd * 0.35), (-L * 0.25, Wd * 0.35), (-L * 0.36, Wd * 0.40)],
                         fc="#0d1b2a", alpha=0.8 * alpha, transform=tr, zorder=z + 1))  # rear window
    for sy in (-Wd * 0.33, Wd * 0.33):
        ax.add_patch(Circle((L / 2 - 0.12, sy), 0.15, fc="#fff3b0", alpha=alpha, transform=tr, zorder=z + 2))
        ax.add_patch(Rectangle((-L / 2 + 0.02, sy - 0.11), 0.12, 0.22, fc="#ff3b3b", alpha=alpha, transform=tr, zorder=z + 2))


class Scene:
    """Static road drawing for one geometry (computed once), plus per-frame signals."""

    def __init__(self, geo: G.Geometry, lim=46.0, labels=None):
        self.geo, self.lim, self.labels = geo, lim, labels or {}
        far = lim * 1.6
        strips, walks = [], []
        self.leg = {}
        for a in range(4):
            if not geo.present[a]:
                continue
            u = geo.u[a]
            r = np.array([-u[1], u[0]])  # right of the inbound heading; lateral coordinate "lat" runs along r
            nin, nout = geo.nin[a], geo.nout[a]
            off_in = 0.0 if nout else nin * W / 2
            off_out = 0.0 if nin else nout * W / 2
            lo = -(nout * W - off_out) if nout else -off_in
            hi = nin * W - off_in if nin else off_out
            self.leg[a] = (u, r, lo, hi, off_in, off_out)
            P = lambda d, lat: u * d + r * lat  # noqa: E731
            strips.append(SPoly([P(0, lo), P(far, lo), P(far, hi), P(0, hi)]))
            walks.append(SPoly([P(0, lo - 2.0), P(far, lo - 2.0), P(far, hi + 2.0), P(0, hi + 2.0)]))
        self.road = unary_union(strips).buffer(4, join_style=1).buffer(-4, join_style=1)
        self.walk = unary_union(walks).buffer(5, join_style=1).buffer(-5, join_style=1)

    def draw(self, ax, sig=None, stop_sign=None):
        geo, lim = self.geo, self.lim
        ax.add_patch(Rectangle((-lim, -lim), 2 * lim, 2 * lim, fc=GRASS, zorder=0))
        for shape, col, z in ((self.walk, WALK, 1), (self.road, ASPHALT, 1.2)):
            for g in getattr(shape, "geoms", [shape]):
                ax.add_patch(Polygon(np.asarray(g.exterior.coords), fc=col, zorder=z))
        far = lim * 1.6
        for a, (u, r, lo, hi, off_in, off_out) in self.leg.items():
            H = geo.Hs[a]
            P = lambda d, lat: u * d + r * lat  # noqa: E731
            seg = lambda d0, d1, lat, **kw: ax.plot(*np.stack([P(d0, lat), P(d1, lat)]).T, **kw)  # noqa: E731
            nin, nout = geo.nin[a], geo.nout[a]
            if nin and nout:
                for dl in (-0.15, 0.15):
                    seg(H, far, dl, color="#e3b341", lw=1.1, zorder=2)
            dividers = [k * W - off_in for k in range(1, nin)] + [-(k * W - off_out) for k in range(1, nout)]
            for lat in dividers:
                for d0 in np.arange(H + 3, far, 6):
                    seg(d0, d0 + 3, lat, color="#c9d1d9", lw=0.8, alpha=0.7, zorder=2)
            controlled = sig is not None or (stop_sign is not None and stop_sign[a * 4: a * 4 + 4].any())
            if nin and controlled:  # stop line only where traffic is stopped (signal or stop sign)
                ax.plot(*np.stack([P(H, -off_in + 0.1), P(H, nin * W - off_in - 0.1)]).T, color="white", lw=2.2, zorder=2)
            n_st = max(int((hi - lo) / 1.6), 2)
            for k in range(n_st):  # crosswalk
                lat0 = lo + (k + 0.25) * (hi - lo) / n_st
                c = [P(H + 0.8, lat0), P(H + 3.4, lat0), P(H + 3.4, lat0 + 0.55 * (hi - lo) / n_st), P(H + 0.8, lat0 + 0.55 * (hi - lo) / n_st)]
                ax.add_patch(Polygon(c, fc="#c9d1d9", alpha=0.5, zorder=2))
            if not nin:
                continue
            if stop_sign is not None and stop_sign[a * 4: a * 4 + 4].any():
                x, y = P(H + 1.0, hi + 1.6)
                ax.add_patch(RegularPolygon((x, y), 8, radius=1.35, orientation=np.pi / 8, fc="#d1242f", ec="white", lw=0.8, zorder=6))
                ax.text(x, y, "STOP", color="white", fontsize=4.2, ha="center", va="center", weight="bold", zorder=7)
            elif sig is not None:
                heads = [k for k in (1, 3, 0) if geo.path_valid[a * 4 + k]][:1] + ([0] if geo.path_valid[a * 4] else [])
                for j, k in enumerate(dict.fromkeys(heads)):
                    x, y = P(H + 5.2 + 1.7 * j, hi + 1.3)
                    ax.add_patch(Circle((x, y), 0.75, fc=LIGHT[int(sig[a * 4 + k])], ec="#0d1117", lw=0.6, zorder=6))
                    if k == 0:
                        ax.text(x, y, "←", color="#0d1117", fontsize=6, ha="center", va="center", weight="bold", zorder=7,
                                rotation=np.degrees(np.arctan2(-u[1], -u[0])) - 90)
        for a, txt in self.labels.items():  # street labels along each leg, outside the road
            if a not in self.leg:
                continue
            u, r, lo, hi, *_ = self.leg[a]
            ang = np.degrees(np.arctan2(u[1], u[0]))
            ang = (ang + 90) % 180 - 90
            side = hi + 4.0
            H = self.geo.Hs[a]
            x, y = u * min(lim * 0.68, H + 22) + r * side
            if abs(x) > lim - 6 or abs(y) > lim - 4:
                x, y = u * (H + 14) + r * side
            ax.text(x, y, txt, color="#e6edf3", fontsize=6.6, ha="center", va="center", rotation=ang, zorder=11,
                    rotation_mode="anchor", bbox=dict(boxstyle="round,pad=0.25", fc="#0d1117", ec="#30363d", alpha=0.85))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axis("off")


def find_crash(beh, lay=S.CANON, n_envs=1024, seconds=300, seed=3, warm=60):
    """Pass 1: run many copies of one intersection, keeping only per-step crash counts; return (env, step) of a crash."""
    st = jax.vmap(S.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), n_envs), beh, lay)
    step = jax.jit(jax.vmap(S.step, in_axes=(0, None, None)))
    counts = []
    for _ in range(int((warm + seconds) / S.DT)):
        st = step(st, beh, lay)
        counts.append(np.asarray(st.m[:, 2:7].sum(1)))
    c = np.array(counts)
    new = np.diff(c, axis=0) > 0
    new[: int((warm + 15) / S.DT)] = False
    ts, es = np.nonzero(new)
    if len(ts) == 0:
        return 0, len(c) // 2, False
    return int(es[0]), int(ts[0]) + 1, True


def record(beh, env, t_end, lay=S.CANON, n_envs=1024, seconds=24, seed=3, lead=0.6):
    """Pass 2: re-run the same (deterministic) batch and keep only that intersection for `seconds` around t_end."""
    st = jax.vmap(S.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), n_envs), beh, lay)
    step = jax.jit(jax.vmap(S.step, in_axes=(0, None, None)))
    n = int(seconds / S.DT)
    t0 = max(t_end - int(lead * n), 0)
    hist = []
    for k in range(t0 + n + 1):
        st = step(st, beh, lay)
        if k + 1 >= t0:
            hist.append(jax.tree.map(lambda a: np.asarray(a)[env], st))
    return hist


class Replay:
    """Draws frames of one recorded intersection onto an axis (keeps crash wrecks on screen for a few seconds)."""

    def __init__(self, geo, lay, hist, scene):
        self.geo, self.lay, self.hist, self.scene = geo, lay, hist, scene
        self.wrecks = []
        self.stop = np.asarray(lay.stop_sign)
        self.signal = not self.stop.any()

    def draw(self, ax, k):
        h, h1 = self.hist[k], self.hist[k + 1]
        lay_np = jax.tree.map(jnp.asarray, self.lay)
        sig = np.asarray(S.signals(jnp.asarray(h.t), jnp.asarray(h.offset), lay_np)[0])
        self.scene.draw(ax, sig if self.signal else None, self.stop)
        path = np.asarray(S.intended_path(jax.tree.map(jnp.asarray, h), lay_np))
        dm = h1.m[2:7] - h.m[2:7]
        gone = h.alive & ~h1.alive
        plen = np.asarray(self.lay.path_len)
        for i in np.where(h.alive)[0]:
            (x, y), th = self.geo.xy(int(path[i]), float(h.s[i]))
            if abs(x) > self.scene.lim + 5 or abs(y) > self.scene.lim + 5:
                continue
            color = AV if h.av[i] else HUMAN[i % len(HUMAN)]
            outline = "#ff4d4d" if h.ran_red[i] else ("#ffb020" if h.dist_t[i] > 0 else None)
            car(ax, x, y, th, color, outline=outline)
            if gone[i] and dm.sum() > 0 and h.s[i] < plen[path[i]] - 3 and not any(w[0] == k and w[5] == i for w in self.wrecks):
                self.wrecks.append((k, x, y, th, G.CRASH_NAMES[int(np.argmax(dm))], i))
        labelled = set()
        for (kk, x, y, th, name, _i) in self.wrecks:
            if 0 <= k - kk < 40:
                car(ax, x, y, th, "#f0883e", alpha=0.9, outline="white")
                ax.add_patch(Circle((x, y), 4.0, fc="#f0883e", alpha=0.18, zorder=8))
                if kk not in labelled:  # one label per crash (both cars are wrecks)
                    labelled.add(kk)
                    ax.text(x, y + 6.5, f"CRASH: {name}", color="#ffa657", fontsize=10, weight="bold", zorder=12, ha="center",
                            bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#f0883e", alpha=0.9))
        return h


def render(beh, out, geo=G.CANON, lay=S.CANON, title="av_world intersection", subtitle=None, labels=None,
           footer=None, seconds=24, fps=10, n_envs=1024):
    e, t_c, found = find_crash(beh, lay, n_envs=n_envs)
    hist = record(beh, e, t_c, lay, n_envs=n_envs, seconds=seconds)
    events = [k for k in range(len(hist) - 1) if hist[k + 1].m[2:7].sum() > hist[k].m[2:7].sum()]
    print(f"{out}: crash {'found' if found else 'NOT found'} in env {e} at step {t_c}; crash frames: {events}", flush=True)
    scene = Scene(geo, labels=labels)
    rep = Replay(geo, lay, hist, scene)
    frames = range(0, len(hist) - 1, max(int(1 / (fps * S.DT)), 1))
    fig, ax = plt.subplots(figsize=(6.4, 7.0), facecolor=BG)
    fig.subplots_adjust(0, 0.035, 1, 0.9)
    t1 = fig.text(0.02, 0.962, title, color="#e6edf3", fontsize=11.5, weight="bold")
    t2 = fig.text(0.02, 0.925, "", color="#8b949e", fontsize=8)
    fig.text(0.02, 0.012, footer or "red outline: entered on red   amber outline: distracted   blue: AV   (risky behaviour amplified for the clip)",
             color="#8b949e", fontsize=6.6)

    def draw(k):
        ax.cla()
        h = rep.draw(ax, k)
        clock = f"t = {h.t:.0f} s"
        if rep.signal:
            clock += f"  ·  signal cycle {(h.t + h.offset) % float(lay.cycle):4.1f}/{float(lay.cycle):.0f} s"
        t2.set_text(f"{subtitle + '  ·  ' if subtitle else ''}{clock}")
        return []

    FuncAnimation(fig, draw, frames=list(frames)).save(out, writer=PillowWriter(fps=fps), dpi=95, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", out, flush=True)
    return found


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/intersection.gif")
    ap.add_argument("--amplify", type=float, default=100.0)  # crashes are rare: amplify so one happens in the clip
    ap.add_argument("--av", type=float, default=0.15)
    a = ap.parse_args()
    import json

    knobs = json.load(open("results/calibration.json"))["calibrated"]
    render(S.Behaviour(**{k: min(v * a.amplify, 1.0) for k, v in knobs.items()}, p_av=a.av), a.out)
