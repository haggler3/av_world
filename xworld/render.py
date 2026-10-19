"""Top-down 2D GIF of one simulated intersection: multi-lane approaches, signals, turning cars, and crashes.

    python -m xworld.render --out docs/intersection.gif
"""
import argparse

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

from . import geometry as G  # noqa: E402
from . import sim as S  # noqa: E402

BG, ASPHALT, WALK, GRASS, BUILD = "#0d1117", "#2b2f36", "#3a3f47", "#1b2a1e", "#21262d"
LIGHT = {0: "#ff4d4d", 1: "#ffb020", 2: "#3ddc84"}
HUMAN = ["#8b949e", "#6e7681", "#a5d6ff", "#d2a8ff", "#ffa198", "#7ee787", "#e3b341"]
AV = "#2f81f7"
W = G.LANE_W
R = 3 * W  # road half-width


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


def scene(ax, sig):
    lim = 48
    ax.add_patch(Rectangle((-lim, -lim), 2 * lim, 2 * lim, fc=GRASS, zorder=0))
    for sx in (-1, 1):  # building blocks in the four corners
        for sy in (-1, 1):
            ax.add_patch(Rectangle((sx * (R + 4) if sx > 0 else -lim, sy * (R + 4) if sy > 0 else -lim), lim - R - 4, lim - R - 4,
                                   fc=BUILD, ec="#30363d", lw=0.8, zorder=0.5))
    ax.add_patch(Rectangle((-lim, -R - 1.6), 2 * lim, 2 * R + 3.2, fc=WALK, zorder=1))
    ax.add_patch(Rectangle((-R - 1.6, -lim), 2 * R + 3.2, 2 * lim, fc=WALK, zorder=1))
    ax.add_patch(Rectangle((-lim, -R), 2 * lim, 2 * R, fc=ASPHALT, zorder=1.2))
    ax.add_patch(Rectangle((-R, -lim), 2 * R, 2 * lim, fc=ASPHALT, zorder=1.2))
    for a in range(4):  # markings per approach, drawn in the approach-0 frame and rotated
        rot = Affine2D().rotate_deg(90 * a) + ax.transData
        ax.plot([-0.15, -0.15], [-lim, -G.H], color="#e3b341", lw=1.1, transform=rot, zorder=2)
        ax.plot([0.15, 0.15], [-lim, -G.H], color="#e3b341", lw=1.1, transform=rot, zorder=2)
        for x in (W, 2 * W, -W, -2 * W):
            for y0 in np.arange(-lim, -G.H - 3, 6):
                ax.plot([x, x], [y0, y0 + 3], color="#c9d1d9", lw=0.8, alpha=0.7, transform=rot, zorder=2)
        ax.plot([0, R], [-G.H, -G.H], color="white", lw=2.2, transform=rot, zorder=2)  # stop line
        for k in range(8):  # crosswalk
            ax.add_patch(Rectangle((-R + k * 2 * R / 8 + 0.4, -G.H - 3.4), 1.6, 2.6, fc="#c9d1d9", alpha=0.55, transform=rot, zorder=2))
        # signal head: left arrow + through/right
        for j, (k, x) in enumerate(((0, W * 0.5), (1, W * 2))):
            st = int(sig[a * 4 + k])
            ax.add_patch(Circle((x, -G.H - 5.2), 0.7, fc=LIGHT[st], transform=rot, zorder=6))
            if k == 0:
                ax.add_patch(Polygon([(x - 0.5, -G.H - 5.2), (x + 0.3, -G.H - 4.7), (x + 0.3, -G.H - 5.7)], fc="#0d1117", transform=rot, zorder=7))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def find_crash(beh, n_envs=1024, seconds=300, seed=3, warm=60):
    """Pass 1: run many intersections, keeping only per-step crash counts; return (env, step) of a crash after warm-up."""
    st = jax.vmap(S.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(seed), n_envs), beh)
    step = jax.jit(jax.vmap(S.step, in_axes=(0, None)))
    counts = []
    for _ in range(int((warm + seconds) / S.DT)):
        st = step(st, beh)
        counts.append(np.asarray(st.m[:, 2:7].sum(1)))
    c = np.array(counts)
    new = np.diff(c, axis=0) > 0
    new[: int((warm + 15) / S.DT)] = False
    ts, es = np.nonzero(new)
    if len(ts) == 0:
        return 0, len(c) // 2
    return int(es[0]), int(ts[0]) + 1


def record(beh, env, t_crash, n_envs=1024, seconds=24, seed=3):
    """Pass 2: re-run the same (deterministic) batch and keep only that intersection around the crash."""
    st = jax.vmap(S.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(seed), n_envs), beh)
    step = jax.jit(jax.vmap(S.step, in_axes=(0, None)))
    n = int(seconds / S.DT)
    t0 = max(t_crash - int(0.6 * n), 0)
    hist = []
    for k in range(t0 + n + 1):
        st = step(st, beh)
        if k + 1 >= t0:
            hist.append(jax.tree.map(lambda a: np.asarray(a)[env:env + 1], st))
    return hist


def render(beh, out, seconds=24, fps=10):
    e, t_c = find_crash(beh)
    hist = record(beh, e, t_c, seconds=seconds)
    env = 0
    events = [k for k in range(len(hist) - 1) if hist[k + 1].m[0, 2:7].sum() > hist[k].m[0, 2:7].sum()]
    print(f"crash found in env {e} at step {t_c}; crash frames in the replay: {events}")
    frames = range(0, len(hist) - 1, max(int(1 / (fps * S.DT)), 1))
    fig, ax = plt.subplots(figsize=(6.4, 6.6), facecolor=BG)
    fig.subplots_adjust(0, 0, 1, 0.93)
    title = fig.text(0.02, 0.965, "", color="#e6edf3", fontsize=11, weight="bold")
    wrecks = []  # (frame index, x, y, th, type) kept on screen after a crash

    def draw(k):
        ax.cla()
        h, h1 = hist[k], hist[k + 1]
        sig = np.asarray(S.signals(jnp.asarray(h.t[env]), jnp.asarray(h.offset[env]))[0])
        scene(ax, sig)
        alive = h.alive[env]
        st_e = jax.tree.map(lambda a: a[env], h)
        path = np.asarray(S.intended_path(jax.tree.map(jnp.asarray, st_e)))
        dm = h1.m[env, 2:7] - h.m[env, 2:7]
        gone = alive & ~h1.alive[env]
        for i in np.where(alive)[0]:
            (x, y), th = G.xy(int(path[i]), float(h.s[env, i]))
            color = AV if h.av[env, i] else HUMAN[i % len(HUMAN)]
            outline = "#ff4d4d" if h.ran_red[env, i] else ("#ffb020" if h.dist_t[env, i] > 0 else None)
            car(ax, x, y, th, color, outline=outline)
            if gone[i] and dm.sum() > 0 and h.s[env, i] < G.PATH_LEN[path[i]] - 3:
                wrecks.append((k, x, y, th, G.CRASH_NAMES[int(np.argmax(dm))]))
        for (kk, x, y, th, name) in wrecks:
            if 0 <= k - kk < 40:
                car(ax, x, y, th, "#f0883e", alpha=0.9, outline="white")
                ax.add_patch(Circle((x, y), 4.0, fc="#f0883e", alpha=0.18, zorder=8))
                ax.text(x, y + 5.5, f"CRASH: {name}", color="#ffa657", fontsize=10, weight="bold", zorder=12, ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#f0883e", alpha=0.9))
        tau = (h.t[env] + h.offset[env]) % S.CYCLE
        title.set_text(f"av_world intersection  ·  t = {h.t[env]:.0f} s  ·  signal cycle {tau:4.1f}/{S.CYCLE:.0f} s")
        ax.text(-47, -46.5, "red outline: entered on red   amber outline: distracted right now   blue: AV   (risky behaviour amplified for the clip)", color="#8b949e", fontsize=7.5, zorder=9)
        return []

    FuncAnimation(fig, draw, frames=list(frames)).save(out, writer=PillowWriter(fps=fps), dpi=95, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/intersection.gif")
    ap.add_argument("--amplify", type=float, default=100.0)  # crashes are rare: amplify so one happens in the clip
    ap.add_argument("--av", type=float, default=0.15)
    a = ap.parse_args()
    import json

    knobs = json.load(open("results/calibration.json"))["calibrated"]
    render(S.Behaviour(**{k: min(v * a.amplify, 1.0) for k, v in knobs.items()}, p_av=a.av), a.out)
