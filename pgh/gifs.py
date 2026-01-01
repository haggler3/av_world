"""Labelled GIFs of mined Pittsburgh intersections: one crash clip per chosen intersection, and a 2x3 grid of six
intersections simulated together in one vmapped batch.

    python -m pgh.gifs        # -> docs/pgh_*.gif
"""
import json
import re

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from xworld import calibrate as C  # noqa: E402
from xworld import render as R  # noqa: E402
from xworld import sim as S  # noqa: E402

from . import layouts as P  # noqa: E402

CLIPS = ["Forbes Avenue & Murray Avenue", "South Craig Street & Forbes Avenue",
         "Reizenstein Way & Penn Avenue & East Liberty Boulevard"]
GRID = ["Forbes Avenue & Murray Avenue", "South Craig Street & Forbes Avenue",
        "Reizenstein Way & Penn Avenue & East Liberty Boulevard", "Liberty Avenue & Wood Street"]
FOOT = "* inferred (Overture/OSM has no lane counts or signals)   red outline: entered on red   amber: distracted   blue: AV"


def labels(it):
    return {l["slot"]: f"{l['name'] or '(unnamed)'}\n{l['speed_mph']} mph · {l['lanes_in']} in / {l['lanes_out']} out*"
            for l in it["legs"]}


def subtitle(it):
    return f"{'signal*' if it['control'] == 'signal' else 'two-way stop*'} · {it['n_legs']} legs · {it['lat']:.4f}, {it['lon']:.4f}"


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


def behaviour(amplify, av=0.15):
    knobs = json.load(open("results/calibration.json"))["calibrated"]
    return S.Behaviour(**{k: min(v * amplify, 1.0) for k, v in knobs.items()}, p_av=av)


def clip(it, amplify=100.0, out=None):
    geo, lay = P.build(it)
    out = out or f"docs/pgh_{slug(it['name'])}.gif"
    found = R.render(behaviour(amplify), out, geo, lay, title=f"Pittsburgh: {it['name']}", subtitle=subtitle(it),
                     labels=labels(it), footer=FOOT + f"   (behaviour x{amplify:.0f})")
    return out, found


def grid(its, out="docs/pgh_grid.gif", amplify=None, seconds=20, warm=90, fps=5, seed=1):
    """Six real intersections in one batch (one Layout each, stacked and vmapped)."""
    built = [P.build(i) for i in its]
    lay = S.stack([b[1] for b in built])
    amplify = amplify or C.amplification(json.load(open("results/calibration.json"))["calibrated"])
    beh = behaviour(amplify)
    st = jax.vmap(S.reset, in_axes=(0, None, 0))(jax.random.split(jax.random.PRNGKey(seed), len(its)), beh, lay)
    step = jax.jit(jax.vmap(S.step, in_axes=(0, None, 0)))
    n_warm, n = int(warm / S.DT), int(seconds / S.DT)
    hist = []
    for k in range(n_warm + n + 1):
        st = step(st, beh, lay)
        if k >= n_warm:
            hist.append(jax.tree.map(np.asarray, st))
    reps = []
    for j, (geo, lay_j) in enumerate(built):
        h = [jax.tree.map(lambda a: a[j], x) for x in hist]
        reps.append(R.Replay(geo, jax.tree.map(lambda a: a[j], lay), h, R.Scene(geo, labels=labels(its[j]))))
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 9.6), facecolor=R.BG)
    fig.subplots_adjust(0.005, 0.03, 0.995, 0.9, wspace=0.02, hspace=0.12)
    fig.text(0.01, 0.955, f"{len(its)} Pittsburgh intersections, auto-mined from OpenStreetMap (Overture), simulated in one batch",
             color="#e6edf3", fontsize=13, weight="bold")
    clock = fig.text(0.01, 0.925, "", color="#8b949e", fontsize=9)
    fig.text(0.01, 0.008, FOOT + f"   (calibrated behaviour x{amplify:.0f})", color="#8b949e", fontsize=8)

    def draw(k):
        for ax, rep, it in zip(axs.flat, reps, its):
            ax.cla()
            rep.draw(ax, k)
            ax.set_title(f"{it['name']}\n{subtitle(it)}", color="#e6edf3", fontsize=8.5)
        clock.set_text(f"t = {k * S.DT:4.1f} s")
        return []

    frames = list(range(0, len(hist) - 1, max(int(1 / (fps * S.DT)), 1)))
    FuncAnimation(fig, draw, frames=frames).save(out, writer=PillowWriter(fps=fps), dpi=60, savefig_kwargs={"facecolor": R.BG})
    plt.close(fig)
    print("wrote", out, flush=True)


def pick_stop(its):
    """The busiest stop-controlled 4-leg and 3-leg intersections with every leg named (by inferred demand)."""
    out = []
    for legs in (4, 3):
        cand = [i for i in its if i["control"] != "signal" and i["n_legs"] == legs and all(l["name"] for l in i["legs"])]
        out.append(max(cand, key=lambda i: (sum(l["demand_vph"] for l in i["legs"]), i["id"])))
    return out


def main():
    its = P.load()
    by_name = {i["name"]: i for i in its}
    stops = pick_stop(its)
    made = []
    for it in [by_name[n] for n in CLIPS] + stops:
        out, found = clip(it, amplify=100.0 if it["control"] == "signal" else 300.0)
        made.append((it["name"], out, found))
    grid([by_name[n] for n in GRID] + stops)
    print(made)


if __name__ == "__main__":
    main()
