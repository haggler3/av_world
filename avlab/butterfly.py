"""The butterfly effect in the city: how fast does a 1 mm nudge to one car turn into a different day?

    python -m avlab.butterfly --quick     # 4 worlds x 3 nudges x 3 min, ~1 min
    python -m avlab.butterfly             # 10 worlds x 6 nudges x 8 min -> results/butterfly/, docs/img/butterfly.{png,gif}

Why: docs/FIXES.md item 14 found that AVBench scores re-roll when the hardware changes (GPU vs CPU), a per-scenario SD of tens of
points. The suspected cause is that the simulator is chaotic: a floating-point difference of 1e-6 is the same kind of event as
moving one car by a millimetre. This measures it directly, with the random numbers held fixed (a twin shares its world's key).

Per world: warm up 90 s, then make twins of the state: one car moved by eps metres, or every car moved by eps x N(0, 1) m (eps = 0 is the
control: it must stay identical, which also tests determinism). Every 2 s each twin is compared with its eps = 0
reference:
    diverged(t) = share of vehicle slots that are alive in either run and differ (alive flag, track, or position by more than 1 m)
At the end: how different are the twins' 10-minute crash count and network speed from the reference, compared with two independent
worlds (different keys)? memory = mean |twin - reference| / mean |independent - independent| for the speed; 0 means the twin is
the same day, 1 means the nudge erased all memory of the original day.
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from xnet import sim as X

OUT = "results/butterfly"
BLOCK_S = 2.0


def nudge(st, mode, eps):
    """mode 'one': move the first live vehicle forward by eps metres. mode 'all': shift every live vehicle by eps x N(0, 1), the size
    of the noise a different floating-point summation order leaves behind. st is an unbatched State."""
    if mode == "one":
        return st._replace(s=st.s.at[jnp.argmax(st.alive)].add(eps))
    noise = jax.random.normal(jax.random.PRNGKey(12345), st.s.shape)
    return st._replace(s=jnp.where(st.alive, jnp.maximum(st.s + eps * noise, 0.0), st.s))


def divergence(a, b, tol=1.0):
    """Share of slots, alive in either state, whose alive flag, track or position (> tol metres) differ. a, b: batched or not."""
    live = a.alive | b.alive
    diff = (a.alive != b.alive) | (a.trk != b.trk) | (jnp.abs(a.s - b.s) > tol)
    return (diff & live).sum(-1) / jnp.maximum(live.sum(-1), 1)


def speed_and_crashes(m_end, m_start):
    """Network mean speed (km/h) and crash count from two metric vectors [NM]."""
    d = dict(zip(X.METRICS, np.asarray(m_end, float) - np.asarray(m_start, float)))
    return 3.6 * d["veh_m"] / max(d["veh_s"], 1.0), float(sum(d[c] for c in X.CRASH_NAMES))


CONFIGS = (("one", 0.0), ("one", 1e-3), ("all", 1e-3), ("all", 1e-2), ("all", 1e-1), ("all", 1.0))


def spread(a, b):
    """Mean |ds| (metres) over slots alive in both and on the same track: a continuous distance between two states."""
    both = a.alive & b.alive & (a.trk == b.trk)
    return (jnp.abs(a.s - b.s) * both).sum(-1) / jnp.maximum(both.sum(-1), 1)


def run(area="oakland", worlds=10, eps=CONFIGS, minutes=8.0, warm_s=90.0, seed=0, gif_frames=True):
    from av_world import easy
    net, geo, info = easy._build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = easy.world(0.0)
    E = len(eps)
    labels = [f'{m}:{e:g}' for m, e in eps]
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), worlds), J, 1024)
    step_all = jax.jit(jax.vmap(lambda s: X.step(s, J, beh, nh)))
    n_blk = int(round(BLOCK_S / X.DT))

    def block(s, _):
        s, _ = jax.lax.scan(lambda c, __: (jax.vmap(lambda x: X.step(x, J, beh, nh))(c), None), s, None, length=n_blk)
        return s, None

    run_blocks = jax.jit(lambda s, n: jax.lax.scan(block, s, None, length=n)[0], static_argnums=1)
    t0 = time.time()
    st = run_blocks(st, int(warm_s / BLOCK_S))
    m_start = np.asarray(st.m)
    twins = jax.tree.map(lambda *xs: jnp.concatenate(xs), *[jax.vmap(lambda x, m=m, e=e: nudge(x, m, e))(st) for m, e in eps])  # eps-major
    ref = lambda t: jax.tree.map(lambda a: a[:worlds], t)

    @jax.jit
    def one_block(s):
        s, _ = jax.lax.scan(lambda c, __: (step_all(c), None), s, None, length=n_blk)
        r = ref(s)
        div = jnp.stack([divergence(jax.tree.map(lambda a, e=e: a[e * worlds:(e + 1) * worlds], s), r) for e in range(E)])  # [E, worlds]
        dist = jnp.stack([spread(jax.tree.map(lambda a, e=e: a[e * worlds:(e + 1) * worlds], s), r) for e in range(E)])
        pos = (s.trk, s.s, s.length, s.alive, s.v)
        return s, (div, dist), jax.tree.map(lambda a: a[jnp.array([0, (E - 1) * worlds])], pos) if (gif_frames and E > 1) else None

    divs, dists, frames = [], [], []
    for _ in range(int(minutes * 60 / BLOCK_S)):
        twins, div, fr = one_block(twins)
        divs.append(np.asarray(div[0]))
        dists.append(np.asarray(div[1]))
        if fr is not None:
            frames.append(jax.tree.map(np.asarray, fr))
    divs = np.stack(divs)  # [T, E, worlds]
    m_end = np.asarray(twins.m).reshape(E, worlds, -1)
    out = {"area": area, "worlds": worlds, "configs": labels, "dist_by_world": np.stack(dists).tolist(), "eps": [list(c) for c in eps], "minutes": minutes, "seconds_compute": round(time.time() - t0, 1),
           "t": [(i + 1) * BLOCK_S for i in range(len(divs))], "div_by_world": divs.tolist()}
    sc = np.array([[speed_and_crashes(m_end[e, w], m_start[w]) for w in range(worlds)] for e in range(E)])  # [E, worlds, 2]
    out["speed"], out["crashes"] = sc[:, :, 0].tolist(), sc[:, :, 1].tolist()
    return out, frames, (net, geo, info)


def analyse(res, thresholds=(0.1, 0.5), n_boot=4000, seed=0):
    """Per configuration: time to reach a divergence share, the speed memory ratio, whether the crash count changed."""
    labels, t = res["configs"], np.array(res["t"])
    div, dist = np.array(res["div_by_world"]), np.array(res["dist_by_world"])  # [T, E, W]
    sp, cr = np.array(res["speed"]), np.array(res["crashes"])
    W = sp.shape[1]
    rng = np.random.default_rng(seed)
    indep = np.abs(sp[0] - np.roll(sp[0], 1))  # two different worlds (different keys), same minutes
    summ = {}
    for e, lab in enumerate(labels):
        row = {"final_div_share": float(div[-1, e].mean()), "final_mean_ds_m": float(dist[-1, e].mean()),
               "max_mean_ds_m": float(dist[:, e].max())}
        for th in thresholds:
            hit = [(t[np.argmax(div[:, e, w] >= th)] if (div[:, e, w] >= th).any() else np.inf) for w in range(W)]
            row[f"time_to_{int(th * 100)}pct_s"] = {"median": float(np.median(hit)), "reached": int(np.isfinite(hit).sum()), "of": W}
        d_sp = np.abs(sp[e] - sp[0])
        boots = [d_sp[i].mean() / max(indep[i].mean(), 1e-9) for i in (rng.integers(0, W, W) for _ in range(n_boot))]
        row["memory_ratio"] = {"value": float(d_sp.mean() / indep.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        row["crash_count_changed"] = {"k": int((cr[e] != cr[0]).sum()), "of": W}
        row["mean_abs_dspeed_kmh"], row["mean_abs_dcrashes"] = float(d_sp.mean()), float(np.abs(cr[e] - cr[0]).mean())
        summ[lab] = row
    summ["independent_mean_abs_dspeed_kmh"] = float(indep.mean())
    summ["mean_crashes_per_world"] = float(cr[0].mean())
    return summ


def make_gif(res, frames, geo_pack, out, every=1):
    """Reference world (grey) and the 1 mm twin (orange) on the same map: where they coincide the dots overlap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from xnet import network as NW
    from xnet.render import Map, Tracks
    net, geo, info = geo_pack
    mp, tr = Map(net, geo, info), Tracks(net, geo)
    fig, ax = plt.subplots(figsize=(7.2, 7.6), facecolor="#0d1117")
    fig.subplots_adjust(0.01, 0.06, 0.99, 0.9)
    ax.set_facecolor("#161b22")
    mp.draw(ax, detail=False, labels=False)
    x0, y0 = NW.to_m(geo.bbox[0], geo.bbox[1])
    x1, y1 = NW.to_m(geo.bbox[2], geo.bbox[3])
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.015, 0.955, "The same morning, every car shifted by ~1 m", color="#e6edf3", fontsize=12, weight="bold")
    hud = fig.text(0.015, 0.918, "", color="#8b949e", fontsize=8.5)
    fig.text(0.015, 0.025, "grey: the original day    orange: the twin day (identical random numbers, every car shifted by ~1 m)\n"
             "orange dots with no grey dot under them are where the two days have already parted",
             color="#8b949e", fontsize=7.5)
    sc_ref = ax.scatter([], [], s=9, c="#8b949e", zorder=5)
    sc_tw = ax.scatter([], [], s=9, c="#ff9e3d", zorder=6, marker="o", linewidths=0.0)
    div = np.array(res["div_by_world"])[:, -1, :].mean(1)

    def pts(f, k):
        trk, s, ln, al, v = (a[k] for a in f)
        x, y, _ = tr.pose(trk[al], s[al], ln[al])
        return np.c_[x, y]

    def draw(k):
        f = frames[k]
        sc_ref.set_offsets(pts(f, 0))
        sc_tw.set_offsets(pts(f, 1))
        hud.set_text(f"t = {90 + (k + 1) * BLOCK_S:5.0f} s since start   share of vehicles that differ: {100 * div[k]:4.0f} %")
        return []

    idx = list(range(0, len(frames), every))
    FuncAnimation(fig, lambda i: draw(idx[i]), frames=len(idx)).save(out, writer=PillowWriter(fps=10), dpi=70, savefig_kwargs={"facecolor": "#0d1117"})
    plt.close(fig)


def plot(res, summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"
    pal = ["#8b949e", "#56b4e9", "#009e73", "#e69f00", "#cc79a7", "#d55e00"]
    nice = {"one:0.001": "1 car, 1 mm", "all:0.001": "every car, 1 mm", "all:0.01": "every car, 1 cm", "all:0.1": "every car, 10 cm", "all:1": "every car, 1 m"}
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=BG, gridspec_kw={"width_ratios": [1.3, 1]})
    t = np.array(res["t"])
    dist = np.array(res["dist_by_world"])
    for e, lab in enumerate(res["configs"]):
        if lab == "one:0":
            continue
        d = dist[:, e, :].mean(1)
        a0.plot(t, np.maximum(d, 1e-6), color=pal[e], lw=2)
        ytxt = max(d[-1], 1e-6) * {"one:0.001": 0.45, "all:0.001": 2.2}.get(lab, 1.0)  # the two sub-mm curves sit on the floor: stagger labels
        a0.text(t[-1] + 5, ytxt, nice.get(lab, lab) + (" (identical)" if d[-1] < 2e-6 else ""), color=pal[e], fontsize=8.5, va="center")
    a0.set_yscale("log")
    a0.set_xlim(0, t[-1] * 1.5)
    a0.set_xlabel("seconds after the nudge", color=INK2)
    a0.set_ylabel("mean distance between the two days (m)", color=INK2, fontsize=9)
    a0.set_title("How a tiny nudge grows (mean over worlds)", color=INK, fontsize=10, loc="left")
    ks = [k for k in res["configs"] if k != "one:0"]
    x = np.arange(len(ks))
    v = [summ[k]["memory_ratio"]["value"] for k in ks]
    lo = [summ[k]["memory_ratio"]["value"] - summ[k]["memory_ratio"]["ci"][0] for k in ks]
    hi = [summ[k]["memory_ratio"]["ci"][1] - summ[k]["memory_ratio"]["value"] for k in ks]
    a1.bar(x, v, color=[pal[res["configs"].index(k)] for k in ks], yerr=[lo, hi], ecolor=INK, capsize=3, width=0.55)
    a1.axhline(1, color=INK, lw=0.8, ls="--")
    a1.text(len(ks) - 0.5, 1.03, "two different days = 1", color=INK, fontsize=8, ha="right")
    a1.set_xticks(x, [nice.get(k, k).replace(", ", ",\n") for k in ks], color=INK2, fontsize=8)
    a1.set_ylabel("|speed: twin - original| / |speed: two worlds|", color=INK2, fontsize=8.5)
    a1.set_title(f"After {res['minutes']:g} min: how much of the original day is left", color=INK, fontsize=10, loc="left")
    for ax in (a0, a1):
        ax.set_facecolor(BG), ax.tick_params(colors=INK2, labelsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#30363d")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--area", default="oakland")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    kw = dict(worlds=4, eps=(('one', 0.0), ('one', 1e-3), ('all', 1e-3)), minutes=3.0) if a.quick else {}
    res, frames, pack = run(a.area, **kw)
    summ = analyse(res)
    res["summary"] = summ
    name = "quick" if a.quick else "butterfly"
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    if not a.quick:
        plot(res, summ, "docs/img/butterfly.png")
        make_gif(res, frames, pack, "docs/img/butterfly.gif", every=2)


if __name__ == "__main__":
    main()
