"""GIF and figure for avlab.sensornoise / avlab.sensorbench.

    python -m avlab.sensornoise_plot          # docs/img/sensornoise_lane.gif: what the AV sees vs the truth, one scripted lane
    python -m avlab.sensornoise_plot --bench  # docs/img/sensornoise_bench.png from results/sensor/sensor.json

The lane: a leader 30 m ahead at 12 m/s brakes at 6 m/s^2 to a stop at t = 4 s. Two AVs run the same IDM policy
(avlab.teb.idm_policy_masked), one with clean sensors, one behind SensorModel "severe"; no envelope. The noisy AV's perceived
leader (from avlab.sensornoise.perturb, the code the benchmark uses) is drawn as a ghost; a missed detection shows as "MISS".
"""
import functools
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avlab.patchyice_plot import BG, FG, GRID, MUTED  # noqa: E402

CLEANC, NOISYC, LEADC, REDC = "#56B4E9", "#E69F00", "#8b949e", "#D55E00"
DT, L = 0.2, 4.5


def _obs(v, v0, gap, vl):
    o = np.zeros(28, np.float32)
    o[0], o[3], o[16] = v / 15, v0 / 15, 1.0
    o[8], o[9], o[10] = 1, np.clip(gap / 50, -0.1, 3.0), vl / 15
    o[11] = np.clip((v - vl) / 10, -2, 2)
    o[25] = np.clip(max(v - vl, 0) ** 2 / (2 * max(gap - 2, 0.5)) / 8, 0, 2)
    o[26] = np.clip(gap / max(v, 1) / 3, 0, 2)
    return o


@functools.lru_cache(None)
def _fns(level):
    import jax
    from avlab import sensornoise as SN
    from avlab import teb as T
    return jax.jit(lambda o, m: T.idm_policy_masked(o, None, m)[0]), jax.jit(lambda o, k: SN.perturb(o, k, SN.LEVELS[level]))


def scripted(seed=0, T_total=40.0, level="severe"):
    """Per-step rows for the clean and the noisy follower. Stops a follower at contact (gap <= 0)."""
    import jax
    import jax.numpy as jnp
    pol, pert = _fns(level)
    n = int(T_total / DT)
    lx, lv, lead = 30.0 + L, 12.0, []
    for k in range(n):
        lv = max(lv + (-6.0 if k * DT >= 4 else 0.0) * DT, 0.0); lx += lv * DT
        lead.append((lx, lv))
    out = {}
    for name in ("clean", "noisy"):
        x, v, rows, hit = 0.0, 12.0, [], None
        for k in range(n):
            lx, lv = lead[k]
            gap = lx - x - L
            o = jnp.asarray(_obs(v, 13.0, gap, lv))[None]
            seen = (gap, lv, True)
            if name == "noisy":
                o = pert(o, jax.random.PRNGKey(seed * 10000 + k))
                seen = (50 * float(o[0, 9]), 15 * float(o[0, 10]), bool(o[0, 8] > 0.5))
            u = float(pol(o, jnp.ones(1, bool))[0, 0])
            a = 3 * u if u >= 0 else 8 * u
            if hit is None and gap <= 0:
                hit = k * DT
            if hit is not None:
                a, v = 0.0, lv
            rows.append(dict(t=k * DT, x=x, v=v, a=a, lead=(lx, lv), seen=seen, hit=hit))
            if hit is None:
                v = max(v + a * DT, 0.0); x += v * DT
        out[name] = rows
    return out


def creep(seeds=40, level="severe"):
    """Mean standing gap (m) of the noisy follower at t = 8, 16, 24, 32, 40 s over `seeds` noise seeds, the clean gap, and the
    share of seeds with contact by t = 40 s. Shows the ratchet: at v = 0 a too-short perceived gap cannot push the car back
    (speed is clipped at 0), a too-long one or a missed leader moves it forward."""
    g = np.array([[r["lead"][0] - L - r["x"] for r in scripted(s, level=level)["noisy"]] for s in range(seeds)])
    c = scripted(0)["clean"]
    hits = [scripted(s, level=level)["noisy"][-1]["hit"] is not None for s in range(seeds)]
    return {"t_s": [8, 16, 24, 32, 40], "noisy_gap_m": g[:, [40, 80, 120, 160, 199]].mean(0).round(3).tolist(),
            "clean_gap_m": round(c[-1]["lead"][0] - L - c[-1]["x"], 3), "contact_share": float(np.mean(hits)), "seeds": seeds, "level": level}


def lane_gif(path="docs/img/sensornoise_lane.gif", seed=None, fps=10, every=2):
    from matplotlib.animation import FuncAnimation, PillowWriter
    if seed is None:  # the first seed (0, 1, ...) whose noisy run ends in contact, if any within 50
        seed = next((s for s in range(50) if scripted(s)["noisy"][-1]["hit"] is not None), 0)
    d = scripted(seed)
    n = len(d["clean"])
    frames = list(range(0, n, every)) + [n - 1] * 10
    fig = plt.figure(figsize=(7.2, 4.4), facecolor=BG)
    ax = fig.add_axes([0.03, 0.42, 0.94, 0.38]); ax2 = fig.add_axes([0.1, 0.1, 0.86, 0.26])
    for a in (ax, ax2):
        a.set_facecolor(BG)
        for s in a.spines.values():
            s.set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.03, 0.93, "AV sensor noise: what the AV sees vs the truth", color=FG, fontsize=12, weight="bold")
    fig.text(0.03, 0.835, "Leader brakes to a stop at t = 4 s. Same IDM policy, no envelope. Dashed ghost = perceived leader\n"
             "(severe: range \u00b120 %, speed \u00b11 m/s, 15 % of steps missed). Played at 2x.", color=MUTED, fontsize=7.5)
    t_all = [r["t"] for r in d["clean"]]

    def draw(k):
        ax.cla(); ax.set_facecolor(BG); ax.set_yticks([]); ax.set_ylim(-1.0, 2.5)
        x0 = d["clean"][k]["x"] - 10
        ax.set_xlim(x0, x0 + 60)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("position (m)", color=MUTED, fontsize=8, labelpad=1)
        for row, (name, c, lab) in enumerate((("noisy", NOISYC, "noisy sensors"), ("clean", CLEANC, "clean sensors"))):
            r = d[name][k]
            y = row * 1.5
            ax.plot([x0, x0 + 60], [y - 0.45, y - 0.45], color=GRID, lw=0.8)
            lx = r["lead"][0]
            ax.add_patch(plt.Rectangle((lx - L, y - 0.3), L, 0.6, color=LEADC))
            ax.add_patch(plt.Rectangle((r["x"] - L, y - 0.3), L, 0.6, color=c))
            ax.text(x0 + 1, y + 0.42, lab, color=c, fontsize=8, weight="bold")
            if name == "noisy":
                g, _, det = r["seen"]
                if det:
                    ax.add_patch(plt.Rectangle((r["x"] + g, y - 0.3), L, 0.6, fill=False, ec=NOISYC, ls="--", lw=1.2))
                elif r["hit"] is None:
                    ax.text(lx + 1, y + 0.42, "leader MISSED", color=REDC, fontsize=8, weight="bold")
            if r["hit"] is not None and r["t"] >= r["hit"]:
                ax.text(lx - L, y - 0.75, f"CONTACT at t = {r['hit']:.1f} s", color=REDC, fontsize=9, weight="bold", ha="center")
            ax.text(x0 + 59, y + 0.42, (f"t = {r['t']:4.1f} s   " if row == 1 else "") + f"gap {lx - L - r['x']:5.1f} m  v {r['v']:4.1f} m/s", color=FG, fontsize=7.5, ha="right")
        ax2.cla(); ax2.set_facecolor(BG)
        for name, c in (("clean", CLEANC), ("noisy", NOISYC)):
            ax2.plot(t_all[:k + 1], [d[name][j]["a"] for j in range(k + 1)], color=c, lw=1.4)
        ax2.set_xlim(0, t_all[-1]); ax2.set_ylim(-8.5, 3.5)
        ax2.set_ylabel("AV accel (m/s²)", color=MUTED, fontsize=8); ax2.set_xlabel("time (s)", color=MUTED, fontsize=8)
        ax2.tick_params(colors=MUTED, labelsize=8)
        ax2.axhline(0, color=GRID, lw=0.8)
        for s in list(ax.spines.values()) + list(ax2.spines.values()):
            s.set_color(GRID)
    FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames)).save(path, writer=PillowWriter(fps=fps), dpi=72, savefig_kwargs={"facecolor": BG})
    return seed


def bench_fig(src="results/sensor/sensor.json", path="docs/img/sensornoise_bench.png"):
    r = json.load(open(src))["cells"]
    order = [("off:clean", "clean"), ("off:mild", "mild"), ("off:moderate", "moderate"), ("off:severe", "severe"),
             ("off:range", "range only"), ("off:speed", "speed only"), ("off:miss", "missed only"),
             ("idm:clean", "clean"), ("idm:severe", "severe")]
    order = [(c, lab) for c, lab in order if c in r]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), facecolor=BG)
    fig.subplots_adjust(0.1, 0.24, 0.98, 0.84)
    ax.set_facecolor(BG)
    for i, (c, lab) in enumerate(order):
        col = CLEANC if c.startswith("idm") else NOISYC
        k, lo, hi = r[c]["rate_per_h"], *r[c]["ci"]
        ax.plot([i, i], [lo, hi], color=col, lw=2)
        ax.plot(i, k, "o", color=col, ms=7)
        ax.text(i, hi * 1.25, f"{r[c]['av_crash']:.0f}", color=FG, fontsize=8, ha="center", va="bottom")
    ax.set_xticks(range(len(order)), [lab for _, lab in order], rotation=30, ha="right", color=FG, fontsize=8.5)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 30)
    ax.set_ylabel("AV crashes per AV-hour (95 %, log)", color=MUTED, fontsize=9)
    n_off = sum(c.startswith("off") for c, _ in order)
    ymax = 25
    ax.text(1.5, ymax * 0.97, "envelope off", color=NOISYC, ha="center", fontsize=9, weight="bold", va="top")
    ax.text(n_off + 0.5, ymax * 0.97, "envelope on", color=CLEANC, ha="center", fontsize=9, weight="bold", va="top")
    ax.axvline(n_off - 0.5, color=GRID, lw=0.8)
    fig.text(0.02, 0.93, "AV sensor noise vs crash rate (Oakland, half AVs, 16 copies x 600 s per cell)", color=FG, fontsize=11, weight="bold")
    fig.text(0.02, 0.875, "Numbers above bars: AV crash count. Same 16 random streams in every cell.", color=MUTED, fontsize=8)
    fig.savefig(path, dpi=110, facecolor=BG)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_fig()
    elif "--creep" in sys.argv:
        out = {lv: creep(level=lv) for lv in ("mild", "moderate", "severe")}
        json.dump(out, open("results/sensor/creep.json", "w"), indent=1)
        print(json.dumps(out))
    else:
        print("seed", lane_gif())
