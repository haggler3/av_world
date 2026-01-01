"""Figures and GIFs for docs/SIM2REAL.md (docs/img/realism/)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import measure as M
from . import stats as ST

IMG = "docs/img/realism"
C = {"real": "#222222", "default": "#d1495b", "tail_fit": "#edae49", "idm_plus": "#00798c", "patched": "#30638e"}
LBL = {"real": "real AV2", "default": "sim: stored drivers", "tail_fit": "sim: earlier tail fit", "idm_plus": "sim: IDM+ only", "patched": "sim: patched"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def fig_sensor(real, nm, proc, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    k = np.arange(1, 16)
    ax = axs[0]
    for who, col, lab in (("other", C["real"], "real: tracked road users"), ("ego", "#888888", "real: the AV itself (localisation)")):
        ax.plot(k * 0.2, nm[who]["D"], "o", color=col, ms=4, label=lab)
        p = [nm[who][q] for q in ("sm", "rho", "sa", "tau")]
        ax.plot(k * 0.2, M.d_model(p, k), "-", color=col, lw=1)
        ax.plot(k * 0.2, M.d_model([0, 0, p[2], p[3]], k), ":", color=col, lw=1)
    ser = M.tracks(proc["default"]["raw"])
    ax.plot(k * 0.2, M.structure(ser), "s", color=C["default"], ms=4, label="sim (perfect speeds)")
    ser = M.tracks(proc["default"]["sensor"])
    ax.plot(k * 0.2, M.structure([v for v in ser if not M.spiky(v[None])[0]]), "^", color=C["default"], mfc="none", ms=5, label="sim through the real tracker")
    ax.set(xlabel="lag h [s]", ylabel="D(h) = E[(v(t+h) - v(t))^2]  [m²/s²]", title="Velocity structure function\nline: driver + sensor fit, dotted: driver only")
    ax.legend(fontsize=7.5)
    _style(ax)
    ax = axs[1]
    bins = np.linspace(-10, 10, 81)
    a_raw = np.concatenate([ST.measure(s)["a"] for s in real["raw"][:200]])
    a_cl = np.concatenate([ST.measure(s)["a"] for s in real["clean"][:200]])
    a_sim = np.concatenate([ST.measure(s)["a"] for s in proc["default"]["raw"][:300]])
    for a, col, lab, ls in ((a_raw, C["real"], "real, as logged", "-"), (a_cl, C["real"], "real, cleaned (despike + RTS)", "--"),
                            (a_sim, C["default"], "sim, stored drivers", "-")):
        ax.hist(np.clip(a[np.isfinite(a)], -10, 10), bins, density=True, histtype="step", color=col, ls=ls, lw=1.4, label=lab)
    ax.set_yscale("log")
    ax.set(xlabel="acceleration over 0.2 s [m/s²]", ylabel="density", title=f"{100 * nm['other']['spiky_track_share']:.0f} % of real tracks have jumps no car can make")
    ax.legend(fontsize=7.5)
    _style(ax)
    ax = axs[2]
    sb = json.load(open(out))["scoreboard"]["default"]
    names = ["hard_brakes_per_h", "tight_share", "ttc2_share", "ttc1_share"]
    x = np.arange(4)
    for j, (mode, lab, col) in enumerate((("raw", "as logged vs perfect sim", "#bbbbbb"), ("sensor", "sim through the real tracker", "#888888"),
                                          ("clean", "both cleaned", C["default"]))):
        r = [sb[mode]["tails"][n]["ratio"] for n in names]
        lo = [sb[mode]["tails"][n]["ratio_ci"][0] for n in names]
        hi = [sb[mode]["tails"][n]["ratio_ci"][1] for n in names]
        ax.bar(x + (j - 1) * 0.27, r, 0.27, color=col, label=lab, yerr=[np.array(r) - lo, np.array(hi) - r], capsize=2)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x, ["hard brakes", "headway < 0.8 s", "TTC < 2 s", "TTC < 1 s"])
    ax.set(ylabel="sim / real (95 % CI)", title="Stored drivers: what the sensor explains")
    ax.legend(fontsize=7.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{IMG}/sensor.png", dpi=130)
    plt.close(fig)


def fig_dists(real, proc, fc, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    worlds = ("default", "tail_fit", "idm_plus", "patched")
    hw = {"real": np.concatenate([ST.measure(s)["hw"] for s in real["clean"]])}
    for w in worlds:
        hw[w] = np.concatenate([ST.measure(s)["hw"] for s in proc[w]["clean"]])
    ax = axs[0]
    bins = np.linspace(0, 5, 51)
    for n, h in hw.items():
        ax.hist(h[np.isfinite(h)], bins, density=True, histtype="step", lw=2.0 if n in ("real", "patched") else 1.1, color=C[n], label=LBL[n])
    ax.axvline(0.8, color="k", ls=":", lw=0.8)
    ax.set(xlabel="time headway [s] (moving, with a leader)", ylabel="density", title="Headway: the simulator's followers hang back")
    ax.legend(fontsize=7.5)
    _style(ax)
    ax = axs[1]
    for n in ["real"] + list(worlds):
        p = np.sort(np.asarray(fc[n]["p10_hw"]))
        ax.plot(p, np.linspace(0, 1, len(p)), color=C[n], lw=2.0 if n in ("real", "patched") else 1.1, label=LBL[n])
    ax.set(xlim=(0, 5), xlabel="a driver's closest headway (10th percentile of an episode) [s]", ylabel="share of drivers",
           title="Per-driver closest following (the bank's target)")
    ax.legend(fontsize=7.5)
    _style(ax)
    ax = axs[2]
    bins = np.linspace(0, 2.0, 41)
    ti = {"real": np.concatenate([ST.measure(s)["ttc_inv"] for s in real["clean"]])}
    for w in worlds:
        ti[w] = np.concatenate([ST.measure(s)["ttc_inv"] for s in proc[w]["clean"]])
    for n, t in ti.items():
        t = t[np.isfinite(t) & (t > 0.05)]
        ax.hist(np.clip(t, 0, 2), bins, density=False, weights=np.full(len(t), 1.0 / max(np.isfinite(ti[n]).sum(), 1)), histtype="step",
                lw=2.0 if n in ("real", "patched") else 1.1, color=C[n], label=LBL[n])
    ax.set_yscale("log")
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.axvline(1.0, color="k", ls=":", lw=0.8)
    ax.set(xlabel="1 / time to collision [1/s] (closing on a leader)", ylabel="share of follower frames", title="Closing fast: TTC < 2 s and < 1 s")
    ax.legend(fontsize=7.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{IMG}/distributions.png", dpi=130)
    plt.close(fig)


def fig_waterfall(res):
    dec = res["decomposition"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    g = [d["gap"] for d in dec]
    cols = ["#bbbbbb", "#999999", C["default"], "#b5838d", C["idm_plus"], C["patched"]]
    ax.barh(range(len(g))[::-1], g, color=cols)
    for i, v in enumerate(g):
        ax.text(v + 0.02, len(g) - 1 - i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(g))[::-1], [d["step"] for d in dec], fontsize=9)
    ax.set(xlabel="gap (0 = indistinguishable by these statistics)", title="The gap, step by step")
    _style(ax)
    ax = axs[1]
    names = [("hard_brakes_per_h", "hard brakes"), ("tight_share", "headway < 0.8 s"), ("ttc2_share", "TTC < 2 s"), ("ttc1_share", "TTC < 1 s")]
    for (k, lab), m in zip(names, ("o", "s", "^", "D")):
        ax.plot(range(len(dec)), [d[k] for d in dec], marker=m, label=lab)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(dec)), [str(i + 1) for i in range(len(dec))])
    ax.set(xlabel="step (left panel, top to bottom)", ylabel="sim / real", title="Tail rates along the way")
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{IMG}/waterfall.png", dpi=130)
    plt.close(fig)


def fig_where(res):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
    for ax, w in zip(axs, ("default", "patched")):
        rows = [r for r in res["scoreboard"][w]["clean"]["where"] if r["leader"]]
        labels = [f"{r['speed']} m/s, {'turning' if r['turning'] else 'straight'}" for r in rows]
        tr = [r["sim_tight"] / max(r["real_tight"], 1e-4) for r in rows]
        wa = [r["w1_a"] for r in rows]
        y = np.arange(len(rows))
        ax.barh(y - 0.2, wa, 0.4, color="#888888", label="acceleration W1 (real IQR units)")
        ax2 = ax.twiny()
        ax2.barh(y + 0.2, tr, 0.4, color=C[w], label="headway < 0.8 s: sim / real")
        ax2.axvline(1, color=C[w], lw=0.8, ls=":")
        ax2.set_xscale("log")
        ax2.set_xlim(0.01, 10)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set(title=f"{LBL[w]}: where (vehicles with a leader)\n", xlabel="acceleration W1")
        ax2.set_xlabel("tight following sim / real", color=C[w])
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower right")
        _style(ax)
    fig.tight_layout()
    fig.savefig(f"{IMG}/where.png", dpi=130)
    plt.close(fig)


def fig_discriminator(res):
    d = res["discriminator"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [0.8, 1.2]})
    ax = axs[0]
    ws = list(d)
    a = [d[w]["auroc"] for w in ws]
    ci = np.array([d[w]["auroc_ci"] for w in ws])
    ax.bar(range(len(ws)), a, color=[C[w] for w in ws], yerr=[np.array(a) - ci[:, 0], ci[:, 1] - np.array(a)], capsize=3)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(range(len(ws)), [LBL[w].replace("sim: ", "") for w in ws], fontsize=8)
    ax.set(ylabel="AUROC real vs sim (held-out scenes)", title="Can a network tell them apart?")
    _style(ax)
    ax = axs[1]
    names = list(d["default"]["importance"])[:8]
    y = np.arange(len(names))
    for j, w in enumerate(("default", "patched")):
        ax.barh(y + (j - 0.5) * 0.38, [d[w]["importance"][n] for n in names], 0.38, color=C[w], label=LBL[w])
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="AUROC lost when the feature is shuffled", title="What gives the simulator away")
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{IMG}/discriminator.png", dpi=130)
    plt.close(fig)


def _pick(scenes, want_tight=True):
    best, bs = None, -1
    for s in scenes:
        m = ST.measure(s)
        hw = m["hw"][np.isfinite(m["hw"])]
        if len(hw) < 60:
            continue
        score = (hw < 1.0).sum() if want_tight else len(hw)
        if score > bs:
            best, bs = s, score
    return best


def gif_following(real, proc, path, frames=50):
    """Three scenes side by side, each centred on its ego, cars coloured by their current headway (red under 0.8 s)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from autotag.rules import leaders, relative
    picks = [("real AV2 (cleaned)", _pick(real["clean"])), ("sim: stored drivers", _pick(proc["default"]["clean"])),
             ("sim: patched", _pick(proc["patched"]["clean"]))]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
    cmap = plt.get_cmap("RdYlGn")
    data = []
    for (title, s), ax in zip(picks, axs):
        lead, gap = leaders(s, relative(s))
        hw = np.where(lead >= 0, gap / np.maximum(s["v"], 0.5), np.nan)
        data.append((s, hw))
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    T = min(frames, *(d[0]["x"].shape[1] for d in data))

    def draw(t):
        for ax, (title, _), (s, hw) in zip(axs, picks, data):
            ax.cla()
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
            e = s["ego"]
            cx, cy = np.nanmean(s["x"][:, t]) if not np.isfinite(s["x"][e, t]) else s["x"][e, t], s["y"][e, t] if np.isfinite(s["y"][e, t]) else np.nanmean(s["y"][:, t])
            ax.set_xlim(cx - 60, cx + 60), ax.set_ylim(cy - 40, cy + 40)
            for i in range(len(s["x"])):
                if not np.isfinite(s["x"][i, t]):
                    continue
                h = hw[i, t]
                col = "#bbbbbb" if not np.isfinite(h) else cmap(np.clip((h - 0.4) / 2.6, 0, 1))
                L = s["length"][i]
                c, sn = np.cos(s["yaw"][i, t]), np.sin(s["yaw"][i, t])
                box = np.array([[-L / 2, -1], [L / 2, -1], [L / 2, 1], [-L / 2, 1], [-L / 2, -1]])
                xy = box @ np.array([[c, sn], [-sn, c]]) + [s["x"][i, t], s["y"][i, t]]
                ax.fill(xy[:, 0], xy[:, 1], color=col, ec="k" if i == e else "none", lw=1.2)
            tight = np.nansum((hw[:, t] < 0.8) & (s["v"][:, t] > 3))
            ax.text(0.02, 0.02, f"t = {t * 0.2:4.1f} s   headway < 0.8 s: {int(tight)}", transform=ax.transAxes, fontsize=8)
        return []
    anim = FuncAnimation(fig, draw, frames=T, interval=200)
    fig.suptitle("Following, coloured by time headway (red 0.4 s, green 3 s, grey: no leader; black outline: the AV)", fontsize=10)
    anim.save(path, writer=PillowWriter(fps=5))
    plt.close(fig)


def main(quick=False):
    from autotag import dag
    from . import pipeline as PL
    PL.QUICK = quick
    cache = f"{PL.OUT}/cache_quick" if quick else PL.CACHE
    os.makedirs(IMG, exist_ok=True)
    out = f"{PL.OUT}/quick/scoreboard.json" if quick else f"{PL.OUT}/scoreboard.json"
    res = json.load(open(out))
    real = {"raw": dag.run("real_logs", log=lambda *_: None, cache=cache), "clean": dag.run("real_clean", log=lambda *_: None, cache=cache)}
    nm = dag.run("tracker_noise", log=lambda *_: None, cache=cache)
    proc = dag.run("processed", log=lambda *_: None, cache=cache)
    fc = dag.run("following_check", log=lambda *_: None, cache=cache)
    fig_sensor(real, nm, proc, out)
    fig_dists(real, proc, fc, out)
    fig_waterfall(res)
    fig_where(res)
    fig_discriminator(res)
    gif_following(real, proc, f"{IMG}/following.gif")
    print("figures ->", IMG)


if __name__ == "__main__":
    import sys
    main("--quick" in sys.argv)
