"""Figures and GIF for the formal barrier result (formal/AvFormal/Barrier.lean) and avlab.barrierfriction.

    python -m avlab.barrierfriction_plot          # docs/img/barrier_phase.png, docs/img/barrier_phase.gif
    python -m avlab.barrierfriction_plot --sim    # also docs/img/barrierfriction.png from results/barrierfriction
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avlab.patchyice_plot import BG, FG, GRID, MUTED, shrink  # noqa: E402

# Okabe-Ito
SOUND, STOP, INFEAS, KNEE, ACCENT = "#009E73", "#E69F00", "#D55E00", "#56B4E9", "#CC79A7"
B, TAU, D0, VMAX = 6.0, 0.5, 2.0, 30.0


def v_feas(mu):
    Bf = 9 * mu
    return np.where(Bf >= B, np.inf, Bf * B * TAU / np.maximum(B - Bf, 1e-9))


def v_stop(mu):
    """Speed above which a state with h = 0 (obstacle stopped) ends in contact when braking at 9 mu: k v^2 = tau v + d0."""
    Bf = 9 * mu
    k = 1 / (2 * Bf) - 1 / (2 * B)
    return np.where(k <= 0, np.inf, (TAU + np.sqrt(TAU ** 2 + 4 * np.maximum(k, 1e-12) * D0)) / (2 * np.maximum(k, 1e-12)))


def _phase(ax, mu_now=None, small=False):
    mu = np.linspace(0.2, 1.0, 801)
    vf, vs = np.minimum(v_feas(mu), VMAX), np.minimum(v_stop(mu), VMAX)
    ax.set_facecolor(BG)
    ax.fill_between(mu, 0, VMAX, where=mu >= 2 / 3, color=SOUND, alpha=0.22, lw=0)
    ax.fill_between(mu, vf, VMAX, where=mu < 2 / 3, color=STOP, alpha=0.35, lw=0)
    ax.fill_between(mu, vs, VMAX, where=mu < 2 / 3, color=INFEAS, alpha=0.6, lw=0)
    ax.plot(mu[mu < 2 / 3], vf[mu < 2 / 3], color=STOP, lw=2)
    ax.plot(mu[mu < 2 / 3], vs[mu < 2 / 3], color=INFEAS, lw=2)
    ax.axvline(2 / 3, color=FG, lw=1.2, ls="--")
    ax.axvspan(0.57, 0.72, ymin=0.0, ymax=0.035, color=KNEE, lw=0)
    fs = 7.5 if small else 9
    ax.text(0.835, 22, "proved sound\nμ ≥ 2/3", color=SOUND, ha="center", fontsize=fs + 1, weight="bold")
    ax.text(0.27, 27.5, "h = 0 ends in contact\n(stop_gap_unsound)", color=FG, fontsize=fs, va="top")
    ax.text(0.395, 11.8, "needs > 9μ braking\n(bound_infeasible)", color=STOP, fontsize=fs, va="top")
    ax.text(0.672, 13.0, "μ* = 2/3 = b / 9", color=FG, fontsize=fs, rotation=90, va="center")
    ax.text(0.575, 1.6, "measured friction knee 0.57–0.72", color=KNEE, fontsize=fs - 0.5)
    ax.set_xlim(0.2, 1.0); ax.set_ylim(0, VMAX)
    ax.set_xlabel("road friction μ (braking cap 9μ m/s²)", color=FG, fontsize=fs + 0.5)
    ax.set_ylabel("AV speed v (m/s)", color=FG, fontsize=fs + 0.5)
    ax.tick_params(colors=MUTED, labelsize=fs)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.axhline(13.4, color=MUTED, lw=0.8, ls=":")
    ax.text(0.995, 13.9, "30 mph", color=MUTED, fontsize=fs - 1, ha="right")
    if mu_now is not None:
        ax.axvline(mu_now, color=ACCENT, lw=2.4)


def phase_png(path="docs/img/barrier_phase.png"):
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=BG)
    fig.subplots_adjust(0.09, 0.14, 0.98, 0.83)
    _phase(ax)
    fig.text(0.01, 0.95, "When is the AV's braking barrier sound? A theorem in Lean 4", color=FG, fontsize=12.5, weight="bold")
    fig.text(0.01, 0.905, "xnet.sim.cbf_bound assumes braking b = 6 m/s²; the road gives 9μ. Machine-checked: the filter can be obeyed "
             "on every state with h ≥ 0 iff μ ≥ 2/3.", color=MUTED, fontsize=8)
    fig.text(0.01, 0.868, "Orange edge: v_feas = 9μ·b·τ / (b − 9μ).  Red edge: k·v² = τ·v + d₀ with k = 1/(18μ) − 1/12.  "
             "Both apply on the barrier's edge, obstacle stopped.", color=MUTED, fontsize=8)
    fig.text(0.01, 0.015, "formal/AvFormal/Barrier.lean · τ = 0.5 s, d₀ = 2 m · knee band (best estimate 0.64–0.65): earlier daily "
             "runs, five random streams, humans-only cities", color=MUTED, fontsize=7)
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    print("wrote", path)


def phase_gif(path="docs/img/barrier_phase.gif", fps=10, v_demo=12.0):
    """Friction sweeps from 1.0 to 0.3. Left: the phase diagram. Right: an AV on the barrier's edge (h = 0) at 12 m/s
    behind a stopped car: where the barrier promises it stops (b = 6) and where the road lets it stop (9 mu)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle
    mus = np.concatenate([np.linspace(1.0, 0.3, 50), np.full(10, 0.3), np.linspace(0.3, 1.0, 25)])
    fig = plt.figure(figsize=(9.0, 5.0), facecolor=BG)
    ax = fig.add_axes([0.07, 0.12, 0.47, 0.70]); ax2 = fig.add_axes([0.60, 0.12, 0.38, 0.70])
    gap = D0 + TAU * v_demo + v_demo ** 2 / (2 * B)  # h = 0 with a stopped obstacle

    def draw(k):
        mu = mus[k]
        ax.clear(); ax2.clear()
        _phase(ax, mu_now=mu, small=True)
        ax.plot([mu], [v_demo], "o", color=ACCENT, ms=9, mec=FG, zorder=6)
        ax2.set_facecolor(BG); ax2.axis("off")
        ax2.set_xlim(-3, 34); ax2.set_ylim(-6, 6)
        ax2.add_patch(Rectangle((-3, -1.6), 37, 3.2, color="#21262d", lw=0))
        ax2.add_patch(Rectangle((gap, -0.9), 4.5, 1.8, color=MUTED, lw=0))  # stopped car
        ax2.text(gap + 2.25, 1.5, "stopped car", color=MUTED, fontsize=8, ha="center")
        stop_b = TAU * v_demo + v_demo ** 2 / (2 * B)
        stop_r = TAU * v_demo + v_demo ** 2 / (2 * 9 * mu)
        ax2.plot([0, stop_b], [-2.6, -2.6], color=SOUND, lw=4)
        ax2.text(0, -3.5, f"barrier promises: stops in {stop_b:.0f} m", color=SOUND, fontsize=8.5, va="top")
        hit = stop_r > gap
        col = INFEAS if hit else FG
        ax2.plot([0, min(stop_r, 33.5)], [2.6, 2.6], color=col, lw=4)
        ax2.text(0, 3.4, f"road allows: stops in {stop_r:.0f} m" + ("  → contact" if hit else ""), color=col, fontsize=8.5)
        x_av = min(stop_r, gap) - 4.5
        ax2.add_patch(Rectangle((x_av, -0.9), 4.5, 1.8, color=ACCENT, lw=0))
        if hit:
            ax2.plot([gap], [0], marker="*", ms=22, color=INFEAS, mec=FG)
        ax2.text(16, -5.6, f"μ = {mu:.2f}   AV at {v_demo:.0f} m/s, h = 0", color=FG, fontsize=10, ha="center")
        return []

    fig.text(0.01, 0.95, "The barrier assumes 6 m/s² of braking; ice gives 9μ", color=FG, fontsize=12.5, weight="bold")
    fig.text(0.01, 0.905, "green: proved sound (μ ≥ 2/3)   orange: the filter asks for braking the road can't give   red: h = 0 "
             "ends in contact   pink: current μ", color=MUTED, fontsize=7.5)
    fig.text(0.01, 0.015, "theorems: formal/AvFormal/Barrier.lean (Lean 4 + Mathlib); right panel: braking from the barrier's edge, "
             "after the τ = 0.5 s margin", color=MUTED, fontsize=7)
    FuncAnimation(fig, draw, frames=len(mus)).save(path, writer=PillowWriter(fps=fps), dpi=72, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    shrink(path, fps)
    print("wrote", path)


def sim_png(out, path="docs/img/barrierfriction.png"):
    e1 = out["E1"]
    mus = sorted((float(m) for m in e1), reverse=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 4.6), facecolor=BG, gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(0.08, 0.14, 0.98, 0.8, wspace=0.32)
    for a in (a1, a2):
        a.set_facecolor(BG); a.tick_params(colors=MUTED, labelsize=8)
        for s in a.spines.values():
            s.set_color(GRID)
    ins = [100 * e1[str(m)]["inside_share"] for m in mus]
    ov = [100 * e1[str(m)]["over_vfeas_share"] for m in mus]
    a1.plot(mus, ins, "o-", color=STOP, lw=2)
    pos = [(m, o) for m, o in zip(mus, ov) if o > 0]
    a1.plot([p[0] for p in pos], [p[1] for p in pos], "s-", color=INFEAS, lw=2)
    a1.set_yscale("log"); a1.set_ylim(0.05, 200)
    a1.axvline(2 / 3, color=FG, lw=1, ls="--")
    for m, v in zip(mus, ins):
        a1.text(m, v * 1.35, f"{v:.2f}", color=STOP, fontsize=7.5, ha="center")
    for m, v in pos:
        a1.text(m, v * 1.35, f"{v:.0f}", color=INFEAS, fontsize=7.5, ha="center")
    a1.text(0.98, 0.4, "inside the barrier (h < 0)", color=STOP, fontsize=8)
    a1.text(0.98, 30, "faster than v_feas(μ):\n0 % for μ ≥ 0.6", color=INFEAS, fontsize=8)
    a1.text(0.655, 100, "μ* = 2/3", color=FG, fontsize=8)
    a1.invert_xaxis()
    a1.set_xlabel("road friction μ", color=FG, fontsize=9); a1.set_ylabel("% of moving AV steps", color=FG, fontsize=9)
    a1.set_title("default barrier (b = 6), 12 copies per friction", color=FG, fontsize=9)
    e2 = out["E2"]
    labs = ["b = 6\n(default)", "b = 3.6\n(= 9μ)"]
    for i, (k, c) in enumerate((("default_b6", MUTED), ("aware_b36", SOUND))):
        r = 1000 * e2[k]["av_crash"] / e2[k]["av_hours"]
        from avlab.patchyice_plot import poisson_ci
        lo, hi = poisson_ci(e2[k]["av_crash"], e2[k]["av_hours"])
        a2.bar(i, r, color=c, width=0.6); a2.plot([i, i], [lo, hi], color=FG, lw=1.5)
        a2.text(i, hi, f"{r:.0f}\n({int(e2[k]['av_crash'])} crashes)", color=FG, fontsize=8, ha="center", va="bottom")
    a2.set_xticks([0, 1], labs, color=FG, fontsize=8.5)
    a2.set_ylabel("AV crashes per 1000 AV-hours", color=FG, fontsize=9)
    a2.set_ylim(0, a2.get_ylim()[1] * 1.25)
    h = out["H1"]
    a2.set_title(f"friction 0.4, 40 copies each: RR {h['rr']:.2f}\n(95 % {h['lo']:.2f}–{h['hi']:.2f}, p {h['p']:.2g})", color=FG, fontsize=9)
    fig.text(0.01, 0.94, "The theorem in the simulator: half-AV Oakland, every AV behind the barrier", color=FG, fontsize=12, weight="bold")
    fig.text(0.01, 0.885, "left: how often AVs are inside their barrier, and in the region the theorem marks infeasible;  right: the "
             "friction-aware barrier the proof suggests", color=MUTED, fontsize=8)
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    phase_png()
    if "--no-gif" not in sys.argv:
        phase_gif()
    if "--sim" in sys.argv:
        sim_png(json.load(open("results/barrierfriction/barrierfriction.json")))
