"""Figures for docs/WORLD_MODEL_UPGRADES.md from the result files of the ten-idea round.

    python scripts/make_wm_figures.py          # -> docs/img/wmup/*.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "docs/img/wmup"
BLUE, ORANGE, AQUA, GRAY, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c86", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.7, "axes.axisbelow": True,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
J = lambda p: json.load(open(p))


def boot(d, n=2000, seed=0):
    d = np.asarray(d, float)
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def bars(ax, x, y, lo=None, hi=None, color=BLUE, w=0.7, **kw):
    ax.bar(x, y, w, color=color, **kw)
    if lo is not None:
        ax.errorbar(x, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)], fmt="none", ecolor=INK, elinewidth=1.1, capsize=3)


def fig_crn():
    dec, pr = J("results/crn/decay.json"), J("results/crn/paired.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    for k, c, lab in (("crn=0", ORANGE, "slot-indexed dice (before)"), ("crn=1", BLUE, "identity-keyed dice (CRN)")):
        t = [r["t"] for r in dec[k]]
        ax[0].plot(t, [r["overlap"] for r in dec[k]], color=c, lw=2.2, label=lab)
        ax[0].text(t[-1] + 8, dec[k][-1]["overlap"], f"{dec[k][-1]['overlap']:.2f}", color=INK, va="center", fontsize=9)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_xlim(0, 680)
    ax[0].set_xlabel("simulated seconds")
    ax[0].set_ylabel("drivers distracted at 1/60 s that are also\ndistracted at 1/20 s (same seed)")
    ax[0].set_title("Coupled worlds stay coupled")
    ax[0].legend(loc="lower left")
    names = [("crashes", "crashes"), ("speed_kmh", "mean speed"), ("red_runs", "red-light runs"), ("lane_changes", "lane changes"), ("peds", "ped. crossings")]
    x = np.arange(len(names))
    for j, (cmp, lab) in enumerate((("av_share_25", "25 % AVs vs none"), ("distraction_x2", "distraction x2"))):
        pass
    w = 0.2
    for i, (cmp, crn, c, lab) in enumerate((("av_share_25", 0, "#9db9e6", "AV share, before"), ("av_share_25", 1, BLUE, "AV share, CRN"),
                                             ("distraction_x2", 0, "#f2b199", "distraction, before"), ("distraction_x2", 1, ORANGE, "distraction, CRN"))):
        ax[1].bar(x + (i - 1.5) * w, [pr[f"{cmp}/crn={crn}"][k]["ratio"] for k, _ in names], w * 0.92, color=c, label=lab)
    ax[1].axhline(1, color=INK, lw=1)
    ax[1].set_xticks(x, [l for _, l in names])
    ax[1].set_ylabel("variance ratio  Var(A)+Var(B) over Var(A-B)\n(64 copies x 600 s; 1 = no gain)")
    ax[1].set_title("What pairing buys")
    ax[1].legend(ncol=2, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/crn.png", dpi=150)


def fig_multistep():
    r = J("results/wm_multistep/results.json")["configs"]
    names = list(r)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, key, title, unit in ((ax[0], "all/speed_kmh@3s", "Speed after 3 s", "km/h MAE"), (ax[1], "all/gap_m@3s", "Gap to leader after 3 s", "m MAE")):
        y = [r[n]["eval"][key]["mae"] for n in names]
        se = [r[n]["eval"][key]["se"] for n in names]
        bars(a, range(4), y, np.array(y) - 1.96 * np.array(se), np.array(y) + 1.96 * np.array(se), color=[GRAY, ORANGE, AQUA, BLUE])
        for i, v in enumerate(y):
            a.text(i, v + 1.96 * se[i] + 0.01 * max(y), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        a.set_xticks(range(4), [n.replace("+", "\n+") for n in names])
        a.set_ylabel(unit)
        a.set_title(title)
    fig.suptitle("Held-out city vehicles: the rollout loss fixes speed (3.7x) and makes the gap worse", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/multistep.png", dpi=150)


def fig_conservative():
    r = J("results/conservative/results.json")["methods"]
    names = ["naive", "mopo", "mopo_x4", "wimle", "truncate", "mopo_support", "mopo_support_x4", "truncate_support"]
    lab = {"naive": "naive", "mopo": "MOPO\nensemble", "mopo_x4": "MOPO x4\nensemble", "wimle": "WIMLE\nensemble", "truncate": "truncate\nensemble",
           "mopo_support": "MOPO\nsupport", "mopo_support_x4": "MOPO x4\nsupport", "truncate_support": "truncate\nsupport"}
    col = [GRAY] + [ORANGE] * 4 + [BLUE] * 3
    base = np.array(r["naive"]["real_returns"])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    gap = [r[n]["gap"] for n in names]
    gaps = [boot(np.array(r[n]["imagined_returns"]) - np.array(r[n]["real_returns"])) for n in names]
    bars(ax[0], range(8), [g[0] for g in gaps], [g[1] for g in gaps], [g[2] for g in gaps], color=col)
    ax[0].axhline(0, color=INK, lw=1)
    ax[0].set_xticks(range(8), [lab[n] for n in names], fontsize=8)
    ax[0].set_ylabel("imagined return - real return\n(positive = the model flatters the plan)")
    ax[0].set_title("Exploitation gap")
    d = [boot(np.array(r[n]["real_returns"]) - base) for n in names]
    bars(ax[1], range(8), [x[0] for x in d], [x[1] for x in d], [x[2] for x in d], color=col)
    ax[1].axhline(0, color=INK, lw=1)
    ax[1].set_xticks(range(8), [lab[n] for n in names], fontsize=8)
    ax[1].set_ylabel("real return minus the naive planner's\n(paired over 256 start states, 95 % CI)")
    ax[1].set_title("What the plans really earn")
    fig.tight_layout()
    fig.savefig(f"{OUT}/conservative.png", dpi=150)


def fig_wm2():
    r = J("results/wm2/results.json")
    o, c = r["one_step"], r["closed_loop"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    names = ["zero", "idm_fitted", "persistence", "none", "leader", "all"]
    lab = ["zero\n(const. vel.)", "IDM\nfitted", "last step's\nacceleration", "token model\nagent alone", "token model\n+ leader", "token model\n+ 8 neighbours"]
    y = [o[n]["mae_a"] for n in names]
    bars(ax[0], range(6), y, color=[GRAY, GRAY, GRAY, "#9db9e6", BLUE, "#6c9ee0"])
    for i, v in enumerate(y):
        ax[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax[0].set_xticks(range(6), lab, fontsize=8)
    ax[0].set_ylabel("acceleration MAE, m/s^2 (0.1 s ahead)")
    ax[0].set_title("One step: history is almost everything")
    ks = ["constant_velocity", "idm_fitted", "token_none", "token_leader", "token_all"]
    kl = ["constant\nvelocity", "IDM\nfitted", "token\nalone", "token\n+ leader", "token\n+ 8 nbrs"]
    base = np.array(c["constant_velocity"]["fde_per_agent"])
    d = [boot(np.array(c[k]["fde_per_agent"]) - base) for k in ks]
    bars(ax[1], range(5), [x[0] for x in d], [x[1] for x in d], [x[2] for x in d], color=[GRAY, GRAY, "#9db9e6", BLUE, "#6c9ee0"])
    ax[1].axhline(0, color=INK, lw=1)
    ax[1].set_xticks(range(5), kl, fontsize=8)
    ax[1].set_ylabel("3 s position error minus constant velocity, m\n(420 agents, 80 held-out scenes, 95 % CI)")
    ax[1].set_title("Closed loop: 20 % less error with a leader")
    fig.tight_layout()
    fig.savefig(f"{OUT}/wm2.png", dpi=150)


def fig_extras():
    s, t, h = J("results/extras/sigreg.json"), J("results/extras/terminal_value.json"), J("results/extras/learned_humans.json")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    names = ["no_regulariser", "sigreg"]
    bars(ax[0], [0, 1], [s[n]["z_std"] for n in names], color=[ORANGE, BLUE], w=0.55)
    ax[0].set_yscale("log")
    for i, n in enumerate(names):
        ax[0].text(i, s[n]["z_std"], f" {s[n]['z_std']:.4f}\n probe R2: speed {s[n]['probe_r2_speed']:.2f}, gap {s[n]['probe_r2_gap']:.2f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks([0, 1], ["JEPA, prediction\nloss only", "JEPA + SIGReg"])
    ax[0].set_ylim(1e-5, 30)
    ax[0].set_ylabel("embedding spread (mean SD per dimension)")
    ax[0].set_title("Collapse and its cure")
    m = t["methods"]
    ns = ["h5_no_value", "h5_value", "h10_no_value", "h10_value", "h20_plan_only"]
    ls = ["5\nno V", "5\nwith V", "10\nno V", "10\nwith V", "20\nplan only"]
    base = np.array(m["h20_plan_only"]["real_returns"])
    d = [boot(np.array(m[n]["real_returns"])) for n in ns]
    bars(ax[1], range(5), [x[0] for x in d], [x[1] for x in d], [x[2] for x in d], color=[GRAY, BLUE, GRAY, BLUE, ORANGE])
    ax[1].set_xticks(range(5), ls, fontsize=8)
    ax[1].set_xlabel("planning horizon h (steps) and terminal value")
    ax[1].set_ylabel("real 20-step return (95 % CI)")
    ax[1].set_title("Short plans + a value")
    keys = ["hard_brakes_per_h", "tight_follow_share", "ttc_under_2s_share", "ttc_under_1s_share"]
    kl = ["hard brakes\nper hour", "headway\n< 0.8 s", "TTC\n< 2 s", "TTC\n< 1 s"]
    wd = h["worlds"]
    wn = [("default", GRAY, "IDM default"), ("fitted", ORANGE, "IDM, tail knobs fitted"), ("learned", BLUE, "learned follower")]
    wn = [w for w in wn if w[0] in wd]
    for i, (n, c, lb) in enumerate(wn):
        ax[2].bar(np.arange(4) + (i - (len(wn) - 1) / 2) * 0.26, [np.log10((wd[n][k] + 1e-4) / (h["real"][k] + 1e-4)) for k in keys], 0.24, color=c, label=lb)
    ax[2].axhline(0, color=INK, lw=1)
    ax[2].set_xticks(range(4), kl, fontsize=8)
    ax[2].set_ylabel("log10 (simulated / real Argoverse 2)\n0 = matches real traffic")
    ax[2].set_title("Everyday tail vs real traffic")
    ax[2].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/extras.png", dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_crn, fig_multistep, fig_conservative, fig_wm2, fig_extras):
        f()
        print("ok", f.__name__)
