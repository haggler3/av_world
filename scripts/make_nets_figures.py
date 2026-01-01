"""Figures for docs/NEURAL_NETS.md.     python scripts/make_nets_figures.py [n1 n2 n4 n5 n6 n7]   -> docs/img/nets/*.png"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.getcwd())
OUT = "docs/img/nets"
BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8d8c86", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.7, "axes.axisbelow": True,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
J = lambda p: json.load(open(p))
R = "results/nets"


def label_bars(ax, xs, ys, fmt="{:.3f}", dy=0.0):
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, fmt.format(y), ha="center", va="bottom", fontsize=8.5)


def n2():
    r = J(f"{R}/counterfactual.json")
    b = r["budget"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.array(b["budgets"]) * 100
    for k, c, lab, ls in (("interventional", BLUE, "interventional critic (trained on forks)", "-"), ("observational", ORANGE, "observational critic (trained on the log)", "-"),
                          ("random_moments_brake", GRAY, "random moments, brake harder", "--"), ("oracle", AQUA, "oracle (true best action)", ":")):
        ax[0].plot(x, np.array(b[k]) * 100, color=c, lw=2.2, ls=ls, marker="o", ms=4, label=lab)
    ax[0].axhline(b["as_is"] * 100, color=INK, lw=1)
    ax[0].text(x[0], b["as_is"] * 100 * 1.06, "the policy as is", fontsize=8.5, color=MUTED)
    ax[0].set_xscale("log"), ax[0].set_yscale("log")
    ax[0].set_xlabel("share of moments where the critic may change the action (%)")
    ax[0].set_ylabel("bad outcomes per AV-moment (%), true, from the forks")
    ax[0].set_title("Choosing when and how to intervene")
    ax[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    names = ["as is", "brake\nharder", "accel-\nerate", "bigger\ngaps", "smaller\ngaps"]
    cal = r["effect_calibration"]
    w = 0.27
    xs = np.arange(1, 5)
    ax[1].bar(xs - w, np.array(cal["true_mean_effect"][1:]) * 100, w, color=INK, label="true (paired forks)")
    ax[1].bar(xs, np.array(cal["interventional"]["mean_pred_effect"][1:]) * 100, w, color=BLUE, label="interventional critic")
    ax[1].bar(xs + w, np.array(cal["observational"]["mean_pred_effect"][1:]) * 100, w, color=ORANGE, label="observational critic")
    ax[1].axhline(0, color=INK, lw=0.8)
    ax[1].set_xticks(xs, names[1:])
    ax[1].set_ylabel("change in bad-outcome probability\nvs acting as is (percentage points)")
    ax[1].set_title("What each action does")
    ax[1].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/n2_critic.png", dpi=150)


def n7():
    r = J(f"{R}/response.json")
    z = np.load(f"{R}/response_preds.npz")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    names = ["as is\n(others vary)", "brake\nharder", "accel-\nerate", "bigger\ngaps", "smaller\ngaps"]
    w = 0.27
    for j, (m, c) in enumerate((("interventional", BLUE), ("observational", ORANGE), ("zero", GRAY))):
        vals = [r["per_action"][str(k)][m]["mae"][0] for k in range(5)]
        ax[0].bar(np.arange(5) + (j - 1) * w, vals, w, color=c, label={"zero": "the follower ignores us"}.get(m, m + " model"))
    ax[0].set_xticks(range(5), names, fontsize=8.5)
    ax[0].set_ylabel("mean abs. error of the effect on the follower's\nacceleration over 4 s (m/s^2)")
    ax[0].set_title("Predicting the human behind")
    ax[0].legend(fontsize=8.5)
    tmin = (z["true"]).min(1)
    pmin = z["p_int"].min(1)
    omin = z["p_obs"].min(1)
    idx = np.random.default_rng(0).choice(len(tmin), min(3000, len(tmin)), replace=False)
    ax[1].scatter(tmin[idx], omin[idx], s=6, color=ORANGE, alpha=0.35, label="observational", linewidths=0)
    ax[1].scatter(tmin[idx], pmin[idx], s=6, color=BLUE, alpha=0.45, label="interventional", linewidths=0)
    lim = [min(tmin[idx].min(), -4), 0.5]
    ax[1].plot(lim, lim, color=INK, lw=0.8)
    ax[1].set_xlim(lim), ax[1].set_ylim(lim)
    ax[1].set_xlabel("true effect of braking harder on the follower (m/s^2, most negative step)")
    ax[1].set_ylabel("predicted")
    ax[1].set_title("When the AV brakes harder")
    ax[1].legend(fontsize=8.5, markerscale=3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/n7_response.png", dpi=150)


def n1():
    r = J(f"{R}/mindreader.json")["targets"]
    z = np.load(f"{R}/mindreader_preds.npz")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [0.8, 1.2, 1]})
    bins = ["runner", "aggr"]
    w = 0.36
    ax[0].bar(np.arange(2) - w / 2, [r[b]["net_auroc"] for b in bins], w, color=BLUE, label="network")
    ax[0].bar(np.arange(2) + w / 2, [r[b]["linear_auroc"] for b in bins], w, color=GRAY, label="linear on 14 features")
    ax[0].axhline(0.5, color=INK, lw=1)
    ax[0].set_ylim(0.4, 1.0), ax[0].set_xticks(range(2), ["red-light\nrunner", "aggressive"])
    ax[0].set_ylabel("AUROC (0.5 = chance)"), ax[0].set_title("Hidden intentions"), ax[0].legend(fontsize=8)
    regs = ["T", "amax", "b", "v0f", "s0"]
    lab = ["headway T", "accel. a_max", "braking b", "desired speed", "standstill gap"]
    ax[1].bar(np.arange(5) - w / 2, [r[k]["net_r2"] for k in regs], w, color=BLUE, label="network")
    ax[1].bar(np.arange(5) + w / 2, [max(r[k]["linear_r2"], 0) for k in regs], w, color=GRAY, label="linear")
    label_bars(ax[1], np.arange(5) - w / 2, [r[k]["net_r2"] for k in regs], "{:.2f}")
    ax[1].set_xticks(range(5), lab, fontsize=8.5), ax[1].set_ylabel("held-out R^2"), ax[1].set_title("Driving style"), ax[1].legend(fontsize=8)
    t, p = z["yr"][:, 0], z["net"][:, 3]
    idx = np.random.default_rng(0).choice(len(t), 6000, replace=False)
    ax[2].hexbin(t[idx], p[idx], gridsize=30, cmap="Blues", mincnt=1)
    ax[2].plot([-2, 2], [-2, 2], color=INK, lw=0.8)
    ax[2].set_xlabel("true headway T (standardised)"), ax[2].set_ylabel("predicted"), ax[2].set_title("Reading the headway")
    fig.tight_layout()
    fig.savefig(f"{OUT}/n1_mindreader.png", dpi=150)


def n6():
    r = J(f"{R}/occlusion.json")
    z = np.load(f"{R}/occlusion_preds.npz")
    fig = plt.figure(figsize=(13, 4.1))
    gs = fig.add_gridspec(2, 6, width_ratios=[1.4, 1.4, 1, 1, 1, 0.08])
    a0 = fig.add_subplot(gs[:, 0])
    ms = ["network", "shadow geometry", "marginal prior"]
    a0.bar(range(3), [r["grid"][m]["auroc"] for m in ms], 0.6, color=[BLUE, GRAY, "#c4c3bd"])
    label_bars(a0, range(3), [r["grid"][m]["auroc"] for m in ms])
    a0.set_ylim(0.8, 1.0), a0.set_xticks(range(3), ["network", "shadow\ngeometry", "marginal\nprior"]), a0.set_ylabel("AUROC, hidden road user per cell")
    a0.set_title("Where are the hidden ones?")
    a1 = fig.add_subplot(gs[:, 1])
    v = [r["ahead"]["network_auroc"], r["ahead"]["shadow_ahead_auroc"]]
    a1.bar(range(2), v, 0.55, color=[BLUE, GRAY])
    label_bars(a1, range(2), v)
    a1.set_ylim(0.5, 1.0), a1.set_xticks(range(2), ["network", "shadow\nahead"]), a1.set_ylabel("AUROC"), a1.set_title("Someone hidden 25 m ahead?")
    for j in range(3):
        i = j * 7
        for row, (arr, tl) in enumerate(((z["grid"][i], "truth"), (z["pg"][i], "predicted"))):
            a = fig.add_subplot(gs[row, 2 + j])
            a.imshow(arr.reshape(8, 8).T, origin="lower", cmap="Blues" if row else "Greys", vmin=0, vmax=1, extent=(-40, 40, -40, 40))
            tok = z["tok"][i]
            m = np.abs(tok).sum(1) > 0
            a.scatter(tok[m, 0] * 40, tok[m, 1] * 40, s=8, c=ORANGE, linewidths=0)
            a.scatter([0], [0], marker="^", s=40, c=INK)
            a.set_xticks([]), a.set_yticks([]), a.grid(False)
            if j == 0:
                a.set_ylabel(tl, fontsize=9)
    fig.text(0.47, 0.01, "examples: black = hidden road users (truth), blue = predicted probability,\norange = what the AV sees, triangle = the AV facing right; 80 x 80 m", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{OUT}/n6_occlusion.png", dpi=150)


def n5():
    r = J(f"{R}/blame.json")["questions"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw={"width_ratios": [0.8, 1.1, 1.1]})
    qs = [("mistake", "a mistake was a but-for cause"), ("careful", "a careful driver in that seat\nwould have avoided it")]
    ax[0].bar(range(2), [r[q]["share_with_cause"] * 100 for q, _ in qs], 0.55, color=[ORANGE, BLUE])
    label_bars(ax[0], range(2), [r[q]["share_with_cause"] * 100 for q, _ in qs], "{:.1f} %")
    ax[0].set_xticks(range(2), [l for _, l in qs], fontsize=8)
    ax[0].set_ylabel("share of 2,948 crashes"), ax[0].set_title("How often can one driver be blamed?")
    for a, (q, title) in zip(ax[1:], (("careful", "Who could have avoided it?"), ("mistake", "Whose mistake caused it?"))):
        ms = list(r[q]["methods"])
        t1 = [r[q]["methods"][m]["top1"] for m in ms]
        au = [r[q]["methods"][m]["auroc"] for m in ms]
        xs = np.arange(len(ms))
        a.bar(xs - 0.2, [t[0] for t in t1], 0.38, color=BLUE, label="top-ranked is a cause")
        a.errorbar(xs - 0.2, [t[0] for t in t1], yerr=[[t[0] - t[1] for t in t1], [t[2] - t[0] for t in t1]], fmt="none", ecolor=INK, capsize=2.5)
        a.bar(xs + 0.2, au, 0.38, color=GRAY, label="AUROC over candidates")
        a.set_xticks(xs, ["network", "faster of\nthe two", "crossed\non red", "privileged:\nhad a mistake"], fontsize=8)
        a.set_ylim(0, 1.28), a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]), a.set_title(title + f"  ({r[q]['test_crashes_with_cause']} test crashes)")
        a.legend(fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/n5_blame.png", dpi=150)


def n4():
    r = J(f"{R}/sbi.json")
    z = np.load(f"{R}/sbi_posterior.npz")
    from nets import sbi as SB
    post = z["post"]
    names = [k[0] for k in SB.KNOBS]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    parts = ax[0].violinplot([post[:, j] for j in range(len(names))], positions=range(len(names)), showextrema=False, widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor(BLUE), b.set_alpha(0.55)
    ax[0].axhspan(0, 1, color="#e8e7e2", alpha=0.3, zorder=0)
    med, lo_, hi_ = np.median(post, 0), np.quantile(post, 0.05, 0), np.quantile(post, 0.95, 0)
    ax[0].errorbar(range(len(names)), med, yerr=[med - lo_, hi_ - med], fmt="o", color=INK, ms=4, capsize=3, lw=1.2)
    for j in range(len(names)):
        if med[j] < 0.03 or med[j] > 0.97:
            ax[0].text(j, 0.5, "at the\nprior's\nedge", ha="center", va="center", fontsize=7, color=ORANGE)
    ax[0].set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=7.5)
    ax[0].set_ylim(-0.03, 1.03), ax[0].set_ylabel("position in the prior range (0 = low end, 1 = high end)")
    ax[0].set_title("Posterior for real traffic (Argoverse 2)")
    sh = r["posterior_sd_over_prior_sd"]
    ax[1].barh(range(len(names)), [sh[n] for n in names], color=[BLUE if sh[n] < 0.7 else GRAY for n in names])
    ax[1].axvline(1, color=INK, lw=1)
    ax[1].set_yticks(range(len(names)), names, fontsize=8), ax[1].invert_yaxis()
    ax[1].set_xlabel("posterior SD / prior SD (held-out worlds)"), ax[1].set_title("What the statistics can pin down")
    ranks = z["ranks"]
    for j in range(len(names)):
        h, _ = np.histogram(ranks[:, j], bins=10, range=(0, 100))
        ax[2].plot(np.arange(10) * 10 + 5, h / h.sum() * 10, color=BLUE if sh[names[j]] < 0.7 else GRAY, lw=1.2, alpha=0.8)
    ax[2].axhline(1, color=INK, lw=1)
    ax[2].set_ylim(0, 2.5), ax[2].set_xlabel("rank of the true knob among 99 posterior draws"), ax[2].set_ylabel("density (1 = calibrated)")
    ax[2].set_title("Calibration (SBC): U-shape = too confident")
    fig.tight_layout()
    fig.savefig(f"{OUT}/n4_sbi.png", dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["n1", "n2", "n4", "n5", "n6", "n7"]
    for w in which:
        globals()[w]()
        print("ok", w)
