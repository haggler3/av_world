"""Which human mistakes make the crashes? An exact Shapley decomposition of the city's crash rate over its mistake mechanisms.

    python -m avlab.attribution            # 5 mechanisms, 32 worlds x 24 copies x 5 min (~15 min, 4 CPU cores) -> results/attribution/
    python -m avlab.attribution --quick    # 3 mechanisms, a smoke run

Each mechanism is a player that is either on (its calibrated rate) or off (0). The value of a coalition S is the crash rate of the
world where only the mechanisms in S are on, v(S) = crashes / vehicle-hour, measured in its own copies. The Shapley value of
mechanism i is its average marginal effect over every order in which the mechanisms could be switched on:

    phi_i = sum over S not containing i of |S|! (n - |S| - 1)! / n! * (v(S + i) - v(S))

The phi_i add up to v(all) - v(none): the part of the crash rate the mechanisms explain together. v(none) is what is left with every
mechanism off (crashes from gap acceptance, merging, cyclists, the physics: the "structural" crashes). Each v(S) is an independent
Poisson rate, so Var(phi_i) = sum_S c_{i,S}^2 k_S / t_S^2 with c_{i,S} the Shapley coefficient of v(S) in phi_i (exact, no sampling
over orders). The same decomposition per crash type tells which mechanism feeds which kind of crash.
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np

from xnet import sim as X

OUT = "results/attribution"
MECHANISMS = {  # name: Behaviour fields switched off (0) when the mechanism is off
    "red-light running": ("p_red",),
    "not looking": ("p_nolook",),
    "distraction": ("distract_rate",),
    "aggressive drivers": ("p_aggressive",),
    "pedestrians crossing against the light": ("p_ped_violate",),
}


def coalitions(n):
    return [frozenset(c) for r in range(n + 1) for c in itertools.combinations(range(n), r)]


def coefficients(n):
    """c[i][S]: phi_i = sum_S c[i][S] v(S) (Shapley weights, exact)."""
    f = math.factorial
    c = [dict() for _ in range(n)]
    for i in range(n):
        for S in coalitions(n):
            if i in S:
                continue
            w = f(len(S)) * f(n - len(S) - 1) / f(n)
            c[i][S | {i}] = c[i].get(S | {i}, 0.0) + w
            c[i][S] = c[i].get(S, 0.0) - w
    return c


def world(names, on, amplify=1.0):
    """Calibrated humans with the mechanisms in `on` at their rate x amplify (probabilities capped at 0.95) and the others off."""
    from examples import _common as C
    base = C.calibrated(p_av=0.0)
    kw = {}
    for j, n in enumerate(names):
        for f in MECHANISMS[n]:
            v = getattr(base, f)
            kw[f] = 0.0 if j not in on else (v * amplify if f == "distract_rate" else min(v * amplify, 0.95))
    return base._replace(**kw)


def run(names=tuple(MECHANISMS), copies=24, warm_s=120.0, seconds=300.0, area="oakland", seed=0, batch=192, amplify=1.0):
    """-> {coalition (frozenset of mechanism indices): per-copy counters [copies, NM]}."""
    from examples import _common as C
    n = len(names)
    cs = coalitions(n)
    jobs = [(S, world(names, S, amplify)) for S in cs for _ in range(copies)]
    out = {S: [] for S in cs}
    t0 = time.time()
    for b in range(0, len(jobs), batch):
        part = jobs[b:b + batch]
        m = C.run_copies(area, [w for _, w in part], warm_s=warm_s, seconds=seconds, seed=seed + b)
        for (S, _), row in zip(part, np.asarray(m)):
            out[S].append(row)
        print(f"  {min(b + batch, len(jobs))}/{len(jobs)} copies ({time.time() - t0:.0f} s)", flush=True)
    return {S: np.asarray(v) for S, v in out.items()}


def shapley(per_copy, names, types=None):
    """Shapley values with exact standard errors of three outcomes: crashes per 1,000 vehicle-hours (the headline), crashes per
    million vehicle-km, and the network's mean speed. The last two tell an exposure effect (a mechanism that slows traffic lowers
    crashes per hour without making anyone safer per km) from a real one."""
    n = len(names)
    c = coefficients(n)
    col = {k: i for i, k in enumerate(X.METRICS)}
    types = types or X.CRASH_NAMES
    counters = {S: np.asarray(v).sum(0) if np.ndim(v) == 2 else np.asarray(v) for S, v in per_copy.items()}

    def per_km(S):
        m = counters[S]
        k = sum(m[col[t]] for t in types)
        km = m[col["veh_m"]] / 1e9  # million vehicle-km
        return k / km, k / km ** 2

    def speed(S):
        v = np.asarray(per_copy[S])
        if v.ndim < 2:
            m = counters[S]
            return 3.6 * m[col["veh_m"]] / max(m[col["veh_s"]], 1), 0.0
        sp = 3.6 * v[:, col["veh_m"]] / np.maximum(v[:, col["veh_s"]], 1)
        return float(sp.mean()), float(sp.var(ddof=1) / len(sp))

    def rate(S, ty=None):
        m = counters[S]
        k = sum(m[col[t]] for t in (ty or types))
        t = m[col["veh_s"]] / 3600 / 1000
        r, var = k / t, k / t ** 2  # rate per 1,000 vehicle-hours and its Poisson variance
        v = np.asarray(per_copy[S])
        if v.ndim == 2 and len(v) > 2:  # robust (sandwich) variance from the copies: crashes cluster (pile-ups, jams), so the
            ki = sum(v[:, col[x]] for x in (ty or types))  # Poisson variance understates it when the counts are overdispersed
            ti = v[:, col["veh_s"]] / 3600 / 1000
            n = len(ki)
            var = max(var, float(n / (n - 1) * np.sum((ki - r * ti) ** 2) / t ** 2))
        return r, var

    res = {"names": list(names), "v_none": rate(frozenset())[0], "v_all": rate(frozenset(range(n)))[0], "phi": {}, "se": {}, "by_type": {}}
    for i, name in enumerate(names):
        res["phi"][name] = sum(w * rate(S)[0] for S, w in c[i].items())
        res["se"][name] = math.sqrt(sum(w * w * rate(S)[1] for S, w in c[i].items()))
        res["by_type"][name] = {ty: sum(w * rate(S, (ty,))[0] for S, w in c[i].items()) for ty in X.CRASH_NAMES}
    res["v_none_by_type"] = {ty: rate(frozenset(), (ty,))[0] for ty in X.CRASH_NAMES}
    # Holm-corrected p-values of the five headline values (a 2.6-sigma result among five is not rare), and a plausibility budget:
    # how many crashes a value implies over the hours its mechanism was on, against how many crashes that mechanism took part in
    from scipy.stats import norm
    from examples.hypotheses import holm
    p = [float(2 * norm.sf(abs(res["phi"][nm]) / max(res["se"][nm], 1e-12))) for nm in names]
    res["p"], res["p_holm"] = dict(zip(names, p)), dict(zip(names, holm(np.asarray(p)).tolist()))
    for key, f in (("per_vkm", per_km), ("speed_kmh", speed)):
        res[key] = {"v_none": f(frozenset())[0], "v_all": f(frozenset(range(n)))[0],
                    "phi": {nm: sum(w * f(S)[0] for S, w in c[i].items()) for i, nm in enumerate(names)},
                    "se": {nm: math.sqrt(sum(w * w * f(S)[1] for S, w in c[i].items())) for i, nm in enumerate(names)}}
    res["coalitions"] = {",".join(names[j] for j in sorted(S)) or "(none)": {"rate": rate(S)[0], "crashes": float(sum(counters[S][col[t]] for t in types)),
                                                                             "vehicle_hours": float(counters[S][col["veh_s"]] / 3600)}
                         for S in counters}
    return res


def reanalyse(name="shapley_x10"):
    """Recompute every statistic from the saved per-copy counters (no simulation)."""
    z = np.load(f"{OUT}/{name}_counters.npz")
    names = tuple(json.load(open(f"{OUT}/{name}.json"))["names"])
    per_copy = {frozenset(int(x) for x in str(c).split(",") if x != ""): z["counters"][i] for i, c in enumerate(z["coalitions"])}
    old = json.load(open(f"{OUT}/{name}.json"))
    res = shapley(per_copy, names)
    res.update({k: old[k] for k in ("wall_s", "amplify", "settings") if k in old})
    col = {k: i for i, k in enumerate(X.METRICS)}
    on = {nm: [S for S in per_copy if j in S] for j, nm in enumerate(names)}
    res["involvement_check"] = {"distraction": {
        "crashes_with_a_distracted_driver": float(sum(per_copy[S][:, col["tail_distracted"]].sum() for S in on.get("distraction", []))),
        "crashes_its_shapley_value_implies": float(abs(res["phi"]["distraction"]) * sum(per_copy[S][:, col["veh_s"]].sum() for S in on["distraction"]) / 3600 / 1000)}} \
        if "distraction" in names else {}
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1, default=float)
    return res


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"
    names = res["names"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=BG, gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in (a0, a1):
        ax.set_facecolor(BG)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
    bars = [("structural (every mechanism off)", res["v_none"], 0.0)] + [(n, res["phi"][n], res["se"][n]) for n in names]
    y = np.arange(len(bars))[::-1]
    a0.barh(y, [b[1] for b in bars], xerr=[1.96 * b[2] for b in bars], color=["#8b949e"] + ["#3987e5"] * len(names), height=0.6,
            error_kw={"ecolor": INK2, "lw": 1})
    a0.set_yticks(y, [b[0] for b in bars], color=INK, fontsize=9)
    a0.axvline(0, color="#30363d")
    a0.set_xlabel("crashes per 1,000 vehicle-hours (Shapley value, 95 % interval)", color=INK2, fontsize=9)
    amp = res.get("amplify", 1.0)
    world = "calibrated humans" if amp == 1 else f"every mistake rate x{amp:g}"
    a0.set_title(f"Where the crash rate comes from ({world}): {res['v_all']:.1f} per 1,000 h with everything on", color=INK, fontsize=10, loc="left")
    types = list(X.CRASH_NAMES)
    M = np.array([[res["v_none_by_type"][t] for t in types]] + [[res["by_type"][n][t] for t in types] for n in names])
    lim = np.abs(M).max()
    a1.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    a1.set_xticks(range(len(types)), types, rotation=20, ha="right")
    a1.set_yticks(range(len(bars)), [b[0] for b in bars], color=INK, fontsize=8.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a1.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", color=INK if abs(M[i, j]) > lim / 2 else "#111", fontsize=8)
    a1.set_title("... by crash type (per 1,000 vehicle-hours)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def main(quick=False, amplify=1.0):
    os.makedirs(OUT, exist_ok=True)
    names = tuple(MECHANISMS)[:3] if quick else tuple(MECHANISMS)
    kw = dict(copies=2, warm_s=30.0, seconds=60.0) if quick else {}
    t0 = time.time()
    per_copy = run(names, amplify=amplify, **kw)
    name = "quick" if quick else ("shapley" if amplify == 1 else f"shapley_x{amplify:g}")
    cs = coalitions(len(names))
    np.savez_compressed(f"{OUT}/{name}_counters.npz", counters=np.stack([per_copy[S] for S in cs]),
                        coalitions=np.array([",".join(map(str, sorted(S))) for S in cs]))
    res = shapley(per_copy, names)
    res.update(wall_s=time.time() - t0, amplify=amplify, settings=kw or dict(copies=24, warm_s=120.0, seconds=300.0))
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1, default=float)
    if not quick:
        plot(res, f"docs/img/attribution{'' if amplify == 1 else f'_x{amplify:g}'}.png")
    print(f"v(all) = {res['v_all']:.2f}, v(none) = {res['v_none']:.2f} per 1,000 vehicle-hours")
    for n in names:
        print(f"  {n:42s} {res['phi'][n]:+6.2f} +- {1.96 * res['se'][n]:.2f} /1000 vh   {res['per_vkm']['phi'][n]:+6.2f} +- "
              f"{1.96 * res['per_vkm']['se'][n]:.2f} /M vkm   speed {res['speed_kmh']['phi'][n]:+.2f} +- {1.96 * res['speed_kmh']['se'][n]:.2f} km/h")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--amplify", type=float, default=1.0, help="mechanisms that are on run at their calibrated rate x this")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--reanalyse", action="store_true", help="recompute the statistics from the saved counters")
    a = ap.parse_args()
    if a.reanalyse:
        r = reanalyse("shapley" if a.amplify == 1 else f"shapley_x{a.amplify:g}")
        for nm in r["names"]:
            print(f"  {nm:42s} {r['phi'][nm]:+6.2f} +- {1.96 * r['se'][nm]:.2f}  p {r['p'][nm]:.3f}  Holm {r['p_holm'][nm]:.3f}")
        print(r.get("involvement_check"))
    elif a.replot:
        for f, img in (("shapley", "attribution"), ("shapley_x10", "attribution_x10")):
            if os.path.exists(f"{OUT}/{f}.json"):
                plot(json.load(open(f"{OUT}/{f}.json")), f"docs/img/{img}.png")
    else:
        main(a.quick, a.amplify)
