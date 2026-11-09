"""Is the junction box where a city forgets its past? Many kicked twins, not one (docs/DAILY.md, 2026-11-06 next step 3).

    python -m avlab.twinkicks [--quick]   # -> results/twinkicks/twinkicks.json
    python -m avlab.twinkicks plot        # -> docs/img/twinkicks.png

E copies of Oakland, warmed 120 s.  Each copy gets a twin; the twin's car `k` is kicked by +3 m/s (k = the fastest car, or a random
moving car, per cell).  Both run `seconds` s, recording (alive, track, arc position) every second (all copies and twins at once).
displaced(v, t) = alive in both and (|ds| > 1 m or another track) -- avlab.twintrace.first_times.
For each pair: the first OTHER car displaced, at second t1, and whether (i) the kicked car was in a junction box (track >= 2L) or
within 30 m behind/ahead on the same track at t1 - 1 (a car-following link), (ii) the first other car was itself in a box.
A pair 'diverges' if >= 25 cars are displaced at the end.
"""
import argparse
import json
import os

import numpy as np

OUT = "results/twinkicks"


def classify(rec_a, kicked, first, L2, near=30.0):
    """-> dict for one pair: diverged?, t1, link kind of the first other car ('following', 'box', 'other', 'none')."""
    from avlab.twintrace import first_times  # noqa: F401  (first is passed in; kept for symmetry)
    others = np.nonzero((first >= 0) & (np.arange(len(first)) != kicked))[0]
    if len(others) == 0:
        return {"t1": -1, "kind": "none"}
    t1 = int(first[others].min())
    cand = others[first[others] == t1]
    t0 = max(t1 - 1, 0)
    kind = "other"
    for v in cand:
        same = rec_a["alive"][t0, kicked] and rec_a["alive"][t0, v] and rec_a["trk"][t0, kicked] == rec_a["trk"][t0, v] and abs(rec_a["s"][t0, kicked] - rec_a["s"][t0, v]) <= near
        inbox = (rec_a["alive"][t0, kicked] and rec_a["trk"][t0, kicked] >= L2) or (rec_a["alive"][t0, v] and rec_a["trk"][t0, v] >= L2)
        if same:
            kind = "following"; break
        if inbox:
            kind = "box"
    return {"t1": t1, "kind": kind}


def run(E=12, warm_s=120.0, seconds=300.0, kick=3.0, area="oakland", modes=("fastest", "random")):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from avlab.twintrace import first_times
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated()
    L2 = 2 * len(net["link_len"])
    keys = jax.random.split(jax.random.PRNGKey(0), E)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, 1024)

    def many(n):
        @jax.jit
        def f(s):
            return jax.lax.scan(lambda c, _: (jax.vmap(lambda si: X.step(si, J, beh, nh))(c), None), s, None, length=n)[0]
        return f
    c20, c1 = many(int(round(20.0 / X.DT))), many(int(round(1.0 / X.DT)))
    for _ in range(int(round(warm_s / 20.0))):
        st = c20(st)
    rng = np.random.default_rng(0)
    alive, v = np.asarray(st.alive), np.asarray(st.v)
    kicked = {}
    for mode in modes:
        ks = []
        for e in range(E):
            ok = np.nonzero(alive[e] & (v[e] > 3.0))[0]
            ks.append(int(ok[np.argmax(v[e][ok])]) if mode == "fastest" else int(rng.choice(ok)))
        kicked[mode] = ks
    # stack: [copies of mode 0 | twins of mode 0 | copies of mode 1 | twins ...] -- the plain copies are shared, so run one set + one twin set per mode
    twins = []
    for mode in modes:
        idx = jnp.asarray(kicked[mode])
        twins.append(st._replace(v=st.v.at[jnp.arange(E), idx].add(kick)))
    big = jax.tree.map(lambda *x: jnp.concatenate(x, 0), st, *twins)
    recs = {k: [] for k in ("alive", "trk", "s")}
    for _ in range(int(seconds)):
        for k in recs:
            recs[k].append(np.asarray(getattr(big, k)))
        big = c1(big)
    rec = {k: np.stack(v_) for k, v_ in recs.items()}  # [T, (1+M)E, N]
    out = {"E": E, "seconds": seconds, "kick": kick, "modes": {}}
    for mi, mode in enumerate(modes):
        pairs = []
        for e in range(E):
            ra = {k: rec[k][:, e] for k in rec}
            rb = {k: rec[k][:, (1 + mi) * E + e] for k in rec}
            first, disp = first_times(ra, rb)
            c = classify(ra, kicked[mode][e], first, L2)
            c["displaced_final"] = int(disp[-1].sum()); c["diverged"] = bool(disp[-1].sum() >= 25)
            c["kicked_alive_s"] = int(ra["alive"][:, kicked[mode][e]].sum())
            pairs.append(c)
        out["modes"][mode] = pairs
    return out


def summarize(res):
    from scipy import stats
    out = {}
    for mode, pairs in res["modes"].items():
        dv = [p for p in pairs if p["diverged"]]
        kinds = {k: sum(p["kind"] == k for p in dv) for k in ("following", "box", "other", "none")}
        allk = {k: sum(p["kind"] == k for p in pairs) for k in ("following", "box", "other", "none")}
        n = max(len(dv), 1)
        ci = stats.binomtest(kinds["box"], len(dv)).proportion_ci(0.95, method="exact") if dv else None
        out[mode] = {"pairs": len(pairs), "diverged": len(dv), "kinds_diverged": kinds, "kinds_all": allk,
                     "box_share_of_diverged": kinds["box"] / n, "box_ci": [ci.low, ci.high] if ci else None,
                     "median_t1_s": float(np.median([p["t1"] for p in pairs if p["t1"] >= 0])) if any(p["t1"] >= 0 for p in pairs) else None}
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, FG, GR = "#0d1117", "#e6edf3", "#30363d"
    R = json.load(open(f"{OUT}/twinkicks.json"))
    S = R["summary"]
    cols = {"following": "#e69f00", "box": "#56b4e9", "other": "#009e73", "none": "#8b949e"}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=BG)
    for a in ax:
        a.set_facecolor(BG); a.tick_params(colors=FG); [s.set_color(GR) for s in a.spines.values()]
        a.xaxis.label.set_color(FG); a.yaxis.label.set_color(FG); a.title.set_color(FG)
    for i, (mode, s) in enumerate(S.items()):
        bot = 0
        for k, col in cols.items():
            v = s["kinds_all"][k]
            if v:
                ax[0].bar(i, v, bottom=bot, color=col, label=k if i == 0 else None); ax[0].text(i, bot + v / 2, f"{k} {v}", color="#0d1117", ha="center", va="center", fontsize=8); bot += v
    ax[0].set_xticks(range(len(S))); ax[0].set_xticklabels([f"kick the {m} car" for m in S], color=FG)
    ax[0].set_ylabel("kicked twins"); ax[0].set_title("How the first other car is reached")
    for mode, col in zip(R["modes"], ("#e69f00", "#56b4e9")):
        pr = R["modes"][mode]
        ax[1].plot(sorted(p["displaced_final"] for p in pr), "o-", color=col); ax[1].text(len(pr) - 1.3, max(p["displaced_final"] for p in pr) + 12, mode, color=col, ha="right", fontsize=9)
    ax[1].set_xlabel("twin, sorted"); ax[1].set_ylabel(f"cars displaced after {int(R['seconds'])} s"); ax[1].set_title("How far one kick spreads")
    fig.tight_layout(); fig.savefig("docs/img/twinkicks.png", dpi=130, facecolor=BG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="run", choices=["run", "plot"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what == "plot":
        return plot()
    res = run(**(dict(E=2, seconds=40.0, warm_s=40.0) if a.quick else {}))
    res["summary"] = summarize(res)
    print(json.dumps(res["summary"], indent=1))
    if not a.quick:
        json.dump(res, open(f"{OUT}/twinkicks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
