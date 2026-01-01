"""How does one kicked car change a whole city? A trace of the twin that diverged (docs/DAILY.md, 2026-11-03 next step 3).

    python -m avlab.twintrace           # seed 9 of the twin experiment, 3 m/s kick to the fastest car, 600 s at 1 s resolution -> results/twintrace/
    python -m avlab.twintrace --quick   # a smoke run (copy 0, 120 s)
    python -m avlab.twintrace --plot    # figure and GIF from the saved trace -> docs/img/twintrace.png|gif

    warm copy `seed` of the 16-copy twin experiment 120 s; twin = copy; kick the fastest car of the twin by +3 m/s
    run both 600 s in 1 s steps, recording (alive, track, arc position, speed) of every slot
    displaced(v, t) = alive in both and (|ds| > 1 m or another track).  first[v] = first t when v is displaced
    parent(v) = the vehicle already displaced at first[v] that is closest to v, if it is within 30 m on the same track (a car-following
    link), or on a track that leads into v's track (a junction link), else "none" (a signal or random-stream effect)
    growth: log2 of the number of displaced vehicles against time; the doubling time is from the fastest-growing stretch.
"""
import argparse
import json
import os

import numpy as np

import jax
import jax.numpy as jnp

OUT = "results/twintrace"


def first_times(rec_a, rec_b, tol=1.0):
    """Per slot: the first frame at which it is displaced (alive in both and further than tol metres or on another track); -1 if never.
    rec_*: dict of arrays [T, N] (alive, trk, s)."""
    both = rec_a["alive"] & rec_b["alive"]
    far = (np.abs(rec_a["s"] - rec_b["s"]) > tol) | (rec_a["trk"] != rec_b["trk"])
    d = far & both
    ever = d.any(0)
    return np.where(ever, d.argmax(0), -1), d


def parents(rec_a, first, kicked, near=30.0):
    """For each displaced slot (not the kicked one), the slot that most plausibly passed the disturbance on: among the slots
    already displaced one frame earlier, the nearest one ahead or behind on the same track within `near` m. Returns {slot: (parent or -1, gap)}."""
    out = {}
    for v in np.argsort(first):
        if first[v] < 0 or v == kicked:
            continue
        t = first[v]
        prev = np.nonzero((first >= 0) & (first < t))[0]  # displaced strictly earlier
        prev = prev[rec_a["alive"][t, prev]]
        same = prev[rec_a["trk"][t, prev] == rec_a["trk"][t, v]]
        if len(same):
            gap = np.abs(rec_a["s"][t, same] - rec_a["s"][t, v])
            j = np.argmin(gap)
            out[int(v)] = (int(same[j]), float(gap[j])) if gap[j] <= near else (-1, float(gap[j]))
        else:
            out[int(v)] = (-1, float("nan"))
    return out


def run(seed_idx=9, E=16, warm_s=120.0, seconds=600.0, kick=3.0, area="oakland", keep_states=False):
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated()
    keys = jax.random.split(jax.random.PRNGKey(0), E)[seed_idx:seed_idx + 1]
    st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, 1024)
    n20 = int(round(20.0 / X.DT))
    n1 = int(round(1.0 / X.DT))

    def many(n):
        @jax.jit
        def f(s):
            return jax.lax.scan(lambda c, _: (jax.vmap(lambda si: X.step(si, J, beh, nh))(c), None), s, None, length=n)[0]
        return f
    c20, c1 = many(n20), many(n1)
    for _ in range(int(round(warm_s / 20.0))):
        st = c20(st)
    idx = int(jnp.argmax(jnp.where(st.alive, st.v, -1.0)[0]))
    tw = st._replace(v=st.v.at[0, idx].add(kick))
    keysr = ("alive", "trk", "s", "v")
    rec = {"a": {k: [] for k in keysr}, "b": {k: [] for k in keysr}}
    states = []
    a, b = st, tw
    for k in range(int(seconds)):
        for name, s_ in (("a", a), ("b", b)):
            for kk in keysr:
                rec[name][kk].append(np.asarray(getattr(s_, kk)[0]))
        if keep_states and k % 10 == 0:
            states.append((jax.tree.map(lambda x: np.asarray(x[0]), a), jax.tree.map(lambda x: np.asarray(x[0]), b)))
        a, b = c1(a), c1(b)
    rec = {n: {k: np.stack(v) for k, v in r.items()} for n, r in rec.items()}
    first, disp = first_times(rec["a"], rec["b"])
    par = parents(rec["a"], first, idx)
    nd = disp.sum(1)
    res = {"seed_idx": seed_idx, "kicked_slot": idx, "kick": kick, "seconds": seconds, "n_displaced_t": nd.tolist(),
           "n_alive_t": (rec["a"]["alive"] & rec["b"]["alive"]).sum(1).tolist(),
           "first": {int(v): int(first[v]) for v in np.nonzero(first >= 0)[0]},
           "parent": {int(v): [p[0], p[1]] for v, p in par.items()}}
    res["summary"] = summarize(res, first, par, rec)
    L2 = 2 * len(net["link_len"])
    firsts = sorted((v for v in res["first"] if v != idx), key=lambda v: res["first"][v])[:6]
    res["first_events"] = [{"slot": int(v), "t": res["first"][v], "in_junction_box": bool(rec["a"]["trk"][res["first"][v], v] >= L2),
                            "kicked_in_junction_box": bool(rec["a"]["trk"][res["first"][v], idx] >= L2) if rec["a"]["alive"][res["first"][v], idx] else None,
                            "dv_mps": float(rec["b"]["v"][res["first"][v], v] - rec["a"]["v"][res["first"][v], v]),
                            "ds_m": float(rec["b"]["s"][res["first"][v], v] - rec["a"]["s"][res["first"][v], v]),
                            "same_track_as_kicked": bool(rec["a"]["trk"][res["first"][v], v] == rec["a"]["trk"][res["first"][v], idx])} for v in firsts]
    res["kicked_alive_until_s"] = int(rec["a"]["alive"][:, idx].sum())
    res["kicked_gap_to_first_other"] = [float(np.abs(rec["a"]["s"][res["first"][v], v] - rec["a"]["s"][res["first"][v], idx])) for v in firsts]
    return (res, rec, states, (net, geo, info)) if keep_states else res


def summarize(res, first, par, rec):
    nd = np.array(res["n_displaced_t"], float)
    ok = nd >= 1
    t = np.arange(len(nd))
    # doubling time: slope of log2(N) over the span where N goes from 4 to the max/2 (least squares)
    sel = (nd >= 4) & (nd <= max(nd.max() / 2, 5))
    dbl = float("nan")
    if sel.sum() >= 5:
        slope = np.polyfit(t[sel], np.log2(nd[sel]), 1)[0]
        dbl = float(1 / slope) if slope > 0 else float("nan")
    kicked = res["kicked_slot"]
    others = [v for v in res["first"] if v != kicked]
    kinds = {"car-following (same track, <= 30 m)": 0, "no displaced vehicle within 30 m on its track": 0}
    for v in others:
        kinds["car-following (same track, <= 30 m)" if res["parent"][v][0] >= 0 else "no displaced vehicle within 30 m on its track"] += 1
    t_first_other = min((res["first"][v] for v in others), default=-1)
    first_other = [v for v in others if res["first"][v] == t_first_other]
    return {"displaced_final": int(nd[-1]), "alive_both_final": int(res["n_alive_t"][-1]), "first_other_s": int(t_first_other),
            "n_displaced_at_60s": int(nd[min(60, len(nd) - 1)]), "n_displaced_at_300s": int(nd[min(300, len(nd) - 1)]),
            "doubling_time_s": dbl, "links": kinds, "first_other_track_same_as_kicked": bool(
                all(rec["a"]["trk"][t_first_other, v] == rec["a"]["trk"][t_first_other, kicked] for v in first_other)) if t_first_other >= 0 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.plot:
        from avlab import twintrace_plot
        return twintrace_plot.main()
    res = run(seed_idx=0 if a.quick else 9, seconds=120.0 if a.quick else 600.0)
    print(json.dumps(res["summary"], indent=1))
    print(json.dumps(res["first_events"], indent=1), res["kicked_alive_until_s"])
    if not a.quick:
        json.dump(res, open(f"{OUT}/trace.json", "w"))


if __name__ == "__main__":
    main()
