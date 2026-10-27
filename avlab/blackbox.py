"""Black box: find a crash, replay it frame by frame, and write the incident report an investigator would (who, where, how fast, the
signal, what each driver was doing, what contributed), with a zoomed animation of the last seconds.

    python -m avlab.blackbox --area oakland                       # the first crash in 8 copies of the city -> results/blackbox/
    python -m avlab.blackbox --area chicago --scenario bus_dartout --av    # the first crash involving an AV
    av-world blackbox --area oakland                              # the same from the CLI

It reads only the simulator's state (nothing in the simulator changes): every vehicle slot carries its kind, speed, applied
acceleration and the human-factor flags the crash model uses (ran the red, distracted, does not look, aggressive, brake pulse, AV
perception fault, broken down), so a report can say which of them were present at the moment of impact.
"""
import argparse
import json
import os
import time

import numpy as np

from xnet import sim as X

OUT = "results/blackbox"
KINDS = {X.CAR: "car", X.BUS: "bus", X.VAN: "delivery van", X.BIKE: "cyclist"}
LIGHT = {0: "red", 1: "yellow", 2: "green"}


def hunt(net, beh, seed=0, warm=90.0, chunk=20.0, budget=600.0, copies=8, av_only=False, policy=None):
    """Run copies until one crashes in a chunk, then replay that chunk of that copy with recording on (the replay steps one at a
    time and is checked to contain the crash). -> (frames, copy) or (None, None)."""
    key = "av_crash" if av_only else None
    idx = [X.METRICS.index(k) for k in ((key,) if key else X.CRASH_NAMES)]
    sim = X.Sim(net, n_envs=copies, beh=beh, seed=seed, reroute_s=0, policy=policy)
    sim.run(warm)
    for _ in range(max(int(budget / chunk), 1)):
        snap = sim.st
        sim.run(chunk)
        after = sim.st
        hit = np.nonzero(np.asarray(after.m)[:, idx].sum(1) > np.asarray(snap.m)[:, idx].sum(1))[0]
        for e in hit:
            sim.st = snap
            frames = sim.run(chunk + 8.0, record=(int(e), 1))  # 8 s more so the report can show the aftermath
            m = np.array([f.m[idx].sum() for f in frames])
            if (np.diff(m) > 0).any():
                return frames, int(e)
        sim.st = after
    return None, None


def _light(net, link, t):
    return LIGHT[int(np.asarray(X.signal_state({k: np.asarray(v) for k, v in net.items()}, float(t)))[link])]


def investigate(net, geo, frames, beh=None):
    """The report of the first crash in `frames` (one copy's per-step states). -> dict, or None if the frames hold no crash."""
    from xnet import render as R
    L = len(net["link_len"])
    names = list(X.METRICS)
    crash_idx = [names.index(c) for c in X.CRASH_NAMES]
    m = np.array([f.m[crash_idx] for f in frames])
    ks = np.nonzero((np.diff(m, axis=0) > 0).any(1))[0]
    if not len(ks):
        return None
    k = int(ks[0]) + 1  # the frame the crash is first counted in
    types = [X.CRASH_NAMES[j] for j in np.nonzero(m[k] - m[k - 1])[0]]
    st, prev = frames[k], frames[k - 1]
    new = np.nonzero((st.wreck_t > 0) & (prev.wreck_t <= 0))[0]
    tracks = R.Tracks(net, geo)
    involved = []
    for i in new:
        same = [j for j in range(max(0, k - 25), k) if frames[j].alive[i] and abs(frames[j].t0[i] - st.t0[i]) < 1e-3]
        v = np.array([frames[j].v[i] for j in same])
        a = np.array([frames[j].acc[i] for j in same])
        trk = int(prev.trk[i])
        x, y, _ = tracks.pose(np.array([trk]), np.array([prev.s[i]]), np.array([prev.length[i]]))
        if trk < 2 * L:
            where, link = f"on link {trk // 2} ({net['link_len'][trk // 2]:.0f} m long, {prev.s[i]:.0f} m along)", trk // 2
        else:
            mv = trk - 2 * L
            turn = {0: "straight", 1: "left", 2: "right"}.get(int(net["mov_turn"][mv]), "a turn")
            where, link = f"inside junction {int(net['mov_node'][mv])}, turning {turn}", int(net["mov_in"][mv])
        hard = np.nonzero(a < -3.0)[0]
        passed = None
        if trk < 2 * L:  # on a link: the light that mattered is the one it came through, not the one at the far end
            hist = [j for j in range(max(0, k - 75), k) if frames[j].alive[i] and abs(frames[j].t0[i] - st.t0[i]) < 1e-3 and frames[j].trk[i] >= 2 * L]
            if hist:
                last = hist[-1]
                first = last
                while first - 1 in hist and frames[first - 1].trk[i] == frames[last].trk[i]:
                    first -= 1
                mv = int(frames[last].trk[i]) - 2 * L
                node = int(net["mov_node"][mv])
                light = _light(net, int(net["mov_in"][mv]), frames[first].t) if net["node_signal"][node] else "no signal"
                passed = {"junction": node, "turn": {0: "straight", 1: "left", 2: "right"}.get(int(net["mov_turn"][mv]), "turn"), "light_at_entry": light, "s_before_impact": round((k - first) * X.DT, 1),
                          "ran_red_there": bool(frames[last].ran_red[i])}
        factors = []
        if prev.ran_red[i]:
            factors.append("entered on red")
        if prev.dist_t[i] > 0:
            factors.append("driver distracted at impact")
        if prev.nolook[i] and trk >= 2 * L:
            factors.append("a driver who turns without looking, mid-turn")
        if prev.aggr[i]:
            factors.append("aggressive driver (accepts short gaps)")
        if prev.pulse_t[i] > 0:
            factors.append("unexplained brake pulse")
        if prev.blind_t[i] > 0:
            factors.append("AV perception blackout")
        if prev.blind_t[i] < 0:
            factors.append("AV phantom obstacle")
        if prev.broken[i]:
            factors.append("broken down")
        involved.append({
            "slot": int(i), "kind": "AV" if prev.av[i] else KINDS.get(int(prev.kind[i]), "vehicle"),
            "where": where, "xy": [round(float(x[0]), 1), round(float(y[0]), 1)],
            "light_on_approach": _light(net, link, prev.t) if trk < 2 * L or net["node_signal"][int(net["mov_node"][trk - 2 * L])] else "none",
            "came_through": passed,
            "node": int(net["link_end"][trk // 2]) if trk < 2 * L else int(net["mov_node"][trk - 2 * L]),
            "speed_at_impact_kmh": round(3.6 * float(v[-1] if len(v) else 0), 1),
            "speed_3s_before_kmh": round(3.6 * float(v[max(0, len(v) - 16)] if len(v) else 0), 1),
            "peak_braking_mps2": round(float(min(a.min(), 0.0)) if len(a) else 0.0, 1),
            "braked_hard_s_before_impact": round((len(a) - 1 - hard[0]) * X.DT, 1) if len(hard) else None,
            "contributing": factors})
    peds = int((st.ped_t > 0).sum())
    if "pedestrian" in types:
        J = {kk: np.asarray(vv) for kk, vv in net.items()}
        for v in involved:
            legs = [g for g in np.nonzero(np.asarray(net["leg_node"]) == v["node"])[0] if prev.ped_t[g] > 0]
            notes = []
            for g in legs:  # when did this crossing start, and was WALK shown then?
                j = k - 1
                while j > max(0, k - 100) and frames[j - 1].ped_t[g] > 0 and frames[j - 1].ped_t[g] >= frames[j].ped_t[g]:
                    j -= 1
                walk = bool(np.asarray(X.leg_walk(J, float(frames[j].t)))[g])
                notes.append(walk)
            if notes and not any(notes):
                v["contributing"].append("struck a pedestrian who started crossing without the WALK signal")
            elif notes:
                v["contributing"].append("struck a pedestrian crossing with the WALK signal")
            else:
                v["contributing"].append("struck a pedestrian on the crosswalk ahead")
    friction = float(beh.friction) if beh is not None else 1.0
    if friction < 0.9:
        for v in involved:
            v["contributing"].append(f"slippery road (friction {friction:.2f})")
    xy = np.mean([v["xy"] for v in involved], 0) if involved else np.zeros(2)
    closing = abs(involved[0]["speed_at_impact_kmh"] - involved[1]["speed_at_impact_kmh"]) if len(involved) == 2 else None
    return {"t_s": round(float(st.t), 1), "frame": k, "types": types, "vehicles": involved, "closing_speed_kmh": closing, "pedestrians_crossing_nearby_city_wide": peds,
            "centre_xy": [round(float(xy[0]), 1), round(float(xy[1]), 1)], "summary": _summary(types, involved, closing)}


def _summary(types, inv, closing=None):
    who = " and ".join(f"a {v['kind']} ({v['speed_at_impact_kmh']:.0f} km/h)" for v in inv) or "a vehicle"
    if closing is not None:
        who += f", closing at {closing:.0f} km/h"
    t = " / ".join(types)
    why = sorted({f for v in inv for f in v["contributing"]})
    return f"{t.capitalize()} crash: {who}. " + (f"Contributing: {'; '.join(why)}." if why else "No human-factor flag was set: a gap or physics crash.")


def markdown(rep, title="Incident report"):
    L = [f"# {title}", "", f"**{rep['summary']}**", "", f"Time {rep['t_s']:.1f} s into the run; crash type: {', '.join(rep['types'])}.", "",
         "| | kind | where | light (on approach, or the junction it just came through) | speed 3 s before | at impact | peak braking | braked hard | contributing |", "|---|---|---|---|---|---|---|---|---|"]
    for j, v in enumerate(rep["vehicles"], 1):
        bh = f"{v['braked_hard_s_before_impact']:.1f} s before" if v["braked_hard_s_before_impact"] is not None else "no"
        came = v.get("came_through")
        light = (f"came through junction {came['junction']} ({came['turn']}) on {came['light_at_entry']} {came['s_before_impact']:.1f} s before"
                 if came else v["light_on_approach"])
        L.append(f"| {j} | {v['kind']} | {v['where']} | {light} | {v['speed_3s_before_kmh']:.0f} km/h | {v['speed_at_impact_kmh']:.0f} km/h | "
                 f"{v['peak_braking_mps2']:.1f} m/s² | {bh} | {'; '.join(v['contributing']) or '-'} |")
    return "\n".join(L) + "\n"


def run(area="oakland", scenario=None, seed=0, av_only=False, out=OUT, gif=True, budget=600.0, av_share=0.25, policy=None):
    from av_world import easy
    net, geo, info = easy._build(area)
    beh = easy.world(av_share, scenario)
    t0 = time.time()
    frames, e = hunt(net, beh, seed=seed, av_only=av_only, budget=budget, policy=policy)
    if frames is None:
        print(f"no {'AV ' if av_only else ''}crash in {budget:.0f} s x 8 copies")
        return None
    rep = investigate(net, geo, frames, beh)
    os.makedirs(out, exist_ok=True)
    name = f"{area}_{scenario or 'everyday'}_s{seed}{'_av' if av_only else ''}"
    rep.update(area=area, scenario=scenario, copy=e, seed=seed, search_s=round(time.time() - t0, 1))
    json.dump(rep, open(f"{out}/{name}.json", "w"), indent=1, default=float)
    open(f"{out}/{name}.md", "w").write(markdown(rep, f"Incident report: {area}{', ' + scenario if scenario else ''}"))
    if gif:
        from xnet import render as R
        k = rep["frame"]
        clip = frames[max(0, k - 40):k + 25]
        R.animate(net, geo, info, clip, f"{out}/{name}.gif", zoom=(tuple(rep["centre_xy"]), 45.0),
                  title=f"Black box: {rep['types'][0]} crash", fps=10)
        rep["gif"] = f"{out}/{name}.gif"
    print(markdown(rep))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="av-world blackbox")
    ap.add_argument("--area", default="oakland")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--av", action="store_true", help="only a crash that involves an AV")
    ap.add_argument("--budget", type=float, default=600.0, help="simulated seconds to search (x 8 copies)")
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args(argv)
    run(a.area, a.scenario, a.seed, a.av, gif=not a.no_gif, budget=a.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
