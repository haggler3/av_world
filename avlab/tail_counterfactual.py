"""Idea 2: counterfactual stress tests on real traffic.

Start from a real moment (an Argoverse 2 scenario, 2.5 s in), put the logged vehicles on the imported map (xnet.avdata), replace the logged ego with
the AV under test, and then change what one real neighbour does: the car ahead brakes hard, the car ahead stops dead, the car behind closes in
to a tailgate, a pedestrian steps out at the next crosswalk. The scene is real (real geometry, real speeds, real spacing); the event is not.
The outcome is read off the ego: did it crash, how close did it come. Control runs without any perturbation tell which failures the
perturbation caused.

    python -m avlab.tail_counterfactual /path/to/av2/val [n_scenarios]    # ~10 min -> results/tail/counterfactual.json (+ failure scenes)
"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from av_world import data_path  # noqa: E402
from avrl import xppo as P  # noqa: E402
from xnet import avdata as AV  # noqa: E402
from xnet import carla_export as CE  # noqa: E402
from xnet import sim as X  # noqa: E402

OUT = "results/tail"
PERTURBATIONS = ("none", "lead_brake", "lead_stop", "tailgater", "ped_dartout")
SECONDS = 8.0
FRAME = 25


def neighbours(st, ego):
    """Slots directly ahead of and behind the ego on its track (or -1)."""
    trk, s, alive = (np.asarray(getattr(st, f))[0] for f in ("trk", "s", "alive"))
    same = alive & (trk == trk[ego])
    same[ego] = False
    ahead = np.nonzero(same & (s > s[ego]))[0]
    behind = np.nonzero(same & (s < s[ego]))[0]
    lead = int(ahead[np.argmin(s[ahead])]) if len(ahead) else -1
    foll = int(behind[np.argmax(s[behind])]) if len(behind) else -1
    return lead, foll


def edit(st, field, slot, value):
    return st._replace(**{field: getattr(st, field).at[0, slot].set(value)})


def run_one(sim, net, st0, ego, kind, lead, foll):
    """Run SECONDS from st0 with the perturbation; -> (frames, outcome dict)."""
    st = st0
    L = len(net["link_len"])
    frames, fired = [], False
    for k in range(int(SECONDS / X.DT)):
        t = k * X.DT
        if kind == "lead_brake" and lead >= 0 and t < 2.5:
            st = edit(st, "v", lead, jnp.maximum(st.v[0, lead] - 5.0 * X.DT, 0.0))
        elif kind == "lead_stop" and lead >= 0 and k == 0:
            st = edit(edit(st, "stop_t", lead, 60.0), "broken", lead, True)
        elif kind == "tailgater" and foll >= 0 and k == 0:
            st = edit(edit(edit(edit(st, "T", foll, 0.3), "v0f", foll, 1.5), "amax", foll, 3.0), "s0", foll, 1.0)
        elif kind == "ped_dartout" and not fired:
            trk_e = int(np.asarray(st.trk)[0, ego])
            leg = int(net["link_leg_in"][trk_e // 2]) if trk_e < 2 * L else -1
            if leg >= 0:  # the pedestrian steps out when the ego is 1.5 s from the line
                dist = float(net["link_len"][trk_e // 2]) - float(np.asarray(st.s)[0, ego])
                if 2.0 < dist < 1.5 * max(float(np.asarray(st.v)[0, ego]), 1.0):
                    st = st._replace(ped_t=st.ped_t.at[0, leg].set(10.0))
                    fired = True
        st = sim.one(st, sim.nh)
        frames.append(jax.tree.map(lambda a: np.asarray(a[0]), st))
    crashed = np.array([f.wreck_t[ego] > 0 for f in frames])
    lead_gap = []
    for f in frames:
        same = f.alive & (f.trk == f.trk[ego])
        same[ego] = False
        ahead = same & (f.s > f.s[ego])
        lead_gap.append(float(np.min(f.s[ahead] - f.length[ahead] - f.s[ego])) if ahead.any() else np.inf)
    t_crash = int(np.argmax(crashed)) * X.DT if crashed.any() else None
    return frames, {"dartout_fired": fired, "ego_crashed": bool(crashed.any()), "t_crash": t_crash, "min_gap_m": float(np.min(lead_gap)), "ego_hard_brake": bool(any(f.acc[ego] < -4.0 for f in frames))}


def main(root, limit=40):
    os.makedirs(f"{OUT}/counterfactual_scenes", exist_ok=True)
    pol_A = P.deterministic(P.load("results/famous/policy_A.npz"))
    drivers = {"rule-based AV (IDM envelope)": (None, "idm"), "learned policy A, no filter": (pol_A, "off")}
    rows, scenes_saved = [], 0
    t0 = time.time()
    for d in sorted(glob.glob(os.path.join(root, "*")))[:limit]:
        sc = AV.read_av2(d)
        scene = AV.av2_to_scene(sc, 2)
        try:
            net, geo, _ = AV.av2_network(sc)
        except Exception:
            continue
        m = AV.Matcher(net, geo)
        for name, (pol, shield) in drivers.items():
            X.SHIELD = shield
            st, info = AV.seed_state(net, geo, scene, FRAME, matcher=m, speed_from_log=True)
            ego_ids = [v["id"] for v in scene["vehicles"] if v["av"]]
            if not ego_ids or ego_ids[0] not in info["slots"]:
                break
            ego = info["slots"][ego_ids[0]]
            if not bool(np.asarray(st.av)[ego]) or float(np.asarray(st.v)[ego]) < 2.0:
                break  # the ego must be on the network and moving
            lead, foll = neighbours(jax.tree.map(lambda a: a[None], st), ego)
            sim = X.Sim(net, n_envs=1, beh=X.Behaviour(demand=0.0, p_av=0.0), seed=1, reroute_s=0, policy=pol)
            st1 = jax.tree.map(lambda a: a[None], st)
            for kind in PERTURBATIONS:
                if kind.startswith("lead") and lead < 0 or kind == "tailgater" and foll < 0:
                    continue
                frames, out = run_one(sim, net, st1, ego, kind, lead, foll)
                rows.append({"scenario": sc["id"], "city": sc["city"], "driver": name, "perturbation": kind, **out})
                if out["ego_crashed"] and kind != "none" and scenes_saved < 6:
                    CE.export_frames(net, geo, frames, f"{OUT}/counterfactual_scenes/{kind}_{sc['id'][:8]}")
                    scenes_saved += 1
        print(f"{sc['id'][:8]} {sc['city']}: {len(rows)} runs, {time.time() - t0:.0f} s", flush=True)
        json.dump(rows, open(f"{OUT}/counterfactual_rows.json", "w"))
    summary = {}
    for name in drivers:
        for kind in PERTURBATIONS:
            r = [x for x in rows if x["driver"] == name and x["perturbation"] == kind]
            if r:
                summary[f"{name} | {kind}"] = {"runs": len(r), "ego_crash_rate": float(np.mean([x["ego_crashed"] for x in r])), "mean_min_gap_m": float(np.mean([min(x["min_gap_m"], 60.0) for x in r])),
                                               "hard_brake_rate": float(np.mean([x["ego_hard_brake"] for x in r])),
                                               "perturbation_applied": float(np.mean([x["dartout_fired"] for x in r])) if kind == "ped_dartout" else 1.0}
    json.dump(summary, open(f"{OUT}/counterfactual.json", "w"), indent=1)
    for k, v in summary.items():
        print(f"{k:55s} runs {v['runs']:3d}  ego crashes {100 * v['ego_crash_rate']:5.1f} %  min gap {v['mean_min_gap_m']:5.1f} m  hard brake {100 * v['hard_brake_rate']:5.1f} %")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
