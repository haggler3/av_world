"""The fundamental diagram of a whole city: raise the demand on Oakland step by step and watch the traffic it carries rise, saturate
at the network's capacity, and start to fall as it jams. The shape traffic engineers measure on single roads, here for a whole street
network. (Measured: flow saturates at ~140 vehicle-km/h per lane-km from demand x2 to x4 while speed drops to 4 km/h, and falls past
that; it never reaches zero because the simulator removes vehicles stuck for minutes, over 1,000 an hour at x6.)

    python examples/gridlock.py            # ~5-6 min on 4 CPU cores -> docs/img/example_gridlock.png, results/examples/gridlock.json
    python examples/gridlock.py --quick    # a 30 s smoke version

Each demand level is its own run (2 copies) with the simulator's default re-routing every 5 min from measured link speeds, warmed up
10 min so the city reaches its steady state, then measured for 10 min. With fixed routes instead, Oakland at calibrated demand slowly
fills up on its own; re-routing is what real drivers do.
  density   vehicles on the network per lane-km
  flow      vehicle-km driven per hour per lane-km (= density x mean speed)
  trips     completed per hour
Gridlock is resolved by the simulator's teleport rule (a vehicle stuck for minutes is removed); teleports are reported, since past
the peak they are part of how the city still "moves".
"""
import argparse
import time

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from examples import _common as C  # noqa: E402
except ImportError:
    import _common as C  # noqa: E402
from xnet import network as NW  # noqa: E402
from xnet import render as R  # noqa: E402
from xnet import sim as X  # noqa: E402

BG, INK, INK2, LINE = "#0d1117", "#e6edf3", "#8b949e", "#2f81f7"


def main(demands=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), copies=2, warm_s=600.0, seconds=600.0, area="oakland", n_slots=4096,
         out="docs/img/example_gridlock.png", save_name="gridlock"):
    t0 = time.time()
    net, geo, info = NW.build(area)
    lane_km = float((net["link_len"] * net["link_lanes"]).sum()) / 1000
    rows, states = [], []
    for dmd in demands:
        sim = X.Sim(net, n_slots=n_slots, n_envs=copies, beh=C.calibrated(p_av=0.0, demand=dmd), seed=7)
        sim.run(warm_s)
        m0, alive = np.asarray(sim.st.m), []
        for _ in range(int(seconds // 60)):
            sim.run(60.0)
            alive.append(float(np.asarray(sim.st.alive).sum(1).mean()))
        d = dict(zip(X.METRICS, (np.asarray(sim.st.m) - m0).sum(0) / copies))
        veh_km_h = d["veh_m"] / 1000 / (seconds / 3600)
        rows.append({"demand": dmd, "vehicles": float(np.mean(alive)), "density_per_lane_km": float(np.mean(alive)) / lane_km,
                     "flow_veh_km_per_h_per_lane_km": veh_km_h / lane_km, "speed_kmh": 3.6 * d["veh_m"] / max(d["veh_s"], 1),
                     "trips_per_hour": d["finished"] / (seconds / 3600), "trips_started_per_hour": d["spawned"] / (seconds / 3600),
                     "teleports_per_hour": d["teleports"] / (seconds / 3600), "blocked_arrivals_per_hour": d["blocked"] / (seconds / 3600)})
        states.append(jax.tree.map(lambda a: np.asarray(a[0]), sim.st))
        r = rows[-1]
        if r["vehicles"] > 0.9 * n_slots:
            print(f"  warning: {r['vehicles']:.0f} of {n_slots} vehicle slots in use: arrivals are being refused, density is capped", flush=True)
        print(f"  demand x{dmd:<4g} {r['vehicles']:6.0f} vehicles  {r['speed_kmh']:5.1f} km/h  trips/h {r['trips_per_hour']:6.0f}  "
              f"teleports/h {r['teleports_per_hour']:5.0f}  ({time.time() - t0:.0f} s)", flush=True)
    peak = int(np.argmax([r["flow_veh_km_per_h_per_lane_km"] for r in rows]))
    picks = (0, peak, len(rows) - 1)
    plot(rows, picks, states, net, geo, info, lane_km, area, out)
    C.save(save_name, {"area": area, "lane_km": lane_km, "copies": copies, "warm_s": warm_s, "seconds": seconds, "rows": rows,
                       "peak_demand": rows[peak]["demand"], "wall_s": time.time() - t0})
    print(f"peak flow at demand x{rows[peak]['demand']:g}: {rows[peak]['flow_veh_km_per_h_per_lane_km']:.0f} veh-km/h per lane-km at "
          f"{rows[peak]['speed_kmh']:.1f} km/h; at x{rows[-1]['demand']:g}: {rows[-1]['flow_veh_km_per_h_per_lane_km']:.0f} at "
          f"{rows[-1]['speed_kmh']:.1f} km/h ({time.time() - t0:.0f} s)")
    return rows


def plot(rows, picks, states, net, geo, info, lane_km, area, out):
    fig = plt.figure(figsize=(15, 8.6), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    a0, a1 = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2])
    k = [r["density_per_lane_km"] for r in rows]
    q = [r["flow_veh_km_per_h_per_lane_km"] for r in rows]
    for ax in (a0, a1):
        ax.set_facecolor(BG)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(color="#21262d")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
    a0.plot(k, q, color=LINE, marker="o")
    for r, x, y in zip(rows, k, q):
        a0.annotate(f"x{r['demand']:g}", (x, y), textcoords="offset points", xytext=(4, 6), color=INK2, fontsize=8)
    a0.set_xlabel("density: vehicles per lane-km", color=INK2, fontsize=9)
    a0.set_ylabel("flow: vehicle-km per hour per lane-km", color=INK2, fontsize=9)
    a0.set_title(f"The fundamental diagram of {area}: flow rises with density, saturates at capacity, then starts to fall (labels: demand x)",
                 color=INK, fontsize=10, loc="left")
    dm = [r["demand"] for r in rows]
    a1.plot(dm, [r["trips_started_per_hour"] for r in rows], color=INK2, ls="--", marker=".", label="trips started")
    a1.plot(dm, [r["trips_per_hour"] for r in rows], color=LINE, marker="o", label="trips completed")
    a1.plot(dm, [r["teleports_per_hour"] for r in rows], color="#f85149", marker="x", label="stuck vehicles removed (jam release)")
    a1.set_xlabel("demand multiplier", color=INK2, fontsize=9)
    a1.set_title("Trips per hour", color=INK, fontsize=10, loc="left")
    a1.legend(facecolor=BG, edgecolor="#30363d", labelcolor=INK, fontsize=8)
    mp = R.Map(net, geo, info)
    tracks, heads = R.Tracks(net, geo), R.signal_heads(net, geo)
    names = ("light traffic", "peak flow", "past the peak")
    for j, (i, nm) in enumerate(zip(picks, names)):
        ax = fig.add_subplot(gs[1, j])
        ax.set_facecolor(R.GROUND)
        mp.draw(ax, detail=False, labels=False)
        R.snapshot_artists(ax, net, geo, tracks, states[i], False, heads)
        ax.set_aspect("equal")
        ax.axis("off")
        r = rows[i]
        ax.set_title(f"{nm}: demand x{r['demand']:g}, {r['vehicles']:.0f} vehicles, {r['speed_kmh']:.0f} km/h", color=INK, fontsize=9, loc="left")
    fig.text(0.01, 0.005, "Vehicles coloured by speed relative to the limit (blue fast, red stopped).", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=105, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        main(demands=(0.5, 2.0), copies=1, warm_s=60, seconds=60, n_slots=1024, out="/tmp/example_gridlock_quick.png", save_name="gridlock_quick")
    else:
        main()
