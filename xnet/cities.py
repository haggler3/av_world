"""More cities: Midtown Manhattan, the Chicago Loop and central Boston, from the same open data as Pittsburgh (Overture Maps,
built largely from OpenStreetMap). Each is about the size of the Oakland area.

    from xnet import cities
    net, geo, info = cities.build("new_york")            # also "chicago", "boston"; av_world.make("new_york") works too
    python -m xnet.cities fetch                           # refresh data/cities/*.parquet from Overture's public bucket (~10 s each)

The road segments are bundled with the package (av_world/_data), so nothing needs fetching. The metre grid is centred on each city.
"""
import os
import sys

import pyarrow.parquet as pq

from pgh import mine

from . import network as NW
from . import osm

CITIES = {  # (west, south, east, north)
    "new_york": dict(bbox=(-73.997, 40.748, -73.972, 40.763), name="Midtown Manhattan, New York (Times Square to Grand Central)"),
    "chicago": dict(bbox=(-87.640, 41.874, -87.618, 41.889), name="the Loop, Chicago"),
    "boston": dict(bbox=(-71.078, 42.345, -71.055, 42.360), name="Back Bay to Downtown Crossing, Boston"),
}
COLS = ["id", "names", "class", "subclass", "connectors", "access_restrictions", "speed_limits", "geometry", "bbox"]


def path(name):
    """The repository's data/cities file if present, else the copy bundled with the package."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cities", f"{name}_segments.parquet")
    if os.path.exists(p):
        return p
    from av_world import data_path
    return data_path(f"{name}_segments.parquet")


def rows(name):
    w, s, e, n = CITIES[name]["bbox"]
    rs = pq.read_table(path(name), columns=COLS).to_pylist()
    return [r for r in rs if r["class"] in mine.DRIVE and r["subclass"] != "link" and r["bbox"]["xmax"] >= w and r["bbox"]["xmin"] <= e
            and r["bbox"]["ymax"] >= s and r["bbox"]["ymin"] <= n]


def build(name, **kw):
    """-> (net, geo, info) like xnet.network.build, for a city in CITIES."""
    bbox = CITIES[name]["bbox"]
    old = (mine.LAT0, mine.LON0)
    origin = ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
    osm.set_origin(*origin)
    try:
        net, geo, info = NW.build(bbox, rows=rows(name), **kw)
    finally:
        osm.set_origin(*old)
    return net, geo, {**info, "city": CITIES[name]["name"], "origin_lat_lon": origin}


def fetch(names=None):
    from pgh import fetch as F
    os.makedirs("data/cities", exist_ok=True)
    for n in names or CITIES:
        F.fetch("segment", f"data/cities/{n}_segments.parquet", CITIES[n]["bbox"])


def bundle():
    """Trim each city's file to the columns and rows the builder reads, into av_world/_data."""
    import pyarrow as pa
    for n in CITIES:
        keep = {r["id"] for r in rows(n)}
        t = pq.read_table(f"data/cities/{n}_segments.parquet", columns=COLS)
        t = t.filter(pa.array([i in keep for i in t["id"].to_pylist()]))
        pq.write_table(t, f"av_world/_data/{n}_segments.parquet", compression="zstd")
        print(n, t.num_rows, "segments")




def snapshot(name, out, seconds=300.0, seed=0):
    """Run the city for `seconds` (calibrated humans, 25 % AVs) and draw every vehicle on the map."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from . import render as R
    from . import sim as X
    net, geo, info = build(name)
    from av_world import data_path
    knobs = json.load(open(data_path("calibration.json")))["calibrated"]
    sim = X.Sim(net, n_slots=1024, n_envs=1, beh=X.Behaviour(**knobs, p_av=0.25), seed=seed)
    sim.run(seconds)
    st = __import__("jax").tree.map(lambda a: np.asarray(a[0]), sim.st)
    fig, ax = plt.subplots(figsize=(8.4, 6.6), facecolor=R.BG)
    fig.subplots_adjust(0.01, 0.06, 0.99, 0.92)
    ax.set_facecolor(R.GROUND)
    R.Map(net, geo, info).draw(ax, detail=False, labels=True)
    R.snapshot_artists(ax, net, geo, R.Tracks(net, geo), st, False, R.signal_heads(net, geo))
    xs = np.concatenate([p[:, 0] for p in geo.link_pts])
    ys = np.concatenate([p[:, 1] for p in geo.link_pts])
    ax.set_xlim(xs.min() - 20, xs.max() + 20)
    ax.set_ylim(ys.min() - 20, ys.max() + 20)
    ax.set_aspect("equal")
    ax.axis("off")
    nv = int(st.alive.sum())
    fig.text(0.015, 0.95, f"{CITIES[name]['name']}: every vehicle after {seconds / 60:.0f} simulated minutes", color="#e6edf3", fontsize=11, weight="bold")
    fig.text(0.015, 0.015, f"{nv} vehicles ({int(st.av[st.alive].sum())} AVs) on {len(net['link_len'])} links, {int(net['node_signal'].sum())} signals; "
             "colour = speed vs limit (blue fast ... red stopped). Map: Overture Maps / OpenStreetMap; lanes, signals and demand inferred",
             color="#8b949e", fontsize=7)
    fig.savefig(out, dpi=90, facecolor=R.BG)
    plt.close(fig)
    return out


def survey(n_envs=8, warm=300.0, seconds=600.0):
    """Every city (and Oakland for reference): network size, throughput, and crashes by type with calibrated humans and 25 % rule-based AVs."""
    import json
    import time

    import numpy as np

    from av_world import data_path

    from . import sim as X
    knobs = json.load(open(data_path("calibration.json")))["calibrated"]
    beh = X.Behaviour(**knobs, p_av=0.25)
    out = {}
    for name in ["oakland"] + list(CITIES):
        net, geo, info = NW.build(name) if name == "oakland" else build(name)
        sim = X.Sim(net, n_envs=n_envs, beh=beh, seed=1, reroute_s=300)
        sim.run(warm)
        m0 = np.asarray(sim.st.m).sum(0)
        t0 = time.time()
        sim.run(seconds)
        dt = time.time() - t0
        d = dict(zip(X.METRICS, np.asarray(sim.st.m).sum(0) - m0))
        hours = n_envs * seconds / 3600
        out[name] = {"links": int(len(net["link_len"])), "junction_moves": int(len(net["mov_len"])), "signals": int(net["node_signal"].sum()),
                     "crosswalks": int(len(net["leg_node"])), "km_of_lanes": float((net["link_len"] * net["link_lanes"]).sum() / 1000),
                     "vehicles_mean": float(d["veh_s"] / (n_envs * seconds)), "mean_speed_kmh": 3.6 * d["veh_m"] / max(d["veh_s"], 1),
                     "simulated_s_per_s": n_envs * seconds / dt, "trips_per_area_hour": d["finished"] / hours,
                     "crashes_per_area_hour": {c: d[c] / hours for c in X.CRASH_NAMES}, "av_crashes": d["av_crash"], "av_hours": d["av_veh_s"] / 3600}
        print(name, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out[name].items() if k != "crashes_per_area_hour"}, flush=True)
    os.makedirs("results/cities", exist_ok=True)
    json.dump(out, open("results/cities/survey.json", "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "snapshots":
        for n in CITIES:
            print(snapshot(n, f"docs/img/city_{n}.png"))
    else:
        {"fetch": fetch, "bundle": bundle, "survey": survey}[cmd]()
