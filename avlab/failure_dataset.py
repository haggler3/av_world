"""The failure-log dataset across cities: index, cross-city rate table, exports, figure and an atlas GIF.

    python -m avlab.failurelog --cities all --cond dry wet      # 1. simulate and cut scenarios -> data/failures/
    python -m avlab.failure_dataset index                        # 2. index + rates -> results/failures/{index.parquet, rates.json, rates.md}
    python -m avlab.failure_dataset export --formats all         # 3. every scenario in every format -> data/failures_export/<format>/<cond>/<city>/
    python -m avlab.failure_dataset export --per-city 2          #    or a sample per (city, condition)
    python -m avlab.failure_dataset sample                       # 4. the small sample pack kept in git -> results/failures/sample/
    python -m avlab.failure_dataset figures                      # 5. docs/img/failures_rates.png, docs/img/failures_atlas.gif

Cross-comparable means: every city ran with the same drivers, AV share, AV policy, warm-up, copies and duration, every crash
is cut with the same window and radius, and every rate has exposure (vehicle-hours, AV-hours) as its denominator, with an
exact Poisson 95 % interval.
"""
import argparse
import glob
import json
import os

import numpy as np

from avlab import failure_export as FX
from avlab import failurelog as F

ROOT = "data/failures"
OUT = "results/failures"
TYPES = F.CRASH_TYPES
SHORT = {"oakland": "Pittsburgh, Oakland", "squirrel_hill": "Pittsburgh, Squirrel Hill", "downtown": "Pittsburgh, Downtown",
         "new_york": "New York, Midtown", "chicago": "Chicago, Loop", "boston": "Boston, Back Bay", "san_francisco": "San Francisco, SoMa",
         "phoenix": "Phoenix, downtown", "austin": "Austin, downtown", "los_angeles": "Los Angeles, downtown", "paris": "Paris, Opéra"}


def _ci(k):
    from scipy.stats import chi2
    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    return lo, chi2.ppf(0.975, 2 * k + 2) / 2


def scenario_paths(root=ROOT, cond="*", city="*"):
    return sorted(glob.glob(f"{root}/{cond}/{city}/*.json.gz"))


def index(root=ROOT, out=OUT):
    import pandas as pd
    os.makedirs(out, exist_ok=True)
    rows = [F.index_row(F.load(p)) | {"path": os.path.relpath(p, root)} for p in scenario_paths(root)]
    df = pd.DataFrame(rows)
    df.to_parquet(f"{out}/index.parquet", index=False)
    runs = [json.load(open(p)) for p in sorted(glob.glob(f"{root}/*/*/_run.json"))]
    rates = []
    for r in runs:
        sub = df[(df.city == r["city"]) & (df.condition == r["cond"])] if len(df) else df
        vh, ah = sum(r["veh_s"]) / 3600, sum(r["av_veh_s"]) / 3600
        k, ka = len(sub), int(sub.av_involved.sum()) if len(sub) else 0
        lo, hi = _ci(k)
        alo, ahi = _ci(ka)
        rates.append(dict(city=r["city"], name=r["name"], condition=r["cond"], veh_hours=round(vh, 1), av_hours=round(ah, 1),
                          crashes=k, per_1000_veh_h=round(1000 * k / vh, 2), ci=[round(1000 * lo / vh, 2), round(1000 * hi / vh, 2)],
                          av_crashes=ka, av_per_1000_av_h=round(1000 * ka / ah, 2) if ah else None,
                          av_ci=[round(1000 * alo / ah, 2), round(1000 * ahi / ah, 2)] if ah else None,
                          types={t: int((sub.type == t).sum()) for t in TYPES} if len(sub) else {},
                          median_impact_kmh=round(float(sub.impact_speed_mps.median()) * 3.6, 1) if len(sub) else None,
                          mean_speed_kmh=round(3.6 * sum(r["veh_m"]) / sum(r["veh_s"]), 2),
                          sim_crashes=round(sum(r["crashes_by_type"].values())), sim_types={t: round(v) for t, v in r["crashes_by_type"].items()},
                          type_mismatch=int(sum(abs((int((sub.type == t).sum()) if len(sub) else 0) - round(r["crashes_by_type"].get(t, 0)))
                                                for t in TYPES)), scenarios_cut=r["scenarios"],
                          lost_at_edges=round(sum(r["crashes_by_type"].values())) - r["scenarios"]))
    ratio = []
    for c in sorted({r["city"] for r in rates}):
        d = {r["condition"]: r for r in rates if r["city"] == c}
        if "dry" in d and "wet" in d:
            from avlab.whocrashes import rate_ratio
            rr = rate_ratio(d["wet"]["crashes"], d["wet"]["veh_hours"], d["dry"]["crashes"], d["dry"]["veh_hours"])
            ratio.append(dict(city=c, wet_over_dry=round(rr["rr"], 2), ci=[round(float(rr["lo"]), 2), round(float(rr["hi"]), 2)],
                              p=float(rr["p"])))
    res = {"settings": F.BASE | {"conditions": F.CONDITIONS, "window_s": [F.PRE_S, F.POST_S], "track_radius_m": F.TRACK_R,
                                 "map_radius_m": F.MAP_R},
           "rates": rates, "wet_over_dry": ratio}
    json.dump(res, open(f"{out}/rates.json", "w"), indent=1)
    md = ["| city | condition | veh-h | crashes | per 1000 veh-h (95 %) | AV crashes | per 1000 AV-h (95 %) | median impact km/h | "
          "rear-end / angle / left-turn / sideswipe / merge / pedestrian / bicycle |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rates, key=lambda r: (r["condition"], -r["per_1000_veh_h"])):
        ty = " / ".join(str(r["types"].get(t, 0)) for t in TYPES)
        md.append(f"| {r['name']} | {r['condition']} | {r['veh_hours']:.0f} | {r['crashes']} | {r['per_1000_veh_h']:.1f} "
                  f"({r['ci'][0]:.1f}-{r['ci'][1]:.1f}) | {r['av_crashes']} | {r['av_per_1000_av_h']:.1f} ({r['av_ci'][0]:.1f}-{r['av_ci'][1]:.1f}) "
                  f"| {r['median_impact_kmh']} | {ty} |")
    open(f"{out}/rates.md", "w").write("\n".join(md) + "\n")
    return res


def export(formats, root=ROOT, out="data/failures_export", per_city=None, cities=None):
    """Every (or `per_city` of every) scenario in each format; plus one Waymo TFRecord shard per (condition, city) when the
    Scenario proto is available."""
    pb2 = FX.scenario_pb2_module()
    n = 0
    for cond_dir in sorted(glob.glob(f"{root}/*")):
        cond = os.path.basename(cond_dir)
        for city_dir in sorted(glob.glob(f"{cond_dir}/*")):
            city = os.path.basename(city_dir)
            if cities and city not in cities:
                continue
            ps = sorted(glob.glob(f"{city_dir}/*.json.gz"))[:per_city]
            scs = [F.load(p) for p in ps]
            for sc in scs:
                FX.export(sc, f"{out}/{cond}/{city}", formats=[f for f in formats if f != "waymo_tfrecord"], mapdir=f"{out}/maps")
                n += 1
            if "waymo_tfrecord" in formats and pb2 is not None and scs:
                os.makedirs(f"{out}/{cond}/{city}/waymo_tfrecord", exist_ok=True)
                FX.to_waymo_tfrecord(scs, f"{out}/{cond}/{city}/waymo_tfrecord/{city}_{cond}.tfrecord", pb2)
    return n


def figures(res=None, root=ROOT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, FG, MUTED, GRID = "#0d1117", "#e6edf3", "#8b949e", "#30363d"
    TCOL = {"rear-end": "#56B4E9", "angle": "#E69F00", "left-turn": "#D55E00", "sideswipe": "#8b949e", "merge": "#CC79A7",
            "pedestrian": "#009E73", "bicycle": "#F0E442"}
    res = res or json.load(open(f"{OUT}/rates.json"))
    rates = res["rates"]
    dry = sorted([r for r in rates if r["condition"] == "dry"], key=lambda r: r["per_1000_veh_h"])
    wet = {r["city"]: r for r in rates if r["condition"] == "wet"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=BG, gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax, bx = axs
    for i, r in enumerate(dry):
        ax.plot(r["ci"], [i + 0.12] * 2, color="#56B4E9", lw=1.4)
        ax.plot(r["per_1000_veh_h"], i + 0.12, "o", color="#56B4E9", ms=6)
        w = wet.get(r["city"])
        if w:
            ax.plot(w["ci"], [i - 0.12] * 2, color="#E69F00", lw=1.4)
            ax.plot(w["per_1000_veh_h"], i - 0.12, "s", color="#E69F00", ms=5)
        left = 0.0
        tot = max(sum(r["types"].values()), 1)
        for t in TYPES:
            if t == "sideswipe":
                continue
            v = r["types"].get(t, 0) / tot
            bx.barh(i, v, left=left, color=TCOL[t], height=0.6)
            left += v
        bx.text(1.02, i, f"{sum(r['types'].values())}", color=MUTED, fontsize=7.5, va="center")
    names = [SHORT.get(r["city"], r["city"]) for r in dry]
    for z in axs:
        z.set_yticks(range(len(dry)))
        z.set_facecolor(BG)
        z.tick_params(colors=MUTED, labelsize=8)
        for s in z.spines.values():
            s.set_color(GRID)
    ax.set_yticklabels(names, color=FG, fontsize=8)
    bx.set_yticklabels([])
    ax.set_xscale("log")
    ax.set_xlabel("crashes per 1000 vehicle-hours (95 % interval, log scale)", color=MUTED, fontsize=8)
    ax.text(0.02, 0.98, "circle = dry (friction 1.0)\nsquare = wet (friction 0.5)", transform=ax.transAxes, color=MUTED, fontsize=7.5, va="top")
    bx.set_xlim(0, 1.12)
    bx.set_xticks([0, 0.5, 1])
    bx.set_xticklabels(["0", "50 %", "100 %"])
    bx.set_xlabel("crash types, dry (number = scenarios)", color=MUTED, fontsize=8)
    x = 0.0
    for t in TYPES:
        if t == "sideswipe":
            continue
        bx.text(x, len(dry) - 0.25, t, color=TCOL[t], fontsize=7, va="bottom")
        x += 0.175
    s = res["settings"]
    fig.text(0.01, 0.96, "Failure logs across cities: the same drivers, AVs and settings everywhere", color=FG, fontsize=12, weight="bold")
    fig.text(0.01, 0.915, f"{s['E']} copies x {s['seconds']:.0f} s per city and condition, {int(s['p_av'] * 100)} % rule-based AVs with the "
             "envelope, calibrated humans. Each crash is cut as an 11 s scenario and exported to 9 formats.", color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.84, bottom=0.1, wspace=0.08)
    fig.savefig("docs/img/failures_rates.png", dpi=110, facecolor=BG)
    plt.close(fig)
    atlas(root)


def atlas(root=ROOT, path="docs/img/failures_atlas.gif", n=6):
    """A grid of BEV replays, one AV-involved crash per city (the first one with an AV, else any), dry condition."""
    from PIL import Image, ImageDraw, ImageFont
    import pandas as pd
    df = pd.read_parquet(f"{OUT}/index.parquet")
    df = df[df.condition == "dry"]
    picks = []
    for city in df.city.unique():
        sub = df[df.city == city].sort_values(["av_involved", "n_tracks"], ascending=[False, False])
        picks.append(sub.iloc[0])
    picks = sorted(picks, key=lambda r: -r.n_tracks)[:n]
    tiles = []
    for r in picks:
        sc = F.load(os.path.join(root, r.path))
        arr = FX.bev(sc, size=128, res=0.5, every=2)
        tiles.append((sc, arr))
    T = min(a.shape[0] for _, a in tiles)
    cols = {"lanes": (48, 54, 61), "intersections": (68, 76, 86), "crosswalks": (110, 118, 129), "red_stop_lines": (255, 77, 77),
            "vehicles": (86, 180, 233), "ego": (230, 159, 0), "pedestrians": (0, 158, 115)}
    W, H, cw = 3, (len(tiles) + 2) // 3, 200
    frames = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = big = ImageFont.load_default()
    for k in range(T):
        canvas = Image.new("RGB", (W * cw + (W + 1) * 8, H * (cw + 34) + 52), (13, 17, 23))
        d = ImageDraw.Draw(canvas)
        d.text((8, 6), "Failure atlas: one crash per city, same settings everywhere", fill=(230, 237, 243), font=big)
        d.text((8, 27), f"t = {k * 0.2 - 8.0:+.1f} s from the crash   orange = ego (AV if involved)   blue = others   green = pedestrians",
               fill=(139, 148, 158), font=font)
        for i, (sc, arr) in enumerate(tiles):
            f = arr[k]
            img = np.zeros((*f.shape[1:], 3), np.float32)
            img[:] = (13, 17, 23)
            for c, name in enumerate(FX.BEV_CH[:-1]):
                m = f[c][..., None] / 255.0
                img = img * (1 - m) + np.array(cols[name]) * m
            im = Image.fromarray(img.astype(np.uint8)).resize((cw, cw), Image.NEAREST)
            x0, y0 = 8 + (i % W) * (cw + 8), 52 + (i // W) * (cw + 34)
            canvas.paste(im, (x0, y0 + 18))
            ev = sc["event"]
            lab = f"{SHORT.get(sc['city']['key'], sc['city']['key']).replace('Pittsburgh, ', 'Pgh ').split(',')[0]}: {ev['type']}" + (" (AV)" if ev["av_involved"] else "")
            d.text((x0, y0 + 2), lab, fill=(230, 237, 243), font=font)
            if k * 2 >= sc["event_index"]:
                d.rectangle([x0, y0 + 18, x0 + cw - 1, y0 + 18 + cw - 1], outline=(255, 77, 77), width=3)
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    return path


def sample(root=ROOT, out=f"{OUT}/sample", picks=None):
    """The small sample pack kept in git: a few scenarios in every format (city maps for OpenSCENARIO / CARLA are not kept:
    `python -m avlab.failure_dataset maps` writes them)."""
    import pandas as pd
    import shutil
    df = pd.read_parquet(f"{OUT}/index.parquet")
    if picks is None:
        picks = []
        for city, typ in (("san_francisco", None), ("paris", "pedestrian"), ("oakland", "rear-end")):
            sub = df[(df.city == city) & (df.condition == "dry")]
            if typ:
                sub = sub[sub.type == typ] if (sub.type == typ).any() else sub
            sub = sub.sort_values("av_involved", ascending=False)
            if len(sub):
                picks.append(sub.iloc[0].path)
    if os.path.exists(out):
        shutil.rmtree(out)
    for p in picks:
        sc = F.load(os.path.join(root, p))
        os.makedirs(f"{out}/canonical", exist_ok=True)
        shutil.copy(os.path.join(root, p), f"{out}/canonical/")
        FX.export(sc, out, mapdir="data/failures_export/maps")
        pb2 = FX.scenario_pb2_module()
        if pb2 is not None:
            os.makedirs(f"{out}/waymo_tfrecord", exist_ok=True)
            FX.to_waymo_tfrecord([sc], f"{out}/waymo_tfrecord/{sc['id']}.tfrecord", pb2)
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "export", "sample", "figures", "maps"])
    ap.add_argument("--formats", nargs="+", default=["all"])
    ap.add_argument("--per-city", type=int, default=None)
    ap.add_argument("--cities", nargs="+", default=None)
    a = ap.parse_args()
    fm = list(FX.FORMATS) + ["waymo_tfrecord"] if a.formats == ["all"] else a.formats
    if a.cmd == "index":
        res = index()
        for r in res["rates"]:
            print(r["city"], r["condition"], r["crashes"], r["per_1000_veh_h"], r["ci"], r["av_crashes"], r["types"])
        for r in res["wet_over_dry"]:
            print(r)
    elif a.cmd == "export":
        print(export(fm, per_city=a.per_city, cities=a.cities), "scenarios exported")
    elif a.cmd == "sample":
        print(sample())
    elif a.cmd == "figures":
        figures()
    elif a.cmd == "maps":
        for c in F.all_cities():
            print(FX.city_xodr(c, "data/failures_export/maps"))


if __name__ == "__main__":
    main()
