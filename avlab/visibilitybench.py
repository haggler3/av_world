"""Limited AV sight distance and the phantom obstacle (avlab.visibility), envelope off.

Pre-registration: results/daily/2026-11-23/predictions.md

    python -m avlab.visibilitybench --quick          # 2 copies, 40 s, two cells (a smoke test)
    python -m avlab.visibilitybench --cell blind8    # one cell -> results/visibility/blind8.json (~2 min)
    python -m avlab.visibilitybench                  # analysis of the saved cells -> results/visibility/visibility.json
"""
import argparse
import json
import os

OUT = "results/visibility"
SEED = 44
CELLS = {"clear": (1e9, False), **{f"blind{s}": (float(s), False) for s in (6, 8, 10, 14)},
         **{f"phantom{s}": (float(s), True) for s in (8, 14)}}
TESTS = {"H1": ("phantom8", "blind8"), "S1": ("blind8", "clear"), "S2": ("phantom8", "clear")}


def analyse(cells):
    from avlab.sensorbench import _poisson_ci
    from avlab.whocrashes import holm, rate_ratio
    h = lambda c: sum(cells[c]["av_veh_s"]) / 3600  # noqa: E731
    k = lambda c: sum(cells[c]["av_crash"])  # noqa: E731
    rows = {c: {"sight_m": CELLS[c][0], "phantom": CELLS[c][1], "av_hours": h(c), "av_crash": k(c),
                "rate_per_h": k(c) / h(c), "ci": [x / h(c) for x in _poisson_ci(k(c))],
                "av_speed_kmh": 3.6 * sum(cells[c]["av_veh_m"]) / sum(cells[c]["av_veh_s"]),
                "av_km": sum(cells[c]["av_veh_m"]) / 1000, "rate_per_1000km": 1000 * k(c) / (sum(cells[c]["av_veh_m"]) / 1000),
                "av_hard": sum(cells[c]["av_hard"]), "city_crashes": sum(cells[c]["crashes"])} for c in cells}
    tests = {}
    for name, (a, b) in TESTS.items():
        if a in cells and b in cells:
            tests[name] = {"cells": [a, b], **rate_ratio(k(a), h(a), k(b), h(b))}
    sec = [n for n in ("S1", "S2") if n in tests]
    if sec:
        for n, p in zip(sec, holm([tests[n]["p"] for n in sec])):
            tests[n]["p_holm"] = p
    return {"cells": rows, "tests": tests}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cell")
    a = ap.parse_args()
    from avlab import visibility as VI
    if a.quick:
        for c in ("blind8", "phantom8"):
            r = VI.run(VI.Visibility(*CELLS[c]), E=2, warm_s=40, seconds=40)
            print(c, "AV crashes", sum(r["av_crash"]), "AV-hours %.2f" % (sum(r["av_veh_s"]) / 3600))
        return
    os.makedirs(OUT, exist_ok=True)
    if a.cell:
        r = VI.run(VI.Visibility(*CELLS[a.cell]), seed=SEED)
        r["cell"] = a.cell
        json.dump(r, open(f"{OUT}/{a.cell}.json", "w"))
        print(a.cell, "AV crashes", sum(r["av_crash"]), "AV-hours %.1f" % (sum(r["av_veh_s"]) / 3600), flush=True)
        return
    cells = {c: json.load(open(f"{OUT}/{c}.json")) for c in CELLS if os.path.exists(f"{OUT}/{c}.json")}
    res = analyse(cells)
    json.dump(res, open(f"{OUT}/visibility.json", "w"), indent=1)
    for c, r in res["cells"].items():
        print(f"{c:9s} {r['av_hours']:7.1f} h {r['av_crash']:6.0f} crashes {r['rate_per_h']:.3f}/h  {r['av_speed_kmh']:.2f} km/h")
    for n, t in res["tests"].items():
        print(n, t)


if __name__ == "__main__":
    main()
