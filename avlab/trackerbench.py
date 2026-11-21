"""Bursty missed detections and a hold-last tracker for the IDM AV (avlab.tracker), envelope off.

Pre-registration: results/daily/2026-11-21/predictions.md

    python -m avlab.trackerbench --quick            # 2 copies, 40 s, two cells (a smoke test)
    python -m avlab.trackerbench --cell iid_hold    # one cell -> results/tracker/iid_hold.json (~2 min)
    python -m avlab.trackerbench                    # analysis of the saved cells -> results/tracker/tracker.json
"""
import argparse
import json
import os

OUT = "results/tracker"
CELLS = {"clean": (0.0, 0.2, 0.0), "iid": (0.15, 0.2, 0.0), "iid_hold": (0.15, 0.2, 1.0), "burst": (0.15, 1.0, 0.0),
         "burst_hold": (0.15, 1.0, 1.0), "burst_hold2": (0.15, 1.0, 2.0)}
TESTS = {"H1": ("iid_hold", "iid"), "S1": ("burst", "iid"), "S2": ("burst_hold", "burst")}


def analyse(cells):
    from avlab.sensorbench import _poisson_ci
    from avlab.whocrashes import holm, rate_ratio
    h = lambda c: sum(cells[c]["av_veh_s"]) / 3600  # noqa: E731
    k = lambda c: sum(cells[c]["av_crash"])  # noqa: E731
    rows = {c: {"av_hours": h(c), "av_crash": k(c), "rate_per_h": k(c) / h(c),
                "ci": [x / h(c) for x in _poisson_ci(k(c))],
                "av_speed_kmh": 3.6 * sum(cells[c]["av_veh_m"]) / sum(cells[c]["av_veh_s"]),
                "city_crashes": sum(cells[c]["crashes"])} for c in cells}
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
    from avlab import tracker as TR
    if a.quick:
        for c in ("iid", "iid_hold"):
            r = TR.run(TR.Perception(*CELLS[c]), E=2, warm_s=40, seconds=40)
            print(c, "AV crashes", sum(r["av_crash"]), "AV-hours %.2f" % (sum(r["av_veh_s"]) / 3600))
        return
    os.makedirs(OUT, exist_ok=True)
    if a.cell:
        r = TR.run(TR.Perception(*CELLS[a.cell]))
        r["cell"] = a.cell
        json.dump(r, open(f"{OUT}/{a.cell}.json", "w"))
        print(a.cell, "AV crashes", sum(r["av_crash"]), "AV-hours %.1f" % (sum(r["av_veh_s"]) / 3600))
        return
    cells = {c: json.load(open(f"{OUT}/{c}.json")) for c in CELLS if os.path.exists(f"{OUT}/{c}.json")}
    res = analyse(cells)
    json.dump(res, open(f"{OUT}/tracker.json", "w"), indent=1)
    for c, r in res["cells"].items():
        print(f"{c:12s} {r['av_hours']:7.1f} h {r['av_crash']:6.0f} crashes {r['rate_per_h']:.3f}/h  {r['av_speed_kmh']:.2f} km/h")
    for n, t in res["tests"].items():
        print(n, t)


if __name__ == "__main__":
    main()
