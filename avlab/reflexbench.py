"""Reflex sensor + 2.0 s delayed view (avlab.reflex), envelope off. Pre-registration: results/daily/2026-11-27/predictions.md

    python -m avlab.reflexbench --quick         # 2 copies, 40 s (smoke test)
    python -m avlab.reflexbench --cell rf10     # one cell -> results/reflex/rf10.json (~3 min)
    python -m avlab.reflexbench                 # tests over the saved cells -> results/reflex/reflex.json
"""
import argparse
import json
import os

OUT = "results/reflex"
CELLS = {"rf0": 0.0, "rf5": 5.0, "rf10": 10.0, "rf20": 20.0}
REF = {"comp2.0": "results/latency/comp2.0.json", "clean": "results/latency/clean.json"}
TESTS = {"H1": ("rf10", "comp2.0"), "S1": ("rf20", "rf10"), "S2": ("rf10", "clean")}


def load():
    cells = {c: json.load(open(f"{OUT}/{c}.json")) for c in CELLS if os.path.exists(f"{OUT}/{c}.json")}
    cells.update({c: json.load(open(p)) for c, p in REF.items()})
    return cells


def analyse(cells):
    from avlab.sensorbench import _poisson_ci
    from avlab.whocrashes import holm, rate_ratio
    h = lambda c: sum(cells[c]["av_veh_s"]) / 3600  # noqa: E731
    k = lambda c: sum(cells[c]["av_crash"])  # noqa: E731
    rows = {c: {"range_m": CELLS.get(c), "av_hours": h(c), "av_crash": k(c), "rate_per_h": k(c) / h(c),
                "ci": [x / h(c) for x in _poisson_ci(k(c))],
                "av_speed_kmh": 3.6 * sum(cells[c]["av_veh_m"]) / sum(cells[c]["av_veh_s"]),
                "av_hard": sum(cells[c]["av_hard"]), "city_crashes": sum(cells[c]["crashes"])} for c in cells}
    tests = {n: {"cells": [a, b], **rate_ratio(k(a), h(a), k(b), h(b))} for n, (a, b) in TESTS.items()
             if a in cells and b in cells}
    sec = [n for n in ("S1", "S2") if n in tests]
    for n, p in zip(sec, holm([tests[n]["p"] for n in sec])):
        tests[n]["p_holm"] = p
    check = None
    if "rf0" in cells:
        check = {"rf0_equals_comp2.0": all(cells["rf0"][f] == cells["comp2.0"][f] for f in ("av_crash", "av_veh_s", "crashes"))}
    return {"cells": rows, "tests": tests, "check": check}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cell")
    a = ap.parse_args()
    from avlab import reflex as RF
    if a.quick:
        for c in ("rf0", "rf10"):
            r = RF.run(RF.Reflex(range_m=CELLS[c]), E=2, warm_s=40, seconds=40)
            print(c, "AV crashes", sum(r["av_crash"]), "AV-hours %.2f" % (sum(r["av_veh_s"]) / 3600))
        return
    os.makedirs(OUT, exist_ok=True)
    if a.cell:
        r = RF.run(RF.Reflex(range_m=CELLS[a.cell]))
        r["cell"] = a.cell
        json.dump(r, open(f"{OUT}/{a.cell}.json", "w"))
        print(a.cell, "AV crashes", sum(r["av_crash"]), "AV-hours %.1f" % (sum(r["av_veh_s"]) / 3600), flush=True)
        return
    res = analyse(load())
    json.dump(res, open(f"{OUT}/reflex.json", "w"), indent=1)
    for c, r in res["cells"].items():
        print(f"{c:8s} {r['av_hours']:7.1f} h {r['av_crash']:5.0f} crashes {r['rate_per_h']:.3f}/h  {r['av_speed_kmh']:.2f} km/h")
    for n, t in res["tests"].items():
        print(n, t)
    print("check", res["check"])


if __name__ == "__main__":
    main()
