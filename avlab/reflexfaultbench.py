"""Imperfect reflex (avlab.reflexfault) + 2.0 s delayed view, envelope off. Pre-registration:
results/daily/2026-11-28/predictions.md

    python -m avlab.reflexfaultbench --quick          # 2 copies, 40 s (smoke test)
    python -m avlab.reflexfaultbench --cell fb        # one cell -> results/reflexfault/fb.json (~3-5 min)
    python -m avlab.reflexfaultbench                  # tests over the saved cells -> results/reflexfault/reflexfault.json
"""
import argparse
import json
import os

OUT = "results/reflexfault"
CELLS = {"fb": dict(p_miss=0.15, burst_s=1.0), "fi": dict(p_miss=0.15, burst_s=0.2), "fn": dict(sigma=0.2)}
REF = {"rf10": "results/reflex/rf10.json", "comp2.0": "results/latency/comp2.0.json", "clean": "results/latency/clean.json"}
H1 = ("fb", "rf10")
SEC = {"S1": ("fi", "rf10"), "S2": ("fn", "rf10"), "S3": ("fb", "comp2.0")}


def load():
    cells = {c: json.load(open(f"{OUT}/{c}.json")) for c in CELLS if os.path.exists(f"{OUT}/{c}.json")}
    cells.update({c: json.load(open(p)) for c, p in REF.items()})
    return cells


def analyse(cells):
    from avlab.sensorbench import _poisson_ci
    from avlab.whocrashes import holm, rate_ratio
    h = lambda c: sum(cells[c]["av_veh_s"]) / 3600  # noqa: E731
    k = lambda c: sum(cells[c]["av_crash"])  # noqa: E731
    rows = {c: {"model": CELLS.get(c), "av_hours": h(c), "av_crash": k(c), "rate_per_h": k(c) / h(c),
                "ci": [x / h(c) for x in _poisson_ci(k(c))],
                "av_speed_kmh": 3.6 * sum(cells[c]["av_veh_m"]) / sum(cells[c]["av_veh_s"]),
                "av_hard": sum(cells[c]["av_hard"]), "city_crashes": sum(cells[c]["crashes"])} for c in cells}
    tests = {n: {"cells": [a, b], **rate_ratio(k(a), h(a), k(b), h(b))}
             for n, (a, b) in {"H1": H1, **SEC}.items() if a in cells and b in cells}
    sec = [n for n in SEC if n in tests]
    for n, p in zip(sec, holm([tests[n]["p"] for n in sec])):
        tests[n]["p_holm"] = p
    return {"cells": rows, "tests": tests}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cell")
    a = ap.parse_args()
    from avlab import reflexfault as RX
    if a.quick:
        for c in ("fb", "fn"):
            r = RX.run(RX.Fault(**CELLS[c]), E=2, warm_s=40, seconds=40)
            print(c, "AV crashes", sum(r["av_crash"]), "AV-hours %.2f" % (sum(r["av_veh_s"]) / 3600))
        return
    os.makedirs(OUT, exist_ok=True)
    if a.cell:
        r = RX.run(RX.Fault(**CELLS[a.cell]))
        r["cell"] = a.cell
        json.dump(r, open(f"{OUT}/{a.cell}.json", "w"))
        print(a.cell, "AV crashes", sum(r["av_crash"]), "AV-hours %.1f" % (sum(r["av_veh_s"]) / 3600), flush=True)
        return
    res = analyse(load())
    json.dump(res, open(f"{OUT}/reflexfault.json", "w"), indent=1)
    for c, r in res["cells"].items():
        print(f"{c:8s} {r['av_hours']:7.1f} h {r['av_crash']:5.0f} crashes {r['rate_per_h']:.3f}/h  {r['av_speed_kmh']:.2f} km/h")
    for n, t in res["tests"].items():
        print(n, t)


if __name__ == "__main__":
    main()
