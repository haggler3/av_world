"""Sim-to-real from Python (`av_world.easy.realism`) and the shell (`av-world realism`).

    av-world realism run [--quick]     build or refresh the DAG (cached: only stale stages run)
    av-world realism explain           the DAG and which stages are fresh
    av-world realism report            the gap, step by step, and what is still different
Python:
    from realism import api
    beh = api.preset()                 # the patched drivers as an xnet Behaviour; also loads their trait bank into the simulator
"""
import json
import os

OUT = "results/realism"


def run(target="report", force=(), quick=False):
    from . import pipeline
    return pipeline.run(target, force=force, quick=quick)


def explain(target="report"):
    from autotag import dag
    from . import pipeline
    text = dag.explain(target, cache=pipeline.CACHE)
    print(text)
    return text


def results(quick=False):
    return json.load(open(f"{OUT}/quick/scoreboard.json" if quick else f"{OUT}/scoreboard.json"))


def preset(quick=False, load_bank=True, **override):
    """The patched human drivers: a Behaviour with the refit's knobs (IDM+, trait bank share, OU residual, pulses). With load_bank the fitted
    trait table is set into the simulator (xnet.sim.set_trait_bank), which must happen before a Sim is built or step is traced."""
    import numpy as np
    from xnet import sim as X
    d = f"{OUT}/quick" if quick else OUT
    knobs = json.load(open(f"{d}/preset.json"))
    if load_bank:
        X.set_trait_bank(np.load(f"{d}/bank.npy"))
    return X.Behaviour(**dict(knobs, **override))


def report(quick=False):
    r = results(quick)
    L = ["the gap, step by step (gap = mean |log sim/real| of four tail rates + mean W1 of six distributions; 0 = identical):"]
    for s in r["decomposition"]:
        L.append(f"  {s['step']:38} gap {s['gap']:.2f}   hard brakes x{s['hard_brakes_per_h']:.2f}  tight x{s['tight_share']:.2f}  "
                 f"TTC<2 x{s['ttc2_share']:.2f}  TTC<1 x{s['ttc1_share']:.2f}")
    L.append("can a network tell real from simulated vehicles? (AUROC, 0.5 = no)")
    for n, d in r["discriminator"].items():
        top = ", ".join(list(d["importance"])[:3])
        L.append(f"  {n:10} {d['auroc']:.3f} [{d['auroc_ci'][0]:.3f}, {d['auroc_ci'][1]:.3f}]   gives it away: {top}")
    rp = r["replay"]
    if "default" in rp and "5s" in rp.get("default", {}):
        L.append(f"log replay error at 5 s ({rp['scenes']} scenes): default {rp['default']['5s']['sim_m']:.2f} m, patched {rp['patched']['5s']['sim_m']:.2f} m, "
                 f"constant velocity {rp['default']['5s']['const_velocity_m']:.2f} m")
    L.append(f"preset: {r['preset']}")
    text = "\n".join(L)
    print(text)
    return text


def cli(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="av-world realism", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--stage", default="report")
    r.add_argument("--force", nargs="*", default=())
    r.add_argument("--quick", action="store_true")
    sub.add_parser("explain")
    p = sub.add_parser("report")
    p.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.stage, tuple(a.force), a.quick)
        report(a.quick)
    elif a.cmd == "explain":
        explain()
    elif a.cmd == "report":
        report(a.quick)
    else:
        ap.print_help()
    return 0
