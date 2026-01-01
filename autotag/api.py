"""The auto-tagger from Python (`av_world.easy.tag`) and the shell (`av-world tag`).

    av-world tag run                 build or refresh every stage of the DAG (cached: only stale stages run)
    av-world tag explain             the DAG and which stages are fresh
    av-world tag query "cut_in & hidden_ahead" [--source av2] [--overlap]
    av-world tag similar SCENE_ID [--k 5]
    av-world tag report              per-tag quality on the simulated golden set, real-vs-sim tag rates, the sim-gap share
"""
import json
import os
import sys

OUT = "results/autotag"


def run(target="store", force=()):
    from . import dag
    from . import pipeline  # noqa: F401  (registers the stages)
    return dag.run(target, force=force)


def explain(target="store"):
    from . import dag
    from . import pipeline  # noqa: F401
    text = dag.explain(target)
    print(text)
    return text


def tags(source="av2"):
    from .schema import TagStore
    p = f"{OUT}/tags_{source}.parquet"
    if not os.path.exists(p) and os.path.exists(p.replace(".parquet", ".jsonl")):
        p = p.replace(".parquet", ".jsonl")
    return TagStore.load(p).frame()


def query(q, source="av2", overlap=False):
    from . import search
    return search.query(tags(source), q, overlap=overlap)


def similar(scene_id, k=5):
    import numpy as np
    from . import search
    z = np.load(f"{OUT}/index.npz")
    return search.Index(z["ids"], z["vectors"]).similar(scene_id, k)


def report():
    r = json.load(open(f"{OUT}/eval.json"))
    lines = ["rule taggers against simulator truth (held-out copies):"]
    for k, v in r["rules"].items():
        lines.append(f"  {k:12} precision {v['precision']:.2f}  recall {v['recall']:.2f}  ({v['n_tags']} tags, {v['n_truth']} true events)")
    lines.append("learned taggers (held-out copies):")
    for k, v in r["learned"].items():
        lines.append(f"  {k:12} AUROC {v['auroc']:.3f}  precision@threshold {v['precision_at_threshold']}  recall {v['recall_at_threshold']}  ECE {v['ece_raw']:.3f} -> {v['ece_calibrated']:.3f}")
    lines.append(f"domain: {100 * r['domain']['av2_share_sim_gap']:.0f} % of real scenes outside the simulator's range ({100 * r['domain']['sim_test_share_sim_gap']:.1f} % of held-out sim scenes)")
    text = "\n".join(lines)
    print(text)
    return text


def cli(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="av-world tag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--stage", default="store")
    r.add_argument("--force", nargs="*", default=())
    sub.add_parser("explain")
    q = sub.add_parser("query")
    q.add_argument("q")
    q.add_argument("--source", default="av2")
    q.add_argument("--overlap", action="store_true")
    q.add_argument("--limit", type=int, default=20)
    s = sub.add_parser("similar")
    s.add_argument("scene_id")
    s.add_argument("--k", type=int, default=5)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.stage, tuple(a.force))
    elif a.cmd == "explain" or a.cmd is None:
        explain()
    elif a.cmd == "query":
        hits = query(a.q, a.source, a.overlap)
        print(f"{len(hits)} scenes match {a.q!r} in {a.source}")
        for sid, win in hits[:a.limit]:
            print(" ", sid, "" if win is None else " ".join(f"[{x:.1f}-{y:.1f} s]" for x, y in win[:3]))
    elif a.cmd == "similar":
        for sid, sim in similar(a.scene_id, a.k):
            print(f"  {sim:.3f}  {sid}")
    elif a.cmd == "report":
        report()
    return 0


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))
