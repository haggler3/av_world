"""The networks and world-model upgrades as one small API (also `av_world.easy.nets` and `av-world nets`).

    from av_world import easy
    easy.nets.list()                         # a table: every feature, its question, time, result, trained or not
    easy.nets.info("occlusion")              # what it is, what it needs and writes, how to use it
    easy.nets.train("occlusion", quick=True) # ~3 min; quick results go to results/nets/quick
    net = easy.nets.load("occlusion")        # the trained network (full run), or load(..., quick=True)
"""
import sys

from . import registry as RG


def list(kind=None):  # noqa: A001  (the natural name in easy.nets.list())
    rows = []
    for f in RG.FEATURES:
        if kind and f.kind != kind:
            continue
        rows.append({"name": f.name, "kind": f.kind, "question": f.question, "minutes_quick": f.minutes[0], "minutes_full": f.minutes[1],
                     "trained": RG.trained(f), "quick_trained": RG.trained(f, RG.QUICK_OUT), "result": f.result})
    try:
        import pandas as pd
        return pd.DataFrame(rows).set_index("name")
    except ImportError:
        return rows


def info(name):
    f = RG.get(name)
    miss = RG.missing_data(f)
    lines = [f"{f.title}  [{f.kind}]", f"  question : {f.question}", f"  result   : {f.result}",
             f"  time     : ~{f.minutes[0]} min quick, ~{f.minutes[1]} min full (4-core CPU)",
             f"  needs    : {', '.join(f.needs) or 'nothing beyond the package'}" + (f"  -> MISSING: {'; '.join(miss)}" if miss else ""),
             f"  writes   : {', '.join(f.outputs)}", f"  trained  : {'yes' if RG.trained(f) else 'no'} (quick: {'yes' if RG.trained(f, RG.QUICK_OUT) else 'no'})",
             f"  train    : av-world nets train {f.name} [--quick]", f"  use      : {f.use}", f"  docs     : {f.doc}"]
    text = "\n".join(lines)
    print(text)
    return text


def train(name, quick=False, out_dir=None):
    f = RG.get(name)
    miss = RG.missing_data(f)
    if miss:
        raise RuntimeError(f"{name} needs: " + "; ".join(miss))
    return RG.train(name, quick=quick, out_dir=out_dir)


def load(name, quick=False, out_dir=None, **kw):
    from . import use
    return use.load(name, out_dir or (RG.QUICK_OUT if quick else RG.FULL_OUT), **kw)


def cli(argv):
    """av-world nets list | info NAME | train NAME [--quick] [--out DIR] | figures | gifs [NAME ...]"""
    import argparse
    ap = argparse.ArgumentParser(prog="av-world nets", description=cli.__doc__)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list")
    i = sub.add_parser("info")
    i.add_argument("name")
    t = sub.add_parser("train")
    t.add_argument("name")
    t.add_argument("--quick", action="store_true", help="a few minutes, results in results/nets/quick")
    t.add_argument("--out", default=None)
    sub.add_parser("figures")
    g = sub.add_parser("gifs")
    g.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd in (None, "list"):
        print(f"{'name':15} {'kind':8} {'quick':>6} {'full':>6} {'trained':>8}  question")
        for f in RG.FEATURES:
            mark = "yes" if RG.trained(f) else ("quick" if RG.trained(f, RG.QUICK_OUT) else "-")
            print(f"{f.name:15} {f.kind:8} {f.minutes[0]:>4}m {f.minutes[1]:>4}m {mark:>8}  {f.question}")
        print("\nav-world nets info NAME   |   av-world nets train NAME --quick   |   docs/TRAINING_NETS.md")
        return 0
    if a.cmd == "info":
        info(a.name)
        return 0
    if a.cmd == "train":
        train(a.name, quick=a.quick, out_dir=a.out)
        print(f"\ndone: av-world nets info {a.name}")
        return 0
    if a.cmd == "figures":
        from . import figures
        figures.main([])
        return 0
    if a.cmd == "gifs":
        from . import gifs
        gifs.main(a.names)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))
