"""A small, cached DAG runner: each stage is a pure function of its inputs and its config, so it only reruns when one of them changes.

    @stage("rules_av2", deps=("av2_scenes",), config={"version": 1})
    def rules_av2(av2_scenes): ...

Cache key of a stage = sha256(stage name, config, the keys of its inputs, the source code of its function). Outputs are pickled to
results/autotag/cache/<name>-<key>.pkl. run(target) resolves the inputs depth-first (a topological order), loads what is cached and
computes the rest, logging each stage's time. explain() prints the graph and what is fresh; mermaid() draws it.
"""
import hashlib
import inspect
import json
import os
import pickle
import time

STAGES = {}
CACHE = "results/autotag/cache"


class Stage:
    def __init__(self, name, fn, deps, config):
        self.name, self.fn, self.deps, self.config = name, fn, tuple(deps), dict(config or {})

    def key(self, keys):
        src = inspect.getsource(self.fn)
        h = hashlib.sha256(json.dumps([self.name, self.config, [keys[d] for d in self.deps], src], sort_keys=True, default=str).encode())
        return h.hexdigest()[:12]


def stage(name, deps=(), config=None):
    def wrap(fn):
        STAGES[name] = Stage(name, fn, deps, config)
        return fn
    return wrap


def order(target):
    seen, out = set(), []

    def visit(n):
        if n in seen:
            return
        if n not in STAGES:
            raise KeyError(f"no stage {n!r}")
        seen.add(n)
        for d in STAGES[n].deps:
            visit(d)
        out.append(n)
    visit(target)
    return out


def keys_for(target):
    keys = {}
    for n in order(target):
        keys[n] = STAGES[n].key(keys)
    return keys


def run(target, force=(), log=print):
    keys, vals = keys_for(target), {}
    os.makedirs(CACHE, exist_ok=True)
    for n in order(target):
        s = STAGES[n]
        path = os.path.join(CACHE, f"{n}-{keys[n]}.pkl")
        if n not in force and os.path.exists(path):
            with open(path, "rb") as f:
                vals[n] = pickle.load(f)
            log(f"  [cached] {n}")
            continue
        t0 = time.time()
        vals[n] = s.fn(*[vals[d] for d in s.deps], **s.config)
        with open(path, "wb") as f:
            pickle.dump(vals[n], f)
        log(f"  [ran {time.time() - t0:6.1f}s] {n}")
    return vals[target]


def explain(target):
    keys = keys_for(target)
    lines = []
    for n in order(target):
        fresh = os.path.exists(os.path.join(CACHE, f"{n}-{keys[n]}.pkl"))
        lines.append(f"{'cached' if fresh else 'stale ':6}  {n:22} <- {', '.join(STAGES[n].deps) or '(source)'}")
    return "\n".join(lines)


def mermaid(target):
    out = ["graph LR"]
    for n in order(target):
        for d in STAGES[n].deps:
            out.append(f"  {d} --> {n}")
    return "\n".join(out)
