"""Scaling laws for the world models trained on the area simulator (xnet): held-out error vs training steps (compute)
and vs simulated data, across architectures.

Error = skill: the model's CRPS on speed and gap at 1/2/3 s, divided by the physics (kinematic) model's, on a fixed
held-out set of vehicles (lower is better; 1 = physics). Each curve is fitted with L(x) = a * x^-b in log-log space.

    DWM_SOURCE=xnet python -m dwm.scaling compute     # error vs training steps   (~5 min)
    DWM_SOURCE=xnet python -m dwm.scaling data        # error vs simulated data   (~5 min)
"""
import json
import os
import sys
import time

os.environ.setdefault("DWM_SOURCE", "xnet")

import numpy as np  # noqa: E402

from . import model as M  # noqa: E402
from . import train as T  # noqa: E402

OUT = "results/scaling"
DATA = "data/dwm_xnet"
BASE = dict(knots=5, sample_steps=5)  # 5 control points over the 3 s: smooth futures, works for every architecture
ARCHS = {
    "MLP 128x2": dict(net="mlp", width=128, depth=2),
    "MLP 256x3": dict(net="mlp"),
    "transformer 64x2": dict(net="attn", width=64, depth=2),
    "flow matching, MLP": dict(net="mlp", kind="flow", sample_steps=8),
}
# simulated data behind the logged windows: 4 copies x 15 min at 5 Hz, every AV decision logged, windows every 3rd step
SIM_STEPS_TOTAL = 4 * 15 * 60 * 5


def load(n_train=None):
    d = {f[:-4]: np.load(os.path.join(DATA, f), mmap_mode="r") for f in os.listdir(DATA) if f.endswith(".npy")}
    te = np.where(d["test"])[0]
    tr = np.where(~np.asarray(d["test"]))[0]
    n_all = len(tr)
    vid = np.asarray(d["vid"])
    tr = tr[np.argsort(vid[tr], kind="stable")]  # in simulation-time order (vehicle ids are assigned as AVs appear)
    if n_train:
        tr = tr[:n_train]
    ev = np.random.default_rng(1).choice(te, 2000, replace=False)
    keep = np.sort(np.concatenate([tr, ev]))
    out = {k: np.asarray(v[keep]) for k, v in d.items()}
    if "kin" not in out:
        out["kin"] = T.kinematic(out["cur"], out["hist"], out["act"])
    return out, n_all


def run(name, d, steps):
    T._CACHE.clear()
    cfg = T.with_latent(M.Cfg(**(BASE | ARCHS[name])))
    st = T.Stats(d)
    idx = np.where(d["test"])[0]
    ref = T.metrics(T.baselines(d, idx)["kinematic"], d, idx)
    t0 = time.time()
    res = T.train(cfg, d, st, 1e9, verbose=False, max_steps=steps, log_every=min(100, steps))
    train_s = time.time() - t0
    pred, _ = T.sample_physical(res, cfg, st, d, idx, 8)
    return T.skill(T.metrics(pred, d, idx), ref), train_s, res["n_params"]


def fit(xs, ys):
    """error = a * x^-b, least squares in log-log space; returns [a, b, r2]."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    slope, icpt = np.polyfit(lx, ly, 1)
    r2 = 1 - ((ly - (icpt + slope * lx)) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-12)
    return [float(np.exp(icpt)), float(-slope), float(r2)]


def compute(steps=(250, 600, 1500, 4000), archs=None):
    d, _ = load(200_000)
    rows = []
    for name in archs or ARCHS:
        for s in steps:
            sk, sec, npar = run(name, d, s)
            rows.append({"arch": name, "steps": s, "skill": sk, "train_s": sec, "params": npar})
            print(json.dumps(rows[-1]), flush=True)
    save(f"compute_{slug(archs)}" if archs else "compute", rows, None)


def data(sizes=(10_000, 30_000, 100_000, 300_000, 730_000), steps=1500, archs=None):
    rows = []
    for n in sizes:
        d, n_all = load(n)
        for name in archs or ARCHS:
            sk, sec, npar = run(name, d, steps)
            rows.append({"arch": name, "windows": n, "sim_steps": n / n_all * SIM_STEPS_TOTAL, "skill": sk, "train_s": sec})
            print(json.dumps(rows[-1]), flush=True)
    save(f"data_{slug(archs)}" if archs else "data", rows, None)


def slug(archs):
    return "_".join(a.split()[0].lower() + a.split()[-1].replace("x", "") for a in archs)


def merge():
    """Combine the per-architecture result files and fit the power laws."""
    for kind, x in (("compute", "steps"), ("data", "sim_steps")):
        rows = []
        for f in sorted(os.listdir(OUT)):
            if f.startswith(kind + "_") and f.endswith(".json"):
                rows += json.load(open(os.path.join(OUT, f)))["rows"]
        if rows:
            archs = list(dict.fromkeys(r["arch"] for r in rows))
            fits = {a: fit([r[x] for r in rows if r["arch"] == a], [r["skill"] for r in rows if r["arch"] == a]) for a in archs}
            save(kind, rows, fits)


def save(kind, rows, fits):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{kind}.json"), "w") as f:
        json.dump({"rows": rows, "fits": fits, "form": "skill = a * x^-b (fit: [a, b, r2])"}, f, indent=1)
    print(kind, "fits", json.dumps(fits))


if __name__ == "__main__":
    cmd, archs = sys.argv[1], [a for a in ARCHS if len(sys.argv) < 3 or sys.argv[2] in a]
    if cmd == "merge":
        merge()
    else:
        {"compute": compute, "data": data}[cmd](archs=archs)
