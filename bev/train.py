"""Train MiniBEVFormer on rendered av_world data (CPU-sized), or just prove the pipeline learns.

    python -m bev.train --smoke                       # overfit one batch of 4 for 80 steps (~1 min): the plumbing works if loss falls
    python -m bev.train data/bev/*.npz --steps 3000   # a real run (not done yet; see docs/BEV.md for the budget)

Each step takes pairs of consecutive frames (t-1, t) of one ego: the BEV of t-1 is encoded without history and without gradient,
warped by the ego motion, and used by temporal self-attention at t (BEVFormer trains the same way, on a short queue).
With --backbone bevformer_r50 the frozen ResNet features are computed once per sample and cached (the backbone is ~90 % of the
compute and never changes), so training costs only the BEV encoder and head.
"""
import argparse
import glob
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import backbone as BB
from . import data as D
from . import metrics as ME
from . import model as M
from .rig import ego_motion

OUT = "results/bev"


def pairs(d):
    """Indices (prev, now) of consecutive frames of the same ego sequence."""
    return [(i - 1, i) for i in range(1, len(d["seq"])) if d["seq"][i] == d["seq"][i - 1]]


def batch_arrays(cfg, d, idx, feats=None):
    """-> dict of arrays for the (prev, now) pairs idx."""
    prev, now = np.array([i for i, _ in idx]), np.array([j for _, j in idx])
    tg = [M.targets(cfg, d["boxes"][j], d["cls"][j]) for j in now]
    mot = [ego_motion(d["pose"][i], d["pose"][j]) for i, j in idx]
    out = {"heat": np.stack([t[0] for t in tg]), "reg": np.stack([t[1] for t in tg]), "mask": np.stack([t[2] for t in tg]),
           "R": np.stack([m[0] for m in mot]).astype(np.float32), "t": np.stack([m[1] for m in mot]).astype(np.float32),
           "e2i": np.broadcast_to(d["ego2img"], (len(idx),) + d["ego2img"].shape).astype(np.float32)}
    if feats is not None:
        out["f_prev"], out["f_now"] = feats[prev], feats[now]
    else:
        out["img_prev"], out["img_now"] = d["imgs"][prev], d["imgs"][now]
    return out


def make_step(cfg, opt, img_hw, r50=None):
    def fwd(params, b):
        if "f_now" in b:
            f_prev, f_now = b["f_prev"], b["f_now"]
        else:
            f_prev, f_now = M.features(params, cfg, b["img_prev"], r50), M.features(params, cfg, b["img_now"], r50)
        prev = jax.lax.stop_gradient(M.encode(params, cfg, f_prev, b["e2i"], img_hw)) if cfg.temporal else None
        bev = M.encode(params, cfg, f_now, b["e2i"], img_hw, prev, b["R"], b["t"])
        return M.head(params, cfg, bev)

    def lossf(params, b):
        h, r = fwd(params, b)
        return M.loss(h, r, b["heat"], b["reg"], b["mask"])

    @jax.jit
    def step(params, st, b):
        (l, parts), g = jax.value_and_grad(lossf, has_aux=True)(params, b)
        up, st = opt.update(g, st, params)
        return optax.apply_updates(params, up), st, l, parts
    return step, jax.jit(fwd)


def cache_features(cfg, d, r50, chunk=8):
    f = jax.jit(lambda x: BB.r50_apply(r50, x))
    out = []
    for i in range(0, len(d["imgs"]), chunk):
        x = d["imgs"][i:i + chunk]
        y = f(jnp.asarray(x.reshape((-1,) + x.shape[2:])))
        out.append(np.asarray(y).reshape(x.shape[:2] + y.shape[1:]).astype(np.float16))
    return np.concatenate(out)


def evaluate(cfg, fwd, params, d, idx, feats=None, bs=8):
    preds, gts = [], []
    for s in range(0, len(idx), bs):
        b = {k: jnp.asarray(v) for k, v in batch_arrays(cfg, d, idx[s:s + bs], feats).items()}
        h, r = fwd(params, b)
        for i, (_, j) in enumerate(idx[s:s + bs]):
            preds.append(M.decode(cfg, h[i], r[i]))
            keep = (d["cls"][j] >= 0) & (np.abs(d["boxes"][j][:, 0]) < cfg.range_m) & (np.abs(d["boxes"][j][:, 1]) < cfg.range_m)
            gts.append((d["boxes"][j][keep], d["cls"][j][keep]))
    return ME.evaluate(preds, gts, D.CLASSES)


def train(paths, cfg=M.Config(), steps=2000, bs=4, lr=1e-3, seed=0, out=OUT, smoke=False, smoke_pairs=None):
    d = {}
    for p in paths:  # concatenate files, keeping ego sequences apart
        x = D.load(p)
        off = (d["seq"].max() + 1) if d else 0
        x["seq"] = x["seq"] + off
        for k, v in x.items():
            d[k] = v if k in ("ego2img", "dt") else (np.concatenate([d[k], v]) if k in d else v)
    img_hw = d["imgs"].shape[2:4]
    r50 = BB.r50_load() if cfg.backbone == "bevformer_r50" else None
    t0 = time.time()
    feats = cache_features(cfg, d, r50) if r50 is not None else None
    t_cache = time.time() - t0
    idx = pairs(d)
    rng = np.random.default_rng(seed)
    seqs = sorted(set(d["seq"][j] for _, j in idx))
    val_seq = set(seqs[-max(1, len(seqs) // 5):]) if len(seqs) > 1 and not smoke else set()
    tr = [p for p in idx if d["seq"][p[1]] not in val_seq]
    va = [p for p in idx if d["seq"][p[1]] in val_seq]
    if smoke:
        tr = va = list(smoke_pairs) if smoke_pairs else tr[:bs]
    params = M.init(jax.random.PRNGKey(seed), cfg)
    opt = optax.adamw(optax.cosine_decay_schedule(lr, steps, 0.05), weight_decay=1e-4)
    st = opt.init(params)
    step, fwd = make_step(cfg, opt, img_hw, r50)
    log, t_steps = [], []
    for s in range(steps):
        sel = tr if smoke else [tr[i] for i in rng.choice(len(tr), bs, replace=False)]
        b = {k: jnp.asarray(v) for k, v in batch_arrays(cfg, d, sel, feats).items()}
        t1 = time.time()
        params, st, l, parts = step(params, st, b)
        l = float(l)
        t_steps.append(time.time() - t1)
        if s % max(steps // 20, 1) == 0 or s == steps - 1:
            log.append({"step": s, "loss": l, "heat": float(parts["heat"]), "reg": float(parts["reg"])})
            print(f"step {s:5d}  loss {l:.3f}  heat {float(parts['heat']):.3f}  reg {float(parts['reg']):.3f}", flush=True)
    ev = evaluate(cfg, fwd, params, d, va, feats)
    res = {"config": cfg.__dict__ | {"cell": cfg.cell}, "params": M.n_params(params), "train_pairs": len(tr), "val_pairs": len(va),
           "steps": steps, "batch": bs, "s_per_step": float(np.median(t_steps[1:])) if len(t_steps) > 1 else None,
           "first_step_s (compile)": t_steps[0], "feature_cache_s": t_cache, "log": log, "eval": ev, "smoke": smoke}
    os.makedirs(out, exist_ok=True)
    name = ("smoke_" if smoke else "") + cfg.backbone
    json.dump(res, open(f"{out}/{name}.json", "w"), indent=1, default=float)
    np.savez(f"{out}/{name}_params.npz", **{"/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path): np.asarray(v)
                                           for path, v in jax.tree_util.tree_flatten_with_path(params)[0]})
    return res, params


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--backbone", default="scratch", choices=("scratch", "bevformer_r50"))
    ap.add_argument("--no-temporal", action="store_true")
    a = ap.parse_args()
    paths = a.paths or sorted(glob.glob(f"{D.OUT}/*.npz"))
    cfg = M.Config(backbone=a.backbone, temporal=not a.no_temporal)
    r, _ = train(paths, cfg, steps=a.steps or (80 if a.smoke else 2000), smoke=a.smoke)
    print(json.dumps({k: v for k, v in r.items() if k not in ("log", "config")}, indent=1, default=float)[:1500])
