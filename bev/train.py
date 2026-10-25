"""Train MiniBEVFormer on rendered av_world data (CPU-sized), or just prove the pipeline learns.

    python -m bev.train --smoke                       # overfit one batch of 4 for 80 steps (~1 min): the plumbing works if loss falls
    python -m bev.train --backbone scratch            # train on data/bev/* except Boston, validate on Boston (held-out city)
    python -m bev.train --backbone bevformer_r50      # the same with BEVFormer-tiny's frozen ResNet stem (features cached once)

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


# ------------------------------------------------------------------------------------------------------------- experiments
def load_many(paths):
    """Concatenate .npz files from bev.data, keeping each file's ego sequences apart."""
    d = {}
    for p in paths:
        x = D.load(p)
        off = (d["seq"].max() + 1) if d else 0
        x["seq"] = x["seq"] + off
        for k, v in x.items():
            d[k] = v if k in ("ego2img", "dt") else (np.concatenate([d[k], v]) if k in d else v)
    return d


def r50_pca_cache(datasets, r50, dims=64, fit_samples=200, chunk=8, seed=0, pca=None):
    """BEVFormer-tiny R50 features of every sample, reduced 512 -> dims channels by a PCA fitted on a sample of training pixels
    (a fixed linear map, so the frozen backbone is run exactly once). -> (list of float16 arrays [N, 6, h, w, dims], (mean, W), seconds)."""
    t0 = time.time()
    f = jax.jit(lambda x: BB.r50_apply(r50, x))
    run = lambda imgs: np.asarray(f(jnp.asarray(imgs.reshape((-1,) + imgs.shape[2:]))))  # noqa: E731
    if pca is not None:  # reuse a fitted reduction (e.g. a trained run's, to evaluate it on new data)
        mu, W, explained = pca
    else:
        rng = np.random.default_rng(seed)
        n = min(fit_samples, len(datasets[0]["imgs"]))
        fit = np.concatenate([run(datasets[0]["imgs"][i:i + 1]).reshape(-1, 512) for i in rng.choice(len(datasets[0]["imgs"]), n, replace=False)])
        mu = fit.mean(0)
        _, sv, vt = np.linalg.svd(fit - mu, full_matrices=False)
        W = vt[:dims].T / (sv[:dims] / np.sqrt(len(fit)))[None]  # whitened: unit variance per component
        explained = float((sv[:dims] ** 2).sum() / (sv ** 2).sum())
    out = []
    for d in datasets:
        parts = []
        for i in range(0, len(d["imgs"]), chunk):
            x = d["imgs"][i:i + chunk]
            y = run(x)
            parts.append(((y.reshape(-1, 512) - mu) @ W).reshape(x.shape[:2] + y.shape[1:3] + (dims,)).astype(np.float16))
        out.append(np.concatenate(parts))
    return out, (mu, W, explained), time.time() - t0


def _gt(cfg, d, j):
    keep = (d["cls"][j] >= 0) & (np.abs(d["boxes"][j][:, 0]) < cfg.range_m) & (np.abs(d["boxes"][j][:, 1]) < cfg.range_m)
    return d["boxes"][j][keep], d["cls"][j][keep], d["occluded"][j][keep]


def predict(cfg, fwd, params, d, idx, feats=None, bs=8, blind=False):
    preds = []
    for s in range(0, len(idx), bs):
        b = batch_arrays(cfg, d, idx[s:s + bs], feats)
        if blind:  # no image information: zero features (cached) or black images
            for k in ("f_prev", "f_now", "img_prev", "img_now"):
                if k in b:
                    b[k] = np.zeros_like(b[k])
        h, r = fwd(params, {k: jnp.asarray(v) for k, v in b.items()})
        preds += [M.decode(cfg, h[i], r[i]) for i in range(len(h))]
    return preds


def report(cfg, d, idx, preds):
    """nuScenes-style metrics on all, plus by distance band, and recall of occluded vs visible objects (2 m, score > 0.3)."""
    gts = [_gt(cfg, d, j) for _, j in idx]
    out = {"all": ME.evaluate(preds, [(b, c) for b, c, _ in gts], D.CLASSES)}
    for lo, hi in ((0, 16), (16, cfg.range_m)):
        band = lambda b: (np.hypot(b[:, 0], b[:, 1]) >= lo) & (np.hypot(b[:, 0], b[:, 1]) < hi)  # noqa: E731
        out[f"{lo}-{hi:g} m"] = ME.evaluate([(pb[band(pb)], ps[band(pb)], pc[band(pb)]) for pb, ps, pc in preds],
                                            [(b[band(b)], c[band(b)]) for b, c, _ in gts], D.CLASSES)
    hit = {"occluded": [0, 0], "visible": [0, 0]}
    for (pb, ps, pc), (b, c, o) in zip(preds, gts):
        q = pb[ps > 0.3]
        for bi, oi in zip(b, o):
            k = "occluded" if oi else "visible"
            hit[k][1] += 1
            hit[k][0] += bool(len(q)) and bool((np.hypot(q[:, 0] - bi[0], q[:, 1] - bi[1]) < 2.0).any())
    out["recall@2m"] = {k: (v[0] / v[1] if v[1] else float("nan"), v[1]) for k, v in hit.items()}
    return out


def experiment(train_paths, val_paths, cfg=M.Config(), steps=3000, bs=4, lr=1e-3, seed=0, eval_every=500, val_sub=160,
               name=None, out=OUT):
    """Train on some cities, validate on another; log the training loss and a validation curve, then the full report, the same
    model blind (no image information), and the training-set score (how much it memorised)."""
    name = name or f"{cfg.backbone}{'' if cfg.temporal else '_notemporal'}"
    dtr, dva = load_many(train_paths), load_many(val_paths)
    img_hw = dtr["imgs"].shape[2:4]
    ftr = fva = None
    pca_info, t_cache = None, 0.0
    if cfg.backbone == "bevformer_r50":
        (ftr, fva), (mu, W, explained), t_cache = r50_pca_cache([dtr, dva], BB.r50_load(), dims=cfg.feat_dims or 64, seed=seed)
        pca_info = {"dims": W.shape[1], "variance_explained": explained}
        os.makedirs(out, exist_ok=True)
        np.savez(f"{out}/exp_{name}_pca.npz", mu=mu, W=W, explained=explained)
    tr, va = pairs(dtr), pairs(dva)
    vsub = [va[i] for i in np.linspace(0, len(va) - 1, min(val_sub, len(va))).astype(int)]
    trsub = [tr[i] for i in np.linspace(0, len(tr) - 1, min(val_sub, len(tr))).astype(int)]
    rng = np.random.default_rng(seed)
    params = M.init(jax.random.PRNGKey(seed), cfg)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(lr / 10, lr, min(100, steps // 10), steps, lr / 20), weight_decay=1e-4)
    st = opt.init(params)
    step, fwd = make_step(cfg, opt, img_hw)
    log, curve, t_steps, run_l = [], [], [], []
    t0, s0 = time.time(), 1
    ckpt = f"{out}/exp_{name}_ckpt.pkl"
    if os.path.exists(ckpt):  # resume after an interruption: same data order (the generator state is saved too)
        import pickle
        c = pickle.load(open(ckpt, "rb"))
        params, st, log, curve, rng = c["params"], c["opt"], c["log"], c["curve"], c["rng"]
        s0 = c["step"] + 1
        print(f"[{name}] resumed at step {s0}", flush=True)
    import pickle
    snaps = f"{out}/snaps"
    os.makedirs(snaps, exist_ok=True)
    if s0 == 1:  # the untrained model, for the gallery of predictions along training
        pickle.dump({"params": jax.device_get(params), "step": 0}, open(f"{snaps}/{name}_step0.pkl", "wb"))
    for s in range(s0, steps + 1):
        sel = [tr[i] for i in rng.choice(len(tr), bs, replace=False)]
        b = {k: jnp.asarray(v) for k, v in batch_arrays(cfg, dtr, sel, ftr).items()}
        t1 = time.time()
        params, st, l, parts = step(params, st, b)
        run_l.append((float(l), float(parts["heat"]), float(parts["reg"])))
        t_steps.append(time.time() - t1)
        if s % 100 == 0:
            m = np.mean(run_l[-100:], 0)
            log.append({"step": s, "loss": m[0], "heat": m[1], "reg": m[2]})
        if s % eval_every == 0 or s == steps:
            ev = ME.evaluate(predict(cfg, fwd, params, dva, vsub, fva), [(_gt(cfg, dva, j)[0], _gt(cfg, dva, j)[1]) for _, j in vsub], D.CLASSES)
            curve.append({"step": s, "val_mAP": ev["mAP"], "val_NDS": ev["NDS"], "train_loss_100": log[-1]["loss"] if log else None,
                          "wall_s": time.time() - t0})
            print(f"[{name}] step {s:5d}  loss {np.mean([x[0] for x in run_l[-eval_every:]]):.3f}  val mAP {ev['mAP']:.3f}  "
                  f"NDS {ev['NDS']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
            import pickle
            os.makedirs(out, exist_ok=True)
            pickle.dump({"params": jax.device_get(params), "opt": jax.device_get(st), "log": log, "curve": curve, "rng": rng, "step": s},
                        open(ckpt, "wb"))
            pickle.dump({"params": jax.device_get(params), "step": s}, open(f"{snaps}/{name}_step{s}.pkl", "wb"))
    full = report(cfg, dva, va, predict(cfg, fwd, params, dva, va, fva))
    blind = ME.evaluate(predict(cfg, fwd, params, dva, va, fva, blind=True), [(_gt(cfg, dva, j)[0], _gt(cfg, dva, j)[1]) for _, j in va], D.CLASSES)
    on_train = ME.evaluate(predict(cfg, fwd, params, dtr, trsub, ftr), [(_gt(cfg, dtr, j)[0], _gt(cfg, dtr, j)[1]) for _, j in trsub], D.CLASSES)
    counts = {split: {c: int((dd["cls"] == i).sum()) for i, c in enumerate(D.CLASSES)} for split, dd in (("train", dtr), ("val", dva))}
    res = {"name": name, "config": cfg.__dict__ | {"cell": cfg.cell}, "params_trainable": M.n_params(params), "train_files": train_paths,
           "val_files": val_paths, "train_pairs": len(tr), "val_pairs": len(va), "class_counts": counts, "steps": steps, "batch": bs,
           "s_per_step": float(np.median(t_steps[5:])), "train_wall_s": time.time() - t0, "feature_cache_s": t_cache, "pca": pca_info,
           "loss_log": log, "val_curve": curve, "val": full, "val_blind": blind, "train_subset": on_train}
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/exp_{name}.json", "w"), indent=1, default=float)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    np.savez(f"{out}/exp_{name}_params.npz", **{"/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path): np.asarray(v)
                                               for path, v in jax.tree_util.tree_flatten_with_path(params)[0]})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--val", nargs="*", default=None, help="validation files (another city); default: every data/bev/boston_*.npz")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--backbone", default="scratch", choices=("scratch", "bevformer_r50"))
    ap.add_argument("--no-temporal", action="store_true")
    a = ap.parse_args()
    paths = a.paths or sorted(glob.glob(f"{D.OUT}/*_s[0-9]*.npz"))  # datasets only (not the backbone weights kept alongside)
    if a.smoke:
        cfg = M.Config(backbone=a.backbone, temporal=not a.no_temporal)
        r, _ = train(paths, cfg, steps=a.steps or 80, smoke=True)
        print(json.dumps({k: v for k, v in r.items() if k not in ("log", "config")}, indent=1, default=float)[:1500])
    else:
        val = a.val if a.val is not None else sorted(glob.glob(f"{D.OUT}/boston_*.npz"))
        tr = [p for p in paths if p not in val]
        cfg = M.Config(backbone=a.backbone, temporal=not a.no_temporal, feat_dims=64 if a.backbone == "bevformer_r50" else 0)
        r = experiment(tr, val, cfg, steps=a.steps or 3000)
        print(json.dumps({k: r[k] for k in ("name", "s_per_step", "val_curve")}, indent=1, default=float))
        print("val:", {k: r["val"]["all"][k] for k in ("mAP", "NDS")}, "blind:", {k: r["val_blind"][k] for k in ("mAP", "NDS")},
              "train subset:", {k: r["train_subset"][k] for k in ("mAP", "NDS")})
