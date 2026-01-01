"""One sample through the whole pipeline, front to back, with every tensor's shape, each stage's time, and a figure wherever a
silent mistake could hide (calibration, which cameras a BEV cell sees, what the features look like, the ego-motion warp, the output).

    python -m bev.trace data/bev/new_york_s0.npz      # -> results/bev/trace.json, docs/img/bev_trace_*.png (~2 min with the smoke fit)
"""
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Wedge  # noqa: E402

from . import backbone as BB  # noqa: E402
from . import data as D  # noqa: E402
from . import metrics as ME  # noqa: E402
from . import model as M  # noqa: E402
from .rig import Rig, box_corners, ego_motion  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
ORDER = (1, 0, 2, 4, 3, 5)  # front-left, front, front-right / back-left, back, back-right
PROBE_COL = ("#00e5ff", "#ff4fd8", "#ffd400")
BG, INK = "#0d1117", "#e6edf3"


def _timed(f, *a, n=3):
    jax.block_until_ready(f(*a))
    t = time.time()
    for _ in range(n):
        r = jax.block_until_ready(f(*a))
    return r, 1000 * (time.time() - t) / n


def _up(a):
    """A BEV array [x index, y index, ...] -> an image with x (forward) up and y (left) to the left, matching the label panel."""
    return a[::-1, ::-1]


def _style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.tick_params(colors="#8b949e", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#30363d")


def probes(cfg, boxes, cls):
    """Three BEV cells to follow through the cameras: on the nearest object ahead, 15 m ahead-left, 20 m behind."""
    ok = (cls >= 0) & (boxes[:, 0] > 3)
    a = boxes[ok][np.argmin(np.hypot(boxes[ok][:, 0], boxes[ok][:, 1]))][:2] if ok.any() else np.array([12.0, 0.0])
    pts = [a, np.array([10.6, 10.6]), np.array([-20.0, 1.0])]
    return [(int((p[0] + cfg.range_m) // cfg.cell), int((p[1] + cfg.range_m) // cfg.cell)) for p in pts]


def run(path, t=None, out_img="docs/img", out_json="results/bev/trace.json", fit_steps=80):
    d = D.load(path)
    if t is None:  # the frame where the ego moves most since the previous one, so the temporal warp is actually exercised
        mv = np.array([np.hypot(*(d["pose"][i, :2] - d["pose"][i - 1, :2])) if d["seq"][i] == d["seq"][i - 1] else -1 for i in range(1, len(d["seq"]))])
        n_obj = (d["cls"][1:] >= 0).sum(1)
        t = 1 + int(np.argmax(np.where(mv > 1.0, n_obj, -1)))  # the busiest frame among those where the ego moved > 1 m
    rig = Rig(width=d["imgs"].shape[3], height=d["imgs"].shape[2])
    cfg = M.Config()
    stages = []

    def log(stage, what, shape, ms=None, note=""):
        stages.append({"stage": stage, "what": what, "shape": shape, "ms": None if ms is None else round(ms, 1), "note": note})

    # 1-3 data
    imgs, boxes, cls, occ = d["imgs"][t], d["boxes"][t], d["cls"][t], d["occluded"][t]
    n = int((cls >= 0).sum())
    log("1 sample", "6 rendered camera images (uint8 RGB), the ego's world pose", list(imgs.shape), note=f"pose {np.round(d['pose'][t], 2).tolist()}")
    log("2 labels", "3-D boxes in the ego frame: x, y, z, l, w, h, yaw, vx, vy + class + occluded", [n, 9],
        note=", ".join(f"{c}: {int((cls == i).sum())}" for i, c in enumerate(D.CLASSES) if (cls == i).any()) + f"; occluded {int(occ[cls >= 0].sum())}")
    E2I = d["ego2img"]
    log("3 calibration", "ego2img = K [R | t] per camera (BEVFormer's lidar2img)", list(E2I.shape), note="K: f = %.1f px" % E2I[0][0, 0])

    # 4 backbone
    p = M.init(jax.random.PRNGKey(0), cfg)
    x = jnp.asarray(imgs[None])
    fs, ms = _timed(jax.jit(lambda pp, xx: M.features(pp, cfg, xx)), p, x)
    log("4a features (scratch CNN)", "stride 8, 64 channels, trained with the model", list(fs.shape), ms, note=f"{M.n_params(p['backbone'])} params")
    have_r50 = os.path.exists(BB.R50_NPZ)
    if have_r50:
        r50 = BB.r50_load()
        cfg_r = M.Config(backbone="bevformer_r50")
        fr, ms_r = _timed(jax.jit(lambda xx: M.features(None, cfg_r, xx, r50)), x)
        log("4b features (BEVFormer-tiny R50 stem + layer1-2, frozen)", "stride 8, 512 channels, folded batch norm", list(fr.shape), ms_r,
            note="1.44 M params, identical to PyTorch within 2e-5")

    # 5 BEV queries, pillars, projection
    loc, valid = M.project(cfg, jnp.asarray(E2I[None]), imgs.shape[1:3])
    hits = np.asarray(valid.any(-1).sum(1))[0].reshape(cfg.bev, cfg.bev)
    log("5 BEV queries + pillars", f"{cfg.bev}x{cfg.bev} cells of {cfg.cell:.0f} m, each lifted to {len(cfg.pillar_z)} heights {cfg.pillar_z} m",
        [cfg.bev * cfg.bev, len(cfg.pillar_z), 3])
    log("6 projection (hit views)", "pillar points -> every camera's feature map; a camera is a hit view if any point lands in it",
        list(loc.shape[1:]), note="cells seen by 0 / 1 / 2 / 3+ cameras: " + " / ".join(f"{(hits == k).mean():.0%}" for k in (0, 1, 2)) + f" / {(hits >= 3).mean():.0%}")

    # 7 temporal alignment, checked on the label heatmaps themselves
    R, tt = ego_motion(d["pose"][t - 1], d["pose"][t])
    h_prev = M.targets(cfg, d["boxes"][t - 1], d["cls"][t - 1])[0].max(-1)
    h_now = M.targets(cfg, boxes, cls)[0].max(-1)
    warped = np.asarray(M.align(cfg, jnp.asarray(h_prev.reshape(1, -1, 1)), jnp.asarray(R[None], jnp.float32), jnp.asarray(tt[None], jnp.float32)))[0, :, 0].reshape(cfg.bev, cfg.bev)
    corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])  # noqa: E731
    log("7 temporal alignment", "previous BEV warped by the ego motion (R, t) into the current frame", [cfg.bev * cfg.bev, cfg.dim],
        note=f"ego moved {np.hypot(*tt):.2f} m, turned {math.degrees(math.atan2(R[1, 0], R[0, 0])):.1f} deg; label heatmaps correlate "
             f"{corr(h_prev, h_now):.2f} unwarped -> {corr(warped, h_now):.2f} warped")

    # 8 encoder + head (untrained)
    enc = jax.jit(lambda pp, f, e: M.head(pp, cfg, M.encode(pp, cfg, f, e, imgs.shape[1:3])))
    (h0, r0), ms_e = _timed(enc, p, fs, jnp.asarray(E2I[None]))
    log("8 BEV encoder", f"{cfg.layers} x (temporal self-attn -> spatial cross-attn -> FFN), {cfg.heads} heads, {cfg.dim} channels",
        [cfg.bev * cfg.bev, cfg.dim], ms_e, note=f"{M.n_params({k: v for k, v in p.items() if k not in ('backbone', 'head')})} params")
    log("9 head", "class heatmaps + 10 regression channels per cell", [[cfg.bev, cfg.bev, cfg.classes], [cfg.bev, cfg.bev, M.N_REG]],
        note=f"{M.n_params(p['head'])} params")
    heat, reg, mask = M.targets(cfg, boxes, cls)
    L, parts = M.loss(h0[0], r0[0], jnp.asarray(heat), jnp.asarray(reg), jnp.asarray(mask))
    log("10 targets + loss", "Gaussian heatmaps at object centres (focal loss) + L1 regression where mask = 1", [int(mask.sum())],
        note=f"untrained: loss {float(L):.2f} (heat {float(parts['heat']):.2f}, reg {float(parts['reg']):.2f})")
    gb, gs, gc = M.decode(cfg, np.log(heat / (1 - heat + 1e-6) + 1e-9), reg, thresh=0.5)
    keep = (cls >= 0) & (np.abs(boxes[:, 0]) < cfg.range_m) & (np.abs(boxes[:, 1]) < cfg.range_m)
    rt = ME.evaluate([(gb, gs, gc)], [(boxes[keep], cls[keep])], D.CLASSES)
    log("11 decode + metrics", "3x3 peak picking -> boxes; nuScenes-style mAP / NDS", [len(gb), 9],
        note=f"labels -> targets -> decode round trip: mAP {rt['mAP']:.2f}, NDS {rt['NDS']:.2f} (must be 1.00)")

    # 12 smoke fit (one batch) to show the output stage learning
    fitted = None
    if fit_steps:
        from . import train as T
        res, fitted = T.train([path], cfg, steps=fit_steps, smoke=True, out=os.path.dirname(out_json) or ".", smoke_pairs=[(t - 1, t)] + [
            (i - 1, i) for i in range(max(1, t - 6), t, 2) if d["seq"][i] == d["seq"][i - 1]][:3])
        log("12 smoke fit", f"overfit one batch of 4 pairs for {fit_steps} steps (plumbing, not training)", [], res["s_per_step"] * 1000,
            note=f"loss {res['log'][0]['loss']:.2f} -> {res['log'][-1]['loss']:.2f}; mAP on that batch {res['eval']['mAP']:.2f}")

    os.makedirs(out_img, exist_ok=True)
    pr = probes(cfg, boxes, cls)
    figures(cfg, rig, d, t, imgs, boxes, cls, hits, pr, loc, valid, h_prev, h_now, warped, fs, fr if have_r50 else None, h0, fitted, heat, out_img)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    json.dump({"source": path, "frame": int(t), "stages": stages}, open(out_json, "w"), indent=1, default=float)
    for s in stages:
        print(f"{s['stage']:52s} {str(s['shape']):24s} {'' if s['ms'] is None else str(s['ms']) + ' ms':>10s}  {s['note']}")
    return stages


def figures(cfg, rig, d, t, imgs, boxes, cls, hits, pr, loc, valid, h_prev, h_now, warped, fs, fr, h0, fitted, heat, out):
    # ---- cameras: labels and the three probe pillars
    fig, axs = plt.subplots(2, 3, figsize=(15, 5.6), facecolor=BG)
    loc_px = (np.asarray(loc)[0] + 0.5) * cfg.stride
    val = np.asarray(valid)[0]
    for a, i in zip(axs.ravel(), ORDER):
        a.imshow(imgs[i])
        _style(a, rig.cams[i][0])
        a.set_xticks([])
        a.set_yticks([])
        for b, c in zip(boxes, cls):
            if c < 0:
                continue
            uv, z, ok = rig.project(box_corners(b))
            if (z[i] > 0.5).all() and ok[i].any():
                for e0, e1 in EDGES:
                    a.plot(*zip(uv[i][e0], uv[i][e1]), color="#ff5050", lw=0.7)
        for (ci, cj), col in zip(pr, PROBE_COL):
            q = ci * cfg.bev + cj
            m = val[i, q]
            if m.any():
                a.scatter(loc_px[i, q, m, 0], loc_px[i, q, m, 1], s=22, c=col, edgecolors="black", linewidths=0.5, zorder=5)
        a.set_xlim(0, rig.width)
        a.set_ylim(rig.height, 0)
    fig.suptitle("Stages 1-3, 6: rendered cameras, the labelled 3-D boxes projected through the calibration (red), and three BEV cells' "
                 "pillar points where they land (dots)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{out}/bev_trace_cameras.png", dpi=110, facecolor=BG)
    plt.close(fig)

    # ---- BEV: labels + camera wedges + probes | hit views | target heatmap
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=BG)
    ext = [-cfg.range_m, cfg.range_m, -cfg.range_m, cfg.range_m]
    a = axs[0]
    _style(a, "Labels in the ego frame (x forward = up), camera fields of view, the probe cells")
    for i in range(rig.n):
        yaw, fov = rig.cams[i][1], rig.cams[i][2]
        a.add_patch(Wedge((0, 0), cfg.range_m * 1.4, 90 + yaw - fov / 2, 90 + yaw + fov / 2, color="#2f81f7", alpha=0.07))
    for b, c in zip(boxes, cls):
        if c >= 0:
            cr = box_corners(b)[:4, :2]
            a.add_patch(Polygon(np.c_[-cr[:, 1], cr[:, 0]], fc=["#a5d6ff", "#d2a8ff", "#f2cc60", "#3ddc84", "#f778ba"][c], ec="none"))
    a.add_patch(Polygon(np.c_[-np.array([1, -1, -1, 1]) * 0.95, np.array([2.3, 2.3, -2.3, -2.3])], fc="white"))
    for (ci, cj), col in zip(pr, PROBE_COL):
        x0, y0 = ci * cfg.cell - cfg.range_m, cj * cfg.cell - cfg.range_m
        a.add_patch(Polygon([(-y0, x0), (-y0 - cfg.cell, x0), (-y0 - cfg.cell, x0 + cfg.cell), (-y0, x0 + cfg.cell)], fc="none", ec=col, lw=2))
    a.set_xlim(-cfg.range_m, cfg.range_m)
    a.set_ylim(-cfg.range_m, cfg.range_m)
    a.set_aspect("equal")
    a = axs[1]
    _style(a, "Hit views: how many cameras each cell's pillar lands in")
    im = a.imshow(_up(hits), extent=ext, cmap="magma", vmin=0, vmax=3)
    plt.colorbar(im, ax=a, fraction=0.046)
    a = axs[2]
    _style(a, "Target: class heatmaps (max over classes)")
    a.imshow(_up(heat.max(-1)), extent=ext, cmap="viridis")
    fig.tight_layout()
    fig.savefig(f"{out}/bev_trace_bev.png", dpi=110, facecolor=BG)
    plt.close(fig)

    # ---- features: image, scratch (untrained), BEVFormer R50 (PCA to RGB)
    cams = (0, 3)
    fig, axs = plt.subplots(len(cams), 3, figsize=(13, 4.6), facecolor=BG)
    for r, i in enumerate(cams):
        axs[r, 0].imshow(imgs[i])
        _style(axs[r, 0], f"{rig.cams[i][0]}: input")
        e = np.abs(np.asarray(fs)[0, i]).mean(-1)
        axs[r, 1].imshow(e, cmap="magma")
        _style(axs[r, 1], "scratch CNN, untrained: mean |feature|")
        if fr is not None:
            f = np.asarray(fr)[0, i].reshape(-1, 512)
            f = f - f.mean(0)
            _, _, vt = np.linalg.svd(f, full_matrices=False)
            pc = f @ vt[:3].T
            pc = (pc - pc.min(0)) / (np.ptp(pc, 0) + 1e-9)
            axs[r, 2].imshow(pc.reshape(np.asarray(fr).shape[2:4] + (3,)))
            _style(axs[r, 2], "BEVFormer-tiny R50 (layer2): top 3 principal components")
        for a in axs[r]:
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{out}/bev_trace_features.png", dpi=110, facecolor=BG)
    plt.close(fig)

    # ---- temporal alignment + output
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4), facecolor=BG)
    rgb = lambda r_, g_: _up(np.stack([r_, g_, np.zeros_like(r_)], -1).clip(0, 1))  # noqa: E731
    axs[0].imshow(rgb(h_prev, h_now), extent=ext)
    _style(axs[0], "Previous labels (red) vs current (green), unwarped")
    axs[1].imshow(rgb(warped, h_now), extent=ext)
    _style(axs[1], "Previous labels warped by the ego motion: overlap (yellow)")
    axs[2].imshow(_up(1 / (1 + np.exp(-np.asarray(h0)[0].max(-1)))), extent=ext, cmap="viridis", vmin=0, vmax=1)
    _style(axs[2], "Model heatmap, untrained")
    if fitted is not None:
        b = {k: jnp.asarray(v) for k, v in __import__("bev.train", fromlist=["x"]).batch_arrays(cfg, d, [(t - 1, t)]).items()}
        prev = M.encode(fitted, cfg, M.features(fitted, cfg, b["img_prev"]), b["e2i"], imgs.shape[1:3])
        h1, _ = M.head(fitted, cfg, M.encode(fitted, cfg, M.features(fitted, cfg, b["img_now"]), b["e2i"], imgs.shape[1:3], prev, b["R"], b["t"]))
        axs[3].imshow(_up(1 / (1 + np.exp(-np.asarray(h1)[0].max(-1)))), extent=ext, cmap="viridis", vmin=0, vmax=1)
        _style(axs[3], "Model heatmap after the 80-step smoke fit on a batch with this frame")
    fig.tight_layout()
    fig.savefig(f"{out}/bev_trace_temporal_output.png", dpi=110, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else sorted(__import__("glob").glob(f"{D.OUT}/*_s[0-9]*.npz"))[0])
