"""What the BEV looks like against the ground truth along training: the same held-out frames, one row per saved training step.

    python -m bev.gallery scratch          # results/bev/snaps/scratch_step*.pkl -> docs/img/bev_gallery_scratch.png and .gif

Each cell: the model's heatmap (max over classes, dark = nothing, yellow = sure), the labelled boxes (green outlines) and the
detections with score > 0.3 (orange), in the ego frame with x forward up, on a Boston frame never seen in training.
"""
import glob
import json
import pickle
import re
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from . import backbone as BB  # noqa: E402
from . import model as M  # noqa: E402
from . import train as T  # noqa: E402
from .rig import box_corners  # noqa: E402

OUT = "results/bev"
BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"


def snapshots(run):
    """[(step, params)] sorted by step: the run's saved snapshots, plus its final parameters if the run finished."""
    out = []
    for p in glob.glob(f"{OUT}/snaps/{run}_step*.pkl"):
        c = pickle.load(open(p, "rb"))
        out.append((int(re.search(r"step(\d+)", p).group(1)), c["params"]))
    return sorted(out, key=lambda x: x[0])


def frames(d, cfg, k=4, seed=0):
    """k validation frame pairs spread over different sequences, each with several labelled objects in the grid."""
    pairs = T.pairs(d)
    good = [p for p in pairs if len(T._gt(cfg, d, p[1])[0]) >= 4]
    by_seq = {}
    for p in good:
        by_seq.setdefault(int(d["seq"][p[1]]), []).append(p)
    seqs = sorted(by_seq)
    pick = [by_seq[s][len(by_seq[s]) // 2] for s in seqs[:: max(1, len(seqs) // k)]][:k]
    return pick


def draw_cell(ax, cfg, heat, pred, gt_boxes):
    ext = [-cfg.range_m, cfg.range_m, -cfg.range_m, cfg.range_m]
    p = 1 / (1 + np.exp(-np.asarray(heat).max(-1)))
    ax.imshow(p[::-1, ::-1], extent=ext, cmap="magma", vmin=0, vmax=1)
    for b in gt_boxes:
        cr = box_corners(b)[:4, :2]
        ax.add_patch(Polygon(np.c_[-cr[:, 1], cr[:, 0]], fc="none", ec="#3ddc84", lw=1.0))
    pb, ps, _ = pred
    for b, s in zip(pb, ps):
        if s > 0.3:
            cr = box_corners(b)[:4, :2]
            ax.add_patch(Polygon(np.c_[-cr[:, 1], cr[:, 0]], fc="#f0883e", ec="none", alpha=0.85))
    ax.add_patch(Polygon([(-0.95, 2.3), (0.95, 2.3), (0.95, -2.3), (-0.95, -2.3)], fc="white"))
    ax.set_xlim(-cfg.range_m, cfg.range_m)
    ax.set_ylim(-cfg.range_m, cfg.range_m)
    ax.set_xticks([])
    ax.set_yticks([])


def main(run):
    r = json.load(open(f"{OUT}/exp_{run}.json")) if glob.glob(f"{OUT}/exp_{run}.json") else None
    snaps = snapshots(run)
    if r is not None:  # the final parameters
        flat = dict(np.load(f"{OUT}/exp_{run}_params.npz"))
        snaps = [s for s in snaps if s[0] < r["steps"]] + [(r["steps"], flat)]
    cfg_d = (r or {}).get("config") or {}
    backbone = "bevformer_r50" if run.startswith("bevformer_r50") else "scratch"
    cfg = M.Config(backbone=backbone, temporal="notemporal" not in run, feat_dims=64 if backbone == "bevformer_r50" else 0,
                   **{k: v for k, v in cfg_d.items() if k in ("bev", "range_m", "dim", "layers")})
    val = (r or {}).get("val_files") or sorted(glob.glob("data/bev/boston_*_s[0-9]*.npz"))
    d = T.load_many(val)
    feats = None
    if backbone == "bevformer_r50":
        z = np.load(f"{OUT}/exp_{run}_pca.npz")
        (feats,), _, _ = T.r50_pca_cache([d], BB.r50_load(), pca=(z["mu"], z["W"], float(z["explained"])))
    if not snaps or snaps[0][0] != 0:  # training starts from M.init(PRNGKey(seed 0)): the untrained model is exactly reproducible
        snaps = [(0, M.init(jax.random.PRNGKey(0), cfg))] + snaps
    pick = frames(d, cfg)
    _, fwd = T.make_step(cfg, None, d["imgs"].shape[2:4])
    template = M.init(jax.random.PRNGKey(0), cfg)
    leaves, tdef = jax.tree_util.tree_flatten_with_path(template)

    def as_tree(p):
        if isinstance(p, dict) and any("/" in k for k in p):  # flat npz of the final parameters ("layers/0/sca/...")
            return jax.tree_util.tree_unflatten(tdef, [jnp.asarray(p["/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)])
                                                       for path, _ in leaves])
        return jax.tree.map(jnp.asarray, p)
    rows = []
    for step, p in snaps:
        params = as_tree(p)
        b = {k: jnp.asarray(v) for k, v in T.batch_arrays(cfg, d, pick, feats).items()}
        h, rg = fwd(params, b)
        rows.append((step, [np.asarray(h[i]) for i in range(len(pick))], [M.decode(cfg, h[i], rg[i]) for i in range(len(pick))]))
    gts = [T._gt(cfg, d, j)[0] for _, j in pick]
    curve = {c["step"]: c["val_mAP"] for c in (r or {}).get("val_curve", [])}
    # ---- the grid
    n_r, n_c = len(rows) + 1, len(pick)
    fig, axs = plt.subplots(n_r, n_c, figsize=(2.6 * n_c, 2.55 * n_r), facecolor=BG)
    axs = np.atleast_2d(axs)
    for c, (_, j) in enumerate(pick):
        axs[0, c].imshow(d["imgs"][j, 0])
        axs[0, c].set_xticks([])
        axs[0, c].set_yticks([])
        axs[0, c].set_title(f"frame {c + 1}: front camera", color=INK2, fontsize=8)
    for rr, (step, hs, ps) in enumerate(rows, start=1):
        for c in range(n_c):
            draw_cell(axs[rr, c], cfg, hs[c], ps[c], gts[c])
        lab = f"step {step}" + (f"\nval mAP {curve[step]:.3f}" if step in curve else "")
        axs[rr, 0].set_ylabel(lab, color=INK, fontsize=9)
    fig.suptitle(f"{run}: predicted BEV (heatmap, orange detections) vs labels (green) on held-out Boston frames, along training",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"docs/img/bev_gallery_{run}.png", dpi=100, facecolor=BG)
    plt.close(fig)
    # ---- the same as an animation over training steps
    from PIL import Image
    ims = []
    for step, hs, ps in rows:
        fig, axs = plt.subplots(1, n_c, figsize=(2.8 * n_c, 3.1), facecolor=BG)
        for c in range(n_c):
            draw_cell(axs[c], cfg, hs[c], ps[c], gts[c])
        fig.suptitle(f"{run}  step {step}" + (f"   val mAP {curve[step]:.3f}" if step in curve else "   (untrained)" if step == 0 else ""),
                     color=INK, fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        ims.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    ims[0].save(f"docs/img/bev_gallery_{run}.gif", save_all=True, append_images=ims[1:] + [ims[-1]] * 2, duration=700, loop=0)
    return [s for s, _, _ in rows]


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else "scratch"))
