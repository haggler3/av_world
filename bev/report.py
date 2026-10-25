"""Summarise the training runs: curves, a results table with its sanity references, and a clip of predictions on the held-out city.

    python -m bev.report        # results/bev/exp_*.json -> docs/img/bev_training.png, results/bev/summary.md, docs/img/bev_boston.gif
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "results/bev"
BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"
COL = {"scratch": "#a5d6ff", "bevformer_r50": "#f0883e", "scratch_notemporal": "#6e7681", "bevformer_r50_notemporal": "#ffb38a"}


def load():
    return {r["name"]: r for r in (json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/exp_*.json")))}


def plot(runs, out="docs/img/bev_training.png"):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), facecolor=BG)
    for ax, title in zip(axs, ("training loss (mean of 100 steps)", "validation mAP on the held-out city (160 frames)")):
        ax.set_facecolor(BG)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(color="#21262d")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
    for n, r in runs.items():
        c = COL.get(n, "#d2a8ff")
        axs[0].plot([x["step"] for x in r["loss_log"]], [x["loss"] for x in r["loss_log"]], color=c, label=n)
        axs[1].plot([x["step"] for x in r["val_curve"]], [x["val_mAP"] for x in r["val_curve"]], color=c, marker="o", ms=3, label=n)
        axs[1].axhline(r["val_blind"]["mAP"], color=c, ls=":", lw=1)
    axs[0].set_yscale("log")
    axs[1].legend(facecolor=BG, edgecolor="#30363d", labelcolor=INK, fontsize=8)
    axs[1].text(0.99, 0.02, "dotted: the same model blind (image features zeroed)", transform=axs[1].transAxes, ha="right", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def table(runs):
    L = ["| run | trainable params | s / step | val mAP | val NDS | blind mAP (NDS) | train-city mAP | 0-16 m mAP | 16-32 m mAP | recall@2m visible / occluded |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in runs.items():
        v = r["val"]
        rc = v["recall@2m"]
        L.append(f"| {n} | {r['params_trainable']:,} | {r['s_per_step']:.2f} | **{v['all']['mAP']:.3f}** | {v['all']['NDS']:.3f} | "
                 f"{r['val_blind']['mAP']:.3f} ({r['val_blind']['NDS']:.3f}) | {r['train_subset']['mAP']:.3f} | {v['0-16 m']['mAP']:.3f} | "
                 f"{v['16-32 m']['mAP']:.3f} | {rc['visible'][0]:.2f} / {rc['occluded'][0]:.2f} |")
    L += ["", "Per class (val AP, mean over 0.5 / 1 / 2 / 4 m; n = ground-truth objects):", "",
          "| run | " + " | ".join(["car", "truck", "bus", "cyclist", "pedestrian"]) + " | mATE | mASE | mAOE | mAVE |", "|---|" + "---|" * 9]
    for n, r in runs.items():
        pc = r["val"]["all"]["per_class"]
        cells = [f"{pc[c]['AP']:.3f} (n={pc[c]['n_gt']})" if c in pc else "-" for c in ("car", "truck", "bus", "cyclist", "pedestrian")]
        a = r["val"]["all"]
        L.append(f"| {n} | " + " | ".join(cells) + f" | {a['mATE']:.2f} | {a['mASE']:.2f} | {a['mAOE']:.2f} | {a['mAVE']:.2f} |")
    return "\n".join(L)


def clip(run, out="docs/img/bev_boston.gif", n=40):
    """The held-out city: front camera + BEV with labels (outlines) and predictions (filled), for one validation sequence."""
    import jax
    import jax.numpy as jnp
    from matplotlib.patches import Polygon
    from PIL import Image

    from . import backbone as BB
    from . import data as D
    from . import model as M
    from . import train as T
    from .rig import box_corners
    r = json.load(open(f"{OUT}/exp_{run}.json"))
    cfg = M.Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in r["config"].items() if k in M.Config.__dataclass_fields__})
    flat = dict(np.load(f"{OUT}/exp_{run}_params.npz"))
    params = M.init(jax.random.PRNGKey(0), cfg)
    leaves, tdef = jax.tree_util.tree_flatten_with_path(params)
    params = jax.tree_util.tree_unflatten(tdef, [jnp.asarray(flat["/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in p)]) for p, _ in leaves])
    d = T.load_many(r["val_files"])
    feats = None
    if cfg.backbone == "bevformer_r50":  # the run's own fitted feature reduction
        z = np.load(f"{OUT}/exp_{run}_pca.npz")
        (feats,), _, _ = T.r50_pca_cache([d], BB.r50_load(), pca=(z["mu"], z["W"], float(z["explained"])))
    seq = d["seq"][len(d["seq"]) // 3]
    idx = [(i - 1, i) for i in np.nonzero(d["seq"] == seq)[0][1:]][:n]
    _, fwd = T.make_step(cfg, None, d["imgs"].shape[2:4])
    preds = T.predict(cfg, fwd, params, d, idx, feats)
    frames = []
    for (_, j), (pb, ps, pc) in zip(idx, preds):
        fig = plt.figure(figsize=(9.6, 4.0), facecolor=BG)
        a0 = fig.add_axes([0.0, 0.0, 0.58, 0.92])
        a0.imshow(d["imgs"][j, 0])
        a0.axis("off")
        a0.set_title("CAM_FRONT (Boston, never seen in training)", color=INK, fontsize=9, loc="left")
        a1 = fig.add_axes([0.62, 0.04, 0.36, 0.86], facecolor=BG)
        a1.set_title("BEV: labels (outline), detections (filled)", color=INK, fontsize=9, loc="left")
        gb, gc, _ = T._gt(cfg, d, j)
        for b in gb:
            cr = box_corners(b)[:4, :2]
            a1.add_patch(Polygon(np.c_[-cr[:, 1], cr[:, 0]], fc="none", ec="#3ddc84", lw=1))
        for b, s in zip(pb, ps):
            if s > 0.3:
                cr = box_corners(b)[:4, :2]
                a1.add_patch(Polygon(np.c_[-cr[:, 1], cr[:, 0]], fc="#f0883e", ec="none", alpha=min(1.0, 0.3 + s)))
        a1.add_patch(Polygon([(-0.95, 2.3), (0.95, 2.3), (0.95, -2.3), (-0.95, -2.3)], fc="white"))
        a1.set_xlim(-cfg.range_m, cfg.range_m)
        a1.set_ylim(-cfg.range_m, cfg.range_m)
        a1.set_aspect("equal")
        a1.tick_params(colors=INK2, labelsize=7)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=400, loop=0, optimize=True)
    return out


if __name__ == "__main__":
    runs = load()
    plot(runs)
    md = table(runs)
    open(f"{OUT}/summary.md", "w").write(md + "\n")
    print(md)
    best = max(runs, key=lambda n: runs[n]["val"]["all"]["mAP"])
    print(clip(best))
