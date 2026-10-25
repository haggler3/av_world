"""Image features for the BEV encoder, stride 8.

  scratch        a 4-layer CNN (3x3 convs, ReLU), 64 channels, ~45 k parameters, trained with everything else
  bevformer_r50  BEVFormer-tiny's own ResNet-50 stem + layer1 + layer2 (torchvision layout), 512 channels, frozen, with its batch
                 norms folded into the convolutions (BEVFormer-tiny freezes them too). Weights from the released checkpoint:

    python -m bev.backbone fetch      # downloads bevformer_tiny_epoch_24.pth (400 MB, Apache-2.0) -> data/bev/bevformer_tiny_r50_l2.npz (6 MB)

Images go in as uint8 [..., H, W, 3] RGB; features come out as [..., H/8, W/8, C].
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

CKPT_URL = "https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth"
R50_NPZ = "data/bev/bevformer_tiny_r50_l2.npz"
MEAN, STD = np.array([123.675, 116.28, 103.53], np.float32), np.array([58.395, 57.12, 57.375], np.float32)  # BEVFormer's img_norm_cfg
LAYERS = (("layer1", 3, 1), ("layer2", 4, 2))  # (name, bottleneck blocks, stride of the first block)


def conv(x, w, b=None, stride=1, pad=None):
    """NHWC x HWIO."""
    k = w.shape[0]
    p = (k - 1) // 2 if pad is None else pad
    y = jax.lax.conv_general_dilated(x, w, (stride, stride), [(p, p), (p, p)], dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y if b is None else y + b


# ------------------------------------------------------------------------------------------------------------- scratch
def scratch_init(key, chans=(24, 32, 48, 64)):
    ps, cin = [], 3
    for i, c in enumerate(chans):
        key, k = jax.random.split(key)
        ps.append({"w": jax.random.normal(k, (3, 3, cin, c)) * np.sqrt(2.0 / (9 * cin)), "b": jnp.zeros(c)})
        cin = c
    return ps


def scratch_apply(ps, img):
    x = img.astype(jnp.float32) / 255.0 - 0.5
    for i, p in enumerate(ps):
        x = jax.nn.relu(conv(x, p["w"], p["b"], stride=2 if i < 3 else 1))
    return x


# ------------------------------------------------------------------------------------------------------------- ResNet-50
def _fold(sd, conv_key, bn_key, eps=1e-5):
    w = sd[conv_key + ".weight"]  # [O, I, kh, kw]
    g, beta, mu, var = (sd[f"{bn_key}.{n}"] for n in ("weight", "bias", "running_mean", "running_var"))
    s = g / np.sqrt(var + eps)
    return {"w": np.transpose(w * s[:, None, None, None], (2, 3, 1, 0)).astype(np.float32), "b": (beta - mu * s).astype(np.float32)}


def extract_r50(ckpt, out=R50_NPZ, prefix="img_backbone."):
    """BEVFormer checkpoint -> folded stem + layer1 + layer2 weights (.npz)."""
    from .torch_ckpt import state_dict
    sd = {k[len(prefix):]: v for k, v in state_dict(ckpt, prefix).items()}
    flat = {f"stem.{k}": v for k, v in _fold(sd, "conv1", "bn1").items()}
    for name, n, _ in LAYERS:
        for i in range(n):
            for c in (1, 2, 3):
                flat.update({f"{name}.{i}.conv{c}.{k}": v for k, v in _fold(sd, f"{name}.{i}.conv{c}", f"{name}.{i}.bn{c}").items()})
            if i == 0:
                flat.update({f"{name}.0.down.{k}": v for k, v in _fold(sd, f"{name}.0.downsample.0", f"{name}.0.downsample.1").items()})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, **flat)
    return out


def r50_load(path=R50_NPZ):
    d = np.load(path)
    p = {}
    for k in d.files:
        *mod, leaf = k.split(".")
        p.setdefault(".".join(mod), {})[leaf] = jnp.asarray(d[k])
    return p


def r50_apply(p, img):
    """uint8 RGB [B, H, W, 3] -> [B, H/8, W/8, 512]: conv1 (7x7/2) + max-pool (3x3/2) + layer1 + layer2 (pytorch style: the stride
    is on the 3x3 conv)."""
    x = (img.astype(jnp.float32) - MEAN) / STD
    x = jax.nn.relu(conv(x, p["stem"]["w"], p["stem"]["b"], stride=2, pad=3))
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), [(0, 0), (1, 1), (1, 1), (0, 0)])
    for name, n, s0 in LAYERS:
        for i in range(n):
            s = s0 if i == 0 else 1
            q = lambda c: p[f"{name}.{i}.{c}"]  # noqa: E731
            y = jax.nn.relu(conv(x, q("conv1")["w"], q("conv1")["b"]))
            y = jax.nn.relu(conv(y, q("conv2")["w"], q("conv2")["b"], stride=s))
            y = conv(y, q("conv3")["w"], q("conv3")["b"])
            sc = conv(x, q("down")["w"], q("down")["b"], stride=s, pad=0) if i == 0 else x
            x = jax.nn.relu(y + sc)
    return x


def fetch(out=R50_NPZ, ckpt="data/bev/bevformer_tiny_epoch_24.pth"):
    if not os.path.exists(ckpt):
        import urllib.request
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        urllib.request.urlretrieve(CKPT_URL, ckpt)
    return extract_r50(ckpt, out)


if __name__ == "__main__" and sys.argv[1:2] == ["fetch"]:
    print(fetch(*sys.argv[2:3]))
