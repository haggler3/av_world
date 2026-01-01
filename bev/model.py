"""MiniBEVFormer: BEVFormer's encoder at CPU scale, in pure JAX, with a dense box head.

Kept from BEVFormer (Li et al., ECCV 2022):
  BEV queries     a learned grid of queries Q [bev*bev, dim], one per BEV cell, plus a learned positional embedding
  spatial cross-attention (SCA)
                  each BEV cell is lifted to a pillar of `pillar_z` heights; the pillar points are projected into every camera with
                  ego2img (BEVFormer's lidar2img); in the cameras they land in ("hit views"), the query samples image features at
                  the projected points plus learned offsets (deformable attention: `sca_points` per pillar point and head), weighted
                  by learned attention, then averaged over the hit views
  temporal self-attention (TSA)
                  the previous frame's BEV is warped into the current ego frame (ego motion), and each query samples it and itself
                  at learned offsets; with no history the queries attend to themselves (as BEVFormer does on a sequence's first frame)
  layer order     TSA -> add & norm -> SCA -> add & norm -> FFN -> add & norm, repeated `layers` times
Changed for CPU scale:
  sizes           32 x 32 BEV over +-32 m (2 m cells; BEVFormer-tiny: 50 x 50 over +-51.2 m), 64 channels (256), 2 layers (3),
                  4 heads (8), images 176 x 96 (800 x 450)
  head            a dense CenterPoint-style head on the BEV map (class heatmaps + box regression at object centres) instead of
                  the deformable-DETR decoder with 900 queries and Hungarian matching, which costs far more and converges slowly
"""
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import backbone as BB

N_REG = 10  # dx, dy (in-cell offset), z, log l, log w, log h, sin yaw, cos yaw, vx, vy


@dataclass(frozen=True)
class Config:
    bev: int = 32
    range_m: float = 32.0
    dim: int = 64
    heads: int = 4
    layers: int = 2
    ffn: int = 128
    pillar_z: tuple = (0.4, 1.0, 1.6, 2.6)
    sca_points: int = 2
    tsa_points: int = 4
    classes: int = 5
    stride: int = 8
    backbone: str = "scratch"  # or "bevformer_r50" (frozen, 512 channels)
    temporal: bool = True

    @property
    def cell(self):
        return 2 * self.range_m / self.bev

    @property
    def feat_in(self):
        return 512 if self.backbone == "bevformer_r50" else 64


def grid_xy(cfg):
    """Cell centres in the ego frame [bev, bev, 2]; axis 0 runs along x (forward), axis 1 along y (left)."""
    c = -cfg.range_m + (np.arange(cfg.bev) + 0.5) * cfg.cell
    return np.stack(np.meshgrid(c, c, indexing="ij"), -1).astype(np.float32)


def pillars(cfg):
    """The SCA reference points: every cell lifted to the pillar heights -> [Q, Z, 3]."""
    g = grid_xy(cfg).reshape(-1, 2)
    Z = len(cfg.pillar_z)
    return np.concatenate([np.repeat(g[:, None], Z, 1), np.broadcast_to(np.array(cfg.pillar_z, np.float32)[None, :, None], (len(g), Z, 1))], -1)


# ------------------------------------------------------------------------------------------------------------- pieces
def _dense(key, i, o, scale=1.0, bias=0.0):
    return {"w": jax.random.normal(key, (i, o)) * scale / np.sqrt(i), "b": jnp.full((o,), bias, jnp.float32)}


def lin(p, x):
    return x @ p["w"] + p["b"]


def norm(p, x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * p["g"] + p["b"]


def bilinear(v, x, y):
    """v [G, h, w, D], x / y [G, N] in pixel coordinates (0 = the centre of the first pixel) -> [G, N, D]; zero outside."""
    G, h, w, D = v.shape
    flat = v.reshape(G, h * w, D)
    x0, y0 = jnp.floor(x), jnp.floor(y)
    out = 0.0
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi, yi = x0 + dx, y0 + dy
        wgt = (1 - jnp.abs(x - xi)) * (1 - jnp.abs(y - yi)) * ((xi >= 0) & (xi < w) & (yi >= 0) & (yi < h))
        idx = (jnp.clip(yi, 0, h - 1) * w + jnp.clip(xi, 0, w - 1)).astype(jnp.int32)
        out = out + jnp.take_along_axis(flat, idx[..., None], axis=1) * wgt[..., None]
    return out


# ------------------------------------------------------------------------------------------------------------- init
def init(key, cfg=Config()):
    D, H, Z, P, T = cfg.dim, cfg.heads, len(cfg.pillar_z), cfg.sca_points, cfg.tsa_points
    ks = iter(jax.random.split(key, 64))
    p = {"query": jax.random.normal(next(ks), (cfg.bev * cfg.bev, D)) * 0.1, "pos": jax.random.normal(next(ks), (cfg.bev * cfg.bev, D)) * 0.1,
         "layers": []}
    if cfg.backbone == "scratch":
        p["backbone"] = BB.scratch_init(next(ks))
    for _ in range(cfg.layers):
        p["layers"].append({
            "sca": {"value": _dense(next(ks), cfg.feat_in, D), "off": _dense(next(ks), D, H * Z * P * 2, 0.01),
                    "attn": _dense(next(ks), D, H * Z * P, 0.01), "out": _dense(next(ks), D, D)},
            "tsa": {"value": _dense(next(ks), D, D), "off": _dense(next(ks), 2 * D, 2 * H * T * 2, 0.01),
                    "attn": _dense(next(ks), 2 * D, 2 * H * T, 0.01), "out": _dense(next(ks), D, D)},
            "ffn": {"a": _dense(next(ks), D, cfg.ffn), "b": _dense(next(ks), cfg.ffn, D)},
            "n1": {"g": jnp.ones(D), "b": jnp.zeros(D)}, "n2": {"g": jnp.ones(D), "b": jnp.zeros(D)}, "n3": {"g": jnp.ones(D), "b": jnp.zeros(D)}})
    p["head"] = {"conv": {"w": jax.random.normal(next(ks), (3, 3, D, D)) * np.sqrt(2.0 / (9 * D)), "b": jnp.zeros(D)},
                 "heat": _dense(next(ks), D, cfg.classes, 0.01, -2.19), "reg": _dense(next(ks), D, N_REG, 0.1)}
    return p


# ------------------------------------------------------------------------------------------------------------- attention
def project(cfg, ego2img, img_hw):
    """Pillar points -> feature-map coordinates [B, C, Q, Z, 2] and validity [B, C, Q, Z] (in front of the camera, inside the image)."""
    ref = jnp.asarray(pillars(cfg))
    h = jnp.concatenate([ref, jnp.ones(ref.shape[:-1] + (1,))], -1)
    q = jnp.einsum("bcij,qzj->bcqzi", ego2img, h)
    z = q[..., 2]
    uv = q[..., :2] / jnp.maximum(z, 1e-3)[..., None]
    H_, W_ = img_hw
    valid = (z > 0.1) & (uv[..., 0] >= 0) & (uv[..., 0] < W_) & (uv[..., 1] >= 0) & (uv[..., 1] < H_)
    return uv / cfg.stride - 0.5, valid


def sca(p, cfg, x, pos, feats, loc, valid):
    """Spatial cross-attention. x [B, Q, D]; feats [B, C, h, w, F]; loc [B, C, Q, Z, 2]; valid [B, C, Q, Z]."""
    B, C, h, w, _ = feats.shape
    Q, D, H, Z, P = x.shape[1], cfg.dim, cfg.heads, len(cfg.pillar_z), cfg.sca_points
    v = lin(p["value"], feats).reshape(B, C, h, w, H, D // H).transpose(0, 1, 4, 2, 3, 5)      # [B, C, H, h, w, Dh]
    off = lin(p["off"], x + pos).reshape(B, Q, H, Z, P, 2)
    att = jax.nn.softmax(lin(p["attn"], x + pos).reshape(B, Q, H, Z * P), -1).reshape(B, Q, H, Z, P)
    pts = loc[:, :, :, None, :, None, :] + off[:, None]                                           # [B, C, Q, H, Z, P, 2]
    pts = pts.transpose(0, 1, 3, 2, 4, 5, 6).reshape(B * C * H, Q * Z * P, 2)
    s = bilinear(v.reshape(B * C * H, h, w, D // H), pts[..., 0], pts[..., 1]).reshape(B, C, H, Q, Z, P, D // H)
    wgt = att.transpose(0, 2, 1, 3, 4)[:, None] * valid[:, :, None, :, :, None]                   # [B, C, H, Q, Z, P]
    per_cam = (s * wgt[..., None]).sum((4, 5))                                                    # [B, C, H, Q, Dh]
    hits = valid.any(-1).sum(1)                                                                   # [B, Q] cameras that see the pillar
    out = per_cam.sum(1) / jnp.maximum(hits, 1)[:, None, :, None]                                 # [B, H, Q, Dh]
    return lin(p["out"], out.transpose(0, 2, 1, 3).reshape(B, Q, D))


def align(cfg, prev, R, t):
    """Warp the previous BEV [B, Q, D] into the current ego frame. R [B, 2, 2], t [B, 2]: p_now = R p_prev + t."""
    B = prev.shape[0]
    g = jnp.asarray(grid_xy(cfg)).reshape(-1, 2)
    p_prev = jnp.einsum("bji,bqj->bqi", R, g[None] - t[:, None])                                   # R^T (p_now - t)
    ix = (p_prev[..., 0] + cfg.range_m) / cfg.cell - 0.5
    iy = (p_prev[..., 1] + cfg.range_m) / cfg.cell - 0.5
    return bilinear(prev.reshape(B, cfg.bev, cfg.bev, -1), iy, ix)                                 # x along rows, y along columns


def tsa(p, cfg, x, pos, aligned):
    """Temporal self-attention over [the queries, the aligned previous BEV]."""
    B, Q, D = x.shape
    H, T = cfg.heads, cfg.tsa_points
    v = lin(p["value"], jnp.stack([x, aligned], 1)).reshape(B, 2, cfg.bev, cfg.bev, H, D // H).transpose(0, 1, 4, 2, 3, 5)
    qin = jnp.concatenate([x + pos, aligned], -1)
    off = lin(p["off"], qin).reshape(B, Q, 2, H, T, 2)
    att = jax.nn.softmax(lin(p["attn"], qin).reshape(B, Q, H, 2 * T), -1).reshape(B, Q, H, 2, T)
    g = jnp.stack(jnp.meshgrid(jnp.arange(cfg.bev), jnp.arange(cfg.bev), indexing="ij"), -1).reshape(Q, 2).astype(jnp.float32)
    pts = g[None, :, None, None, None, :] + off                                                    # [B, Q, 2, H, T, (row, col)]
    pts = pts.transpose(0, 2, 3, 1, 4, 5).reshape(B * 2 * H, Q * T, 2)
    s = bilinear(v.reshape(B * 2 * H, cfg.bev, cfg.bev, D // H), pts[..., 1], pts[..., 0]).reshape(B, 2, H, Q, T, D // H)
    out = (s * att.transpose(0, 3, 2, 1, 4)[..., None]).sum((1, 4))                               # [B, H, Q, Dh]
    return lin(p["out"], out.transpose(0, 2, 1, 3).reshape(B, Q, D))


# ------------------------------------------------------------------------------------------------------------- forward
def features(params, cfg, imgs, r50=None):
    """imgs uint8 [B, C, H, W, 3] -> [B, C, H/8, W/8, F]."""
    B, C = imgs.shape[:2]
    flat = imgs.reshape((B * C,) + imgs.shape[2:])
    f = jax.lax.stop_gradient(BB.r50_apply(r50, flat)) if cfg.backbone == "bevformer_r50" else BB.scratch_apply(params["backbone"], flat)
    return f.reshape((B, C) + f.shape[1:])


def encode(params, cfg, feats, ego2img, img_hw, prev=None, R=None, t=None):
    """feats [B, C, h, w, F], ego2img [B, C, 4, 4] -> BEV [B, Q, D]. prev / R / t: the previous BEV and the ego motion (or None)."""
    B = feats.shape[0]
    x = jnp.broadcast_to(params["query"], (B,) + params["query"].shape)
    pos = params["pos"][None]
    loc, valid = project(cfg, ego2img, img_hw)
    aligned = align(cfg, prev, R, t) if (cfg.temporal and prev is not None) else None
    for p in params["layers"]:
        x = norm(p["n1"], x + tsa(p["tsa"], cfg, x, pos, x if aligned is None else aligned))
        x = norm(p["n2"], x + sca(p["sca"], cfg, x, pos, feats, loc, valid))
        x = norm(p["n3"], x + lin(p["ffn"]["b"], jax.nn.relu(lin(p["ffn"]["a"], x))))
    return x


def head(params, cfg, bev):
    """BEV [B, Q, D] -> heatmap logits [B, bev, bev, classes], regression [B, bev, bev, N_REG]."""
    B = bev.shape[0]
    m = bev.reshape(B, cfg.bev, cfg.bev, cfg.dim)
    m = jax.nn.relu(BB.conv(m, params["head"]["conv"]["w"], params["head"]["conv"]["b"]))
    return lin(params["head"]["heat"], m), lin(params["head"]["reg"], m)


# ------------------------------------------------------------------------------------------------------------- targets, loss, decoding
def targets(cfg, boxes, cls):
    """Labels of one sample (bev.data: boxes [N, 9], cls [N]) -> heatmap [bev, bev, classes], reg [bev, bev, N_REG], mask [bev, bev]."""
    heat = np.zeros((cfg.bev, cfg.bev, cfg.classes), np.float32)
    reg = np.zeros((cfg.bev, cfg.bev, N_REG), np.float32)
    mask = np.zeros((cfg.bev, cfg.bev), np.float32)
    ii, jj = np.meshgrid(np.arange(cfg.bev), np.arange(cfg.bev), indexing="ij")
    for b, c in zip(boxes, cls):
        if c < 0:
            continue
        gx, gy = (b[0] + cfg.range_m) / cfg.cell, (b[1] + cfg.range_m) / cfg.cell
        if not (0 <= gx < cfg.bev and 0 <= gy < cfg.bev):
            continue
        i, j = int(gx), int(gy)
        sigma = max(0.6, 0.25 * max(b[3], b[4]) / cfg.cell)
        heat[..., c] = np.maximum(heat[..., c], np.exp(-((ii + 0.5 - gx) ** 2 + (jj + 0.5 - gy) ** 2) / (2 * sigma ** 2)))
        heat[i, j, c] = 1.0
        reg[i, j] = [gx - i - 0.5, gy - j - 0.5, b[2], math.log(b[3]), math.log(b[4]), math.log(b[5]), math.sin(b[6]), math.cos(b[6]), b[7], b[8]]
        mask[i, j] = 1.0
    return heat, reg, mask


def loss(heat_logit, reg_pred, heat, reg, mask):
    """CenterNet focal loss on the heatmaps + L1 on the regression at object centres, each normalised by the object count."""
    p = jax.nn.sigmoid(heat_logit)
    pos = heat >= 1.0
    lp = jax.nn.log_sigmoid(heat_logit)
    ln = jax.nn.log_sigmoid(-heat_logit)
    focal = jnp.where(pos, (1 - p) ** 2 * lp, (1 - heat) ** 4 * p ** 2 * ln)
    n = jnp.maximum(pos.sum(), 1.0)
    l_heat = -focal.sum() / n
    l_reg = (jnp.abs(reg_pred - reg).sum(-1) * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return l_heat + 0.5 * l_reg, {"heat": l_heat, "reg": l_reg}


def decode(cfg, heat_logit, reg_pred, k=60, thresh=0.1):
    """One sample's head outputs -> boxes [n, 9] (ego frame, as in bev.data), scores [n], classes [n]."""
    p = 1 / (1 + np.exp(-np.asarray(heat_logit)))
    pad = np.pad(p, ((1, 1), (1, 1), (0, 0)), constant_values=0)
    mx = np.max([pad[i:i + cfg.bev, j:j + cfg.bev] for i in range(3) for j in range(3)], 0)
    peak = (p == mx) & (p > thresh)
    idx = np.argwhere(peak)
    sc = p[peak]
    order = np.argsort(-sc)[:k]
    idx, sc = idx[order], sc[order]
    r = np.asarray(reg_pred)[idx[:, 0], idx[:, 1]]
    x = (idx[:, 0] + 0.5 + r[:, 0]) * cfg.cell - cfg.range_m
    y = (idx[:, 1] + 0.5 + r[:, 1]) * cfg.cell - cfg.range_m
    boxes = np.stack([x, y, r[:, 2], np.exp(r[:, 3]), np.exp(r[:, 4]), np.exp(r[:, 5]), np.arctan2(r[:, 6], r[:, 7]), r[:, 8], r[:, 9]], 1)
    return boxes, sc, idx[:, 2]


def n_params(tree):
    return int(sum(np.size(x) for x in jax.tree.leaves(tree)))
