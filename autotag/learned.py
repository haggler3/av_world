"""Learned taggers: tags no human can annotate from video, taught by the simulator's hidden truth.

Inputs are computable from trajectories alone, the same on real logs and simulated scenes: per vehicle and frame
  (speed, acceleration, has a leader, gap, closing speed, leader's acceleration, |yaw rate|, vehicles within 20 m),
over the last 5 s (25 frames at 0.2 s). Labels come from the simulator state at the window's end:
  distracted   the driver is in a distraction episode (dist_t > 0): it does not react
  tailgater    the driver's preferred IDM headway T is under 1.25 s (a style, constant per driver)
  crash_soon   this vehicle is in a crash within the next 3 s
Network: causal dilated convolutions written as shifted copies + one matmul (as nets.mindreader), one logit per tag.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from nets import common as CM

from . import rules as RU

W, STRIDE, HORIZON = 25, 5, 15
F = 8
TAGS = ("distracted", "tailgater", "crash_soon")


def agent_features(sc):
    """-> [V, T, F] per-frame features from trajectories only."""
    lead, gap = RU.leaders(sc)
    v = np.nan_to_num(sc["v"])
    V, T = v.shape
    a = np.zeros_like(v)
    a[:, 1:] = np.diff(v, axis=1) / sc["dt"]
    vl = np.where(lead >= 0, v[np.maximum(lead, 0), np.arange(T)[None]], v)
    al = np.where(lead >= 0, a[np.maximum(lead, 0), np.arange(T)[None]], 0.0)
    yr = np.zeros_like(v)
    yr[:, 1:] = np.abs(RU.wrap(np.diff(np.nan_to_num(sc["yaw"]), axis=1))) / sc["dt"]
    d = np.hypot(sc["x"][:, None] - sc["x"][None], sc["y"][:, None] - sc["y"][None])
    dens = np.sum(np.where(np.isfinite(d), d < 20.0, False), 1) - 1
    has = lead >= 0
    f = np.stack([v / 15.0, np.clip(a, -8, 5) / 3.0, has, np.where(has, np.minimum(np.nan_to_num(gap, nan=60.0), 60.0), 60.0) / 30.0,
                  np.where(has, np.clip(v - vl, -10, 10), 0.0) / 10.0, np.clip(al, -8, 5) / 3.0 * has, np.minimum(yr, 1.5), dens / 10.0], -1)
    return f.astype(np.float32)


def windows(sc, labels=True):
    """Windows ending at t = W-1, W-1+STRIDE, ... for vehicles alive over the whole window (and the horizon, when labelled)."""
    f = agent_features(sc)
    V, T, _ = f.shape
    al = sc["alive"]
    xs, ys, meta = [], [], []
    tr = sc.get("truth")
    for t in range(W - 1, T - (HORIZON if labels and tr is not None else 0), STRIDE):
        for i in np.nonzero(al[:, t - W + 1:t + 1].all(1) & (sc["kind"] != "bike"))[0]:
            xs.append(f[i, t - W + 1:t + 1])
            meta.append((i, t))
            if labels and tr is not None:
                ys.append([tr["distracted"][i, t], tr["T_head"][i] < 1.25, tr["crash"][i, t + 1:t + 1 + HORIZON].any()])
    X_ = np.asarray(xs, np.float32).reshape(-1, W, F)
    Y_ = np.asarray(ys, np.float32).reshape(-1, len(TAGS)) if ys else None
    return X_, Y_, meta


def init(key, ch=48):
    ks = jax.random.split(key, 4)
    conv = lambda k, i, o: {"w": jax.random.normal(k, (5, i, o)) * np.sqrt(2 / (5 * i)), "b": jnp.zeros(o)}
    return {"c": [conv(ks[0], F, ch), conv(ks[1], ch, ch), conv(ks[2], ch, ch)], "head": CM.init_mlp(ks[3], [2 * ch, 64, len(TAGS)])}


def apply(p, x):
    h = x
    for l, dil in zip(p["c"], (1, 2, 4)):
        T = h.shape[1]
        hp = jnp.pad(h, ((0, 0), (4 * dil, 0), (0, 0)))
        taps = jnp.concatenate([hp[:, j * dil:j * dil + T] for j in range(5)], -1)
        y = taps @ l["w"].reshape(-1, l["w"].shape[-1]) + l["b"]
        h = jax.nn.gelu(y) + (h if h.shape[-1] == y.shape[-1] else 0.0)
    return CM.mlp(p["head"], jnp.concatenate([h.mean(1), h[:, -1]], -1))


def train(X_, Y_, steps=3000, seed=0):
    pos = Y_.mean(0)
    wpos = np.clip(0.5 / np.maximum(pos, 1e-4), 1.0, 50.0)                  # rare tags are up-weighted, capped

    def loss(p, b):
        l = optax.sigmoid_binary_cross_entropy(apply(p, b["x"]), b["y"])
        return (l * (1 + (wpos - 1) * b["y"])).mean()
    p = init(jax.random.PRNGKey(seed))
    return CM.fit(loss, p, {"x": X_, "y": Y_}, steps=steps, batch=1024)


def scores(p, X_, chunk=8192):
    if len(X_) == 0:
        return np.zeros((0, len(TAGS)))
    f = jax.jit(apply)
    return np.concatenate([np.asarray(jax.nn.sigmoid(f(p, jnp.asarray(X_[i:i + chunk])))) for i in range(0, len(X_), chunk)])


def hidden_ahead_scores(sc, occ, every=5):
    """N6 from the ego's seat every `every` frames: P(a road user hidden within 25 m ahead in its lane). -> list of (t, p)."""
    e = sc["ego"]
    out = []
    for t in range(0, sc["T"], every):
        al = np.nonzero(sc["alive"][:, t])[0]
        if e not in al or len(al) < 2:
            continue
        kind = np.array([{"car": 0, "av": 0, "bus": 1, "van": 2, "bike": 3}.get(k, 0) for k in sc["kind"][al]])
        ei = int(np.nonzero(al == e)[0][0])
        tok, _ = occ.from_scene(ei, sc["x"][al, t], sc["y"][al, t], sc["yaw"][al, t], sc["length"][al], np.nan_to_num(sc["v"][al, t]), kind)
        _, a = occ.belief(tok)
        out.append((t, float(a[0])))
    return out


def hidden_ahead_truth(sc, t):
    """Sim only: is a road user (present in the full scene) hidden from the ego within 25 m ahead in its lane at frame t?"""
    from nets import occlusion as OC
    e = sc["ego"]
    al = np.nonzero(sc["alive"][:, t])[0]
    al = al[al != e]
    if len(al) == 0:
        return False
    x, y, h, L = sc["x"][al, t], sc["y"][al, t], sc["yaw"][al, t], sc["length"][al]
    ex, ey, eh = sc["x"][e, t], sc["y"][e, t], sc["yaw"][e, t]
    vis = OC.visibility(ex, ey, x, y, h, L)
    c, s = np.cos(eh), np.sin(eh)
    fx, fy = (x - ex) * c + (y - ey) * s, -(x - ex) * s + (y - ey) * c
    return bool((~vis & (fx > 0) & (fx < 25.0) & (np.abs(fy) < 2.5)).any())
