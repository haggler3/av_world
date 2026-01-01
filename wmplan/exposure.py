"""Exposure bias in an action-conditioned world model: teacher forcing vs self-forcing, on the simulator's own driving data.

    python -m wmplan.exposure --quick      # ~1 min
    python -m wmplan.exposure              # 4 variants x 3 seeds -> results/wmplan/exposure.json, docs/img/exposure.png

A world model that is trained to predict the next scene from the *true* current scene (teacher forcing) is never shown its own
mistakes, then is asked to run on its own output for a whole closed-loop rollout. Errors compound. Autoregressive video and world
models fix this by training on their own rollouts ("Self-Forcing"), which is also what makes few-step / causal students stable over
long horizons. This tests the idea in the smallest honest setting: the 28-feature scene of one AV (own motion, leader, stop target,
signal, right of way, junction conflicts, merge gaps) and its action (acceleration, gap acceptance), from wmplan/data.py.

Model, identical in every variant: x_{t+1} = x_t + g(x_t, u_t) on standardised features, g a 2-hidden-layer MLP (192 units).
    TF        loss on one step from the true x_t at every t in a 15-step window
    TF+noise  the same, with Gaussian noise (0.15 sd) added to the input x_t (the classic cheap fix)
    SF-stop   unroll the 15 steps on the model's own output, loss at every step, gradient stopped between steps
    SF        the same unroll with gradients through all 15 steps (backprop through time)
Evaluation on a held-out copy of the city (env 7): open-loop 15-step rollouts from the true x_0 with the logged actions; RMSE per step
in standardised units, on windows whose 15 steps are all valid; 95 % intervals over seeds (t) and over windows (bootstrap).
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from wmplan import model as M

OUT = "results/wmplan"
H, HID = M.H, 192
VARIANTS = ("TF", "TF+noise", "SF-stop", "SF")


def init(key, nin, nout):
    ks = jax.random.split(key, 3)
    dims = [nin, HID, HID, nout]
    return [{"w": jax.random.normal(k, (i, o)) * np.sqrt(2.0 / i) * (0.1 if j == 2 else 1.0), "b": jnp.zeros(o)}
            for j, (k, i, o) in enumerate(zip(ks, dims[:-1], dims[1:]))]


def g(p, x, u):
    h = jnp.concatenate([x, u], -1)
    for lyr in p[:-1]:
        h = jax.nn.gelu(h @ lyr["w"] + lyr["b"])
    return h @ p[-1]["w"] + p[-1]["b"]


def rollout(p, x0, us):
    """x0 [B, D], us [B, H, A] -> predicted x_1..x_H [B, H, D] (the model on its own output)."""
    def body(x, u):
        x = x + g(p, x, u)
        return x, x
    _, xs = jax.lax.scan(body, x0, jnp.moveaxis(us, 1, 0))
    return jnp.moveaxis(xs, 0, 1)


def loss_fn(p, b, variant, key):
    """b: x [B, H+1, D] true standardised scenes, u [B, H, A], valid [B, H+1]."""
    x, u, v = b["x"], b["u"], b["valid"][:, 1:]
    if variant in ("TF", "TF+noise"):
        xin = x[:, :-1]
        if variant == "TF+noise":
            xin = xin + 0.15 * jax.random.normal(key, xin.shape)
        pred = xin + g(p, xin, u)
    elif variant == "SF-stop":
        def body(xc, u_t):
            xn = jax.lax.stop_gradient(xc) + g(p, jax.lax.stop_gradient(xc), u_t)
            return xn, xn
        _, pr = jax.lax.scan(body, x[:, 0], jnp.moveaxis(u, 1, 0))
        pred = jnp.moveaxis(pr, 0, 1)
    elif variant.startswith("SFK"):  # self-forcing on the first K steps only (ablation of the unroll length)
        k = int(variant[3:])
        pred = rollout(p, x[:, 0], u[:, :k])
        v, x = v[:, :k], x[:, :k + 1]
    else:
        pred = rollout(p, x[:, 0], u)
    err = ((pred - x[:, 1:]) ** 2).mean(-1)
    return (err * v).sum() / jnp.maximum(v.sum(), 1.0)


def prepare(path=f"{OUT}/data.npz"):
    d, starts = M.load_data(path)
    d = dict(d)
    d["obs"], d["u"] = d["obs"].astype(np.float32), d["u"].astype(np.float32)  # stored as float16: statistics in float32
    mu, sd = d["obs"].mean(0), d["obs"].std(0) + 1e-6
    d["obs"] = ((d["obs"] - mu) / sd).astype(np.float32)
    umu, usd = d["u"].mean(0), d["u"].std(0) + 1e-6
    d["u"] = ((d["u"] - umu) / usd).astype(np.float32)
    tr = starts[d["env"][starts] != M.HOLDOUT_ENV]
    te = starts[d["env"][starts] == M.HOLDOUT_ENV]
    return d, tr, te


def batch(d, idx):
    j = np.minimum(idx[:, None] + np.arange(H + 1)[None], d["n"] - 1)
    valid = np.cumprod(np.concatenate([np.ones((len(idx), 1), bool), d["cont"][j[:, :-1]]], 1), 1).astype(np.float32)
    return {"x": jnp.asarray(d["obs"][j]), "u": jnp.asarray(d["u"][j][:, :H]), "valid": jnp.asarray(valid)}


def train(d, tr, variant, steps, seed, bs=512, lr=2e-3):
    key = jax.random.PRNGKey(seed)
    nin, nu = d["obs"].shape[1], d["u"].shape[1]
    p = init(key, nin + nu, nin)
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    st = opt.init(p)

    @jax.jit
    def upd(p, st, b, k):
        l, gr = jax.value_and_grad(loss_fn)(p, b, variant, k)
        u, st = opt.update(gr, st)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    for i in range(steps):
        b = batch(d, rng.choice(tr, bs))
        p, st, l = upd(p, st, b, jax.random.fold_in(key, i))
    return p


def evaluate(p, d, te, max_windows=4000, seed=0):
    """Per-step open-loop RMSE [H] on windows whose H steps are all valid, plus per-window squared error [W, H] for the bootstrap."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(te, min(max_windows, len(te)), replace=False)
    b = batch(d, idx)
    full = np.asarray(b["valid"][:, -1]) > 0
    pred = rollout(p, b["x"][:, 0], b["u"])
    se = np.asarray(((pred - b["x"][:, 1:]) ** 2).mean(-1))[full]
    pers = np.asarray(((b["x"][:, :1] - b["x"][:, 1:]) ** 2).mean(-1))[full]
    return se, pers


def ablate(ks=(3, 5, 8), steps=1000, seeds=3):
    """Ablation: how many steps must the self-rollout be? SF trained on the first K steps of each window (K = 1 is TF, K = 15 is SF,
    both from the main run). Same data, steps, seeds, held-out evaluation. -> results/wmplan/exposure_ablation.json"""
    d, tr, te = prepare()
    main_res = json.load(open(f"{OUT}/exposure.json"))["variants"]
    out = {"TF (K=1)": main_res["TF"]["rmse_by_step_seeds"], "SF (K=15)": main_res["SF"]["rmse_by_step_seeds"]}
    for k in ks:
        t0 = time.time()
        ps = np.array([np.sqrt(evaluate(train(d, tr, f"SFK{k}", steps, s), d, te)[0].mean(0)) for s in range(seeds)])
        out[f"SF (K={k})"] = ps.tolist()
        print(f"K={k}: step 1 {ps[:, 0].mean():.3f}  step 15 {ps[:, -1].mean():.3f} [{ps[:, -1].min():.3f}, {ps[:, -1].max():.3f}] ({time.time() - t0:.0f} s)", flush=True)
    json.dump(out, open(f"{OUT}/exposure_ablation.json", "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablate", action="store_true", help="unroll-length ablation (needs the main run's exposure.json)")
    a = ap.parse_args()
    if a.ablate:
        ablate()
        return
    steps, seeds = (300, 1) if a.quick else (1000, 3)
    d, tr, te = prepare()
    print(f"{len(tr)} training windows, {len(te)} held-out windows", flush=True)
    res = {"steps": steps, "seeds": seeds, "variants": {}}
    for var in VARIANTS:
        t0 = time.time()
        per_seed = []
        for s in range(seeds):
            p = train(d, tr, var, steps, s)
            se, pers = evaluate(p, d, te)
            per_seed.append(np.sqrt(se.mean(0)))
        ps = np.array(per_seed)  # [seeds, H]
        res["variants"][var] = {"rmse_by_step": ps.mean(0).tolist(), "rmse_by_step_seeds": ps.tolist(), "windows": int(len(se))}
        print(f"{var:9s} RMSE step 1 {ps[:, 0].mean():.3f}  step 5 {ps[:, 4].mean():.3f}  step 15 {ps[:, -1].mean():.3f}  ({time.time() - t0:.0f} s)", flush=True)
    res["persistence_rmse_by_step"] = np.sqrt(pers.mean(0)).tolist()
    json.dump(res, open(f"{OUT}/{'exposure_quick' if a.quick else 'exposure'}.json", "w"), indent=1)
    if not a.quick:
        plot(res, "docs/img/exposure.png")


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"
    pal = {"TF": "#e69f00", "TF+noise": "#cc79a7", "SF-stop": "#56b4e9", "SF": "#009e73"}
    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor=BG)
    ks = np.arange(1, H + 1) * 0.2
    for var, r in res["variants"].items():
        ps = np.array(r["rmse_by_step_seeds"])
        ax.plot(ks, ps.mean(0), color=pal[var], lw=2)
        if len(ps) > 1:
            ax.fill_between(ks, ps.min(0), ps.max(0), color=pal[var], alpha=0.2, lw=0)
        ax.text(ks[-1] + 0.05, ps.mean(0)[-1] + {"SF": -0.012, "SF-stop": 0.012}.get(var, 0.0), var, color=pal[var], fontsize=9, va="center")
    ax.plot(ks, res["persistence_rmse_by_step"], color=INK2, lw=1.2, ls="--")
    ax.text(ks[-1] + 0.05, res["persistence_rmse_by_step"][-1], "no model (hold the scene)", color=INK2, fontsize=8, va="center")
    ax.set_xlim(0, ks[-1] * 1.3)
    ax.set_xlabel("seconds into the open-loop rollout", color=INK2)
    ax.set_ylabel("RMSE, standardised features (held-out city copy)", color=INK2, fontsize=9)
    ax.set_title("A world model that trains on its own rollouts drifts less", color=INK, fontsize=11, loc="left")
    ax.set_facecolor(BG)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#30363d")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
