"""N4: amortised inference of how real people drive. Which simulator knobs explain real Argoverse 2 traffic, and how sure can we be?

Simulation-based inference: draw knob vectors theta from a prior, simulate each world, summarise it with the statistics that can also
be measured on real logs, and train a network q(theta | stats) on the pairs. Applied to the real statistics it gives a whole posterior
in one forward pass, instead of the single point the cross-entropy fit of avlab/tail_calibrate finds after minutes of simulation.

  theta (9)   p_red, log10 distraction rate, p_aggressive, T_med, T_sigma, accel_noise, pulse_rate, log10 ped_scale, p_ped_violate
  stats (6)   log10 of hard brakes per hour, headway < 0.8 s share, TTC < 2 s share, TTC < 1 s share (avlab.incidents.scene_metrics, the
              code that measured real.json) + median moving speed + SD of moving acceleration, on 60 s of one copy of Oakland
  q           a mixture of 8 diagonal Gaussians in the logit space of the prior box (a mixture-density network)

Checks: simulation-based calibration (the rank of the true theta among posterior draws must be uniform on held-out worlds), the
posterior on the real statistics, and a posterior-predictive check against the point fit.

    python -m nets.sbi [n_worlds]          # ~25 min -> results/nets/sbi.json
"""
import glob
import json
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from avlab import incidents as IN  # noqa: E402
from avrl import ladder as LD  # noqa: E402
from xnet import carla_export as CE  # noqa: E402
from xnet import sim as X  # noqa: E402

from . import collect as C  # noqa: E402
from . import common as CM  # noqa: E402

OUT = "results/nets"
KNOBS = (("p_red", 0.0, 0.4, False), ("distract_rate", -7.0, -3.5, True), ("p_aggressive", 0.0, 0.6, False), ("T_med", 0.6, 1.6, False),
         ("T_sigma", 0.0, 0.7, False), ("accel_noise", 0.0, 1.0, False), ("pulse_rate", 0.0, 0.02, False), ("ped_scale", -0.5, 0.5, True),
         ("p_ped_violate", 0.0, 0.4, False))
STATS = ("hard_brakes_per_h", "tight_follow_share", "ttc_under_2s_share", "ttc_under_1s_share", "median_speed", "acc_sd")
EPS = (1.0, 2e-4, 2e-4, 5e-5)
NC = 4
LS_MIN = -2.5  # floor on a component's log-SD in logit units: stops single components collapsing onto training worlds
D = len(KNOBS)


def to_theta(u):
    """Unit cube [n, 9] -> knob values."""
    out = {}
    for j, (k, lo, hi, lg) in enumerate(KNOBS):
        v = lo + (hi - lo) * u[:, j]
        out[k] = 10.0 ** v if lg else v
    return out


def behaviours(u):
    base = X.Behaviour(**LD.knobs(), p_av=0.0)
    th = to_theta(u)
    rows = [base._replace(**{k: float(th[k][i]) for k in th}) for i in range(len(u))]
    return jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *rows)


def feat(s, tr, pc):
    return {"alive": s.alive, "trk": s.trk, "s": s.s, "length": s.length.astype(jnp.float32), "t0": s.t0, "av": s.av, "kind": s.kind.astype(jnp.int8),
            "v": s.v, "wreck_t": s.wreck_t.astype(jnp.float16), "ped_t": s.ped_t.astype(jnp.float16)}


def summary(scenes):
    """Pooled statistics of a list of scenes: the four tail rates (as avlab.incidents) + median speed and acceleration SD while moving."""
    tot, vs, accs = None, [], []
    for sc in scenes:
        m = IN.scene_metrics(sc)
        tot = m if tot is None else {k: tot[k] + m[k] for k in m}
        x, y, yaw, v, ln, alive = IN.arrays(sc)
        a = np.diff(v, axis=1) / sc["dt"]
        mv = (v[:, 1:] > 1.0) & alive[:, 1:] & alive[:, :-1]
        vs.append(v[:, 1:][mv]), accs.append(a[mv])
    r = IN.rates(tot)
    v, a = np.concatenate(vs), np.concatenate(accs)
    a = a[np.abs(a) < 10]
    return np.array([np.log10(r[k] + e) for k, e in zip(STATS[:4], EPS)] + [np.median(v) / 10.0, a.std()], np.float32)


def simulate(w, u, warm=100.0, seconds=60.0, seed=0):
    beh = behaviours(u)
    st = C.run(w, w.reset(len(u), seed), beh, warm)
    _, f = C.record(w, st, beh, seconds, 1, feat)
    f = C.to_host(f, np.float32)
    out = []
    for e in range(len(u)):
        frames = [SimpleNamespace(**{k: f[k][t, e] for k in f}) for t in range(f["alive"].shape[0])]
        out.append(summary([CE.trajectories(w.net_np, w.geo, frames)]))
    return np.stack(out)


def real_stats(root, limit=400):
    from xnet import avdata as AV
    scenes = [AV.av2_to_scene(AV.read_av2(d), 2) for d in sorted(glob.glob(os.path.join(root, "*")))[:limit]]
    return summary(scenes)


# ---------------------------------------------------------------- the mixture-density posterior (in logit space of the box)
def logit(u):
    u = np.clip(u, 1e-4, 1 - 1e-4)
    return np.log(u / (1 - u))


def init(key, d_in=len(STATS), h=128):
    return CM.init_mlp(key, [d_in, h, h, NC * (1 + 2 * D)])


def mdn(p, s):
    y = CM.mlp(p, s)
    w = jax.nn.log_softmax(y[:, :NC], -1)
    mu = y[:, NC:NC + NC * D].reshape(-1, NC, D)
    ls = jnp.clip(y[:, NC + NC * D:].reshape(-1, NC, D), LS_MIN, 2.0)
    return w, mu, ls


def nll(p, b):
    w, mu, ls = mdn(p, b["s"])
    z = (b["z"][:, None, :] - mu) / jnp.exp(ls)
    lp = (-0.5 * z ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1) + w
    # density of u (the box) = density of z times the logit Jacobian, which is constant in the parameters: the loss in z is enough
    return -jax.scipy.special.logsumexp(lp, -1).mean()


def sample(p, s, n, key):
    w, mu, ls = mdn(p, jnp.asarray(s[None]))
    k1, k2 = jax.random.split(key)
    c = jax.random.categorical(k1, w[0], shape=(n,))
    z = mu[0, c] + jnp.exp(ls[0, c]) * jax.random.normal(k2, (n, D))
    return np.asarray(jax.nn.sigmoid(z))                               # back to the unit cube


def main(n_worlds=1024, batch=32):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cache = f"{OUT}/sbi_sims.npz"
    w = C.World()
    if os.path.exists(cache):
        z = np.load(cache)
        U, S = z["U"], z["S"]
    else:
        rng = np.random.default_rng(0)
        U = rng.uniform(size=(n_worlds, D)).astype(np.float32)
        S = np.zeros((n_worlds, len(STATS)), np.float32)
        for i in range(0, n_worlds, batch):
            S[i:i + batch] = simulate(w, U[i:i + batch], seed=i)
            print(f"{i + batch} worlds ({time.time() - t0:.0f}s)", flush=True)
        np.savez(cache, U=U, S=S)
    mu_s, sd_s = S.mean(0), S.std(0) + 1e-6
    Sn = (S - mu_s) / sd_s
    n_te = len(U) // 8
    tr, te = slice(n_te, None), slice(0, n_te)
    p = init(jax.random.PRNGKey(0))
    # 8 noisy copies of every training world (stat noise 0.1 SD): a smoother, less over-confident posterior from 900 simulations
    rng = np.random.default_rng(1)
    s_aug = np.concatenate([Sn[tr] + 0.1 * rng.normal(size=Sn[tr].shape) for _ in range(8)]).astype(np.float32)
    z_aug = np.concatenate([logit(U[tr])] * 8).astype(np.float32)
    p, curve = CM.fit(nll, p, {"s": s_aug, "z": z_aug}, steps=2500, batch=256, lr=1e-3, wd=1e-3)
    held = float(nll(p, {"s": jnp.asarray(Sn[te]), "z": jnp.asarray(logit(U[te]).astype(np.float32))}))
    # simulation-based calibration: rank of the truth among 99 posterior draws, per knob
    ranks = np.zeros((n_te, D), int)
    for i in range(n_te):
        dr = sample(p, Sn[te][i], 99, jax.random.PRNGKey(i))
        ranks[i] = (dr < U[te][i][None]).sum(0)
    hist = np.stack([np.histogram(ranks[:, j], bins=10, range=(0, 100))[0] for j in range(D)])
    # how much does the posterior shrink from the prior? (posterior SD in the unit cube / prior SD 0.289), averaged over held-out worlds
    shrink = np.zeros(D)
    for i in range(min(n_te, 64)):
        shrink += sample(p, Sn[te][i], 400, jax.random.PRNGKey(1000 + i)).std(0) / 0.2887
    shrink /= min(n_te, 64)
    real = np.asarray(json.load(open("results/nets/real_stats.json"))["stats"], np.float32) if os.path.exists("results/nets/real_stats.json") else None
    if real is not None:
        lo_, hi_ = S.min(0), S.max(0)
        print("real statistics outside every simulated world:", [n for n, r_, a, b in zip(STATS, real, lo_, hi_) if r_ < a or r_ > b], flush=True)
    if real is None:
        real = real_stats("/tmp/claude-0/-home-user/55bcdaf2-8ef3-51cf-9658-ec3ffd940c9e/scratchpad/av2")
        json.dump({"stats": real.tolist(), "names": STATS}, open("results/nets/real_stats.json", "w"))
    post = sample(p, (real - mu_s) / sd_s, 4000, jax.random.PRNGKey(7))
    th = to_theta(post)
    summ = {k: {"median": float(np.median(v)), "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95))} for k, v in th.items()}
    res = {"n_worlds": int(len(U)), "held_out_nll": held, "sbc_hist": hist.tolist(), "posterior_sd_over_prior_sd": dict(zip([k[0] for k in KNOBS], shrink.tolist())),
           "real_stats": real.tolist(), "stat_names": STATS, "posterior_real": summ, "sim_stats_range": [S.min(0).tolist(), S.max(0).tolist()],
           "sbc_chi2": [float(x) for x in (((hist - hist.sum(1, keepdims=True) / 10) ** 2) / (hist.sum(1, keepdims=True) / 10)).sum(1)],
           "outside_sim_range": [n for n, r_, a, b in zip(STATS, real, S.min(0), S.max(0)) if r_ < a or r_ > b]}
    print(json.dumps({k: v for k, v in res.items() if k != "sbc_hist"}, indent=1), flush=True)
    np.savez_compressed(f"{OUT}/sbi_posterior.npz", post=post, U=U, S=S, real=real, ranks=ranks)
    # posterior-predictive check: simulate 32 posterior draws and 32 prior draws, compare their statistics with the real ones
    pp = simulate(w, post[:32].astype(np.float32), seed=99)
    pr = simulate(w, np.random.default_rng(5).uniform(size=(32, D)).astype(np.float32), seed=99)
    dist = lambda X_: np.abs((X_ - real) / sd_s).mean(1)
    res["ppc"] = {"posterior_draws_mean_abs_z": CM.boot(dist(pp)), "prior_draws_mean_abs_z": CM.boot(dist(pr)), "posterior_stats": pp.tolist(), "prior_stats": pr.tolist()}
    print("posterior-predictive distance to the real statistics (in SDs):", res["ppc"]["posterior_draws_mean_abs_z"], "prior:", res["ppc"]["prior_draws_mean_abs_z"], flush=True)
    res["curve"] = curve[::40].tolist()
    with open(f"{OUT}/sbi.json", "w") as f:
        json.dump(res, f)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
