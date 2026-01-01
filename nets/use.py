"""Use the trained networks: load the saved weights and ask each one its question in plain units.

    from av_world import easy
    critic = easy.nets.load("critic")              # or nets.use.load("critic", out_dir="results/nets/quick")
    critic.risk(obs, u)                             # P(bad outcome in the next 4 s) for each of the 5 actions

Inputs are the arrays the training used; each class documents its shapes and has a `from_*` helper where the input is not simply the AV
observation (`xnet.sim` with `rl=` returns the 28-feature observation of every slot as tr["obs"]).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import common as CM

ACTIONS = ("as is", "brake harder", "accelerate", "bigger gaps", "smaller gaps")


def _path(out_dir, name):
    p = os.path.join(out_dir, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found: train it first (av-world nets train ... ; quick runs live in results/nets/quick)")
    return p


class RiskCritic:
    """N2. risk(obs [B, 28], u [B, 2]) -> P(crash, near miss or > 1 m inside the barrier in 4 s) for each action: [B, 5].
    best_action(obs, u, min_gain) -> index of the action to take (0 = keep the policy's) where the predicted gain exceeds min_gain."""

    def __init__(self, out_dir="results/nets", which="interventional"):
        p, extra = CM.load_params(_path(out_dir, "critic_params.pkl"))
        self.ens, self.deltas, self.which = p[which], np.asarray(extra["deltas"]), which

    def risk(self, obs, u):
        obs, u = np.asarray(obs, np.float32), np.asarray(u, np.float32)
        out = []
        for d in self.deltas:
            x = jnp.asarray(np.concatenate([obs, u + d], 1))
            out.append(np.asarray(jnp.mean(jnp.stack([jax.nn.sigmoid(CM.mlp(m, x)[:, 0]) for m in self.ens]), 0)))
        return np.stack(out, 1)

    def best_action(self, obs, u, min_gain=0.002):
        r = self.risk(obs, u)
        k = r.argmin(1)
        return np.where(r[:, 0] - r.min(1) > min_gain, k, 0)


class ResponseModel:
    """N7. follower_effect(obs [B, 28], follower [B, 6], u [B, 2], action) -> [B, 19]: change of the follower's acceleration (m/s^2)
    over the next 0.2-3.8 s if the AV takes `action` (index into ACTIONS) instead of acting as is.
    follower = (gap behind m, follower speed m/s, follower acceleration, follower is an AV 0/1, AV speed, AV acceleration)."""

    def __init__(self, out_dir="results/nets"):
        p, extra = CM.load_params(_path(out_dir, "response_params.pkl"))
        self.ens, self.deltas = p["interventional"], np.asarray(extra["deltas"])

    def follower_effect(self, obs, follower, u, action=1):
        from .response import x_of
        x = jnp.asarray(x_of(np.asarray(obs, np.float32), np.asarray(follower, np.float32), np.asarray(u, np.float32) + self.deltas[action]))
        return np.asarray(jnp.mean(jnp.stack([CM.mlp(m, x) for m in self.ens]), 0))


class OcclusionNet:
    """N6. belief(tokens [B, 32, 10]) -> (grid [B, 8, 8] of P(a hidden road user in that 10 m cell), ahead [B] = P(someone hidden within 25 m
    ahead in the lane)). Grid axes: x ahead from -40 to 40 m (first index), y left from -40 to 40 m. from_scene builds the tokens."""

    def __init__(self, out_dir="results/nets"):
        self.p, _ = CM.load_params(_path(out_dir, "occlusion_params.pkl"))

    def belief(self, tokens):
        from .occlusion import apply
        y = np.asarray(apply(self.p, jnp.asarray(np.asarray(tokens, np.float32))))
        sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return sig(y[:, :-1]).reshape(-1, 8, 8), sig(y[:, -1])

    @staticmethod
    def from_scene(ego, x, y, heading, length, speed, kind):
        """One AV's view from world positions (metres, radians): index ego of arrays over the road users -> (tokens [1, 32, 10], visible mask)."""
        from . import occlusion as OC
        x, y, heading, length, speed, kind = (np.asarray(a, float) for a in (x, y, heading, length, speed, kind))
        c, s = np.cos(heading[ego]), np.sin(heading[ego])
        fx, fy = (x - x[ego]) * c + (y - y[ego]) * s, -(x - x[ego]) * s + (y - y[ego]) * c
        near = np.nonzero((np.abs(fx) < OC.RANGE) & (np.abs(fy) < OC.RANGE) & (np.arange(len(x)) != ego))[0]
        vis = OC.visibility(x[ego], y[ego], x[near], y[near], heading[near], length[near])
        v = near[vis][:OC.KMAX]
        dh = heading[v] - heading[ego]
        tok = np.zeros((1, OC.KMAX, 10), np.float32)
        k = kind[v]
        tok[0, :len(v)] = np.stack([fx[v] / OC.RANGE, fy[v] / OC.RANGE, np.cos(dh), np.sin(dh), speed[v] / 15.0, length[v] / 12.0,
                                    k == 0, k == 1, k == 2, k == 3], 1)
        visible = np.zeros(len(x), bool)
        visible[v] = True
        return tok, visible


class MindReader:
    """N1. read(windows [B, 25, 8]) -> dict of the driver's IDM style in real units (T s, amax m/s^2, b m/s^2, v0f, s0 m) and the
    probabilities runner / aggressive (at chance level: kept for completeness). A window is 10 s at 0.4 s of
    (speed, acceleration, gap ahead (150 = none), leader speed, distance to the line, light ahead 0/1/2, speed limit, lane change 0/1),
    as nets.mindreader.windows builds from a recording."""

    def __init__(self, out_dir="results/nets"):
        from . import mindreader as MR
        self.MR = MR
        self.p, extra = CM.load_params(_path(out_dir, "mindreader_params.pkl"))
        if "reg_mean" in extra:
            self.mu, self.sd = np.asarray(extra["reg_mean"]), np.asarray(extra["reg_sd"])
        else:                                             # weights saved before the scales were: take them from the training windows
            z = np.load(os.path.join(out_dir, "mindreader_windows.npz"))
            yr = z["y"][z["env"] < CM.split_env(z["env"])][:, list(MR.REGI)]
            self.mu, self.sd = yr.mean(0), yr.std(0) + 1e-6

    def read(self, windows):
        y = np.asarray(self.MR.apply(self.p, jnp.asarray(self.MR.normalise(np.asarray(windows, np.float32)))))
        sig = lambda z: 1 / (1 + np.exp(-z))
        reg = y[:, 3:] * self.sd + self.mu
        return {"runner": sig(y[:, 0]), "aggressive": sig(y[:, 1]), **{k: reg[:, j] for j, k in enumerate(self.MR.REG)}}


class BlameNet:
    """N5. rank(candidates [B, 8, 18], mask [B, 8]) -> [B, 8] scores: high = a but-for cause. question "careful" (a careful driver in that
    seat avoids the crash) or "mistake" (that driver's mistake caused it). Features as nets.blame.run_branches writes them."""

    def __init__(self, out_dir="results/nets", question="careful"):
        self.p, _ = CM.load_params(_path(out_dir, f"blame_{question}_params.pkl"))

    def rank(self, candidates, mask):
        from .blame import apply
        s = np.asarray(apply(self.p, jnp.asarray(np.asarray(candidates, np.float32)), jnp.asarray(np.asarray(mask, bool))))
        return np.where(mask, 1 / (1 + np.exp(-s)), 0.0)


class RealTrafficPosterior:
    """N4. sample(stats [6], n) -> {knob: [n] values}; real(n) samples the posterior for the real Argoverse 2 statistics.
    stats = (log10 hard brakes per hour + 1, log10 headway < 0.8 s share, log10 TTC < 2 s share, log10 TTC < 1 s share, median moving
    speed / 10, SD of moving acceleration), as nets.sbi.summary computes them from scenes."""

    def __init__(self, out_dir="results/nets"):
        self.p, extra = CM.load_params(_path(out_dir, "sbi_params.pkl"))
        self.mu, self.sd = np.asarray(extra["stats_mean"]), np.asarray(extra["stats_sd"])

    def sample(self, stats, n=1000, seed=0):
        from . import sbi as SB
        u = SB.sample(self.p, (np.asarray(stats, np.float32) - self.mu) / self.sd, n, jax.random.PRNGKey(seed))
        return SB.to_theta(u)

    def real(self, n=1000, seed=0):
        import json
        from . import sbi as SB
        return self.sample(np.asarray(json.load(open(SB.REAL_STATS))["stats"]), n, seed)


LOADERS = {"critic": RiskCritic, "response": ResponseModel, "occlusion": OcclusionNet, "mindreader": MindReader, "blame": BlameNet,
           "sbi": RealTrafficPosterior}


def load(name, out_dir="results/nets", **kw):
    if name not in LOADERS:
        raise KeyError(f"{name!r} has no loader; networks: {', '.join(LOADERS)}")
    return LOADERS[name](out_dir, **kw)
