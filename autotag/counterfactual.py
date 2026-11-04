"""Counterfactual tags (simulator only): moments whose outcome depends on what the AV does.

From the exact forks of nets/forks.py: an AV-moment is "action_sensitive" if its true outcome (bad / not bad in 4 s) differs between
the five actions. That label exists only because the city was forked with every other die held fixed. A tagger for new moments:
the interventional risk critic's spread over actions,  s = max_a Q(o, a) - min_a Q(o, a),  calibrated on held-out copies.
It needs the AV's 28 observation features, so it tags simulated runs and any AV stack that logs them; Argoverse 2 does not.
"""
import os

import numpy as np

from nets import common as CM


def evaluate(forks="results/nets/forks.npz", critic_dir="results/nets"):
    if not (os.path.exists(forks) and os.path.exists(os.path.join(critic_dir, "critic_params.pkl"))):
        return None
    from nets import use
    d = np.load(forks)
    bad = d["bad"][:, :5].astype(bool)
    truth = bad.any(1) & ~bad.all(1)
    env = d["env"]
    test = env >= CM.split_env(env)
    critic = use.RiskCritic(critic_dir)
    idx = np.nonzero(test)[0]
    r = critic.risk(d["obs"][idx], d["u_prior"][idx])
    spread = r.max(1) - r.min(1)
    return {"n": int(len(idx)), "rate": float(truth[idx].mean()), "auroc_spread": CM.auroc(spread, truth[idx]),
            "auroc_risk_as_is": CM.auroc(r[:, 0], truth[idx]), "scores": spread, "truth": truth[idx], "env": env[idx]}
