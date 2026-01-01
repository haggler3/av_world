"""Write your own AV policy in ~15 lines and see where it lands on AVBench.

    python examples/my_policy.py           # ~3 min: nominal + jaywalkers -> results/examples/my_policy.json
    python examples/my_policy.py --quick   # one scenario

A policy is any JAX function f(obs [N, 28], key) -> (u [N, 2], aux). obs are the AV's features (names in av_world.sil.OBS_NAMES:
speed, the leader's gap and speed, the stop target, the signal, ...), u[:, 0] in [-1, 1] is the acceleration command (3 u m/s^2
forward, 8 u m/s^2 braking) and u[:, 1] the gap acceptance at junctions (0 = default, > 0 more cautious). It is scored exactly like
the leaderboard's policies: the same traffic and random numbers, with no safety filter and under the envelope.
"""
import argparse
import json
import time

import jax.numpy as jnp

try:
    from examples import _common as C  # noqa: F401  (puts the repository on the path)
except ImportError:
    import _common as C  # noqa: F401
from avrl import bench as B  # noqa: E402
from avrl import scorecard as SC  # noqa: E402


def careful_idm(obs, key, T=1.5, s0=2.5, a_max=1.5, b=2.0):
    """The Intelligent Driver Model (Treiber et al. 2000) with a 1.5 s headway: follow the leader, and treat an active stop line
    (red light, stop sign, or no gap to enter the junction) as a stopped leader. The junction speed cap lowers the desired speed."""
    v = 15 * obs[:, 0]
    v0 = jnp.maximum(jnp.minimum(15 * obs[:, 3], 15 * obs[:, 16] + 0.5), 0.5)       # desired speed, capped before the junction
    has_lead, gap_lead, v_lead = obs[:, 8] > 0.5, 50 * obs[:, 9], 15 * obs[:, 10]
    has_stop, gap_stop = obs[:, 6] > 0.5, 100 * obs[:, 7]
    def idm_term(gap, dv):  # (s* / s)^2
        s_star = s0 + jnp.maximum(v * T + v * dv / (2 * jnp.sqrt(a_max * b)), 0.0)
        return (s_star / jnp.maximum(gap, 0.1)) ** 2
    inter = jnp.maximum(jnp.where(has_lead, idm_term(gap_lead, v - v_lead), 0.0), jnp.where(has_stop, idm_term(gap_stop, v), 0.0))
    a = a_max * (1 - (v / v0) ** 4 - inter)
    u0 = jnp.clip(jnp.where(a >= 0, a / 3.0, a / 8.0), -1.0, 1.0)
    return jnp.stack([u0, jnp.zeros_like(u0)], 1), jnp.zeros((obs.shape[0], 2))


def main(scenarios=("nominal", "jaywalkers"), policy=careful_idm, name="careful_idm (this example)"):
    t0 = time.time()
    mine = B.evaluate(policy, scenarios)
    lb = json.load(open(f"{B.OUT}/leaderboard.json"))
    others = ("rule-based AV", "PPO + barrier teacher, speed + jaywalkers (phase 3)")
    rows = [(name, mine["scenarios"])] + [(p, {n: lb["scores"][p][n] for n in scenarios}) for p in others if p in lb.get("scores", {})]
    print(f"\n{'policy':52s} {'av_world score':>14s} {'checks':>7s} {'crashes/AV-h unaided':>21s} {'comfort':>8s} {'progress':>9s}")
    out = {}
    for p, sc in rows:
        score = sum(sc[n]["av_score"] for n in scenarios) / len(scenarios)
        checks = sum(sum(SC.absolute_checks(sc[n]["metrics"]).values()) for n in scenarios)
        m = lambda k: sum(sc[n]["metrics"][k] for n in scenarios) / len(scenarios)  # noqa: E731
        print(f"{p:52s} {score:14.1f} {checks:4d}/{8 * len(scenarios):<2d} {m('crash_per_h_unaided'):21.3f} {m('comfort_pass'):8.3f} {m('progress'):9.3f}")
        out[p] = {"av_score": score, "absolute_checks": checks, "scenarios": sc}
    fails = {k: sum(not SC.absolute_checks(mine["scenarios"][n]["metrics"])[k] for n in scenarios) for k in SC.ABSOLUTE}
    print("your policy's failed checks:", {k: v for k, v in fails.items() if v} or "none", f"({time.time() - t0:.0f} s)")
    C.save("my_policy", {"scenarios": scenarios, "rows": out})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(("nominal",) if a.quick else ("nominal", "jaywalkers"))
