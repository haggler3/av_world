"""The Lean theorems of formal/AvFormal/Barrier.lean, checked against the simulator's own cbf_bound (so the Lean model
and the code cannot drift apart silently), and the critical speeds used by avlab.barrierfriction."""
import jax.numpy as jnp
import numpy as np

from avlab.barrierfriction import v_feas
from xnet import sim as X


def test_lean_model_matches_code():
    rng = np.random.default_rng(0)
    g, v, vl, al = rng.uniform(0, 80, 500), rng.uniform(0, 25, 500), rng.uniform(0, 25, 500), rng.uniform(-6, 3, 500)
    a, h = X.cbf_bound(jnp.asarray(g), jnp.asarray(v), jnp.asarray(vl), jnp.asarray(al))
    p = X.CBF
    h_lean = g - p["d0"] - p["tau"] * v - (v ** 2 - vl ** 2) / (2 * p["b"])
    cap_lean = ((vl - v) + vl * al / p["b"] + p["alpha"] * h_lean) / (p["tau"] + v / p["b"])
    assert np.allclose(np.asarray(h), h_lean, rtol=1e-5, atol=1e-3) and np.allclose(np.asarray(a), cap_lean, rtol=1e-4, atol=1e-3)
    assert (p["b"], p["tau"]) == (6.0, 0.5)  # the constants the corollaries in Barrier.lean use


def test_feasible_iff_two_thirds_on_samples():
    rng = np.random.default_rng(1)
    v, vl, al = rng.uniform(0, 30, 4000), rng.uniform(0, 30, 4000), rng.uniform(-6, 3, 4000)
    p = X.CBF
    g = p["d0"] + p["tau"] * v + (v ** 2 - vl ** 2) / (2 * p["b"]) + rng.uniform(0, 5, 4000)  # h >= 0
    a, h = (np.asarray(x) for x in X.cbf_bound(jnp.asarray(g), jnp.asarray(v), jnp.asarray(vl), jnp.asarray(al)))
    assert (h >= -1e-3).all()
    for mu in (2 / 3, 0.8, 1.0):  # bound_feasible
        assert (a > -9 * mu - 1e-4).all()
    for mu in (0.6, 0.5, 0.4):  # bound_infeasible: on the edge, obstacle stopped, just above v_feas
        vv = np.array([v_feas(mu) * 1.01 + 0.01, v_feas(mu) * 2 + 1])
        gg = p["d0"] + p["tau"] * vv + vv ** 2 / (2 * p["b"])
        aa, _ = X.cbf_bound(jnp.asarray(gg), jnp.asarray(vv), jnp.zeros(2), jnp.zeros(2))
        assert (np.asarray(aa) < -9 * mu).all()


def test_critical_speeds_match_lean():
    assert abs(v_feas(0.6) - 27) < 1e-9 and abs(v_feas(0.5) - 9) < 1e-9 and abs(v_feas(0.4) - 4.5) < 1e-9
    assert v_feas(2 / 3) == float("inf") and v_feas(1.0) == float("inf")
