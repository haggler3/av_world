"""dwm.simagents: tokens, the realism likelihood, the policy's output range, and a tiny training run that learns."""
import jax
import jax.numpy as jnp
import numpy as np

from dwm import simagents as SA


def test_tokens_cover_the_range():
    a = np.array([-20.0, -8.0, -0.01, 0.0, 2.99, 3.0, 9.0])
    t = SA.tokenize(a)
    assert t.min() == 0 and t.max() == SA.K - 1
    assert np.all(np.abs(SA.CENTERS[SA.tokenize(SA.CENTERS)] - SA.CENTERS) < 1e-9)


def test_likelihood_is_one_for_a_point_mass_and_ranks_distributions():
    gt = np.array([0, 10, 0, 0])
    assert abs(SA.likelihood(gt, gt, eps=0.0) - 1.0) < 1e-9
    near = np.array([1, 8, 1, 0])
    far = np.array([0, 0, 1, 9])
    assert SA.likelihood(gt, near) > SA.likelihood(gt, far)
    assert SA.likelihood(gt, np.zeros(4)) > 0  # smoothing: an empty sim histogram is not -inf


def test_policy_outputs_valid_commands():
    p = SA.init(jax.random.PRNGKey(0))
    obs = jax.random.uniform(jax.random.PRNGKey(1), (64, 28))
    for mode in ("sample", "argmax", "idm"):
        u, aux = SA.policy(p, mode)(obs, jax.random.PRNGKey(2))
        assert u.shape == (64, 2) and float(jnp.abs(u[:, 0]).max()) <= 1.0
        assert np.allclose(np.asarray(u[:, 1]), SA.U1_HUMAN)


def test_training_learns_a_deterministic_rule():
    rng = np.random.default_rng(0)
    obs = rng.uniform(0, 1, (4000, 28)).astype(np.float32)
    tok = (obs[:, 0] * SA.K).astype(int).clip(0, SA.K - 1)   # token = a function of feature 0
    p = SA.train(obs, tok, steps=400, batch=512, lr=3e-3, log=lambda *a: None)
    assert SA.nll(p, obs, tok) < 0.5 * np.log(SA.K)


def test_diff_ci_detects_a_clear_difference():
    r = SA.diff_ci(np.full(8, 1.0) + np.arange(8) * 0.01, np.full(8, 0.5) + np.arange(8) * 0.01, n=2000)
    assert r["lo"] > 0 and r["p"] < 0.01
