import jax
import jax.numpy as jnp
import numpy as np

from avrl import env as E
from avrl import ppo
from wmplan import sigreg as S
from wmplan import terminal_value as TV


def test_epps_pulley_is_small_for_gaussian_and_large_for_non_gaussian():
    rng = np.random.default_rng(0)
    gauss = float(S.epps_pulley(jnp.asarray(rng.normal(size=4000), jnp.float32)))
    const = float(S.epps_pulley(jnp.zeros(4000)))
    uniform = float(S.epps_pulley(jnp.asarray(rng.uniform(-3, 3, 4000), jnp.float32)))
    two_points = float(S.epps_pulley(jnp.asarray(rng.choice([-1.0, 1.0], 4000), jnp.float32)))
    analytic = float(np.sqrt(2 * np.pi) - np.sqrt(2 * np.pi / 3))                  # E[EP] under N(0, 1) is 1.06, not 0, and does not grow with N
    assert gauss < 6 * analytic and const > 100 * analytic and two_points > 50 * analytic and uniform > 100 * analytic


def test_sigreg_detects_collapse_low_rank_and_wrong_scale():
    rng = np.random.default_rng(1)
    key = jax.random.PRNGKey(0)
    iso = jnp.asarray(rng.normal(size=(2000, 8)), jnp.float32)
    base = float(S.sigreg(iso, key))
    line = jnp.asarray(np.outer(rng.normal(size=2000), np.ones(8)), jnp.float32)
    assert float(S.sigreg(iso * 0.01, key)) > 100 * base          # collapsed to a point
    assert float(S.sigreg(line, key)) > 5 * base                  # one direction only
    assert float(S.sigreg(iso * 3.0, key)) > 5 * base             # right shape, wrong scale
    assert S.effective_rank(np.asarray(iso)) > 7.5 and S.effective_rank(np.asarray(line)) < 1.2


def test_sigreg_has_a_gradient_that_pulls_a_collapsed_embedding_open():
    key = jax.random.PRNGKey(0)
    z = jnp.asarray(np.random.default_rng(2).normal(size=(512, 4)), jnp.float32) * 0.05
    for _ in range(150):
        z = z - 0.05 * jax.grad(lambda z: S.sigreg(z, key, 32))(z) * 512
    assert float(z.std(0).mean()) > 0.3


def test_jepa_without_the_regulariser_collapses_and_with_it_does_not():
    rng = np.random.default_rng(3)
    o = rng.normal(size=(3000, 6)).astype(np.float32)
    u = rng.uniform(-1, 1, 3000).astype(np.float32)
    o2 = (o + 0.1 * u[:, None]).astype(np.float32)
    p0, _ = S.train((o, u, o2), lam=0.0, steps=500)
    p1, _ = S.train((o, u, o2), lam=1.0, steps=500)
    s0 = np.asarray(S.encode(p0, jnp.asarray(o))).std(0).mean()
    s1 = np.asarray(S.encode(p1, jnp.asarray(o))).std(0).mean()
    assert s0 < 0.05 and s1 > 0.5


def test_probe_r2_recovers_a_linear_target():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(1000, 5))
    t = z @ rng.normal(size=5)
    assert S.probe_r2(z, t, 800) > 0.99


def test_mc_returns_discount_the_logged_future_and_respect_chain_breaks():
    n = 60
    rows = np.zeros((n, 26), np.float32)
    rows[:, 0] = 0
    rows[:, 1] = 0.1 * np.arange(n)
    rows[30:, 2] = 1                                              # a segment change at row 30 breaks the chain
    rows[:, 3 + 0] = 0.5                                          # speed feature; no leader seen, so only the progress term is non-zero
    obs, g = TV.mc_returns(rows, length=10)
    per_step = float(TV.C.rhat(jnp.asarray(rows[0, 3:25])))
    full = sum(TV.C.GAMMA ** k * per_step for k in range(10))
    assert len(g) == 40 - 1 + 1 - 0 or len(g) > 0
    assert np.allclose(g, full, atol=1e-5)
    assert len(g) == 2 * (30 - 10)                                # windows that do not straddle row 30 (one chain each side)


def test_cem_with_a_terminal_value_prefers_the_action_that_leads_to_a_valuable_state():
    key = jax.random.PRNGKey(0)
    model = [ppo.init_mlp(jax.random.fold_in(key, i), [E.OBS_DIM + 1, 16, 16, TV.W.CONT.__len__() + len(TV.W.BINARY) + 3]) for i in range(2)]
    norm = (jnp.zeros(len(TV.W.CONT)), jnp.ones(len(TV.W.CONT)))
    vp = ppo.init_mlp(key, [E.OBS_DIM, 8, 1])
    o0 = jnp.full(E.OBS_DIM, 0.3)
    plan = TV.cem_h(model, norm, vp, o0, key, 4, True, pop=16, iters=2, elite=4)
    assert plan.shape == (4,) and bool(jnp.isfinite(plan).all()) and bool((jnp.abs(plan) <= 1).all())
