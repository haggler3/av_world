import jax
import jax.numpy as jnp
import numpy as np

from avrl import env as E
from wmplan import conservative as C


def test_keep_mask_stops_at_first_exceedance():
    s = jnp.array([0.1, 0.2, 0.9, 0.1, 0.1])
    assert np.array_equal(np.asarray(C.keep_mask(s, 0.5)), [1, 1, 0, 0, 0])      # stays off after the first breach
    assert np.array_equal(np.asarray(C.keep_mask(s, 5.0)), [1, 1, 1, 1, 1])


def test_trust_weights_fall_and_start_at_one_when_the_models_agree():
    w = np.asarray(C.trust_weights(jnp.array([0.0, 0.0, 0.3, 0.3]), tau=0.5))
    assert w[0] == 1.0 and w[1] == 1.0 and np.all(np.diff(w) <= 1e-9) and w[-1] < w[2] < 1.0


def test_objectives_reduce_to_the_plain_return_without_disagreement():
    r, s = jnp.array([1.0, -2.0, 0.5, 0.25]), jnp.zeros(4)
    base = float(C.j_naive(r, s))
    for obj in (C.j_mopo, C.j_wimle, C.j_truncate):
        assert abs(float(obj(r, s)) - base) < 1e-6


def test_mopo_prices_disagreement_and_is_monotone_in_lambda():
    r, s = jnp.ones(5), jnp.array([0.0, 0.1, 0.2, 0.4, 0.8])
    js = [float(C.j_mopo(r, s, lam=l)) for l in (0.0, 1.0, 2.0)]
    assert js[0] > js[1] > js[2]


def test_truncate_ignores_everything_after_the_breach():
    r = jnp.array([1.0, 1.0, -100.0, -100.0])
    s = jnp.array([0.0, 0.0, 2.0, 2.0])
    assert float(C.j_truncate(r, s, s_max=1.0)) > 1.0               # the catastrophe past the breach is not counted: that is the cost of truncating


def test_exploitation_gap_sign():
    assert C.exploitation_gap([2.0, 3.0], [1.0, 1.0]) == 1.5          # imagined above real = flattered
    assert C.exploitation_gap([0.0], [1.0]) == -1.0


def test_support_signal_is_zero_on_the_data_and_grows_away_from_it():
    rng = np.random.default_rng(0)
    data = (rng.normal(size=(3000, E.OBS_DIM)).astype(np.float32), rng.uniform(-1, 1, 3000).astype(np.float32))
    sup = C.make_support(data, n_ref=1000)
    near = jnp.asarray(data[0][1000:1010])
    far = near + 25.0
    u = jnp.zeros(10)
    s_near, s_far = np.asarray(C.support_signal(sup, near, u)), np.asarray(C.support_signal(sup, far, u))
    assert s_near.mean() < 0.5 and s_far.min() > 5.0 and s_far.mean() > 10 * s_near.mean()


def test_rhat_penalises_a_collision_and_a_short_gap():
    o = jnp.zeros(E.OBS_DIM).at[0].set(0.5).at[10].set(1.0).at[11].set(0.5).at[20].set(1.0)
    safe = float(C.rhat(o))
    close = float(C.rhat(o.at[11].set(0.02)))
    crash = float(C.rhat(o.at[11].set(-0.05)))
    assert safe > close > crash and crash < -20.0
    unseen = float(C.rhat(o.at[10].set(0.0).at[11].set(-0.05)))      # no leader seen: no gap terms
    assert unseen > crash
