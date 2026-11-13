"""Fast tests for avlab.stopline (2026-11-13): the channel formulas and the friction-tomography estimators."""
import numpy as np

from avlab import stopline as S


def test_commitment_channel_is_off_above_four_ninths():
    v, dist = np.linspace(0, 20, 50)[:, None], np.linspace(0, 80, 50)[None, :]
    dry = S.can_stop(v, dist, 1.0)
    for mu in (0.45, 0.5, 0.7, 0.85):
        assert np.array_equal(S.can_stop(v, dist, mu), dry)
    assert not np.array_equal(S.can_stop(v, dist, 0.3), dry)


def test_stoppable_is_monotone_in_friction():
    v, dist = np.linspace(0, 20, 40)[:, None], np.linspace(0, 40, 40)[None, :]
    tlen = np.full_like(dist, 50.0)
    prev = S.stoppable(v, dist, tlen, 1.0)
    for mu in (0.85, 0.7, 0.5, 0.3):
        cur = S.stoppable(v, dist, tlen, mu)
        assert not (cur & ~prev).any()  # lower friction never makes a car stoppable
        prev = cur
    assert S.stoppable(np.array(30.0), np.array(0.0), np.array(10.0), 0.1)  # short links are always held


def _synthetic(mu, n_bg=400, n_spike=30, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([2.0 + rng.exponential(1.2, n_bg), np.full(n_spike, 9 * mu - 0.01)])
    x = np.minimum(x, 9 * mu - 0.01)  # braking beyond the cap is clipped to it, as in the simulator
    return np.histogram(x, bins=np.arange(2.0, 10.0001, 0.05))[0]


def test_estimators_recover_mu():
    for mu in (1.0, 0.7, 0.5, 0.4):
        h = _synthetic(mu)
        assert abs(S.mu_hat_max(h) - mu) <= 0.01
        assert abs(S.mu_hat_spike(h, sigma=0.3, rng=np.random.default_rng(1)) - mu) <= 0.05
    assert np.isnan(S.mu_hat_max(np.zeros(160)))


def test_ratio_and_slopes():
    r, lo, hi = S.ratio_ci(10, 100, 10, 1000)
    assert abs(r - 10) < 1e-9 and lo < 10 < hi
    mus = np.array([1.0, 0.85, 0.7, 0.6, 0.5, 0.4])
    y = np.where(mus >= 0.7, mus ** -1.0, 0.7 ** -1.0 * (mus / 0.7) ** -4.0)
    s = S.loglog_slopes(mus, y)
    assert abs(s["above"] + 1) < 1e-6 and abs(s["below"] + 4) < 1e-6
