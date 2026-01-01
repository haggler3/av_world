import numpy as np

from avlab import frictionpool as FP
from avlab import whocrashes as W


def test_rate_ratio_equal_rates_and_interval():
    r = W.rate_ratio(10, 100.0, 10, 100.0)
    assert abs(r["rr"] - 1) < 1e-9 and r["lo"] < 1 < r["hi"] and r["p"] > 0.9
    r = W.rate_ratio(2, 100.0, 40, 100.0)
    assert r["rr"] < 0.1 and r["hi"] < 1 and r["p"] < 0.001


def test_rate_ratio_exposure_scaling():
    assert abs(W.rate_ratio(10, 50.0, 10, 100.0)["rr"] - 2.0) < 1e-9
    assert W.rate_ratio(0, 5.0, 0, 5.0)["n"] == 0


def test_holm_monotone():
    a = W.holm([0.01, 0.04, 0.03])
    assert a == [0.03, 0.06, 0.06] or np.allclose(a, [0.03, 0.06, 0.06])


def test_hinge_recovers_knee():
    rng = np.random.default_rng(1)
    mu = np.repeat([1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], 30)
    lam = 9 * np.where(mu > 0.7, 1.0, (0.7 / mu) ** 3.5)
    t = np.full(len(mu), 4.0)
    y = rng.poisson(lam * t * 0.5)  # a count large enough to identify the knee
    f = FP.best_hinge(y.astype(float), t, mu, np.zeros(len(mu)))
    assert f["hinge"] in (0.6, 0.7, 0.8) and f["beta"][-1] > 1.5
