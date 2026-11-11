"""avlab.knee: the hinge design, the fixed-hinge test and the search-corrected bootstrap on synthetic counts (no simulation)."""
import numpy as np

from avlab import knee as K


def _data(extra, seed=0, n=40):
    rng = np.random.default_rng(seed)
    mu = np.repeat([1.0, 0.85, 0.7, 0.6, 0.5, 0.4], n)
    t = np.full(mu.shape, 20.0)
    lr = np.log(0.4) + extra * np.maximum(np.log(0.7) - np.log(mu), 0.0)  # flat above 0.7, slope -extra below
    return rng.poisson(t * np.exp(lr)).astype(float), t, mu, np.zeros_like(mu)


def test_design_columns():
    X = K.design([1.0, 0.5], [0, 1], hinge=0.7, side="below")
    assert X.shape == (2, 4)  # 1, run dummy, log x, hinge
    assert X[0, -1] == 0.0 and np.isclose(X[1, -1], np.log(0.7) - np.log(0.5))
    Xa = K.design([1.0, 1.3], [0, 0], hinge=1.0, side="above")
    assert Xa.shape == (2, 3) and Xa[0, -1] == 0.0 and np.isclose(Xa[1, -1], np.log(1.3))


def test_fixed_hinge_recovers_knee():
    y, t, mu, run = _data(extra=3.0)
    f = K.fixed_hinge(y, t, mu, run, 0.7)
    assert abs(f["slope_above"]) < 3 * f["slope_above_se"] + 0.3
    assert f["extra_lo"] < 3.0 < f["extra_hi"]
    assert f["p_one_sided"] < 1e-3


def test_bootstrap_null_is_calibrated_and_detects_knee():
    y, t, mu, run = _data(extra=0.0, seed=3)  # a pure power law (b = 0): no knee
    r0 = K.boot_sup_p(y, t, mu, run, (0.5, 0.6, 0.7, 0.85), "below", B=49, seed=1)
    assert r0["p_boot"] > 0.05
    y, t, mu, run = _data(extra=3.0, seed=4)
    r1 = K.boot_sup_p(y, t, mu, run, (0.5, 0.6, 0.7, 0.85), "below", B=49, seed=1)
    assert r1["p_boot"] <= 0.02 and r1["hinge"] in (0.6, 0.7)
