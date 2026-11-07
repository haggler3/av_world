import numpy as np

from avlab import crashmix as CM


def test_break_fit_detects_hinge():
    mus = np.repeat([1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3], 1)
    t = np.full(len(mus), 1000.0)
    lm = np.log(mus)
    rate = np.exp(-6 + 0.2 * lm + (-6.0) * np.minimum(lm - np.log(0.55), 0))  # flat, then a steep rise below 0.55
    y = np.round(rate * t * 50)  # large counts: tiny noise
    r = CM.break_fit(mus, y, t * 50)
    assert r["best_break"] == 0.55 and r["delta_aic"] > 10


def test_break_fit_single_power_law_not_favoured():
    mus = np.array([1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3])
    t = np.full(9, 5e4)
    y = np.round(1e-3 * mus ** -3.0 * t)
    assert CM.break_fit(mus, y, t)["delta_aic"] < 2


def test_ratio_ci_brackets_truth():
    rr, lo, hi = CM.ratio_ci(40, 10.0, 10, 10.0)
    assert abs(rr - 4.0) < 1e-9 and lo < 4 < hi and lo > 1.5


def test_type_counts_shapes():
    from xnet import sim as X
    m = np.zeros((3, X.NM)); m[:, X.METRICS.index("veh_s")] = 3600; m[1, X.METRICS.index("pedestrian")] = 2
    d = CM.type_counts(m)
    assert d["pedestrian"] == 2 and d["crashes"] == 2 and abs(d["hours"] - 3) < 1e-9 and CM.rate(d) == 1000 * 2 / 3
