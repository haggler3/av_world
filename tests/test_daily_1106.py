import numpy as np

from avlab import friction as F
from avlab import minibench as MB


def test_irls_recovers_elasticity_and_interaction():
    rng = np.random.default_rng(1)
    mu = np.repeat([1.0, 0.8, 0.6, 0.45, 0.3], 80)
    sh = np.tile(np.repeat([0.0, 1.0], 40), 5)
    t = np.full(len(mu), 4.0)
    rate = 0.01 * mu ** -1.0 * np.where(sh > 0, mu ** -1.0, 1.0)  # slope -1 for humans, -2 for the half-AV city
    y = rng.poisson(rate * t * 100)
    b, cov, phi = F.irls(y, t, F.design(np.log(mu), sh))
    assert abs(b[2] - -1.0) < 0.15 and abs(b[3] - -1.0) < 0.2
    assert phi >= 1.0 and cov.shape == (4, 4)


def test_summarize_shapes():
    rng = np.random.default_rng(0)
    cells = {}
    for mu in (1.0, 0.6, 0.3):
        for p in (0.0, 0.5):
            cells[f"{mu}|{p}"] = {"mu": mu, "p_av": p, "crashes": rng.poisson(5 / mu, 6).tolist(), "av_crashes": [0] * 6,
                                  "hours": [3.0] * 6, "speed": [18.0] * 6}
    s = F.summarize({"cells": cells})
    assert s["humans_only"]["elasticity"] < 0 and "curvature" in s["half_AVs"] and "interaction" in s


def test_minibench_helpers():
    x = np.arange(10.0)
    p = MB.loo_predict(x, 2 * x + 1)
    assert np.allclose(p, 2 * x + 1)
    assert abs(MB.spearman(x, x ** 3) - 1) < 1e-9 and abs(MB.spearman(x, -x) + 1) < 1e-9
    rng = np.random.default_rng(0)
    q = rng.normal(size=(8, 4))
    M = np.concatenate([q, q[:, :1] * 0 + q.mean(1, keepdims=True)], 1)  # column 4 is the row mean
    S, err = MB.best_subset(M, M[:, :].mean(1) * 0 + q.mean(1), 1)
    assert S == (4,) and err < 1e-9
