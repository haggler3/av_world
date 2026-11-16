"""2026-11-16: knee profile (aggregation, profile, knee-vs-curve on synthetic data), gap-confirm summary, doctor interpreter check."""
import sys

import numpy as np

from avlab import knee as K
from avlab import kneeprofile as KP


def _synthetic(hinge=None, seed=0, E=30):
    rng = np.random.default_rng(seed)
    mus = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0])
    mu = np.repeat(mus, E); run = np.zeros(len(mu)); t = np.full(len(mu), 20.0)
    lr = np.log(0.01) - 2.0 * np.log(mu)
    if hinge is not None:
        lr = lr + 4.0 * np.maximum(np.log(hinge) - np.log(mu), 0)
    return rng.poisson(t * np.exp(lr)).astype(float), t, mu, run


def test_aggregate_keeps_deviance_differences():
    y, t, mu, run = _synthetic(hinge=0.6)
    ya, ta, ma, ra = KP.aggregate(y, t, mu, run)
    assert len(ya) == 7 and ya.sum() == y.sum() and np.isclose(ta.sum(), t.sum())
    d_copy = K.fit(y, t, K.design(mu, run))["dev"] - K.fit(y, t, K.design(mu, run, 0.6))["dev"]
    d_cell = K.fit(ya, ta, K.design(ma, ra))["dev"] - K.fit(ya, ta, K.design(ma, ra, 0.6))["dev"]
    assert np.isclose(d_copy, d_cell, rtol=1e-6, atol=1e-6)


def test_profile_finds_a_planted_hinge():
    y, t, mu, run = KP.aggregate(*_synthetic(hinge=0.6, seed=1, E=200))
    grid = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8)
    _, drops = KP.profile(y, t, mu, run, grid)
    assert abs(grid[int(np.argmax(drops))] - 0.6) <= 0.05
    lo, hi = KP.likelihood_set(grid, drops)
    assert lo <= 0.6 <= hi


def test_knee_vs_curve_sign():
    # a strong planted kink beats a quadratic; a pure power law gives no big advantage to either
    y, t, mu, run = KP.aggregate(*_synthetic(hinge=0.6, seed=2, E=300))
    T, qdrop, _ = KP.knee_vs_curve(y, t, mu, run, (0.5, 0.55, 0.6, 0.65, 0.7))
    assert T > 0 and qdrop > 0
    y, t, mu, run = KP.aggregate(*_synthetic(hinge=None, seed=3, E=300))
    T0, _, _ = KP.knee_vs_curve(y, t, mu, run, (0.5, 0.55, 0.6, 0.65, 0.7))
    assert abs(T0) < 10


def test_gapconfirm_summary_and_u1():
    from avlab import gapconfirm as G
    assert np.isclose(G.GSC0 + G.GSC_SPAN * np.tanh(G.U1_MAX), G.GSC_MAX)
    cell = lambda k, h: {"av_crashes": [k / 2, k / 2], "av_hours": [h / 2, h / 2], "av_km": [7.5 * h / 2] * 2,  # noqa: E731
                         "crashes": [k, k], "hours": [h, h], "av_finished": [5, 5], "av_merges": [1, 1]}
    res = {"cells": {"blind@0.4": cell(40, 1000), "gap@0.4": cell(10, 1000)}}
    s = G.summarise(res)
    t = s["tests"]["gap@0.4 vs blind@0.4"]
    assert np.isclose(t["av_rr"]["rr"], 0.25) and t["av_rr"]["p"] < 0.05 and s["confirmed"]
    assert np.isclose(t["av_speed_ratio"], 1.0)


def test_doctor_interpreter_mismatch():
    from av_world.doctor import interpreter_mismatch
    assert interpreter_mismatch(python=sys.executable) is None
    assert interpreter_mismatch(python="") is None
