"""2026-11-17: stratified exact rate ratio (avlab.gappowered), model-predicted ratios and two-cell likelihood (avlab.knee5)."""
import numpy as np

from avlab import knee as K
from avlab.gappowered import stratified_rr
from avlab.knee5 import predicted_ratio, two_cell_loglik
from avlab.whocrashes import rate_ratio


def test_single_stratum_matches_exact_rate_ratio():
    a = stratified_rr([(12, 1000.0, 27, 1050.0)])
    b = rate_ratio(12, 1000.0, 27, 1050.0)
    assert np.isclose(a["rr"], b["rr"])
    assert np.isclose(a["lo"], b["lo"], rtol=1e-3) and np.isclose(a["hi"], b["hi"], rtol=1e-3)
    assert abs(a["p"] - b["p"]) < 0.02  # doubled one-sided tail vs binomtest's two-sided rule


def test_pooling_strata_tightens_and_null_is_flat():
    one = stratified_rr([(10, 500.0, 20, 500.0)])
    three = stratified_rr([(10, 500.0, 20, 500.0)] * 3)
    assert np.isclose(three["rr"], 0.5) and three["hi"] - three["lo"] < one["hi"] - one["lo"] and three["p"] < one["p"]
    null = stratified_rr([(20, 500.0, 20, 500.0), (5, 100.0, 50, 1000.0)])
    assert np.isclose(null["rr"], 1.0) and null["p"] > 0.5 and null["lo"] < 1 < null["hi"]


def test_predicted_ratio_recovers_planted_models():
    mus = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]); run = np.zeros(len(mus)); t = np.full(len(mus), 1e7)
    m = t * np.exp(np.log(0.01) - 2.0 * np.log(mus))   # noise-free power law, slope -2
    assert np.isclose(predicted_ratio(m, t, mus, run, None), (0.65 / 0.75) ** -2.0, rtol=1e-4)
    m = t * np.exp(np.log(0.01) - 1.0 * np.log(mus) + 3.0 * np.maximum(np.log(0.7) - np.log(mus), 0))
    want = np.exp(-1.0 * np.log(0.65 / 0.75) + 3.0 * (np.log(0.7) - np.log(0.65)))
    assert np.isclose(predicted_ratio(m, t, mus, run, 0.7), want, rtol=1e-4)
    assert K.fit(m, t, K.design(mus, run, 0.7))["dev"] < 1e-6


def test_two_cell_loglik_peaks_at_observed_ratio():
    k_a, t_a, k_b, t_b = 40.0, 3000.0, 30.0, 3100.0
    obs = (k_a / t_a) / (k_b / t_b)
    rs = np.exp(np.linspace(-1, 1, 201))
    ll = [two_cell_loglik(k_a, t_a, k_b, t_b, r) for r in rs]
    assert abs(rs[int(np.argmax(ll))] / obs - 1) < 0.02


def test_cochran_q_flags_disagreeing_strata():
    from avlab.gappowered import cochran_q
    same = cochran_q([(30, 1000.0, 60, 1000.0), (31, 1000.0, 61, 1000.0)])
    diff = cochran_q([(10, 1000.0, 90, 1000.0), (90, 1000.0, 10, 1000.0)])
    assert same["p"] > 0.8 and diff["p"] < 1e-6 and diff["df"] == 1
