"""The scorecard: rates from counters, the exposure-aware bound, the composite score's direction, and the absolute bars."""
import numpy as np
import pytest

from avrl import scorecard as SC
from xnet import sim as X


def counters(**kw):
    m = np.zeros(X.NM)
    base = dict(av_veh_s=36000.0, av_veh_m=36000.0 * 5.0, veh_s=360000.0, veh_m=360000.0 * 4.5)  # 10 AV-hours at 18 km/h
    for k, v in {**base, **kw}.items():
        m[X.METRICS.index(k)] = v
    return m


def test_rates_and_bound():
    s = SC.scorecard(counters(av_crash=2, **{"av_pedestrian": 1, "av_rear-end": 1, "rear-end": 5, "pedestrian": 1}))["metrics"]
    assert s["crash_per_h"] == pytest.approx(0.2) and s["crash_per_1000km"] == pytest.approx(2 / 180 * 1000)
    assert s["harm_per_h"] == pytest.approx((10 + 1) / 10)
    assert s["human_crash_per_h"] == pytest.approx(4 / 90)  # 6 crashes, 2 of them the AVs'
    assert s["progress"] == pytest.approx(18.0 / 16.0)  # humans: (1620 - 180) km over 90 h
    assert SC.poisson_ucb(0, 10) == pytest.approx(0.2996, abs=1e-3)  # zero crashes in 10 h: < 0.3 per hour, 95 %
    assert s["crash_ucb95_per_h"] > s["crash_per_h"]


def test_score_prefers_safer_and_smoother():
    clean = SC.scorecard(counters())
    crashy = SC.scorecard(counters(av_crash=1, av_pedestrian=1))
    harsh = SC.scorecard(counters(av_hard=200))
    assert clean["av_score"] == pytest.approx(100.0 * min(clean["metrics"]["progress"], 1.0))
    assert crashy["av_score"] < clean["av_score"] and harsh["av_score"] < clean["av_score"] and crashy["av_score"] > 0
    assert all(0 <= v <= 1 for v in crashy["factors"].values())


def test_absolute_bars_do_not_depend_on_a_baseline():
    ok = SC.absolute_checks(SC.scorecard(counters(pedestrian=4))["metrics"])
    assert all(ok.values())
    bad = SC.absolute_checks(SC.scorecard(counters(av_hard=200, av_red=1, av_near=1))["metrics"])
    assert not bad["comfortable"] and not bad["lawful"] and not bad["right of way"]
