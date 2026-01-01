import numpy as np

from avlab import speedshare as S


def _res(b, p_av=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {"p_av": p_av, "scales": {}}
    for sc in (0.7, 0.85, 1.0, 1.15, 1.3):
        t = np.full(32, 2.0)
        y = rng.poisson(5 * sc ** b * t / 2.0)
        out["scales"][str(sc)] = {"crashes": y.tolist(), "veh_s": (t * 3600).tolist(),
                                  "by_type": {k: float(y.sum()) if k == "pedestrian" else 0.0 for k in S.TYPES}}
    return out


def test_holm_matches_hand_values():
    adj = S.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03]))


def test_elasticity_recovers_known_power():
    b, se, _ = S.elasticity(_res(4.0))
    assert abs(b - 4.0) < 3 * se + 0.3


def test_compare_detects_different_elasticities_and_not_equal_ones():
    rows = S.compare(_res(5.0, seed=1), _res(1.0, p_av=0.5, seed=2))
    assert rows[0]["what"] == "all crashes" and rows[0]["diff"] > 2 and rows[0]["p_holm"] < 0.01
    same = S.compare(_res(3.0, seed=3), _res(3.0, p_av=0.5, seed=4))
    assert same[0]["p"] > 0.01
    assert [r["what"] for r in rows][:2] == ["all crashes", "pedestrian"]   # types with <3 non-zero scales are skipped
