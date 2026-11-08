import numpy as np

from avlab import cliffcause as K


def _cell(mu, rear, ped, k=4):
    return {"mu": mu, "types": {"rear-end": rear, "pedestrian": ped, "merge": 0, "bicycle": 0}, "crashes": rear + ped,
            "hours": 4.0 * k, "per_copy_crashes": [(rear + ped) / k] * k, "per_copy_hours": [4.0] * k, "rear_n": rear, "rear_closing": 2.0 * rear,
            "ped_n": ped, "ped_speed": 3.0 * ped}


def test_summarize_cliff_shares_and_hinge():
    res = {"cells": {str(m): _cell(m, 2, int(8 / m ** 3)) for m in (0.7, 0.6, 0.5, 0.4)}}
    s = K.summarize_cliff(res)
    assert len(s["rows"]) == 4 and 0 <= s["rear_share_mu_le_0.5"] <= 1 and s["hinge"]["hinge_mu"] in (0.6, 0.5)


def test_summarize_limit_recovers_slope():
    rng = np.random.default_rng(0)
    cells = {}
    for sc in (0.85, 1.0, 1.3):
        for p in (0.0, 0.5):
            y = rng.poisson(30 * sc ** 4, 12)
            cells[f"{sc}|{p}"] = {"scale": sc, "p_av": p, "crashes": y.tolist(), "hours": [4.0] * 12, "ped": 0.0, "rear": 0.0}
    s = K.summarize_limit({"cells": cells})
    assert abs(s["humans"]["elasticity"] - 4) < 1.0 and "interaction" in s
