import numpy as np

from avrl import bench as B
from xnet import sim as X


def counters(av_s=3600.0, crash=0, near=0, hard=0, red=0, av_m=5000.0, shield=0, veh_s=7200.0, veh_m=9000.0):
    m = np.zeros((2, X.NM))
    i = {k: j for j, k in enumerate(X.METRICS)}
    for k, v in dict(av_veh_s=av_s, av_crash=crash, av_near=near, av_hard=hard, av_red=red, av_veh_m=av_m, av_shield=shield,
                     veh_s=veh_s, veh_m=veh_m).items():
        m[:, i[k]] = v / 2
    return m


def test_rule_based_clears_its_own_bar():
    base = {"raw": counters(crash=1, hard=10), "shielded": counters(crash=1, hard=10)}
    c = B.checks(base, base)
    assert set(c) == set(B.CHECKS) and all(v[0] for v in c.values())


def test_checks_catch_a_policy_that_leans_on_the_envelope():
    base = {"raw": counters(crash=1, hard=10), "shielded": counters(crash=1, hard=10)}
    reckless = {"raw": counters(crash=50, hard=0), "shielded": counters(crash=0, hard=2, shield=int(0.2 * 3600 / X.DT))}
    c = B.checks(reckless, base)
    assert c["safe"][0] and c["smooth"][0] and c["efficient"][0]
    assert not c["safe unaided"][0] and not c["self-reliant"][0]


def test_checks_catch_a_timid_policy():
    base = {"raw": counters(crash=1, hard=10), "shielded": counters(crash=1, hard=10)}
    timid = {"raw": counters(crash=0, av_m=2500.0), "shielded": counters(crash=0, av_m=2500.0, veh_m=6000.0)}
    c = B.checks(timid, base)
    assert c["safe unaided"][0] and not c["efficient"][0] and not c["good neighbour"][0]
