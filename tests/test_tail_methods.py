"""The five long-tail methods: their building blocks, on constructed inputs (the experiments themselves take minutes to hours)."""
from types import SimpleNamespace

import numpy as np

from avlab import tail_calibrate as TC
from avlab import tail_counterfactual as CF
from avlab import tail_explore as EX
from avlab import tail_novelty as NV
from avlab import tail_qd as QD
from tests.test_longtail import BASE, run
from xnet import sim as X


def test_fitted_driver_knobs_shape_the_everyday_tail():
    _, calm = run(X.Behaviour(**BASE), seconds=120)
    _, wild = run(X.Behaviour(**BASE, T_med=0.8, T_sigma=0.5, accel_noise=1.0, pulse_rate=0.02), seconds=120)
    pick = lambda s: np.asarray(s.st.alive & ~s.st.av)  # noqa: E731
    assert abs(np.median(np.asarray(wild.st.T)[pick(wild)]) - 0.8) < 0.25  # lognormal around T_med
    assert np.asarray(wild.st.T)[pick(wild)].std() > np.asarray(calm.st.T)[pick(calm)].std() * 0.5
    m_calm = float(np.asarray(calm.st.m)[:, X.METRICS.index("av_hard")].sum())
    assert m_calm >= 0
    _, base = run(X.Behaviour(**BASE), seconds=60, n=2)
    assert float(np.asarray(base.st.pulse_t).max()) == 0.0  # nothing happens when the knobs are off


def test_calibration_loss_is_zero_at_the_target_and_grows_with_distance():
    real = {k: 0.01 for k in TC.STATS}
    assert TC.loss(real, real) == 0.0
    far = {k: 0.001 for k in TC.STATS}
    assert TC.loss(far, real) > TC.loss({k: 0.005 for k in TC.STATS}, real) > 0


def test_novelty_ranks_outliers_first_and_enrichment_is_computed():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(5000, 6))
    rows = np.vstack([rng.normal(size=(1000, 6)), rng.normal(size=(20, 6)) * 0.2 + 6.0])
    s = NV.novelty(ref, rows)
    assert s[-20:].min() > np.sort(s[:1000])[-50]  # the 20 outliers all outrank all but the most extreme 5 % of ordinary rows
    y = np.zeros(len(rows), bool)
    y[-20:-10] = True
    e = NV.enrich(s, y, "t")
    assert e["top_0.01"]["enrichment"] > 10 and e["auc"] > 0.95


def test_quality_diversity_genome_decoding_and_description():
    for g in (np.zeros(len(QD.KEYS)), np.ones(len(QD.KEYS)), np.full(len(QD.KEYS), 0.5)):
        kw = QD.decode(g)
        for k, (lo, hi, _) in QD.GENES.items():
            assert lo - 1e-9 <= kw[k] <= hi + 1e-9
    d = {a: 0 for a in X.AV_CRASH} | {"av_pedestrian": 3, "imp_ped_n": 3, "imp_ped_speed": 18.0, "imp_rear_n": 0, "imp_rear_closing": 0.0}
    assert QD.describe(d) == (5, 2)  # pedestrian crashes at a mean 6 m/s
    assert QD.describe({a: 0 for a in X.AV_CRASH} | {"imp_ped_n": 0, "imp_ped_speed": 0, "imp_rear_n": 0, "imp_rear_closing": 0}) is None


def test_go_explore_cells_and_danger():
    n, K = 4, 3
    feat = np.zeros((n, K, 35), np.float32)
    feat[..., 0] = 10 / 15.0  # moving at 10 m/s
    feat[..., 5] = 1.0        # on a link
    feat[..., 28] = 3.0       # 30 m of room
    feat[1, 0, 28] = -0.2     # copy 1: an AV 2 m inside its barrier
    feat[2, 0, 5] = -1.0      # copy 2: an AV in a junction box
    feat[3, 0, 31] = 0.7      # copy 3: a red-runner on the road
    live = np.ones((n, K), bool)
    D, cell = EX.danger_and_cell(feat, live)
    assert D[1] > D[0] and D[0] == 0.0
    assert len({int(c) for c in cell}) == 4  # four different kinds of situation
    D2, cell2 = EX.danger_and_cell(feat, live, "mistakes")
    assert D2[3] > D2[0]


def test_counterfactual_neighbours():
    st = SimpleNamespace(trk=np.array([[5, 5, 5, 7, 5]]), s=np.array([[10.0, 30.0, 4.0, 11.0, 20.0]]), alive=np.ones((1, 5), bool))
    assert CF.neighbours(st, 0) == (4, 2)  # ahead: the one at 20 m; behind: the one at 4 m; slot 3 is on another track
