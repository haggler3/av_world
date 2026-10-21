"""Famous-event scenarios: every knob exists, draws respect the train/test intensity ranges, behaviours build."""
import numpy as np

from avlab import famous as FM
from xnet import sim as X


def test_every_scenario_builds_with_valid_knobs():
    rng = np.random.default_rng(0)
    for name in FM.SCENARIOS:
        for split in ("train", "test"):
            kw, u = FM.draw(name, rng, split)
            lo, hi = FM.TRAIN_U if split == "train" else FM.TEST_U
            assert lo <= u <= hi
            beh = FM.behaviour(kw)
            assert isinstance(beh, X.Behaviour)
            assert 0.0 <= float(beh.p_red) <= 0.95 and float(beh.friction) > 0.2
    FM.behaviour(FM.GENERIC)


def test_scenarios_scale_with_intensity_and_test_reaches_beyond_train():
    for name, (_, _, f, _) in FM.SCENARIOS.items():
        lo, hi = f(0.4), f(1.2)
        moved = [k for k in lo if lo[k] != hi[k]]
        assert moved, name  # intensity changes at least one knob
    assert FM.TEST_U[1] > FM.TRAIN_U[1]


def test_conditions_cover_scenarios_and_nominal_worlds():
    c = FM.conditions("test", seed=3, per=2)
    assert set(FM.SCENARIOS) <= set(c) and set(FM.NOMINAL) <= set(c)
    assert all(len(v) == 2 for v in c.values())
