"""Fast tests of the math in avlab.speedlimit and avlab.twintrace (no simulation)."""
import numpy as np

from avlab import speedlimit as SL
from avlab import twintrace as TT


def test_poisson_fit_recovers_elasticity():
    rng = np.random.default_rng(0)
    scale = np.repeat([0.7, 1.0, 1.3], 400)
    t = np.full(scale.shape, 80.0)
    y = rng.poisson(t * 0.02 * scale ** 3)
    b, se, phi = SL.poisson_fit(y, t, np.log(scale))
    assert abs(b - 3.0) < 4 * se and se < 0.5 and phi >= 1.0


def test_poisson_fit_null_slope():
    rng = np.random.default_rng(1)
    x = np.repeat([-0.3, 0.0, 0.3], 300)
    t = np.full(x.shape, 50.0)
    b, se, _ = SL.poisson_fit(rng.poisson(t * 0.05), t, x)
    assert abs(b) < 4 * se


def _rec(s, trk=None):
    s = np.asarray(s, float)
    return {"alive": np.ones_like(s, bool), "s": s, "trk": np.zeros(s.shape, int) if trk is None else trk, "v": np.zeros_like(s)}


def test_first_times_and_parents():
    a = _rec([[0, 10, 20, 500]] * 4)
    b_s = np.array([[0, 10, 20, 500], [2, 10, 20, 500], [3, 10.5, 20, 500], [4, 12, 22, 500]], float)
    b = _rec(b_s)
    first, d = TT.first_times(a, b)
    assert first.tolist() == [1, 3, 3, -1] and d.sum() == 1 + 1 + 3  # car 0 from frame 1; cars 1 and 2 at frame 3 (> 1 m)
    par = TT.parents(a, first, kicked=0)
    assert par[1][0] == 0 and par[2][0] == 0 and abs(par[1][1] - 10) < 1e-9


def test_parent_none_when_far():
    a = _rec([[0, 400]] * 3)
    b = _rec([[0, 400], [5, 400], [5, 410]])
    first, _ = TT.first_times(a, b)
    par = TT.parents(a, first, kicked=0)
    assert par[1][0] == -1
