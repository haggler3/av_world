"""Fast tests for the 2026-11-02 tools: avlab.butterfly (nudges, divergence) and avlab.watchtime (windowed features, AUC curve)."""
import jax.numpy as jnp
import numpy as np

from avlab import butterfly as B
from avlab import watchtime as W
from xnet import sim as X


class _S:  # a minimal stand-in for State with the fields the helpers read
    def __init__(self, alive, trk, s):
        self.alive, self.trk, self.s = jnp.asarray(alive), jnp.asarray(trk), jnp.asarray(s, float)


def test_divergence_and_spread():
    a = _S([1, 1, 1, 0], [0, 0, 1, 0], [0, 10, 20, 0])
    assert float(B.divergence(a, a)) == 0.0
    b = _S([1, 1, 1, 0], [0, 0, 1, 0], [0, 12, 20.5, 0])  # one slot moved 2 m, one 0.5 m (under the 1 m tolerance)
    assert abs(float(B.divergence(a, b)) - 1 / 3) < 1e-6
    assert abs(float(B.spread(a, b)) - (0 + 2 + 0.5) / 3) < 1e-6
    c = _S([1, 1, 0, 1], [0, 0, 1, 0], [0, 10, 20, 5])  # alive flags differ in two slots
    assert abs(float(B.divergence(a, c)) - 2 / 4) < 1e-6


def test_nudge_modes():
    st = X.reset(__import__("jax").random.PRNGKey(0), {"link_len": jnp.ones(3), "leg_node": jnp.zeros(4)}, 6)
    st = st._replace(alive=jnp.array([0, 0, 1, 1, 0, 1], bool), s=jnp.array([0, 0, 5, 7, 0, 9.0]))
    one = B.nudge(st, "one", 0.5)
    assert np.allclose(np.asarray(one.s), [0, 0, 5.5, 7, 0, 9])
    zero = B.nudge(st, "all", 0.0)
    assert np.allclose(np.asarray(zero.s), np.asarray(st.s))
    al = B.nudge(st, "all", 1e-3)
    d = np.asarray(al.s - st.s)
    assert (d[~np.asarray(st.alive)] == 0).all() and 0 < np.abs(d[np.asarray(st.alive)]).max() < 1e-2


def test_window_features_and_curve():
    n = int(round(10 / X.DT))
    v = np.full(n, 10.0)
    acc = np.zeros(n)
    f = W.window_features(v, acc, 5)
    assert f.shape == (2, 8) and np.allclose(f[:, 0], 10) and np.allclose(f[:, 2], 0)
    rng = np.random.default_rng(0)
    segs = []
    for k in range(40):  # AVs have a noisier acceleration, so longer windows should separate them at least as well
        is_av = k % 2 == 0
        nn = int(round(60 / X.DT))
        segs.append((10 + rng.normal(0, 1, nn), rng.normal(0, 1.0 if is_av else 0.7, nn), is_av, k % 4, k))
    r = W.curve(segs, windows=(3, 60), n_boot=50)
    assert r[60]["auc"] >= r[3]["auc"] - 0.05 and r[3]["auc"] > 0.55
