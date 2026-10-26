"""The examples run (in tiny configurations), so they cannot rot silently."""
import numpy as np
import pytest

from examples import _common as C


@pytest.fixture(autouse=True)
def no_saving(monkeypatch):
    monkeypatch.setattr(C, "save", lambda *a, **k: None)


def test_av_share_sweep_runs(tmp_path):
    from examples import av_share_sweep as A
    res = A.main(shares=(0.0, 1.0), copies=1, warm_s=20, seconds=40, out=str(tmp_path / "a.png"))
    assert len(res) == 2 and res[0]["vehicle_hours"] > 0 and (tmp_path / "a.png").exists()
    assert res[1]["av_share"] == 1.0


def test_per_copy_settings_differ():
    rows = C.per_copy(C.run_copies("oakland", [C.calibrated(p_av=0.0), C.calibrated(p_av=1.0)], warm_s=20, seconds=40))
    assert rows[0]["av_hours"] == 0 and rows[1]["av_hours"] > 0.5 * rows[1]["vehicle_hours"]


def test_gridlock_runs(tmp_path):
    from examples import gridlock as G
    rows = G.main(demands=(0.5, 2.0), copies=1, warm_s=30, seconds=60, n_slots=1024, out=str(tmp_path / "g.png"))
    assert rows[1]["vehicles"] > rows[0]["vehicles"] and (tmp_path / "g.png").exists()


def test_careful_idm_brakes_for_a_stop_line_and_cruises_otherwise():
    import jax.numpy as jnp

    from examples.my_policy import careful_idm
    o = np.zeros((2, 28), np.float32)
    o[:, 0], o[:, 3], o[:, 16] = 10 / 15, 13 / 15, 1.0          # 10 m/s, wants 13 m/s, no junction cap
    o[1, 6], o[1, 7] = 1.0, 15 / 100                            # row 1: a stop line 15 m ahead
    u, _ = careful_idm(jnp.asarray(o), None)
    assert u.shape == (2, 2) and float(u[0, 0]) > 0 and float(u[1, 0]) < -0.3


def test_watch_renders_frames():
    from av_world import watch as W
    v = W.View("oakland", 40, 12)
    v.sim.run(20)
    img, status = v.frame(1.0)
    assert img.shape == (24, 40, 3) and "vehicles" in status
    assert len(W.to_ansi(img).split("\n")) == 12


def test_hypothesis_statistics_recover_known_answers():
    from examples import hypotheses as H
    rng = np.random.default_rng(1)
    x = np.repeat(np.log([1.0, 2.0, 4.0, 8.0]), 200)
    t = np.full(len(x), 50.0)
    y = rng.poisson(t * 0.02 * np.exp(0.5 * x))  # elasticity 0.5
    b, cov, _, phi = H.poisson_glm(y, np.c_[np.ones(len(x)), x], np.log(t))
    assert abs(b[1] - 0.5) < 3 * np.sqrt(cov[1, 1]) and 0.7 < phi < 1.3
    rr, (lo, hi), p = H.rate_ratio(40, 1000.0, 20, 1000.0)
    assert np.isclose(rr, 2.0) and lo < 2.0 < hi and p < 0.05
    assert list(H.holm([0.01, 0.04, 0.03])) == [0.03, 0.06, 0.06]
