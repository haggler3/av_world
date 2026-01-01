import json
import os

import numpy as np
import pytest

from realism import following as FL
from realism import measure as M
from realism import mix as MX
from realism import stats as ST


def scene(V=2, T=50, dt=0.2):
    nan = lambda: np.full((V, T), np.nan)
    return {"id": "t", "source": "test", "dt": dt, "T": T, "x": nan(), "y": nan(), "yaw": np.zeros((V, T)), "v": nan(),
            "kind": np.array(["car"] * V), "av": np.zeros(V, bool), "length": np.full(V, 4.5), "ego": V - 1, "px": np.zeros((0, T)),
            "py": np.zeros((0, T)), "truth": None}


def following_scene(gap=8.0, v=10.0, T=50):
    """Car 0 follows car 1 at a constant speed and bumper gap; car 2 (the ego) is parked far away."""
    s = scene(3, T)
    t = np.arange(T) * 0.2
    s["x"][1] = 100 + v * t
    s["x"][0] = s["x"][1] - gap - 4.5
    s["v"][0] = s["v"][1] = v
    s["y"][:2] = 0.0
    s["x"][2], s["y"][2], s["v"][2] = -500.0, 50.0, 0.0
    return s


def test_headway_and_tail_counts_on_a_hand_built_scene():
    m = ST.measure(following_scene(gap=6.0, v=10.0))
    hw = m["hw"][np.isfinite(m["hw"])]
    assert np.allclose(hw, 0.6)                              # 6 m at 10 m/s
    moving_s, hb, tight, ttc2, ttc1 = m["tail"]
    assert tight == pytest.approx(moving_s / 2, rel=0.05)    # the follower is tight all the time; the leader has no leader
    assert hb == 0 and ttc2 == 0


def test_hard_brake_is_counted_once_per_refractory_period():
    s = following_scene(gap=30.0)
    s["v"][0, 20:] = np.maximum(10.0 - 6.0 * np.arange(30) * 0.2, 0.0)
    m = ST.measure(s)
    assert m["tail"][1] == 1


def test_identical_pools_score_no_gap():
    sc = [following_scene(gap=g) for g in (4.0, 8.0, 15.0)]
    P = ST.pool([ST.measure(s) for s in sc])
    sb = ST.scoreboard(P, P, n_boot=10, where=False)
    assert sb["gap"] < 1e-6 and all(abs(t["ratio"] - 1) < 1e-6 or t["real"] == 0 for t in sb["tails"].values())


def synthetic_tracks(n=300, T=60, sa=1.0, tau=2.0, sm=0.3, rho=0.7, seed=0):
    rng = np.random.default_rng(seed)
    dt = 0.2
    truth, obs = [], []
    for _ in range(n):
        a, e, v = 0.0, rng.normal() * sm, 10.0
        tv, ov = [], []
        for _ in range(T):
            a = np.exp(-dt / tau) * a + rng.normal() * sa * np.sqrt(1 - np.exp(-2 * dt / tau))
            e = rho * e + rng.normal() * sm * np.sqrt(1 - rho ** 2)
            v += a * dt
            tv.append(v), ov.append(v + e)
        truth.append(np.array(tv)), obs.append(np.array(ov))
    return truth, obs


def test_structure_fit_separates_sensor_from_driver():
    truth, obs = synthetic_tracks()
    f = M.fit_structure(M.structure(obs))
    assert 0.2 < f["sm"] < 0.4 and 0.5 < f["rho"] < 0.85 and 0.6 < f["sa"] < 1.5


def test_smoother_beats_the_raw_speeds():
    truth, obs = synthetic_tracks(n=40)
    nm = {"sm": 0.3, "rho": 0.7, "sa": 1.0, "tau": 2.0}
    raw = np.mean([np.mean((o - t) ** 2) for o, t in zip(obs, truth)])
    sm = np.mean([np.mean((M.rts(o, nm) - t) ** 2) for o, t in zip(obs, truth)])
    assert sm < 0.6 * raw


def test_hampel_removes_an_isolated_spike():
    v = np.full(30, 10.0)
    v[15] = 16.0
    assert np.allclose(M.hampel(v), 10.0)


def test_corrupt_adds_noise_of_the_measured_size():
    s = following_scene()
    nm = {"other": {"sm": 0.3, "rho": 0.7, "spike_rate": 0.0, "spike_sizes": [3.0]}, "ego": {"sm": 0.0, "rho": 0.0, "spike_rate": 0.0, "spike_sizes": [3.0]},
          "pos_jitter": 0.0}
    c = M.corrupt(s, nm, seed=1)
    d = (c["v"][0] - s["v"][0])
    assert 0.15 < d.std() < 0.45 and np.allclose(c["x"][0], s["x"][0])


def idm_episode(T=1.2, s0=2.0, amax=1.5, b=2.0, v0=15.0, n=80):
    """A follower driving IDM behind a leader that slows down and speeds up."""
    dt = 0.2
    t = np.arange(n) * dt
    vl = 10 + 3 * np.sin(t / 3)
    xl = np.cumsum(vl) * dt + 30.0
    x, v = 0.0, 10.0
    vs, ss, acc = [], [], []
    for k in range(n):
        s = xl[k] - x - 4.5
        a = FL.idm([T, s0, amax, b, v0], np.array(v), np.array(s), np.array(vl[k]))
        vs.append(v), ss.append(s), acc.append(float(a))
        v = max(v + float(a) * dt, 0.0)
        x += v * dt
    return {"v": np.array(vs), "a": np.array(acc), "s": np.array(ss), "vl": vl}


def test_episode_fit_recovers_idm_parameters_without_noise():
    f = FL.fit_episode(idm_episode())
    assert f["r2"] > 0.95 and abs(f["T"] - 1.2) < 0.25 and abs(f["s0"] - 2.0) < 1.0


def test_deconvolution_recovers_a_blurred_headway_distribution():
    rng = np.random.default_rng(0)
    true_T = rng.lognormal(np.log(1.0), 0.4, 3000)
    blur = lambda T: T + 0.7 + 0.1 * rng.normal(size=len(T))
    real = blur(true_T)
    B, hist = FL.deconvolve(real, lambda bank: blur(bank[:, 0]), iters=3, offset=0.3, log=lambda *_: None)
    assert hist[-1]["mean_abs_quantile_err_s"] < 0.05
    assert abs(np.median(B[:, 0]) - np.median(true_T)) < 0.1


def test_mix_weights_move_the_simulated_context_toward_the_real_one():
    real = [following_scene(v=v) for v in np.linspace(8, 14, 20)]
    sim = [following_scene(v=v) for v in np.linspace(4, 12, 30)]
    w, info = MX.weights(real, sim)
    assert abs(info["sim_context_mean_weighted"]["median_speed"] - info["real_context_mean"]["median_speed"]) < \
        abs(info["sim_context_mean"]["median_speed"] - info["real_context_mean"]["median_speed"])


def test_idm_plus_holds_the_equilibrium_gap_near_the_desired_speed():
    import jax.numpy as jnp
    from xnet import sim as X
    v, v0, T, s0 = 13.0, 14.0, 1.2, 2.0
    gap = s0 + v * T
    assert abs(float(X.idm_plus(jnp.array(v), v0, jnp.array(gap), jnp.array(v), 1.5, 2.0, T, s0))) < 0.3
    assert float(X.idm(jnp.array(v), v0, jnp.array(gap), jnp.array(v), 1.5, 2.0, T, s0)) < -0.3       # plain IDM backs off


def test_ou_residual_has_unit_variance_and_the_right_memory():
    import jax.numpy as jnp
    from xnet import sim as X
    uid = jnp.arange(4000, dtype=jnp.int32)
    tau = 2.0
    a = np.asarray(X.ou_residual(uid, jnp.float32(10.0), tau))
    b = np.asarray(X.ou_residual(uid, jnp.float32(11.0), tau))
    assert 0.8 < a.var() < 1.2
    assert abs(np.corrcoef(a, b)[0, 1] - np.exp(-1.0 / tau)) < 0.08


def test_trait_bank_is_sampled_and_reproducible():
    from nets import collect as C
    from xnet import sim as X
    w = C.World()
    X.set_trait_bank(np.array([[0.5, 1.0, 1.4, 2.2]], np.float32))
    try:
        beh = X.Behaviour(bank=1.0, crn=1.0)
        st = C.run(w, w.reset(1, 0), beh, 30.0)
        st2 = C.run(w, w.reset(1, 0), beh, 30.0)
    finally:
        X.set_trait_bank(None)
    alive, av, kind = np.asarray(st.alive[0]), np.asarray(st.av[0]), np.asarray(st.kind[0])
    cars = alive & ~av & (kind == X.CAR)
    assert cars.sum() > 5 and np.allclose(np.asarray(st.T[0])[cars], 0.5) and np.allclose(np.asarray(st.s0[0])[cars], 1.0)
    X.set_trait_bank(np.array([[0.5, 1.0, 1.4, 2.2]], np.float32))
    try:
        st3 = C.run(w, w.reset(1, 0), X.Behaviour(bank=1.0, crn=1.0, s0_scale=0.5), 30.0)
    finally:
        X.set_trait_bank(None)
    cars3 = np.asarray(st3.alive[0]) & ~np.asarray(st3.av[0]) & (np.asarray(st3.kind[0]) == X.CAR)
    assert np.allclose(np.asarray(st3.s0[0])[cars3], 0.5)
    assert np.array_equal(np.asarray(st.T), np.asarray(st2.T))


def test_new_knobs_off_by_default():
    from xnet import sim as X
    b = X.Behaviour()
    assert b.bank == 0 and b.s0_scale == 1 and b.ou_sigma == 0 and b.idm_plus == 0


# ---------------------------------------------------------------- realism gates on the full run's results
RES = "results/realism/scoreboard.json"


@pytest.fixture(scope="module")
def res():
    if not os.path.exists(RES):
        pytest.skip("run `av-world realism run` first")
    return json.load(open(RES))


def test_gate_patched_is_closer_to_real_than_default(res):
    sb = res["scoreboard"]
    assert sb["patched"]["clean"]["gap"] < sb["default"]["clean"]["gap"]


def test_gate_tight_following_and_ttc1_move_toward_real(res):
    sb = res["scoreboard"]
    for k in ("tight_share", "ttc1_share"):
        d, p = sb["default"]["clean"]["tails"][k]["ratio"], sb["patched"]["clean"]["tails"][k]["ratio"]
        assert abs(np.log(max(p, 1e-6))) < abs(np.log(max(d, 1e-6))), k


def test_gate_hard_brakes_within_a_factor_of_two(res):
    r = res["scoreboard"]["patched"]["clean"]["tails"]["hard_brakes_per_h"]["ratio"]
    assert 0.5 < r < 2.0


def test_gate_discriminator_finds_the_patched_drivers_harder_to_tell_apart(res):
    d = res["discriminator"]
    assert d["patched"]["auroc"] < d["default"]["auroc"]
