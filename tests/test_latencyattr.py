import jax.numpy as jnp
import numpy as np

from avlab import latencyattr as LA


def test_classify_states():
    has = jnp.array([False, True, True, True, False, True])
    g = jnp.array([150.0, 20.0, 20.0, 20.0, 150.0, 20.0])
    shown = jnp.array([False, True, False, True, True, True])
    gs = jnp.array([150.0, 21.5, 150.0, 26.0, 30.0, 15.0])
    assert LA.classify(has, g, shown, gs).tolist() == [0, 1, 2, 3, 4, 4]


def test_state_tallies_match_simulator():
    r = LA.run(LA.Latency(2.0, True), E=2, warm_s=40, seconds=40)
    s = np.asarray(r["state_s"]).sum(1)
    np.testing.assert_allclose(s, r["av_veh_s"], rtol=0.02)  # every alive non-wrecked AV step is in exactly one state
    assert np.array_equal(np.asarray(r["state_crash"]).sum(1), np.round(r["av_crash"]).astype(int))


def test_analyse_tests():
    fake = {"model": {"delay_s": 2.0}, "av_crash": [12.0], "state_s": [[3600.0, 3600.0, 360.0, 360.0, 0.0]],
            "state_crash": [[0, 1, 10, 1, 0]]}
    res = LA.analyse({"cv2.0": fake})
    assert res["cells"]["cv2.0"]["crash_check"]
    assert res["tests"]["H1"]["rr"] == 100.0 and res["tests"]["H1"]["p"] < 1e-6
    assert abs(res["tests"]["S1"]["share"] - 10 / 11) < 1e-9
