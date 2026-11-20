"""avlab.sensornoise: the leader-feature corruption is exact when off, deterministic per key, and touches only leader features."""
import jax
import jax.numpy as jnp
import numpy as np

from avlab import sensornoise as SN


def _obs(n=64, seed=0):
    r = np.random.default_rng(seed)
    o = r.uniform(0, 1, (n, 28)).astype(np.float32)
    o[:, 8] = (r.uniform(size=n) > 0.3).astype(np.float32)  # some rows have no leader
    return jnp.asarray(o)


def test_clean_is_identity():
    o, k = _obs(), jax.random.PRNGKey(1)
    assert SN.perturb(o, k, SN.SensorModel(0.0, 0.0, 0.0)) is o
    p = np.asarray(SN.perturb(o, k, SN.SensorModel(1e-9, 0.0, 0.0)))
    assert np.allclose(p[:, [8, 9, 10]], np.asarray(o)[:, [8, 9, 10]], atol=1e-5)
    f = lambda obs, key, mask: obs  # noqa: E731
    assert SN.noisy(f, SN.CLEAN) is f


def test_only_leader_rows_and_features_change():
    o, k = _obs(), jax.random.PRNGKey(2)
    p = np.asarray(SN.perturb(o, k, SN.LEVELS["severe"]))
    o = np.asarray(o)
    other = [j for j in range(28) if j not in (8, 9, 10, 11, 25, 26)]
    assert np.array_equal(p[:, other], o[:, other])
    no_lead = o[:, 8] < 0.5
    assert np.array_equal(p[no_lead], o[no_lead])
    assert not np.allclose(p[~no_lead, 9], o[~no_lead, 9])


def test_deterministic_and_miss_all():
    o, k = _obs(), jax.random.PRNGKey(3)
    m = SN.LEVELS["moderate"]
    assert np.array_equal(np.asarray(SN.perturb(o, k, m)), np.asarray(SN.perturb(o, k, m)))
    p = np.asarray(SN.perturb(o, k, SN.SensorModel(0.0, 0.0, 1.0)))
    assert p[:, 8].sum() == 0 and np.all(p[np.asarray(o)[:, 8] > 0.5, 9] == 3.0)


def test_range_error_is_relative_lognormal():
    n = 20000
    o = jnp.zeros((n, 28)).at[:, 8].set(1.0).at[:, 9].set(0.4)  # 20 m gap
    p = np.asarray(SN.perturb(o, jax.random.PRNGKey(4), SN.SensorModel(0.2, 0.0, 0.0)))
    z = np.log(p[:, 9] / 0.4)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 0.2) < 0.01


def test_bench_analysis_on_synthetic_cells():
    from avlab import sensorbench as B
    mk = lambda k: {"av_crash": [k / 2, k / 2], "av_veh_s": [36000.0, 36000.0], "av_veh_m": [1e5, 1e5], "av_hard": [1, 1],  # noqa: E731
                    "av_shield": [0, 0], "crashes": [k, k], "av_types": {}}
    cells = {"off:clean": mk(2), "off:severe": mk(100), "off:miss": mk(90), "off:range": mk(3), "idm:severe": mk(2)}
    r = B.analyse(cells)
    assert r["H1_pass"] and r["H1_envelope_rr"]["rr"] == 0.02
    assert r["secondary"]["off:severe"]["pass"] and not r["secondary"]["off:range"]["pass"]
