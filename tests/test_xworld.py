import jax
import jax.numpy as jnp
import numpy as np

from xworld import geometry as G
from xworld import sim as S

CLEAN = S.Behaviour(p_red=0, p_nolook=0, p_nosignal=0, distract_rate=0, p_aggressive=0)


def test_geometry():
    assert len(G.PATHS) == 16 and len(G.CONFLICTS) > 20
    for p in G.PATHS:  # every path is continuous
        assert np.max(np.linalg.norm(np.diff(p["pts"], axis=0), axis=1)) < 5.0
    # opposing left turns pass each other without crossing (US geometry)
    assert not any({c[0], c[1]} == {0, 8} for c in G.CONFLICTS)


def test_signal_plan_never_gives_conflicting_protected_greens():
    for seg in range(len(S.SIG_END)):
        state, perm = np.asarray(S.SIG_STATE[seg]), np.asarray(S.SIG_PERM[seg])
        for p, q, *_ in G.CONFLICTS:
            both = state[p] >= 1 and state[q] >= 1
            assert not (both and not perm[p] and not perm[q]), (seg, p, q)


def test_clean_world_has_almost_no_crashes():
    m, st = S.run_batch(jax.random.PRNGKey(0), 128, 300, CLEAN, warmup=60)
    assert m[0] > 15_000  # vehicles entered
    assert m[2:7].sum() <= 2  # no risky behaviour -> (almost) no crashes
    assert m[8] / m[0] < 1e-3  # and (almost) no red-light running


def test_risky_behaviour_causes_the_matching_crash_types():
    m_red, _ = S.run_batch(jax.random.PRNGKey(1), 128, 300, CLEAN._replace(p_red=1.0), warmup=60)
    m_dis, _ = S.run_batch(jax.random.PRNGKey(1), 128, 300, CLEAN._replace(distract_rate=1 / 30), warmup=60)
    m_look, _ = S.run_batch(jax.random.PRNGKey(1), 128, 300, CLEAN._replace(p_nolook=0.3), warmup=60)
    assert m_red[8] / m_red[0] > 5e-3  # red runners run reds (late entries, which rarely end in a crash)
    assert m_dis[2] > m_dis[3] + m_dis[4]  # distraction -> mostly rear-end crashes
    assert m_look[4] + m_look[5] > m_look[2]  # not looking -> left-turn and sideswipe crashes


def test_deterministic():
    a, _ = S.run_batch(jax.random.PRNGKey(5), 16, 100, S.Behaviour(), warmup=0)
    b, _ = S.run_batch(jax.random.PRNGKey(5), 16, 100, S.Behaviour(), warmup=0)
    assert np.array_equal(a, b)
