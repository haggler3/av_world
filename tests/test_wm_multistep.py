import jax
import jax.numpy as jnp
import numpy as np

from avrl import env as E
from avrl import wm_multistep as M


def synthetic_rows(n_veh=40, steps=60, seed=0):
    """Rows [id, t, segment, obs(22), u] where the speed feature follows v' = v + 0.02 u exactly (clipped), everything else static."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_veh):
        o = rng.uniform(0.2, 0.8, E.OBS_DIM).astype(np.float32)
        o[[4, 5, 6]] = [0, 0, 1]
        o[[7, 8, 9, 10, 15]] = 0
        for t in range(steps):
            u = float(rng.uniform(-1, 1))
            rows.append([v, 0.1 * t, 0, *o, u])
            o = o.copy()
            o[0] = np.clip(o[0] + 0.02 * u, 0, 1)
    return np.asarray(rows, np.float32)


def test_window_starts_respect_vehicle_and_segment_breaks():
    rows = M.sort_rows(synthetic_rows(n_veh=3, steps=20))
    rows[25:, 2] += 1                       # a segment change inside vehicle 1
    s = M.window_starts(rows, 5)
    for i in s:
        w = rows[i:i + 5]
        assert (w[:, 0] == w[0, 0]).all() and (w[:, 2] == w[0, 2]).all()
    assert len(s) > 0 and 25 - 4 not in s and 22 not in s


def test_split_by_vehicle_never_mixes_a_vehicle():
    rows = synthetic_rows()
    te = M.split_by_vehicle(rows)
    assert not (set(rows[te, 0]) & set(rows[~te, 0])) and 0.1 < te.mean() < 0.3


def test_assemble_soft_and_hard_agree_on_confident_predictions():
    o = jnp.asarray(np.random.default_rng(0).uniform(0, 1, (4, E.OBS_DIM)), jnp.float32)
    cont = o[:, M.CONT] + 0.01
    pb = jnp.ones((4, M.NB)) * 0.999
    pl = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (4, 1))
    soft, hard = M.assemble(o, cont, pb, pl, False), M.assemble(o, cont, pb, pl, True)
    assert np.allclose(np.asarray(soft), np.asarray(hard), atol=2e-3)


def test_rollout_shapes_for_every_history_length():
    key = jax.random.PRNGKey(0)
    for hist in (1, 3):
        models = M.init(key, hist, hidden=16, n_models=2)
        o_hist = jnp.zeros((5, hist, E.OBS_DIM))
        u_seq = jnp.zeros((5, hist + 7))
        norm = (jnp.zeros(M.NC), jnp.ones(M.NC))
        preds, _ = M.rollout(models, norm, o_hist, u_seq, True, True)
        assert len(preds) == 7 and preds[0][0].shape == (5, E.OBS_DIM)


def test_rollout_training_learns_the_speed_dynamics_and_beats_a_constant_prediction():
    rows = M.sort_rows(synthetic_rows())
    for hist, K in ((1, 1), (3, 5)):
        models, norm, curve = M.fit(rows, hist, K, jax.random.PRNGKey(0), steps=400, batch=128, n_models=2)
        assert np.isfinite(curve).all() and curve[-20:].mean() < 0.5 * curve[:20].mean()
        ev = M.evaluate(models, norm, rows, hist, horizon=10, n=100)
        # the do-nothing prediction (speed stays) is wrong by 0.02 |u| per step on average; the learned model must beat it at 1 s
        assert ev["all/speed_kmh@1s"]["mae"] < 0.5 * (0.5 * 0.02 * 10) * E.V_SCALE * 3.6
