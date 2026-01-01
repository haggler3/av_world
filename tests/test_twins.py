"""avlab.twins: the displacement measure, the nudges, and the bench hook's defaults (fast, no simulation)."""
import numpy as np
import jax.numpy as jnp

from avlab import twins as T


class S:  # the few State fields the measure reads
    def __init__(self, s, alive, trk=None, v=None, key=None):
        self.s, self.alive = jnp.asarray(s, jnp.float32), jnp.asarray(alive)
        self.trk = jnp.asarray(trk if trk is not None else np.zeros(np.shape(s), int))
        self.v = jnp.asarray(v if v is not None else np.ones(np.shape(s)), jnp.float32)
        self.key = key

    def _replace(self, **kw):
        o = S(self.s, self.alive, self.trk, self.v, self.key)
        for k, x in kw.items():
            setattr(o, k, x)
        return o


def test_displaced_counts_only_vehicles_alive_in_both_and_far_enough():
    a = S([[0, 10, 20, 30]], [[1, 1, 1, 0]])
    b = S([[0.5, 12, 20, 99]], [[1, 1, 1, 1]])
    d, n = T.displaced(a, b)
    assert n[0] == 3 and abs(d[0] - 1 / 3) < 1e-9          # only the vehicle that moved 2 m; the one not alive in a is ignored


def test_a_different_track_counts_as_displaced():
    a, b = S([[5.0]], [[1]], trk=[[3]]), S([[5.0]], [[1]], trk=[[4]])
    assert T.displaced(a, b)[0][0] == 1.0


def test_nudge_changes_only_the_fastest_car_or_everyone():
    st = S([[0, 1, 2]], [[1, 1, 1]], v=[[1.0, 9.0, 3.0]])
    one = T.nudge(st, (1, 0.5), None)
    assert np.allclose(np.asarray(one.v), [[1.0, 9.5, 3.0]]) and np.allclose(np.asarray(st.v), [[1.0, 9.0, 3.0]])
    every = T.nudge(S([[0, 1]], [[1, 0]], v=[[1.0, 2.0]]), ("all", 0.25), None)
    assert np.allclose(np.asarray(every.v), [[1.25, 2.0]])  # dead slots are left alone


def test_bench_runner_defaults_keep_the_leaderboard_settings():
    import inspect
    from avrl import bench as B
    p = inspect.signature(B.runner).parameters
    assert (p["seed"].default, p["key_seed"].default, p["kick"].default) == (123, 7, 0.0)
