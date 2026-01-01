"""The particle estimator is unbiased for any guide: checked on a random walk whose answer is known."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from avlab import rare as R


class S(NamedTuple):
    key: object
    x: object


class Walk:
    """n copies of a biased random walk; 'crash' = reaching +TOP within the run. Same interface as avlab.rare.Rare."""
    TOP, UP = 5, 0.35

    def __init__(self, n):
        self.n = n
        self.rng = np.random.default_rng(0)

    def window(self, st, key):
        steps = np.where(np.random.default_rng(int(np.asarray(jax.random.key_data(key)).ravel()[-1]) % 2**31).random(self.n) < self.UP, 1, -1)
        steps = steps + 0 * np.asarray(st.x)
        x = np.maximum(np.asarray(st.x) + steps, -3)
        crash = (x >= self.TOP).astype(float)
        x = np.where(x >= self.TOP, self.TOP - 1, x)  # the crash is counted once, the walk goes on
        st = S(st.key, jnp.asarray(x))
        return st, crash, x[:, None, None].astype(np.float32), np.ones((self.n, 1), bool), None, None, None, None


class Guide:
    def __init__(self, shape):
        self.shape = shape

    def V(self, f, live):
        x = f[:, 0, 0]
        return np.clip(self.shape * (x - R_TOP), np.log(1e-5), 0.0)


R_TOP = Walk.TOP


def exact(K):
    """P(the walk from 0 reaches TOP within K steps), by dynamic programming (absorbing barrier at TOP, floor at -3)."""
    p = {0: 1.0}
    hit = 0.0
    for _ in range(K):
        q = {}
        for x, m in p.items():
            for d, pr in ((1, Walk.UP), (-1, 1 - Walk.UP)):
                y = max(x + d, -3)
                if y >= Walk.TOP:
                    hit += m * pr
                else:
                    q[y] = q.get(y, 0.0) + m * pr
        p = q
    return hit


def estimates(method, guide, K=8, n=64, reps=300, lam=1.0):
    walk = Walk(n)
    out = []
    for i in range(reps):
        st0 = S(jax.vmap(lambda k: jax.random.key_data(k))(jax.random.split(jax.random.PRNGKey(i), n)) * 0 + 0, jnp.zeros(n))
        out.append(R.run(walk, st0, guide, K=K, lam=lam, seed=i, method=method)["p"])
    return np.array(out)


def test_particle_estimator_is_unbiased_and_helps_with_a_good_guide():
    # run() re-keys the state with fold_in; the toy state carries a dummy key array of the right shape
    truth = exact(9)  # run() takes one window to reach x_0, then K more
    plain = estimates("plain", Guide(0.0))
    for shape in (0.0, 0.6, 1.2):  # any guide, including none (0) and a sharp one
        part = estimates("particles", Guide(shape))
        se = part.std(ddof=1) / np.sqrt(len(part))
        assert abs(part.mean() - truth) < 4 * se + 1e-3, (shape, part.mean(), truth)
    good = estimates("particles", Guide(1.2))
    assert good.std() < 0.8 * plain.std()  # a guide that knows the distance to the barrier lowers the spread at equal cost
