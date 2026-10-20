"""Packing the learner's rows (Config.compact) keeps every valid AV transition."""
import jax
import numpy as np
import optax

from avrl import ladder as LD
from avrl import xppo as P
from xnet import network as NW
from xnet import sim as X


def test_compact_update_keeps_all_valid_rows():
    X.R = X.Reward(**LD.REWARD)
    net, _, _ = NW.build("oakland")
    beh = P.behaviours(2, 0, LD.knobs())
    out = {}
    for comp in (0.0, 0.5):
        cfg = P.Config(n_envs=2, n_steps=8, epochs=1, reward=LD.REWARD, compact=comp)
        opt = optax.adam(1e-4)
        warm, _, update = P.make_fns(cfg, net, beh, opt)
        params = P.init_params(jax.random.PRNGKey(0))
        st = jax.vmap(lambda k: X.reset(k, X.to_jnp(net), cfg.n_slots))(jax.random.split(jax.random.PRNGKey(1), 2))
        st = warm(params, st, jax.random.PRNGKey(2))
        p2, _, _, s = update(params, opt.init(params), st, jax.random.PRNGKey(3))
        out[comp] = (p2, s)
    assert float(out[0.5][1]["dropped"]) == 0.0
    assert float(out[0.5][1]["transitions"]) == float(out[0.0][1]["transitions"]) > 0
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(out[0.5][0]))
