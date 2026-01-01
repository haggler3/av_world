"""Fast data collection shared by every network in nets/.

The old loops stepped the simulator from Python and copied the state to the host every step (`sim.run(0.2)` + `np.asarray`), which
cost more than the simulation. Here everything happens inside one jitted `lax.scan`:

  record(...)   steps E vmapped copies `steps` times; every `every` steps a user function reduces the state (and, with a policy, the
                step's transition dict) to a fixed-shape feature slab, and the scan stacks it. One host transfer per call.
  fork(st, K)   K copies of every state (env-major), so K branches of one moment run as one vmapped batch with one compile.

With Behaviour.crn = 1 the branches of a fork share every human's dice, so whatever is varied per branch (an action, a mask) is the only
difference between them.
"""
import jax
import jax.numpy as jnp
import numpy as np

from xnet import network as NW
from xnet import sim as X


class World:
    """One area compiled once; behaviour is a traced argument (a new knob set does not recompile)."""

    def __init__(self, area="oakland", n_slots=768):
        self.area = area
        self.net_np, self.geo, self.info = NW.build(area)
        self.J = X.to_jnp(self.net_np)
        self.nh = jnp.asarray(self.net_np["nh"])
        self.n_slots = n_slots

    def reset(self, n_envs, seed=0):
        keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
        return jax.vmap(X.reset, in_axes=(0, None, None))(keys, self.J, self.n_slots)


def fork(st, k):
    """[E, ...] -> [E * K, ...]: copy e's K branches are rows e*K .. e*K+K-1, all with the same key (common random numbers)."""
    return jax.tree.map(lambda a: jnp.repeat(a, k, axis=0), st)


def stack_beh(beh, n):
    """A Behaviour with every field broadcast to [n] (vmapped knobs)."""
    return type(beh)(*[jnp.full((n,), float(v)) for v in beh])


def run(world, st, beh, seconds):
    """Advance without recording (warm-up), in one compiled scan. beh: a Behaviour (shared) or a stacked one ([E] fields)."""
    return _run(st, beh, world.J, world.nh, int(round(seconds / X.DT)), _batched(beh))


def _batched(beh):
    return bool(np.ndim(beh[0]) > 0)


from functools import partial  # noqa: E402


@partial(jax.jit, static_argnums=(4, 5))
def _run(st, beh, J, nh, steps, batched):
    f = jax.vmap(lambda s, b: X.step(s, J, b, nh), in_axes=(0, 0 if batched else None))
    return jax.lax.scan(lambda s, _: (f(s, beh), None), st, None, length=steps)[0]


def ndev(n_envs):
    """Devices to shard n_envs copies over: all of them when they divide n_envs (set XLA_FLAGS=--xla_force_host_platform_device_count=4
    before importing jax to make the 4 CPU cores 4 devices; measured x2.1 on this machine), else 1."""
    d = jax.device_count()
    return d if d > 1 and n_envs % d == 0 else 1


def _shard(x, d):
    return jax.tree.map(lambda a: a.reshape((d, a.shape[0] // d) + a.shape[1:]), x)


def _unshard(x):
    return jax.tree.map(lambda a: a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:]), x)


_PMAPS = {}


def record(world, st, beh, seconds, every, feat, policy=None, per_copy=None):
    """Run `seconds`, and every `every` steps store feat(st, tr, per_copy) (tr is the transition dict when a policy drives the AVs, else
    None). policy(obs, key, per_copy_row) -> (u, aux) gets the copy's row of `per_copy` (e.g. an action perturbation or a mask) and the
    step index through per_copy["_t"]. -> (final state, stacked features with leading axes [T, E])."""
    steps = int(round(seconds / X.DT))
    assert steps % every == 0
    per_copy = {} if per_copy is None else per_copy
    batched = _batched(beh)
    d = ndev(st.t.shape[0])
    if d == 1:
        return _record(st, beh, world.J, world.nh, per_copy, steps // every, every, feat, policy, batched)
    key = (id(world), steps // every, every, feat, policy, batched)
    if key not in _PMAPS:
        _PMAPS[key] = jax.pmap(lambda s, b, pc: _record.__wrapped__(s, b, world.J, world.nh, pc, steps // every, every, feat, policy, batched),
                               in_axes=(0, 0 if batched else None, 0))
    s2, out = _PMAPS[key](_shard(st, d), _shard(beh, d) if batched else beh, _shard(per_copy, d))
    # features come back [d, T, E/d, ...]: put the copies back in order -> [T, E, ...]
    out = jax.tree.map(lambda a: jnp.moveaxis(a, 0, 1).reshape((a.shape[1], a.shape[0] * a.shape[2]) + a.shape[3:]), out)
    return _unshard(s2), out


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9))
def _record(st, beh, J, nh, per_copy, n_out, every, feat, policy, batched):
    def one(s, b, pc, t):
        if policy is None:
            return X.step(s, J, b, nh), None
        pc = dict(pc, _t=t)
        return X.step(s, J, b, nh, rl=(lambda obs, k: policy(obs, k, pc), s.key))

    stepv = jax.vmap(one, in_axes=(0, 0 if batched else None, 0, None))

    def inner(c, t):
        s, _ = c
        s2, tr = stepv(s, beh, per_copy, t)
        return (s2, tr), None

    def outer(s, i):
        tr0 = jax.eval_shape(lambda: stepv(s, beh, per_copy, 0)[1])
        tr0 = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), tr0)
        (s, tr), _ = jax.lax.scan(inner, (s, tr0), i * every + jnp.arange(every))
        return s, feat(s, tr, per_copy)

    return jax.lax.scan(outer, st, jnp.arange(n_out))


def to_host(x, dtype=np.float16):
    """Feature slabs to numpy, floats narrowed (labels and ids keep their type)."""
    return jax.tree.map(lambda a: np.asarray(a).astype(dtype) if np.issubdtype(np.asarray(a).dtype, np.floating) else np.asarray(a), x)
