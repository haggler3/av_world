"""Time the simulator: 200 steps of 16 Oakland copies, median of 5 runs (compile excluded).

Used by the daily routine's rule for `xnet/` changes: t1 / t0 <= 1.05, where t0 and t1 are the medians before and after.

    python -m avlab.speedcheck                 # prints the median and the 5 times
    python -m avlab.speedcheck --save a.json   # also saves them, plus per-copy counters for a fixed seed (regression check)
"""
import argparse
import json
import time

import numpy as np


def bench(E=16, steps=200, reps=5, seed=0, area="oakland", n_slots=1024, patch=None):
    """Median wall time [s] of `steps` steps of E copies, and the per-copy counters after the first run [E, NM]."""
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, _, _ = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    if patch is not None:
        J["link_patch"] = jnp.asarray(patch)
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *[C.calibrated() for _ in range(E)])
    st0 = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), E), J, n_slots)

    @jax.jit
    def run(st):
        return jax.lax.scan(lambda s, _: (jax.vmap(lambda si, b: X.step(si, J, b, nh))(s, B), None), st, None, length=steps)[0]

    m = np.asarray(jax.block_until_ready(run(st0)).m)  # compile + first run
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(run(st0))
        times.append(time.perf_counter() - t)
    return float(np.median(times)), times, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    a = ap.parse_args()
    med, times, m = bench()
    print(f"median {med:.3f} s for 200 steps x 16 copies; runs {[round(x, 3) for x in times]}")
    if a.save:
        json.dump({"median_s": med, "times": times, "metrics": m.tolist()}, open(a.save, "w"))


if __name__ == "__main__":
    main()
