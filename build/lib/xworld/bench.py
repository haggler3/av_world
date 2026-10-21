"""Batch throughput: how many intersection-hours one wall-second simulates, for a few batch sizes.

    python -m xworld.bench
"""
import time

import jax

from . import sim as S


def main(sizes=(256, 1024, 2048), seconds=200):
    for n in sizes:
        S.run_batch(jax.random.PRNGKey(0), n, 50, S.Behaviour(), warmup=0)  # compile
        t0 = time.time()
        m, _ = S.run_batch(jax.random.PRNGKey(1), n, seconds, S.Behaviour(), warmup=0)
        wall = time.time() - t0
        print(f"{n:5d} intersections: {n * seconds / 3600 / wall:6.2f} intersection-hours per second, "
              f"{m[0] / wall:8.0f} entering vehicles per second, {n * S.M * seconds / S.DT / wall / 1e6:5.1f} M vehicle-slot updates per second")


if __name__ == "__main__":
    main()
