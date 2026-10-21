"""PPO on every device at once: the area copies are sharded over a mesh (one `shard_map`), each device rolls out its own
copies and trains on its own valid AV rows (packed, `Config.compact`), and the gradients are averaged with `pmean`.
The same script runs on 1 GPU, 8 GPUs, or CPU cores (`--cpu-devices 4`). Start policy and reward are those of avrl.fork
(no safety filter; the barrier function is the teacher).

    python -m avrl.gpu_train --preset gpu --out results/gpu          # 128 copies per device, 1000 updates
    python -m avrl.gpu_train --cpu-devices 4 --preset cpu --bench    # throughput only, 20 s
    python -m avrl.gpu_train --cpu-devices 4 --preset cpu --updates 100 --out results/gpu_cpu

Presets: cpu = 8 copies per device, gpu = 128 copies per device (a few hundred copies per card keep it busy).
GPU numbers in the README are projections: this repository was developed without a GPU.
"""
import argparse
import json
import os
import sys
import time

if "--cpu-devices" in sys.argv:  # must be set before jax is imported
    n = sys.argv[sys.argv.index("--cpu-devices") + 1]
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={n}"
os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as PS  # noqa: E402

from av_world import data_path  # noqa: E402
from xnet import network as NW  # noqa: E402
from xnet import sim as X  # noqa: E402

from . import ladder as LD  # noqa: E402
from . import xppo as P  # noqa: E402

START = data_path("policy.npz")
REWARD = LD.REWARD | {"barrier": 0.3, "inside": 0.05, "veto": 1.0}
PRESETS = {"cpu": dict(per_dev=8, updates=100), "gpu": dict(per_dev=128, updates=1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="cpu", choices=PRESETS)
    ap.add_argument("--cpu-devices", type=int, default=0)
    ap.add_argument("--per-dev", type=int)
    ap.add_argument("--updates", type=int)
    ap.add_argument("--compact", type=float, default=0.25)
    ap.add_argument("--init", default=START)
    ap.add_argument("--out", default="results/gpu")
    ap.add_argument("--bench", action="store_true", help="time a few updates and exit")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pre = PRESETS[a.preset]
    per_dev, updates = a.per_dev or pre["per_dev"], a.updates or pre["updates"]
    devs = jax.devices()
    D = len(devs)
    E = per_dev * D
    print(f"{D} x {devs[0].platform} device, {per_dev} copies each = {E} copies, compact={a.compact}", flush=True)

    X.R = X.Reward(**REWARD)
    X.SHIELD = "off"
    net, geo, _ = NW.build("oakland")
    J = X.to_jnp(net)
    cfg = P.Config(n_envs=per_dev, reward=REWARD, warm=150.0, compact=a.compact, axis_name="copies")
    beh = P.behaviours(E, a.seed, LD.knobs())
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad), optax.adam(cfg.lr))
    warm, _, update = P.make_fns(cfg, net, jax.tree.map(lambda x: x[:per_dev], beh), opt)
    mesh = Mesh(np.array(devs), ("copies",))
    sh, rep = PS("copies"), PS()

    def local(params, opt_state, st, key, b):
        params, opt_state, st, s = update(params, opt_state, st, key[0], b)
        ps = lambda x: jax.lax.psum(x, "copies")
        return params, opt_state, st, {"reward_sum": ps(s["reward_per_step"] * s["transitions"]), "transitions": ps(s["transitions"]),
                                       "dropped": ps(s["dropped"]), "m": ps(s["m"]), "kl": jax.lax.pmean(s["kl"], "copies")}

    step = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(rep, rep, sh, sh, sh), out_specs=(rep, rep, sh, rep), check_vma=False),
                   donate_argnums=(2,))
    warm_s = jax.jit(jax.shard_map(lambda p, st, k, b: warm(p, st, k[0], b), mesh=mesh, in_specs=(rep, sh, sh, sh), out_specs=sh,
                                   check_vma=False), donate_argnums=(1,))
    keys = lambda k: jax.random.split(k, D)
    key = jax.random.PRNGKey(a.seed)
    params = P.load(a.init)
    opt_state = opt.init(params)
    st = jax.jit(jax.vmap(lambda k: X.reset(k, J, cfg.n_slots)), out_shardings=jax.sharding.NamedSharding(mesh, sh))(jax.random.split(key, E))
    for i in range(int(cfg.warm / (X.DT * (cfg.n_steps + 1)))):
        st = warm_s(params, st, keys(jax.random.fold_in(key, 10_000 + i)), beh)
    sim_steps = E * (cfg.n_steps + 1)
    log, t0 = [], time.time()
    out = a.out
    os.makedirs(out, exist_ok=True)
    for i in range(updates):
        if i > 0 and i % cfg.reset_every == 0:  # restart a quarter of the copies, as xppo.train does
            sel = jnp.asarray((np.arange(E) % 4) == (i // cfg.reset_every) % 4)
            new = jax.vmap(lambda k: X.reset(k, J, cfg.n_slots))(jax.random.split(jax.random.fold_in(key, 30_000 + i), E))
            st = jax.tree.map(lambda x, y: jnp.where(sel.reshape((-1,) + (1,) * (x.ndim - 1)), y, x), st, new)
        t1 = time.time()
        params, opt_state, st, s = step(params, opt_state, st, keys(jax.random.fold_in(key, i)), beh)
        jax.block_until_ready(st)
        dt = time.time() - t1
        if i == 0:
            t0 = time.time()  # exclude compilation from the throughput
            continue
        row = {"update": i, "s": round(dt, 2), "area_steps_per_s": round(sim_steps / dt), "transitions": float(s["transitions"]),
               "dropped": float(s["dropped"]), "reward": float(s["reward_sum"] / jnp.maximum(s["transitions"], 1.0)),
               **P.summarize(s["m"])}
        log.append(row)
        if i % 10 == 0 or a.bench:
            print(json.dumps({k: (round(float(x), 3) if isinstance(x, (float, np.floating)) else x) for k, x in row.items()}, default=float), flush=True)
        if a.bench and time.time() - t0 > 20:
            break
        if (i + 1) % 25 == 0 or i + 1 == updates:
            P.save(params, f"{out}/policy_{i + 1}.npz")
    n = len(log)
    tot = sum(r["s"] for r in log)  # update time only (no resets, logging or checkpoints)
    print(f"{n} updates in {tot:.0f} s: {n * sim_steps / tot:,.0f} area-steps/s = {n * sim_steps * X.DT / tot:,.0f} area-s/s", flush=True)
    if not a.bench:
        json.dump(log, open(f"{out}/train_log.json", "w"), default=float)
        P.save(params, f"{out}/policy.npz")


if __name__ == "__main__":
    main()
