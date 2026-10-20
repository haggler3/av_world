# How fast is it, next to other simulators?

One number that every simulator can be reduced to: **vehicle-steps per second**, the count of (vehicle, time step) pairs
the simulator advances in one wall-clock second. It ignores how much each step does, so the table also says what a step
contains. Read it as a rough ordering, not a ranking: these simulators answer different questions.

| simulator | what one step does | hardware | published speed | vehicle-steps / s |
|---|---|---|---|---|
| **CARLA** | Unreal Engine rendering, PhysX vehicle dynamics, sensors | one GPU workstation (GTX 1080 Ti era dev-branch benchmark) | ~1,200 FPS empty world with physics off, falling to ~37 FPS with 250 vehicles | ~10 k with 250 vehicles |
| **MetaDrive** | lightweight 3D physics (Bullet), lidar-style sensors | CPU | ~300 FPS single agent with 10 traffic cars; ~60 FPS with 40 agents | ~2-3 k |
| **Nocturne** | 2D, replays Waymo scenes in C++ | CPU | ~3,400 steps/s | ~3 k per scene (tens of agents) |
| **SUMO** | microscopic car-following and lane changing, single thread | one CPU core | scales with vehicles; real time at ~10^4 vehicles is typical | ~10^5 (order of magnitude) |
| **CityFlow** | SUMO-like, multithreaded C++ | 64 threads | reported 20-30x faster than SUMO | ~10^6 (relative claim) |
| **GPUDrive** | 2D, data-driven agents, CUDA | one A100 | 2.3 M agent-steps/s peak; ~200 k *controlled* agent-steps/s | 2.3 M |
| **Waymax** | kinematic logged-scenario replay in JAX | accelerator | ~64 M controlled-agent-steps/s at 64 scenes x 16 agents (RTX PRO 6000) | 64 M (kinematic) |
| **av_world `xnet`, this repo** | a whole city area: signals, right of way, gap acceptance, IDM, lane changes, pedestrians, buses, 7 crash types, the barrier function | 4 CPU threads, measured | 395 simulated s per wall-clock s, 8 copies of ~380 vehicles | **0.75 M** (rule-based AVs), 0.58 M with the policy |
| same, 4 host devices (`shard_map`) | same | same CPU, measured | 693 simulated s/s with the policy | **1.3 M** |
| same, one A100 | same | **projection, not measured** | 17-34 k area-steps/s from a bandwidth roofline at 15-30 % use | **6-13 M** |

Sources: CARLA's own development-branch benchmark table (the official docs state no real-time factor; I could not open the
page from the build container, so the figures come from search summaries and should be treated as approximate);
[GPUDrive](https://arxiv.org/abs/2408.01584), [Waymax](https://arxiv.org/abs/2310.08710),
[MetaDrive](https://arxiv.org/abs/2109.12674), [Nocturne](https://arxiv.org/abs/2206.09889),
[CityFlow](https://arxiv.org/abs/1905.05217), and the SceneFactory comparison of GPU driving simulators for the 200 k and 64 M numbers.

## What the table does and does not say

* Waymax and GPUDrive step simple kinematic or 2D agents that mostly replay logs; a step is a few arithmetic operations
  per agent. `xnet` steps interacting vehicles that queue, yield, merge and crash, so its per-vehicle step is far more work.
  Compared on work per vehicle-step, the GPU simulators are not 10-100x faster than a CPU `xnet`; they are doing less.
* CARLA is the only one here that renders camera and lidar. If your policy needs pixels, nothing else on the list applies.
  `xnet` has no sensors, which is the main reason it is two orders of magnitude faster per vehicle than CARLA.
* The A100 row is an estimate. This repository was developed on a CPU container. The estimate divides memory bandwidth
  (~2 TB/s) by the compiler's measured 17.6 MB of traffic per copy per step and assumes 15-30 % use. A real GPU run may land outside it,
  and `python -m avrl.gpu_train --bench` measures it in under a minute.

## What was done to speed it up (measured on the 4-thread CPU)

| change | effect |
|---|---|
| learner trains only on valid AV rows, packed (`Config.compact`) | PPO update 1.97 s -> 0.81 s (x2.4); 86 % of the old rows were padding |
| copies sharded over devices with `shard_map` (`avrl/gpu_train.py`) | x2.1 on 4 host devices; the same line spreads copies across GPUs |
| combined PPO throughput | ~210 -> ~1,100 area-steps/s (about x5) |
