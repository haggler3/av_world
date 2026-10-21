# How av_world compares and where it sits in current research

Moved from the README.

## What this does that common tools don't

The driving-simulation field has split into two camps. **Log-replay simulators** (Waymax, GPUDrive, nuPlan) replay
short recorded snippets and are as realistic as their logs, but cannot create situations the logs don't contain.
**Self-play simulators** (Gigaflow) scale to billions of kilometres, but every agent is the same policy, so the
world never contains a distracted human or a red-light runner. av_world sits in between: a **synthetic, open-ended
city whose human drivers are calibrated to real crash statistics**, so the policy learns among the mistakes real
humans make, at the rates they make them.

| | av_world | Waymax | GPUDrive | Gigaflow | MetaDrive | CARLA | SUMO |
|---|---|---|---|---|---|---|---|
| Framework | JAX | JAX | C++/CUDA (Madrona) | GPU batch sim | Python / Panda3D | Unreal | C++ |
| Scenes | whole neighbourhoods from any OSM map, unbounded duration | ~9 s logged snippets (WOMD) | ~9 s logged snippets (WOMD) | procedural maps | procedural + imported | hand-built towns | any OSM map |
| Other road users | calibrated human drivers, pedestrians, buses, vans, breakdowns | logs or IDM | logs or the policy | the policy (self-play) | rule-based traffic | traffic manager | car-following models |
| Crashes | modelled and **calibrated by crash type** | detected | detected | detected | detected | detected | collision-free by default |
| Signals, stop signs, right of way | full conflict-point model | logged light states | from logs | yes | partial | yes | yes |
| RL + world models + scaling laws in one repo | **yes** | sim + baselines | sim + baselines | not released | sim + baselines | sim | sim |
| Runs on a laptop CPU | **380x real time** | yes | GPU-focused | multi-GPU | yes | GPU | yes |

**What's distinctive here:**

1. **Calibrated risk, not just traffic.** Red runners, drivers who don't look, distraction and breakdowns are tuned
   so crash rates *by type* match published intersection data. Most simulators are either collision-free by
   default (SUMO) or inherit whatever their logs contain.
2. **The simulator itself is the experimental variable.** The dynamics ladder switches sources of dynamics on one at
   a time and measures what each buys the learner. Active scenario selection asks which simulated data teaches
   fastest. These are questions about the *teacher*, not the student.
3. **Sim → policy → generative world model → scaling laws in one codebase.** The same logs train PPO and the
   diffusion world models, so the scaling laws for policy and model can be compared directly.
4. **Any city, from open data.** `pgh/` mines 10,205 Pittsburgh intersections automatically; `xnet/` turns a
   bounding box into a simulated neighbourhood.
5. **Inspectable.** Every run can be replayed in the [browser viewer](https://haggler3.github.io/av_world/viewer/):
   key moments are indexed, so you can jump straight to each crash, red run or breakdown.

## Where this sits in current research

- **Scale beats hand-tuning.** Gigaflow ([Cusumano-Towner et al., ICML 2025](https://arxiv.org/abs/2502.03349))
  trained one self-play policy on 1.6 billion km and set the state of the art on CARLA, nuPlan and Waymax without
  ever seeing human data. Here: the same bet at laptop scale, plus calibrated humans as the hard part of the world.
- **Throughput is the bottleneck.** GPUDrive ([Kazemkhani et al., ICLR 2025](https://arxiv.org/abs/2408.01584))
  reaches over a million steps per second on a GPU. `xnet` is jitted end to end with fixed shapes, so the same code
  moves to a GPU.
- **Scaling laws for driving.** Waymo ([Baniodeh et al., 2025](https://waymo.com/research/scaling-laws-of-motion-forecasting-and-planning))
  found power laws in compute for motion forecasting and planning. This repo fits the same laws for policy score vs
  simulator steps and for world-model error vs compute and data, across four architectures.
- **Generative world models.** GAIA-2 ([Wayve, 2025](https://arxiv.org/abs/2503.20523)) generates multi-camera
  driving video with latent diffusion. `dwm/` applies diffusion and flow matching to the traffic state itself: 3 s
  futures in 0.06 ms.
- **Data-driven simulation in JAX.** Waymax ([Gulino et al., NeurIPS 2023](https://arxiv.org/abs/2310.08710)) made
  log-replay simulation differentiable and batched. `xnet` keeps that interface style (pure `step`, vmapped worlds)
  but generates its own traffic.
- **Adaptive curricula.** Prioritised level replay ([Jiang et al., 2021](https://arxiv.org/abs/2010.03934)) selects
  training environments by learning potential. Here it is tested against uniform scenario sampling on a 54-scenario
  grid of the same city.
