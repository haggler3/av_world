# av_world

**A city-scale driving world in pure JAX, calibrated to real crash statistics, with the research stack built on it and an unusually
honest account of what it can and cannot do.**

[**Website**](https://haggler3.github.io/av_world/) · [**3-D scenes**](https://haggler3.github.io/av_world/scenes.html) · [**Browser viewer**](https://haggler3.github.io/av_world/viewer/index.html) · [What you can test](docs/TESTING.md) · [Docs index](#guides)

![learned AVs in Oakland](docs/img/xnet_learned_avs.gif)

## What it is

* **`xnet`**, an area simulator built from open map data (Pittsburgh's Oakland, Squirrel Hill and Downtown, Midtown Manhattan, the Chicago Loop, central Boston, any OpenStreetMap city, Argoverse 2 / Waymo maps):
  car following, lane changes, zipper merges, right of way at every conflict point, signals, pedestrians, cyclists, buses, breakdowns, road
  friction, seven crash types, a control-barrier-function safety filter. About 395 simulated seconds per second for 8 copies of a ~380-vehicle area
  on 4 CPU cores, and it shards across devices.
* **Human drivers calibrated to published crash data by crash type** (`xworld`, `pgh`): a world with the mistakes real people make.
* **A research stack on top**: multi-agent PPO (`avrl`), AVBench, latent world models and a planner (`dwm`, `wmplan`), a stress lab (`avlab`).
* **Software and hardware in the loop**: drive any AV from your own controller over TCP, in lockstep (60x real time) or at wall-clock pace with
  deadlines and fallbacks ([SIL_HIL.md](docs/SIL_HIL.md)).
* **NVIDIA-ready export**: any simulated long-tail scene or real Argoverse 2 log as an OpenUSD stage (Omniverse) and a camera-view
  map-and-box control video of the kind driving world models such as Cosmos are conditioned on, set in the real city: buildings at their
  mapped heights, trees, lamps and signals fused from open map data (Overture / OpenStreetMap) ([NVIDIA.md](docs/NVIDIA.md)).
  Every export also renders this side-by-side clip, and `--find-crash` hunts down an AV crash to film
  ([more scenes](https://haggler3.github.io/av_world/scenes.html)):

  <img src="docs/img/nvidia3d_compare_small.gif" width="724" alt="The same Manhattan scene in av_world (left) and as the exported OpenUSD stage (right)">
* **Long-tail tooling**: sensor faults, aimed human mistakes, dart-outs, famous-incident scenarios, a real-log importer, CARLA export, and
  five methods for teasing novel long-tail data out of the simulator ([report](docs/LONG_TAIL_METHODS.md)).

```bash
pip install .                       # data and a trained starting policy are bundled
av-world info && av-world bench     # simulated seconds per wall-clock second on this machine
```
Contributing: `pip install pre-commit && pre-commit install` turns on the [gitleaks](https://github.com/gitleaks/gitleaks) hook, which
blocks commits that contain API keys, tokens or private keys.
```python
import av_world
env = av_world.make("oakland", n_envs=8)           # or an OpenStreetMap file, or "city.npz"
state = env.reset(seed=0, warm_s=60)
state, traj = env.rollout(state, policy=None, steps=200)     # None: the rule-based AV
print(env.metrics(state)["av_crash"])
```

## What we found, including what did not work

| question | answer | where |
|---|---|---|
| Can a policy drive with no safety filter if the barrier function teaches it? | Yes, mostly: unaided AV crashes fell 15-23x (0.64 to 0.03-0.04 per AV-hour). On AVBench's fixed bars (18 scenarios, 7 of them long-tail incidents) the best passes 123 of 144 checks against the rule-based AV's 104, but the rule-based AV keeps the best core score (77.2 vs 56.8): unaided, the learned policies still crash 1.3-1.6x as often. | [Metrics](docs/METRICS.md#results), [Experiments](docs/EXPERIMENTS.md#training-the-driver-to-need-no-filter-avrlunaidedpy) |
| Is one training run enough to rank policies? | **No.** Retraining the barrier-teacher phases with the same recipe moved them by up to 10 av_world points and 8-15 of 144 checks, about the size of most gaps between policies. | [Metrics](docs/METRICS.md#results) |
| Does planning in a latent world model beat the policy it starts from? | Partly: 0.47 unaided crashes per AV-hour against 0.64 for its PPO prior, but 15-60x the barrier-taught policies. Its model ranks crashes well (AUROC 0.96) but predicts the stop-line gap worse than holding it constant. | [Metrics](docs/METRICS.md#results) |
| Does a more varied world teach better? | Mutating cities did: 8 of 8 (seed, world) pairs had fewer crashes, 17-63 % fewer. Fork-at-danger and world-model crash shaping did not help. | [Experiments](docs/EXPERIMENTS.md) |
| Does a GPU-style trainer help? | Packing the learner's rows and sharding copies: PPO ~5x faster on CPU. On one NVIDIA L4: a PPO phase 2x faster than on CPU (95 s vs 195 s), a latent-planner benchmark run about 5x (under 2 min vs ~10). | [Speed](docs/SPEED.md) |
| Can it estimate a 1-in-a-million crash probability? | **No.** The estimator is unbiased (checked on a random walk) but the learned guide predicts too weakly to beat plain Monte Carlo. | [Long tail](docs/LONG_TAIL.md) |
| Do famous incidents come out of the simulator? | **Mechanisms yes, incidents no**: fidelity 0.2-0.6 by feature count; pedestrian impacts 24 mph against Tempe's 43; everyday near-misses 4-6x thinner than real Argoverse 2 traffic. | [Incident audit](docs/INCIDENT_AUDIT.md) |
| Does training on famous-event scenarios help? | Not shown: two seeds, 0.077 / 0.172 for the mix against 0.084 / 0.093 for ordinary training. They are good *tests*: ~0 crashes in ordinary traffic, 0.25-0.4 per AV-hour in the jaywalker and glare scenarios. | [Famous events](docs/FAMOUS_EVENTS.md) |
| Does the simulator predict real traffic? | Its drivers are about 2x worse than constant velocity at predicting Argoverse 2 vehicles (17 m vs 9 m at 5 s): not calibrated to real driving. | [Data import](docs/DATA_IMPORT.md) |
| Five ways to make novel long-tail data | Drivers fitted to real logs match real hard braking (48 vs 43 per hour) but still under-tailgate 8x; MAP-Elites finds a broader failure catalogue than random search (7 vs 5 kinds), though elites regress on re-runs; novelty mining makes data 3x richer in crashes; counterfactuals on real scenes give a few new failures; go-explore did not beat plain simulation. | [Long-tail methods](docs/LONG_TAIL_METHODS.md) |

## Guides

| | |
|---|---|
| [docs/TESTING.md](docs/TESTING.md) | What AV behaviour you can test, the knobs, metrics and suites, how to plug in your stack, and the limits |
| [docs/METRICS.md](docs/METRICS.md) | AVBench: fixed bars the rule-based AV can fail, the scorecard and av_world score, results on 18 scenarios ([leaderboard](results/bench/leaderboard.md)) |
| [docs/NVIDIA.md](docs/NVIDIA.md) | OpenUSD stages and Cosmos-style control videos from long-tail scenes, side-by-side clips, filming crashes |
| [docs/SIL_HIL.md](docs/SIL_HIL.md) | software / hardware in the loop: protocol, modes, timing, measured latency |
| [docs/CITIES.md](docs/CITIES.md) | New York, Chicago and Boston: what was built, what is real and what is inferred |
| [docs/LONG_TAIL_METHODS.md](docs/LONG_TAIL_METHODS.md) | five methods for novel long-tail data, with pseudocode and results |
| [docs/INCIDENT_AUDIT.md](docs/INCIDENT_AUDIT.md) | does the long tail that matters come out of the simulator? |
| [docs/FAMOUS_EVENTS.md](docs/FAMOUS_EVENTS.md) | ten famous failure patterns as scenarios; do they help training? |
| [docs/DATA_IMPORT.md](docs/DATA_IMPORT.md) | Argoverse 2 and Waymo import; how well the simulator predicts real logs |
| [docs/JUNCTIONS.md](docs/JUNCTIONS.md) | the intersection detector for imported maps, with pseudocode |
| [docs/LONG_TAIL.md](docs/LONG_TAIL.md) | overview of the long-tail tooling: OSM import, CARLA export, sensor faults, adversary, rare events |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | dynamics ladder, scaling laws, active learning, AVBench, unaided training, mutating cities |
| [docs/RL.md](docs/RL.md), [docs/WORLD_MODELS.md](docs/WORLD_MODELS.md), [docs/METHODS.md](docs/METHODS.md) | the RL stack, the world models, the methods |
| [docs/SIMULATORS.md](docs/SIMULATORS.md) | simulator internals, calibration against crash data, running everything, limitations, code layout |
| [docs/SPEED.md](docs/SPEED.md) | speed against CARLA, GPUDrive, Waymax and others (CPU measured, GPU projected) |
| [docs/RESEARCH_CONTEXT.md](docs/RESEARCH_CONTEXT.md) | comparison with other simulators and current research |
| [docs/ROADMAP.md](docs/ROADMAP.md), [docs/HISTORY.md](docs/HISTORY.md) | what is next; how the repository came about |

## Repository map

| package | what it is |
|---|---|
| `av_world/` | the pip package: `make()` environment, the `av-world` command, the SIL/HIL plugin, bundled maps and start policy |
| `xnet/` | the area simulator, road network builder, the extra cities, the NVIDIA exporter, OSM and log importers, junction detector, CARLA export, sensor model |
| `xworld/`, `pgh/` | the calibrated intersection simulator; 10,205 mined Pittsburgh intersections |
| `avrl/` | PPO, AVBench, ladder, unaided training, famous-event training, multi-device trainer |
| `dwm/`, `wmplan/` | diffusion / flow world models with scaling laws; latent world model and planner |
| `avlab/` | stress lab, tail scenarios, incident audit, adversary, rare events, the five long-tail methods |
| `tests/` | 80 tests: dynamics, safety, importers, junctions, sensor model, estimators |
| `docs/` | guides, the GitHub Pages site and the browser viewer |
| `results/` | the JSON behind every number in the docs |

Everything runs in jobs of about ten minutes on 4 CPU cores. `export JAX_DEFAULT_PRNG_IMPL=unsafe_rbg` makes random numbers much cheaper on CPU.
Data sources and licences: [docs/SIMULATORS.md](docs/SIMULATORS.md#layout) (Overture Maps / OpenStreetMap, published crash-rate tables);
Argoverse 2 is CC BY-NC-SA 4.0 and is downloaded on demand, never redistributed.
