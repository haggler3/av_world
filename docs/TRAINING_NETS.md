# Training the simulator-only networks: the easy guide

Six neural networks and seven world-model upgrades, each one command to train and one line to use. They learn from three things a
simulator like this one can give and real data cannot:

* **What is hidden.** Every driver's red-light-running habit, following headway and distraction is in the state, so a network can be
  told the truth about what it is trying to infer.
* **What would have happened.** A fork of the city is a copy of one array tree, and identity-keyed random numbers hold every other
  driver's dice fixed. Change one action and the difference in outcome is caused by that action, with no confounding.
* **Many worlds whose settings are known**, next to real Argoverse 2 traffic to point them at.

The results, with figures and GIFs, are in [NEURAL_NETS.md](NEURAL_NETS.md) and [WORLD_MODEL_UPGRADES.md](WORLD_MODEL_UPGRADES.md).

## Quickstart

```bash
pip install -e .                      # the simulator and everything these need (jax, optax, numpy, scipy, pandas for tables)
av-world nets list                    # thirteen features: their question, time, and whether they are trained
av-world nets train occlusion --quick # ~3 min on a laptop CPU; results in results/nets/quick/
av-world nets info occlusion          # what it learned, what it wrote, how to use it
```

```python
from av_world import easy

easy.nets.list()                                   # the same table as a DataFrame
easy.nets.train("mindreader", quick=True)          # ~3 min
reader = easy.nets.load("mindreader", quick=True)  # the trained network
reader.read(windows)                               # each driver's headway, acceleration, desired speed ... in real units
```

**Quick** runs shrink every collection and training (forks: 12 cycles × 16 copies; cities: 4 copies × 2 min; training: 1/4 of the steps).
They write to `results/nets/quick/`, so they never overwrite the full results. Expect the same direction of results with wider intervals.
**Full** runs reproduce the numbers in the docs and write to `results/nets/`.

## What you need

| feature needs | what to get |
|---|---|
| `city`: the Pittsburgh road network | `python -m pgh.fetch` (all the city features) |
| `av2`: real Argoverse 2 scenarios | `av-world fetch-av2 data/av2 --n 400` (public bucket, no account), or point `AV2_ROOT` at your folder |
| `logs`: AV decision logs `data/world.bin`, `data/env.bin` | made by `small_av_world` and `python -m avrl.worldmodel collect-env` ([WORLD_MODELS.md](WORLD_MODELS.md)) |

`av-world nets info NAME` says which of these a feature needs and whether it is missing.

## The networks

Times are for a 4-core CPU (quick / full). Full times are measured; quick times are estimates, except occlusion (measured: 1 min).

### `critic`: which action makes this moment safe? (N2)
**Learns** `Q(o, a)`: the probability of a crash, a near miss or being more than 1 m inside the safety barrier in the next 4 s, for each of
five AV actions. The data comes from forks of the city with the dice held fixed. A second critic, trained only on what the policy actually did, is
the control. **Result:** at a 10 % intervention budget, bad outcomes −71 %. The log-trained critic makes them +69 % worse, because it learns that
accelerating is harmless.

```bash
av-world nets train critic --quick     # 4 min / 10 min; runs the shared forks first (results/nets/forks.npz, ~92 MB full)
```
```python
critic = easy.nets.load("critic")
p = critic.risk(obs, u)                # obs [B, 28] (tr["obs"] of xnet.sim.step with rl=), u [B, 2] the policy's action -> [B, 5]
k = critic.best_action(obs, u, min_gain=0.002)   # 0 = keep the policy's action; 1 brake harder, 2 accelerate, 3/4 bigger/smaller gaps
```
**Knobs** in `nets/forks.py`: `DELTAS` (the action set), `PERT` and `WIN` (perturb 2 s, watch 4 s), `BAD_MARGIN`, the world `AD.world(5.0)`.

### `response`: what will the driver behind me do? (N7)
**Learns** the causal effect of the AV's action on the following human's acceleration over the next 4 s, from the same forks. **Result:** when
braking, its error is half that of "the follower ignores us"; a log-trained model is worse than that.

```python
resp = easy.nets.load("response")
d = resp.follower_effect(obs, follower, u, action=1)   # follower = (gap m, speed, accel, is AV, AV speed, AV accel) -> [B, 19] m/s^2 over 0.2-3.8 s
```

### `occlusion`: where are the road users the AV cannot see? (N6)
**Learns** from only the visible road users an 8 × 8 grid (10 m cells, 80 × 80 m around the AV) of where the hidden ones are, plus
P(someone hidden 25 m ahead in my lane). Labels come from exact line of sight in the simulator. **Result:** AUROC 0.968 per cell (geometry 0.926);
hidden-ahead 0.978 (0.770).

```python
occ = easy.nets.load("occlusion")
tokens, visible = occ.from_scene(ego, x, y, heading, length, speed, kind)   # world positions of the road users (m, rad)
grid, ahead = occ.belief(tokens)                                            # [1, 8, 8] probabilities, [1]
```
Vehicles and cyclists only: pedestrians in this simulator are crossing processes, not positioned agents.

### `mindreader`: what kind of driver is that? (N1)
**Learns** from 10 s of a car's driving the hidden parameters the simulator gave its driver. **Result:** headway R² 0.41, acceleration
0.44, desired speed 0.37 (a linear model gets ≈ 0). Red-light runners and aggressive drivers stay at chance: ordinary driving does not show them.

```python
reader = easy.nets.load("mindreader")
style = reader.read(windows)    # windows [B, 25, 8]: speed, accel, gap ahead (150 = none), leader speed, distance to line, light (0 red 1 yellow 2 green), limit, lane change
style["T"], style["amax"], style["v0f"], style["s0"]
```
`nets.mindreader.collect` and `nets.mindreader.windows` build windows from any simulated run.

### `blame`: who caused this crash, and who could have avoided it? (N5)
**Learns** but-for causes. Every crash is rewound 6–8 s and replayed once per nearby driver, with that driver's mistakes off, and once
with that driver driving carefully. The network ranks the candidates from what an investigator could see. **Result:** a mistake is a but-for
cause in only 1.6 % of crashes; a careful driver in one seat avoids 46.6 %; ranking AUROC 0.94. "The faster of the two" still names one driver
better (top-1 0.95 vs 0.89).

```python
b = easy.nets.load("blame", question="careful")     # or question="mistake"
scores = b.rank(candidates, mask)                   # candidates [B, 8, 18] as nets.blame.run_branches writes them
```

### `sbi`: which settings explain real traffic? (N4)
**Learns** a posterior over nine simulator knobs from six traffic statistics, trained on simulated worlds whose knobs are known.
**Result:** its draws land 2.24 SD from the real statistics (a point fit 2.60, the prior 3.50). It also shows that three real statistics lie outside
every simulated world: the simulator cannot follow as closely as real drivers do.

```python
post = easy.nets.load("sbi")
knobs = post.real(n=1000)          # {"T_med": [...], "accel_noise": [...], ...} for real Argoverse 2 traffic
knobs = post.sample(stats, n=500)  # for your own statistics (nets.sbi.summary computes them from scenes)
```

## The world-model upgrades

| name | question | quick / full | how to use what it gives |
|---|---|---|---|
| `crn` | how much less noisy is a paired comparison? | 2 / 10 min | `X.Behaviour(crn=1.0)`, then `avlab.paired.compare(net, beh_a, beh_b)` |
| `conservative` | how do you stop a planner exploiting the model? | 1 / 2 min | `wmplan.conservative.cem(..., j_mopo, {"lam": 0.25}, support=..., signal="support")` |
| `multistep` | does training on its own rollouts fix drift? | 2 / 6 min | `avrl.wm_multistep.fit(rows, hist=3, K=10, key=...)` |
| `agent_tokens` | does attending to neighbours predict real driving? | 4 / 8 min | `wm2.model.apply(params, S, NB, visible)` |
| `learned_humans` | are learned car-followers more realistic than IDM? | 3 / 10 min | `X.set_learned_follower(params); X.Behaviour(learned=1.0)` |
| `sigreg` | how do you stop a joint-embedding model collapsing? | 1 / 1 min | `loss + lam * wmplan.sigreg.sigreg(z, key)` |
| `terminal_value` | does a value let a short planner see further? | 1 / 3 min | `wmplan.terminal_value.cem_h(..., h=5, use_v=True)` |

Train any of them the same way: `av-world nets train crn --quick`.

## The auto-tagger

| feature | question | quick / full | use |
|---|---|---|---|
| `autotag` | which real and simulated scenes contain a cut-in, a hidden road user, a tailgater, or are out of the simulator's reach? | 3 / 25 min | `av-world tag run`, then `av-world tag query "cut_in & hidden_ahead" --overlap` or `easy.tag.query(...)` |

It is a cached DAG, so reruns only rebuild stale stages. The design, ten methods and results are in [AUTOTAG.md](AUTOTAG.md).

## Making it fast

* **Use every core.** The network modules set `XLA_FLAGS=--xla_force_host_platform_device_count=4`, so the 4 CPU cores become 4 devices and
  copies run under `pmap` (×2.1 measured). On another machine, set the count to your core count before running.
* **Run one job at a time.** Two simulations on the same cores each run at about half speed.
* **Everything is cached.** Datasets go to `.npz` next to the results (`forks.npz`, `mindreader_windows.npz`, `occlusion_samples.npz`,
  `blame_data.npz`, `sbi_sims.npz`). Retraining reads the cache and never re-simulates; delete the file to collect fresh data.
* **The first call compiles** (10–30 s per new batch shape). After that, a 60 s run of 32 copies takes a few seconds.

## When something goes wrong

| symptom | fix |
|---|---|
| `... needs: the Pittsburgh map data` | `python -m pgh.fetch` |
| `... needs: Argoverse 2 scenarios` | `av-world fetch-av2 data/av2 --n 400`, or `export AV2_ROOT=/path/to/scenarios` |
| `FileNotFoundError: ..._params.pkl` when loading | train it first; for a quick run, `load(name, quick=True)` |
| out of memory during collection | try `--quick` first; for a full run lower the copies or minutes in the module's `collect()` defaults (e.g. `nets/mindreader.py`) |
| the numbers differ from the docs | quick runs use less data and training; full runs on a different CPU or GPU differ slightly (see FIXES.md #14) |

## Under the hood

* `nets/registry.py`: every feature as data (question, time, outputs, needs, how to run it). The CLI, `easy.nets` and this page read it.
* `nets/use.py`: the loaders (`RiskCritic`, `ResponseModel`, `OcclusionNet`, `MindReader`, `BlameNet`, `RealTrafficPosterior`).
* `nets/collect.py`: the fast recorder shared by all of them:
  * `record` steps vmapped copies in one compiled scan and reduces each step to features;
  * `fork` turns K branches of a moment into one batch;
  * `run` warms up.
* One module per network: `nets/forks.py` + `counterfactual.py` + `response.py`, `occlusion.py`, `mindreader.py`, `blame.py`, `sbi.py`. Each
  has `main(out_dir=..., quick=...)` and a docstring with the method.
* Tests: `pytest tests/test_nets.py tests/test_nets_features.py`.
