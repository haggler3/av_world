# Easy mode: `av_world.easy` and `av_world.compat`

The research code is many modules with many knobs. `av_world.easy` does the common jobs in one line each, with defaults that match
the rest of the repository (calibrated human drivers, the bundled PPO policy, AVBench's warm states), so the numbers it gives are
the numbers everything else reports. `av_world.compat` connects av_world to the tools people already use. Walkthrough, executed:
[notebooks/06_easy_mode.ipynb](../notebooks/06_easy_mode.ipynb).

## In ten lines

```python
from av_world import easy

run = easy.simulate("oakland", minutes=5, av_share=0.3)        # 4 copies, calibrated humans, 30 % AVs
run.summary()                                                   # crashes (with a 95 % interval), speeds, trips: plain units
run.show(); run.gif("city.gif", seconds=30); run.save("trips.parquet")

@easy.driver                                     # your own driver, in metres, seconds and m/s^2
def two_second_rule(car):
    jnp = easy.jnp
    follow = jnp.where(car.has_lead & (car.lead_gap < 2 * car.speed + 4), -2 * car.brake_needed_for_lead - 1, 3.0)
    stop = jnp.where(car.has_stop & (car.stop_gap < 2 * car.speed + 8), -2 * car.brake_needed_for_stop - 1, 3.0)
    return jnp.minimum(jnp.minimum(follow, stop), jnp.where(car.speed < car.desired_speed, 1.2, -0.5))

easy.compare({"mine": two_second_rule, "IDM": easy.drivers.idm()})   # AVBench, against the leaderboard
policy = easy.train(updates=25, reward={"headway": 1.0})        # PPO, every AV driven by one policy
easy.export_perception("data/bus", scenario="bus_dartout")      # cameras + 3-D boxes + sensor detections
```

## New tools in plain words (November 2026)

Three additions, each usable in a few lines. Run Python from the repository root with the same interpreter pip used
(`av-world doctor` says which).

### Where and when did each crash happen? (`avlab.crashlog`)

```python
from avlab import crashlog
crashes, exposure = crashlog.quick("oakland", copies=2, minutes=5, friction=0.4)   # any simulator knob works, e.g. p_av=0.5
crashes.head()                       # one row per crash: copy, time_s, x, y, link, vehicles, impact_kmh, av_involved, kind
crashes.groupby("kind").size()       # rear, junction, lane_change, pedestrian, other
exposure.nlargest(5, "vehicle_hours")  # the busiest streets, to turn crash counts into rates per vehicle-hour
```

`kind` is the most specific role among the crash's vehicles (pedestrian, then lane change, junction, rear-end striker, other). Logging does not change the simulation (a test checks the final state
is identical), and the crash count matches the simulator's own counters. Bigger runs and the patchy-ice study:
`python -m avlab.icemap --run`. For full control (any `Behaviour` per copy, your own loop) use `crashlog.Logger`.

### Try a different AV planner: the Timed Elastic Band (`avlab.teb`)

```python
from av_world import easy
easy.compare({"TEB": easy.drivers.teb(), "IDM": easy.drivers.idm()}, scenarios=["nominal"])   # AVBench, ~15 min on 4 CPU cores
```

TEB plans the next 4 s as 8 points and times between them, and picks the fastest plan that respects speed, comfort and the gap to
the car ahead. Under the simulator's safety envelope it drives 10 % faster and smoother than IDM; on its own it crashes, so its
AVBench score is near 0. Treat it as a starting point to improve, not a safe driver. The animation `docs/img/teb_band.gif` shows
the plan being re-made every 0.2 s. Experiments: `python -m avlab.tebbench --quick`.

### Machine-checked math about the simulator (`formal/`)

`formal/AvFormal/Barrier.lean` proves, in Lean 4, when the AVs' braking barrier can always be obeyed: only if the road's friction is
at least 2/3 (because the barrier assumes 6 m/s² of braking and the road gives 9 x friction). You do not need Lean to use the
result: `python -m avlab.barrierfriction_plot --no-gif` draws the picture, and `tests/test_barrier_formal.py` checks the proof's
formulas against the code. To re-check the proofs yourself: install Lean 4.25.0, then `cd formal && lake exe cache get && lake build`
(see [formal/README.md](../formal/README.md)).

## `av_world.easy`

| function | what it does | time (4-core CPU) |
|---|---|---|
| `cities()`, `scenarios()` | what you can simulate: 6 areas, 10 famous-incident archetypes, 18 AVBench scenarios | instant |
| `simulate(city, minutes, av_share, scenario, driver, copies, **knobs)` | a `Run`: `.summary()`, `.show()`, `.gif()`, `.save()`, `.more()` | ~10 s per copy-minute incl. compile |
| `world(av_share, scenario, intensity, risk, **knobs)` | the traffic itself (an `xnet.sim.Behaviour`), for advanced use | |
| `driver(fn)` | `fn(car) -> acceleration` (m/s²) or `(acceleration, caution)` becomes a policy; `car` has 27 named fields (`easy.Car.describe()`) | |
| `drivers.idm(T)`, `.comfortable()`, `.cautious()`, `.ppo()`, `.teb()`, `.load(path)`, `.rule_based` | ready-made drivers; `comfortable` is IDM that brakes only as hard as it must (AVBench nominal 93.1 vs IDM's 81.0, passes the comfort check) | |
| `score(driver, scenarios)` | AVBench scorecard: av_world score, the 8 checks, crash rates with and without the safety filter | ~30-60 s per scenario |
| `compare({name: driver}, scenarios)` | a table of several drivers plus the leaderboard's references | |
| `sweep(param, values)` | `simulate` once per value of one setting (a DataFrame) | |
| `train(updates, city, start, reward)` | PPO from the bundled policy, from scratch, or from a checkpoint; returns a `Policy` (`.plot()`, `.save()`, usable as a driver) | ~2.5 s per update |
| `reward_weights()` | the reward terms you can override | |
| `export_perception(out, scenario, city)` | nuScenes-style images, boxes, detections, calibration (needs `pyvista`) | ~3 s per simulated second |
| `export_3d(out, scenario, city)` | OpenUSD stage + control video + side-by-side clip (NVIDIA Omniverse / Isaac) | ~1-2 min |

## `av_world.compat`

| you have | use | notes |
|---|---|---|
| a Gymnasium / Stable-Baselines3 / CleanRL agent | `compat.GymEnv(area)` | drive one AV in the full city; passes `gymnasium.utils.env_checker.check_env`; SB3's PPO runs ~80 steps/s on a CPU |
| a multi-agent (PettingZoo / RLlib) setup | `compat.ParallelEnv(area)` | every AV is an agent `"av_<slot>"`; agents come and go with trips |
| a numpy function | `compat.from_numpy(fn)` | `fn(obs [N, 28]) -> u [N, 2]`, run through `jax.pure_callback` (works under vmap) |
| a PyTorch module / an SB3 model | `compat.from_torch(module)`, `compat.from_sb3(model)` | back in the full city, every AV |
| the trained JAX policy, elsewhere | `compat.to_torch(path)`, `compat.to_onnx(path)` | identical outputs (max difference ~1e-7); ONNX for C++, ROS, TensorRT |
| trajectories for pandas / Spark / SQL | `run.save("x.csv" / "x.parquet")`, `compat.scene_to_table(scene)` | one row per vehicle per 0.2 s step |
| CARLA | `run.save("x.xodr")` | OpenDRIVE roads, the scene JSON and a replay script |
| NVIDIA Omniverse / Isaac / Cosmos | `easy.export_3d(...)`, `av-world export-nvidia` | [NVIDIA.md](NVIDIA.md) |
| a perception model | `easy.export_perception(...)`, `av-world export-perception` | [BEV.md](BEV.md) |
| a stack in another process or language | `av-world sil` | TCP, lockstep or real time: [SIL_HIL.md](SIL_HIL.md) |
| OpenStreetMap, Argoverse 2, Waymo | `easy.simulate("map.osm")`, `xnet.avdata` | [DATA_IMPORT.md](DATA_IMPORT.md) |

Gymnasium, PettingZoo, torch, SB3 and pandas are optional; `compat` imports them only when used (without Gymnasium, `GymEnv` still
works with the same API and a minimal `Box`).

## What it does not hide

* Crash counts from short runs are noisy: `summary()` reports the 95 % interval (exact Poisson) next to every crash rate. A
  3-minute run of 4 copies gives intervals several times wider than the rate.
* `train` with a few updates shows the mechanics, not a good driver; see notebook 03 and [RL.md](RL.md).
* `GymEnv` controls one AV with the safety envelope on (`shield="off"` for the raw command); its reward is the simulator's
  per-step reward, the same one PPO uses.
