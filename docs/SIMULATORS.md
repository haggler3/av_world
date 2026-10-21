# The simulators: intersections, whole areas, Pittsburgh, calibration, running it

The simulator internals, calibration against crash data, the Pittsburgh intersection survey, how to run everything, limitations and the code layout. Moved from the README.

## The simulators

![Oakland, Pittsburgh, every vehicle](xnet_oakland_overview.gif)

*All of Oakland, Pittsburgh (2.4 x 1.8 km, 313 junctions, 75 km of directed lanes) simulated as one area: every vehicle coloured
by its speed relative to the limit. Below, the same run around Forbes Ave & S Craig St.*

![Oakland street view](xnet_oakland_zoom.gif)

## A whole area: Oakland, Pittsburgh (`xnet/`)

`xnet/network.py` turns the Overture / OpenStreetMap roads inside a bounding box into a network. Roads are cut at the
area's edge, and those cut points become entries and exits. Junctions closer than 30 m are joined into one, as real
traffic simulators do. Lanes, signals, stop signs and demand are inferred from road class, as for the single
intersections below.

| | Oakland |
|---|---|
| junctions / directed links | 313 / 622 (68 links with 2 lanes, 76 one-way; Fifth and Forbes are the one-way pair); 74 entries and 74 exits at the area's edge |
| paths through junction boxes | 1,253, with 1,891 conflict points (crossing or merging into the same lane) |
| control (inferred) | 34 signals (2 phases; approaches whose straight or right-turn paths cross never share a phase), 276 stop-controlled approaches (two-way and all-way stops) |
| crosswalks | 608 |

`xnet/sim.py` moves every vehicle every 0.2 s. Vehicles live on *tracks*: a lane of a link, or a path through a box.

| | |
|---|---|
| **following** | IDM toward whichever is nearest: the leader on the same track, the last car on the next one or two tracks (across a box or a short link), a stop line, a bus stop, a parked van, a wreck |
| **lanes** | mandatory lane changes to reach the turn lane; discretionary ones to pass a slow or stopped car; **zipper merges** where a lane ends (the driver alongside holds back to let one in); passing a stopped car on a single-lane street via the opposing lane when it is clear |
| **junctions** | every move has a priority: signal phase or major road, then straight > right > left, then a fixed tie-break. A driver checks the box (cars before each conflict point) and the cars arriving on higher-priority moves, looking up to three links upstream along their routes. At stop signs, the first to stop goes first. Permissive left turners clear at the change to red. "Don't block the box": enter only if there is room after it |
| **pedestrians** | at signals they start in a 10 s WALK interval; elsewhere when no car is close. A share cross against the signal or into a short gap. Drivers yield to crosswalks up to two links ahead |
| **other road users** | buses (3 %) stop 15-35 s at stops on arterials; delivery vans (5 %) double-park 1-3 min; breakdowns block a lane for 5-15 min; crashed cars stay as wrecks for 2-5 min |
| **risky drivers** | the calibrated knobs from `xworld/calibrate.py`: late red-light running, not looking, not signalling, distraction, aggressive gaps; AVs never do any of these |
| **routing** | next-hop tables to 64 destinations (map exits and inner blocks), re-solved every 5 simulated minutes from measured link speeds; trips that cannot reach their destination leave by the nearest exit |

**Results** (`python -m xnet.experiment`: 8 copies of Oakland x 1 simulated hour each, moderate demand, about 480,000
junction entries and 43,000 trips per world):

| | clean world | calibrated risky drivers (x18.5, scaled back) | 3 % of pedestrians violating |
|---|---|---|---|
| trips finished / mean trip | 40,767 / 8.1 min | 40,498 / 8.1 min | 40,995 / 7.9 min |
| mean speed | 11.2 km/h | 11.2 km/h | 11.6 km/h |
| gridlock teleports (stuck 4 min) | 492 (1.1 % of trips) | 528 | 421 |
| lane changes / passes on the opposing lane | 47,641 / 19,570 | 47,211 / 19,761 | 46,950 / 19,211 |
| bus and delivery stops / pedestrian crossings | 10,799 / 112,172 | 10,748 / 112,503 | 10,626 / 117,348 |
| crashes in the run | 18 (10 rear-end, 7 left-turn, 1 pedestrian) | 31 | 22 |
| **crashes per million junction entries** | **37 (model artefacts)** | **+1.5 over clean** (0.9 rear-end, 0.3 angle, 0.1 sideswipe, 0.1 pedestrian) | **+8 over clean** (pedestrian) |

**Read these with care:**

- **The clean world is not crash-free.** It has 37 crashes per million junction entries from the model itself, all
  at Pittsburgh's irregular junctions: odd geometry, very short links, merged junctions. Getting it this low (from
  567 crashes in the first 10-minute run) took about twenty fixes, listed in the commit history. The single-intersection
  world below is crash-free when clean.
- **Driver crash rates are reported as the excess over that baseline**, the same way the calibration subtracts its
  clean baseline. The driver-caused rate, 1.5 per million entries, is the same order as the single-intersection
  calibration (0.62) and the published 0.78. But it rests on only 13 extra crashes (about ±40 %).
- **The pedestrian violation rate (3 %) is a guess, not calibrated.**

**Speed:** 8 copies of Oakland, with about 600 vehicles each in 1,024 vehicle slots, run at about **380x real time in
total** on 4 CPU cores. That is about 8 simulated hours in 75 s, 2.1x faster than the first version. Profiling the
compiled step found where the time went:

- **Vehicle slots:** cost is linear in slots, and 1,024 is enough for a moderate hour.
- **"Last car on a track" lookups:** these were three binary searches per step. They are now one scatter of each
  track's rearmost car.
- **Free slots:** found with `nonzero` instead of a full sort.
- **Conflict lists:** trimmed to the widest one actually used (10 instead of 16).

Results are identical to before at the same slot count.

**Driving AVs with a learned policy:** `xnet.sim.step(..., rl=(policy_fn, key))` lets a policy drive every AV. The
policy reads a 28-feature observation and returns an acceleration command, capped by a classical safety envelope, and a
gap-acceptance scale. The step also returns the transition (reward, valid, done) for training.
[`av_world_rl`](https://github.com/haggler3/av_world_rl) trains such a policy with multi-agent PPO
(`pip install -e .` here first).

![six Pittsburgh intersections simulated in one batch](pgh_grid.gif)

*Single intersections: six real Pittsburgh intersections, auto-mined from OpenStreetMap, simulated together in one vmapped batch. Pink,
green, grey: human drivers; blue: AVs. Starred values (lanes, signal or stop control) are inferred: the open map data
has no lane counts or signal locations.*

## Real intersections from OpenStreetMap: Pittsburgh

`pgh/` reads Pittsburgh's road network from [Overture Maps](https://overturemaps.org/), an open map built largely from
OpenStreetMap, straight from its public S3 bucket. It then mines every intersection and turns each into a simulator layout.

| | |
|---|---|
| road segments read | 69,155 (drivable: trunk to residential; ramps, service roads and paths left out) |
| junctions with 3+ drivable legs | 12,946 |
| **intersections kept** | **10,205** (2,929 four-leg, 7,276 three-leg); dropped: 5+ legs, legs that don't fit four approach slots within 35 degrees, legs shorter than 25 m, junctions closer than 25 m to another (divided roads, offset T pairs) |
| from the map | leg bearings, street names, road class, one-way direction, speed limits (where mapped), distance to the next junction |
| **inferred** (not in the open data) | lanes (trunk and primary roads: 2 per direction, plus a left-turn bay at signals; others: 1), control (**signal** if both crossing roads are tertiary or above and one is secondary or above, 397 intersections; otherwise a **two-way stop** for the lower-class road, 9,808), demand by road class (average hour: trunk 600 to residential 60 vehicles/h per approach), speed (Pittsburgh's 25 mph default where no limit is mapped) |

![mined intersections](pgh_map.png)

Each intersection keeps its real geometry: legs at their real bearings, one-way legs, T-junctions, 1-3 lanes per
direction. Turning paths are curves sized to the turn. The stop-line distance clears the crossing roads, and conflict
points are computed from the actual paths. Signal timing splits the green by demand. At two-way stops, minor-road
drivers stop, then go when the gap is clear; of two drivers waiting on crossing paths, the first to stop goes first.

| | |
|---|---|
| ![Forbes & Murray](pgh_forbes_avenue_murray_avenue.gif) | ![Penn & Coal](pgh_penn_avenue_coal_street.gif) |
| Forbes Ave & Murray Ave (Squirrel Hill), inferred signal | Penn Ave & Coal St, inferred two-way stop |
| ![Craig & Forbes](pgh_south_craig_street_forbes_avenue.gif) | ![Penn & East Liberty](pgh_reizenstein_way_penn_avenue_east_liberty.gif) |
| S Craig St & Forbes Ave (Oakland), a signalised T | Penn Ave & East Liberty Blvd, a skewed signalised crossing |

*The crash clips amplify risky behaviour (x100 at signals, x300 at stop signs) so that a crash happens within 24 s.
One more clip: [Ohio River Blvd & Shiloh Ave](pgh_ohio_river_boulevard_shiloh_avenue.gif), a two-way-stop T.*

**Checks on the real layouts:** with risky behaviour switched off, 1,024 Pittsburgh intersections (all 397 inferred
signals plus 627 stop-controlled) run 20 minutes each with **zero crashes**. All of them flow: even the worst gets 60 %
of its demand through. Getting there took fixing several problems, all of them found on real Pittsburgh geometry:

* skewed T-junctions set the stop line 47 m back;
* two cars merging into one exit were counted as a rear-end while still 35 m apart;
* on narrow two-lane streets, opposing protected left turns crossed each other;
* two left-turners facing each other each waited for the other, a gridlock in about 5 % of four-leg intersections.

**Crash survey** (`python -m pgh.survey`): 2,048 intersections (all inferred signals plus a random 1,651 stop-controlled),
30 simulated minutes each, ran in 9 minutes. That is 433,000 entering vehicles with the calibrated behaviour. It saw 2
crashes, 0.25 per million entering vehicles after scaling. **That is too few to rank intersections or compare
signal and stop control** (90 % interval 0.04-0.8). It would need roughly 20x more simulated traffic, which is possible
on a GPU.

## What is simulated

| | |
|---|---|
| **Roads** | any 3- or 4-leg layout: real bearings, 1-3 lanes in and out per leg, one-way legs; per approach up to 4 paths (left, two through, right) through the box, with every point where two paths cross or merge |
| **Control** | signals: per road a protected left phase where there are left-turn bays, then through with permissive lefts, 3 s yellow, 1-2 s all-red, right turn on red; two-way stop: the major road never stops, minor-road drivers stop fully and go when clear |
| **Driving** | IDM car following; lane changes to reach the turn lane; yellow decided once at onset; gap acceptance for permissive lefts, right on red and stop signs; start-up lost time; drivers look before going on a fresh green; **evasive braking** after a 0.7-1.5 s reaction time when a car crosses their path |
| **Risky behaviour** | late red-light running; not looking before turning or changing lanes; not signalling (the driver cut in front of reacts late); distraction (1.5-3 s without reacting); aggressive gap acceptance |
| **Crashes** | rear-end (same lane), angle and left-turn (crossing paths in the box), sideswipe (merges, lane changes), detected from car bodies |
| **AVs** | a share of vehicles that always look and signal, never run reds, are never distracted, accept larger gaps and react in 0.3 s |

![the default intersection](intersection.gif)

*The default intersection: 4 legs, 3 lanes each way (left-turn bay, through, through or right). Red outline: entered on red;
amber: distracted right now. Risky behaviour amplified x100 for the clip.*

## Calibrated to real crash data

Crashes are rare, so `xworld/calibrate.py` probes each behaviour on its own at an amplified level. It measures crashes
per million entering vehicles (MEV) per unit of that behaviour, and fits the behaviour rates to the targets with bounded least
squares. A verification run then applies all the calibrated behaviours together. It amplifies them as much as possible while keeping every
probability below 1 (x18.5), then scales the rates back down. Calibration uses the default intersection.

| (urban signalised 4-way intersection) | simulated | published |
|---|---|---|
| **crashes per million entering vehicles** | **0.62** (fit 0.74) | **0.78** (Massachusetts statewide signalised average); 1.26 (generic intersection safety table) |
| rear-end share | 50 % (fit 50 %) | 36 % (HSIS, Michigan + Minnesota, 4-leg signalised) |
| angle share | 21 % (fit 13 %) | 20.3 % (Minnesota), 25.8 % (California) |
| left-turn share | 29 % (fit 26 %) | ~15 % (assumed; not in the sources found) |
| red-light-running share of crashes | 21 % | 16-20 % (HSIS, four states) |
| red-light runs per 1000 entering vehicles | 0.60 | 3.9 (Melbourne, 15 intersections; studies range ~1-19) |

The verification run had 14 crashes, so the rate is uncertain by about ±30 % and the shares by about ±13 points. **The
crash rate and mix match; red-light-run frequency is still 6x too low.** Each simulated red run is more likely to end in
a crash than a real one, so matching the crash data leaves too few red runs. Making red running realistic took three
mechanisms: runners only enter in the first 0.3 s of red, start-up lost time, and drivers who look before going on
green. It is still the main open gap.

**Read the calibrated behaviour rates as effective rates, not observed frequencies.** The fit gives:

* 5.1 % late red-light runners;
* not looking in about 1 in 89,000 drivers;
* very rare distraction.

In the model a distraction is a complete 1.5-3 s blackout and a non-looker never checks; real lapses are far more
common but much milder. Stop-sign running and pedestrians are not modelled, which matters most at the stop-controlled
Pittsburgh intersections.

## AV share

Same calibrated world (default intersection), AVs replacing human drivers (runs amplified x18.5, rates scaled back):

| AV share | 0 % | 25 % | 50 % | 100 % |
|---|---|---|---|---|
| crashes per MEV | 0.62 | 0.75 | 0.62 | 0 |
| crashes in the run | 14 | 17 | 14 | 0 |

With 14-17 crashes per run, **0-50 % AVs are not distinguishable**. Only an all-AV fleet clearly removes crashes, and that
follows from how the model is built: every crash needs a human error, and these AVs make none. The AV here is a simple
careful-driver model. The next step is to drive the AVs with the learned policy from
[`av_world_rl`](https://github.com/haggler3/av_world_rl), so that the result is measured, not assumed, and to run much
longer on a GPU.

## Speed, and a GPU training script

[docs/SPEED.md](SPEED.md) compares `xnet` with CARLA, MetaDrive, Nocturne, SUMO/CityFlow, GPUDrive and Waymax in
vehicle-steps per second (CARLA ~10 k, GPUDrive 2.3 M, this repo 0.75 M on 4 CPU threads, 6-13 M projected for one A100).

`python -m avrl.gpu_train` runs PPO with the copies sharded over every device (`jax.shard_map`, gradients averaged with
`pmean`), and the learner trains on the valid AV rows only (`Config.compact`; 86 % of the old rows were padding).
Measured on the 4-thread CPU (4 forced host devices): a PPO update went from 1.97 s to 0.81 s with packing alone, and the
whole trainer from ~210 to ~1,100 area-steps/s. Trained 100 updates (3 min, 32 copies) from the 234k-step policy with no
safety filter, it reached 0.011 crashes per AV-hour in Oakland (3 replicates, 95 % CI -0.001 to 0.022; one seed) against
0.026-0.031 for the 8-copy trainer after the same 100 updates, at the same speed (15.7 vs 16.1 km/h). That is a hint that more
copies per update help, not a measured effect. **No GPU was available: the GPU rows are projections**; run
`python -m avrl.gpu_train --preset gpu --bench` on a card to measure.

## Run it

```bash
pip install .                                  # or: pip install -r requirements.txt, then run from the checkout
export JAX_DEFAULT_PRNG_IMPL=unsafe_rbg        # much cheaper random numbers on CPU

python -m xworld.calibrate                     # probes, fit, verification (~15 min) -> results/calibration.json
python -m xworld.calibrate av                  # crash rate vs AV share -> results/av_sweep.json
python -m xworld.render --out docs/intersection.gif
python -m xworld.bench
python -m pytest -q tests

python -m pgh.fetch                            # Pittsburgh roads from Overture's public S3 (~1 min) -> data/
python -m pgh.mine                             # mine intersections -> data/pgh_intersections.json, docs/pgh_map.png
python -m pgh.survey                           # calibrated crash survey over 2,048 intersections -> results/pgh_survey.json
python -m pgh.gifs                             # labelled GIFs -> docs/pgh_*.gif

python -m xnet.experiment                      # whole area: clean / risky drivers / pedestrians (~9 min) -> results/xnet_oakland.json
python -m xnet.render                          # whole-area overview and street-view GIFs -> docs/xnet_oakland_*.gif
python -m pytest -q tests/test_xnet.py
```

For another city, change `BBOX` in `pgh/fetch.py`. For another area of Pittsburgh, add a bounding box to `AREAS` in
`xnet/network.py` (`squirrel_hill` and `downtown` are there) and pass its name.

In code:

```python
import jax
from xworld import sim as S
from pgh import layouts as P

beh = S.Behaviour(p_red=0.05, p_av=0.3)                      # any behaviour mix, any AV share
metrics, state = S.run_batch(jax.random.PRNGKey(0), n_envs=1024, seconds=600, beh=beh)   # default intersection
print(S.summary(metrics))                                    # crashes per MEV by type, red runs, lane changes

its = P.load()[:512]                                         # 512 mined Pittsburgh intersections, one per env
lay = S.stack([P.build(i)[1] for i in its])
m, _ = S.run_batch(jax.random.PRNGKey(0), len(its), 600, beh, lay=lay, per_env=True)   # [512, metrics]

from xnet import network as NW, sim as X
net, geo, info = NW.build("oakland")                         # the whole area
sim = X.Sim(net, n_envs=8, beh=X.Behaviour(demand=1.0, p_av=0.2))
sim.run(3600)                                                # one hour, 8 copies, rerouting every 5 min
print(X.summary(sim.st.m))
```

## Limitations

* Lanes, signal locations and demand at the real intersections are inferred from road class, not observed (marked * in
  the GIFs). Signal plans are generic, not Pittsburgh's actual timings.
* The area simulator has a residual crash rate of its own (37 per million junction entries, above); paths through a
  junction are fixed curves, and vehicles do not swerve; at most 2 lanes per direction; no parking manoeuvres, cyclists
  or grades (Pittsburgh's hills).
* Only 3- and 4-leg intersections; no roundabouts or all-way stops; no pedestrians, cyclists, trucks, grades or
  single-vehicle crashes (about 16 % of real crashes are outside the four modelled types).
* Arrivals that find the entry occupied are dropped (8-15 % of demand at the Pittsburgh intersections); rates are per
  vehicle that actually entered.
* Lane changes are instantaneous; vehicles follow fixed paths through the box.
* Calibration uses one target set from a few sources, on the default intersection; red-light-run frequency is 6x below
  observations (above).

## Layout

```
xworld/geometry.py   any 3-/4-leg layout: lanes, paths, conflict points, lane/move tables (CANON = default 4-way)
xworld/sim.py        the batched simulator; Layout = one intersection's tables + control plan (stack to vmap)
xworld/calibrate.py  calibration to crash data, verification, AV sweep
xworld/render.py     top-down GIFs with car sprites, signals, stop signs and street labels
xworld/bench.py      throughput
pgh/fetch.py         Pittsburgh roads from Overture Maps (public S3)
pgh/mine.py          intersection mining and attribute inference
pgh/layouts.py       mined intersection -> Geometry + Layout
pgh/survey.py        calibrated crash survey over many real intersections
pgh/gifs.py          labelled Pittsburgh GIFs
xnet/network.py      a whole map area: links, lanes, junction boxes, moves, conflict points, control, crosswalks, routing
xnet/sim.py          the batched area simulator (following, lane changes, merges, right of way, pedestrians, buses, ...)
xnet/experiment.py   clean / risky drivers / pedestrians in one area
xnet/render.py       overview and street-view GIFs
avrl/xppo.py         multi-agent PPO in xnet (one shared policy drives every AV); xeval.py, xviz.py
avrl/ladder.py       the dynamics ladder (training worlds of increasing dynamics, scored in one test world); ladder_plot.py
avrl/fork.py         PPO with options: fork at danger, world-model crash shaping, mutating cities; dyn_eval.py, dyn_plot.py, fork_viz.py
xnet/mutate.py       mutating cities: per-copy speed limits, signal plans, demand and pedestrian hot spots, closures + rerouting
wmplan/risk.py       the world model's crash head as a reward-shaping potential
av_world/            the pip package: make() environment, `av-world` command line, bundled Pittsburgh data and start policy (_data/)
gpu_train: avrl/gpu_train.py   PPO sharded over every device, learner on packed AV rows
xnet/osm.py          any city from an OpenStreetMap extract
xnet/carla_export.py OpenDRIVE map + trajectories + replay script for CARLA
xnet/perception.py   sensor model on recorded scenes: occlusion, range, fog, dropouts, ghosts
avlab/adversary.py   aimed human mistakes (optional): evolution strategies against the AV crash rate
avlab/tails.py       long-tail survey and scene recorder
avlab/rare.py        particle estimator for rare crash probabilities with a learned per-AV committor
avlab/split.py       importance splitting for rare crash rates (and why risk tilting collapses)
avrl/bench.py        AVBench: 11 scenarios x 8 checks vs the rule-based AV, envelope / barrier function / no filter; bench_plot.py
wmplan/              latent world model (data.py, model.py), CEM planner (planner.py), evaluate.py, viz.py, plot.py
avlab/               condition sweeps (sweep.py), statistics (analyze.py), charts (plot.py)
avrl/active.py       active scenario selection: uniform vs level replay vs failure-focused replay; active_plot.py
xnet/export_web.py   recordings + maps for the browser viewer (docs/viewer/)
docs/index.html      the project site (GitHub Pages from /docs)
avrl/env.py, ppo.py  the corridor driver (small_av_world), export.py: weights and parity vectors for Rust
dwm/                 diffusion / flow world models; DWM_SOURCE=city | xnet; xnet_data.py (logs from the area simulator)
dwm/scaling.py       scaling laws across architectures (compute, simulated data); scaling_plot.py
tests/               geometry, conflict-free signal plans, zero crashes without risky behaviour (default and
                     real-world-like layouts), behaviour -> crash type, determinism; area network consistency,
                     routes, flow, distraction -> rear-ends, determinism
results/             calibration.json, av_sweep.json, pgh_survey.json, xnet_oakland.json, ladder/, active/, bench/, avlab/, wmplan/, scaling/, xnet*/ (policies), dwm/, final/
```

Sources: [Overture Maps](https://docs.overturemaps.org/) (transportation theme, release 2026-09-23.1;
© OpenStreetMap contributors, ODbL),
[Massachusetts crash rate data](https://www.mass.gov/service-details/intersection-and-roadway-crash-rate-data-for-analysis),
[intersection safety course (C04-039)](https://www.cedengineering.com/userfiles/C04-039%20-%20Intersection%20Safety%20-%20US.pdf),
[FHWA HSIS signalised intersection study](https://www.fhwa.dot.gov/publications/research/connectedvehicles/11040/004.cfm),
[IHSDM crash type defaults](https://ihsdm.org/public/html/user/cpm/cpm_em.9.html),
[SWOV: red light running (Melbourne observations)](https://swov.nl/en/publicatie/red-light-running).
