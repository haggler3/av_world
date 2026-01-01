# av_world

A fast, batched **intersection world** for AV research, in pure JAX. It covers signalised and stop-controlled intersections with
turn lanes, and human drivers who run reds, don't look, don't signal and get distracted. Crash rates are **calibrated
to published intersection crash data**. Intersections can be the default 4-way or **real ones auto-mined from
OpenStreetMap** (here: Pittsburgh). Thousands run in parallel (`jax.vmap`), each with its own layout, so rare events
like crashes can be measured.

![six Pittsburgh intersections simulated in one batch](docs/pgh_grid.gif)

*Six real Pittsburgh intersections, auto-mined from OpenStreetMap, simulated together in one vmapped batch. Pink,
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

![mined intersections](docs/pgh_map.png)

Each intersection keeps its real geometry: legs at their real bearings, one-way legs, T-junctions, 1-3 lanes per
direction. Turning paths are curves sized to the turn. The stop-line distance clears the crossing roads, and conflict
points are computed from the actual paths. Signal timing splits the green by demand. At two-way stops, minor-road
drivers stop, then go when the gap is clear; of two drivers waiting on crossing paths, the first to stop goes first.

| | |
|---|---|
| ![Forbes & Murray](docs/pgh_forbes_avenue_murray_avenue.gif) | ![Penn & Coal](docs/pgh_penn_avenue_coal_street.gif) |
| Forbes Ave & Murray Ave (Squirrel Hill), inferred signal | Penn Ave & Coal St, inferred two-way stop |
| ![Craig & Forbes](docs/pgh_south_craig_street_forbes_avenue.gif) | ![Penn & East Liberty](docs/pgh_reizenstein_way_penn_avenue_east_liberty.gif) |
| S Craig St & Forbes Ave (Oakland), a signalised T | Penn Ave & East Liberty Blvd, a skewed signalised crossing |

*The crash clips amplify risky behaviour (x100 at signals, x300 at stop signs) so that a crash happens within 24 s.
One more clip: [Ohio River Blvd & Shiloh Ave](docs/pgh_ohio_river_boulevard_shiloh_avenue.gif), a two-way-stop T.*

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

![the default intersection](docs/intersection.gif)

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

## Speed

`python -m xworld.bench` on 4 CPU cores (default intersection, 700 vehicles/h per approach):

| intersections in parallel | intersection-hours per second | entering vehicles per second |
|---|---|---|
| 256 | 1.5 | 2,800 |
| 1024 | 1.9 | 3,600 |
| 2048 | 1.9 | 3,600 |

About 8 minutes per million entering vehicles on CPU. Supporting any layout per intersection cost about 14 % of the
speed of the earlier fixed-layout version (2.2 intersection-hours per second).

The whole step is one jitted function with fixed shapes, and per-intersection layouts are just batched arrays, so
running thousands of *different* real intersections costs the same as thousands of copies of one. It should be much
faster on a GPU.

## Run it

```bash
pip install -r requirements.txt
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
```

For another city, change `BBOX` in `pgh/fetch.py`.

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
```

## Limitations

* Lanes, signal locations and demand at the real intersections are inferred from road class, not observed (marked * in
  the GIFs). Signal plans are generic, not Pittsburgh's actual timings.
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
tests/               geometry, conflict-free signal plans, zero crashes without risky behaviour (default and
                     real-world-like layouts), behaviour -> crash type, determinism
results/             calibration.json, av_sweep.json, pgh_survey.json
```

Sources: [Overture Maps](https://docs.overturemaps.org/) (transportation theme, release 2026-09-23.1;
© OpenStreetMap contributors, ODbL),
[Massachusetts crash rate data](https://www.mass.gov/service-details/intersection-and-roadway-crash-rate-data-for-analysis),
[intersection safety course (C04-039)](https://www.cedengineering.com/userfiles/C04-039%20-%20Intersection%20Safety%20-%20US.pdf),
[FHWA HSIS signalised intersection study](https://www.fhwa.dot.gov/publications/research/connectedvehicles/11040/004.cfm),
[IHSDM crash type defaults](https://ihsdm.org/public/html/user/cpm/cpm_em.9.html),
[SWOV: red light running (Melbourne observations)](https://swov.nl/en/publicatie/red-light-running).
