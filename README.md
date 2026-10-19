# av_world

A fast, batched **intersection world** for AV research, in pure JAX: signalised 4-way intersections with turn lanes,
human drivers who run reds, don't look, don't signal and get distracted, and crash rates **calibrated to published
intersection crash data**. Thousands of intersections run in parallel (`jax.vmap`), so rare events like crashes can be
measured.

![a simulated intersection](docs/intersection.gif)

*Pink, green, grey: human drivers; blue: AVs; red outline: entered on red; amber: distracted right now. The clip
amplifies risky behaviour so that a crash (here a rear-end in a queue) happens within 24 s.*

## What is simulated

| | |
|---|---|
| **Roads** | 4 approaches × 3 inbound lanes (left-turn bay, through, through-or-right) and 3 outbound lanes; 16 turning paths through the box, with every point where two paths cross or merge |
| **Signals** | 102 s, 4-phase plan: protected then permissive lefts, 3 s yellow, 1-2 s all-red, right turn on red after a stop |
| **Driving** | IDM car following; lane changes on the approach to reach the turn lane; yellow decided once at onset; gap acceptance for permissive lefts and right-on-red; start-up lost time; drivers look before going on a fresh green; **evasive braking** after a 0.7-1.5 s reaction time when a car crosses their path |
| **Risky behaviour** | late red-light running; not looking before turning or changing lanes; not signalling (the driver cut in front of reacts late); distraction (1.5-3 s without reacting); aggressive gap acceptance |
| **Crashes** | rear-end (shared lanes), angle and left-turn (crossing paths in the box), sideswipe (merges, lane changes), detected from car bodies |
| **AVs** | a share of vehicles that always look and signal, never run reds, are never distracted, accept larger gaps and react in 0.3 s |

## Calibrated to real crash data

Crashes are rare, so `xworld/calibrate.py` probes each behaviour on its own at an amplified level, measures crashes per
million entering vehicles (MEV) per unit of that behaviour, and fits the behaviour rates to the targets with bounded least
squares. A verification run then applies all calibrated behaviours together, amplified 25x, and scales the rates back.

| (urban signalised 4-way intersection) | simulated | published |
|---|---|---|
| **crashes per million entering vehicles** | **0.85** | **0.78** (Massachusetts statewide signalised average); 1.26 (generic intersection safety table) |
| rear-end share | 54 % (fit 36 %) | 36 % (HSIS, Michigan + Minnesota, 4-leg signalised) |
| angle share | 23 % (fit 14 %) | 20.3 % (Minnesota), 25.8 % (California) |
| left-turn share | 23 % (fit 19 %) | ~15 % (assumed; not in the sources found) |
| red-light-running share of crashes | 23 % | 16-20 % (HSIS, four states) |
| red-light runs per 1000 entering vehicles | 0.32 | 3.9 (Melbourne, 15 intersections; studies range ~1-19) |

The verification run had 13 crashes, so the shares are uncertain by about ±15 points. **The crash rate and mix match;
red-light-run frequency does not:** each simulated red run is still about 10x more likely to end in a crash than a real
one, so matching the crash data leaves too few red runs. Making red running realistic took three mechanisms (runners
only enter in the first 0.3 s of red, start-up lost time, and drivers who look before going on green) and it is still
the main open gap.

**Read the calibrated behaviour rates as effective rates, not observed frequencies.** The fit gives 2.5 % late
red-light runners, not looking in about 1 in 93,000 drivers, and very rare distraction. In the model a distraction is a
complete 1.5-3 s blackout and a non-looker never checks; real lapses are far more common but much milder.

Before any calibration, the simulator was checked to produce **zero crashes with all risky behaviour switched off**
(it took several bug fixes, listed in the commit history: yield on yellow, arrival-time estimates, lane-change gaps,
spawn spacing, crash detection from car bodies).

## AV share

Same calibrated world, AVs replacing human drivers (rates scaled from runs amplified 25x):

| AV share | 0 % | 25 % | 50 % | 100 % |
|---|---|---|---|---|
| crashes per MEV | 0.59 | 0.39 | 0.07 | 0 |
| crashes in the run | 9 | 6 | 1 | 0 |

This mostly confirms how the model is built: every crash needs a human error, and these AVs make none. The AV here is a
simple careful-driver model; the next step is to drive the AVs with the learned policy from
[`av_world_rl`](https://github.com/haggler3/av_world_rl), so that the result is measured, not assumed.

## Speed

`python -m xworld.bench` on 4 CPU cores:

| intersections in parallel | intersection-hours per second | entering vehicles per second |
|---|---|---|
| 256 | 1.6 | 3,100 |
| 1024 | 2.1 | 3,900 |
| 2048 | 2.2 | 4,100 |

About 7 minutes per million entering vehicles on CPU. The whole step is one jitted function with fixed shapes, so it
should be much faster on a GPU.

## Run it

```bash
pip install -r requirements.txt
export JAX_DEFAULT_PRNG_IMPL=unsafe_rbg        # much cheaper random numbers on CPU

python -m xworld.calibrate                     # probes, fit, verification (~12 min) -> results/calibration.json
python -m xworld.calibrate av                  # crash rate vs AV share -> results/av_sweep.json
python -m xworld.render --out docs/intersection.gif
python -m xworld.bench
python -m pytest -q tests
```

In code:

```python
import jax
from xworld import sim as S

beh = S.Behaviour(p_red=0.025, p_av=0.3)            # any behaviour mix, any AV share
metrics, state = S.run_batch(jax.random.PRNGKey(0), n_envs=1024, seconds=600, beh=beh)
print(S.summary(metrics))                            # crashes per MEV by type, red runs, lane changes
```

## Limitations

* One isolated intersection type (4-way, 3 lanes each way, a fixed signal plan); no pedestrians, cyclists, trucks or
  single-vehicle crashes (about 16 % of real crashes are outside the four modelled types).
* Arrivals that find the entry occupied are dropped (about 15 % of demand at 700 vehicles per hour per approach);
  rates are per vehicle that actually entered.
* Lane changes are instantaneous; vehicles follow fixed paths through the box.
* Calibration uses one target set from a few sources; red-light-run frequency is 10x below observations (above).

## Layout

```
xworld/geometry.py   lanes, the 16 paths, conflict points
xworld/sim.py        the batched simulator (signals, driving, behaviour, crashes, arrivals)
xworld/calibrate.py  calibration to crash data, verification, AV sweep
xworld/render.py     top-down GIF with car sprites
xworld/bench.py      throughput
tests/               geometry, conflict-free signal plan, zero crashes without risky behaviour, behaviour -> crash type, determinism
results/             calibration.json, av_sweep.json
```

Sources: [Massachusetts crash rate data](https://www.mass.gov/service-details/intersection-and-roadway-crash-rate-data-for-analysis),
[intersection safety course (C04-039)](https://www.cedengineering.com/userfiles/C04-039%20-%20Intersection%20Safety%20-%20US.pdf),
[FHWA HSIS signalised intersection study](https://www.fhwa.dot.gov/publications/research/connectedvehicles/11040/004.cfm),
[IHSDM crash type defaults](https://ihsdm.org/public/html/user/cpm/cpm_em.9.html),
[SWOV: red light running (Melbourne observations)](https://swov.nl/en/publicatie/red-light-running).
