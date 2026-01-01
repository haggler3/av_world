# Five ways to tease novel long-tail data out of the simulator

The incident audit ([INCIDENT_AUDIT.md](INCIDENT_AUDIT.md)) said two things about the simulator's tail: it makes discrete failure *mechanisms* on
demand (a dart-out, a sensor blackout, a pile-up), and it is far thinner than real traffic in the *everyday* tail (hard braking, tailgating, short
time-to-collision: 4-6x thinner than Argoverse 2). This report is about doing something about both. Five methods, each implemented, each run, each
scored the same way, with the pseudocode and the results as they came out, including the ones that did not work.

**The yardstick.** A method is good if it yields (a) more *rare events per simulated hour* than plain simulation, (b) more *different* ones, and
(c) events that stay *realistic* (the surrogate-safety statistics of `avlab.incidents.scene_metrics` against real logs). A method that finds a
single freak failure over and over, or failures that vanish on a second look, scores badly.

| # | method | what it makes | verdict |
|---|---|---|---|
| 1 | [Fit the everyday driver to real logs](#1-fit-the-everyday-driver-to-real-logs) | the *continuous* tail: hard braking, short headways | **works for hard braking and TTC < 2 s, falls short on tailgating**; but it did not make the rule-based AV crash more |
| 2 | [Counterfactuals on real traffic](#2-counterfactuals-on-real-traffic) | real scenes with one neighbour behaving badly | **works as a scene generator, weakly: 1-2 new ego failures per perturbation over ~30-40 applicable scenes; the learned policy fails more than the rule-based AV** |
| 3 | [MAP-Elites failure library](#3-a-library-of-different-failures-map-elites) | a catalogue of *different* failures | **works against random search (7 cells vs 5), but elites regress when re-run** |
| 4 | [Novelty mining](#4-mine-for-the-unfamiliar) | a ranking of scenes by unfamiliarity | **modest and real: 3x crash enrichment in the top 5 %** |
| 5 | [Go-explore from near-misses](#5-go-explore-from-near-misses) | many crash scenes per hour | **does not work as built: no more crashes than plain simulation** |

Everything below uses the shared pieces of the repository: the area simulator (`xnet.sim`) with its optional knobs, the 35-number
description of an AV's situation from `avlab.rare`, the scene metrics of `avlab.incidents`, and the Argoverse 2 importer of `xnet.avdata`.
Raw results are in `results/tail/`; each module's docstring has the command to rerun it; the building blocks are tested in
`tests/test_tail_methods.py`.

---

## 1. Fit the everyday driver to real logs

**Why.** The simulator's humans are IDM followers with a headway of 1.0-1.8 s and no noise. Real drivers brake without a reason, follow closer
than that, and vary. That texture is what produces most real near-misses, and it was missing.

**New simulator knobs** (all off by default, so existing results are unchanged; `Behaviour`):
`T_med`, `T_sigma` (headway `T = T_med * exp(T_sigma * z)`, `z ~ N(0,1)` at spawn), `accel_noise` (standard deviation of white noise added to a
human driver's acceleration each step), `pulse_rate` and `pulse_a` (a human driver sometimes brakes at `pulse_a` for one second with no reason).

```
real = scene_metrics over 400 Argoverse 2 scenes -> r*  (hard brakes per hour, share of time with headway < 0.8 s, TTC < 2 s, TTC < 1 s)
theta = (T_med, T_sigma, accel_noise, pulse_rate);  mu, sd = centre and a third of the box
repeat 5 times:
    draw 8 thetas ~ N(mu, sd), clipped to the box
    for each theta:
        simulate Oakland with humans(theta): 2 x (100 s warm-up + 60 s recorded)
        r(theta) = scene_metrics of the recorded scenes
        loss(theta) = sum_k | log( (r_k + eps_k) / (r*_k + eps_k) ) |
    elite = the 3 lowest-loss thetas
    mu, sd = mean and std of the elite   (std floored at 5 % of the box)
```
`avlab/tail_calibrate.py`, about 4 minutes. The fit hit the box edges (acceleration noise 0.94 of 1.0, pulse rate 0.019 of 0.02), so the box
limited it.

| statistic | real (Argoverse 2) | default drivers | fitted drivers | fitted, fresh seeds |
|---|---|---|---|---|
| hard brakes per moving-vehicle hour | 43.2 | 11.3 | **47.9** | 53.4 |
| time with headway under 0.8 s | 1.85 % | 0.04 % | 0.24 % | 0.22 % |
| time with TTC under 2 s | 1.05 % | 0.16 % | **0.69 %** | 0.70 % |
| time with TTC under 1 s | 0.35 % | 0.01 % | 0.03 % | 0.02 % |
| loss (sum of absolute log ratios) | | 10.0 | 4.7 | 5.3 |

Fitted knobs: median headway 0.76 s, `T_sigma` 0.52, acceleration noise 0.94 m/s², brake pulses at 0.019 per second.

**Result.** Hard braking and TTC < 2 s now match the logs; tight following (8x short) and TTC < 1 s (12-17x short) do not: with an IDM follower,
the headway the driver *wants* is not the headway it *keeps*, and no setting of these four knobs reaches the real extremes. A different following
model would be needed.

**What it did to the AV.** The rule-based AV under the IDM envelope in a x5-risk world, 217 AV-hours with each population: **2 crashes with
default drivers, 0 with the fitted ones**, AV hard braking unchanged (18.7 and 18.4 per hour). The realistic everyday tail did not make this
AV crash more: the IDM envelope absorbs noise and brake pulses. So method 1 repairs the *statistics* of the simulator; it does not by itself
produce new failures for a robust AV.

---

## 2. Counterfactuals on real traffic

**Why.** The most believable stress scene is a real one with one thing changed. Argoverse 2 supplies the real scene (geometry, speeds, spacing);
`xnet.avdata` puts it into the simulator; the change is scripted.

```
for scenario in Argoverse 2 validation scenes:
    net = the scenario's lane map -> road network;  matcher = lane lookup for the network
    for driver in {rule-based AV under the IDM envelope, learned policy A with no filter}:
        st = logged vehicles at t = 2.5 s placed on the network, the logged ego replaced by `driver`
        lead, follower = the vehicles directly ahead of / behind the ego on its lane
        for perturbation in {none, lead_brake, lead_stop, tailgater, ped_dartout}:
            run 8 s of the reactive simulator (no new arrivals), applying the perturbation:
                lead_brake   the leader loses 5 m/s of speed per second for 2.5 s
                lead_stop    the leader stops dead (maximum braking, as a breakdown)
                tailgater    the follower gets headway 0.3 s, desired speed x1.5, acceleration 3 m/s^2
                ped_dartout  a pedestrian starts crossing the next crosswalk when the ego is 1.5 s from it
            outcome = ego crashed?, smallest gap, ego hard-braked?
        failures that the run with no perturbation does not have are caused by the perturbation; export them as CARLA scenes
```
`avlab/tail_counterfactual.py`.

150 real Argoverse 2 scenarios in which the ego could be placed on the imported network (the run stopped there when the container restarted;
`results/tail/counterfactual_rows.json`). A perturbation applies only when it can: 29 egos had a car directly ahead on their lane, 40 one behind,
and the pedestrian stepped out in 23-43 runs (the ego had to reach a crosswalk within 8 s). *New failures* are crashes in scenes where the
unperturbed run did not crash.

| driver | perturbation | runs | ego crashes | new failures | mean smallest gap | ego hard-braked |
|---|---|---|---|---|---|---|
| rule-based AV (IDM envelope) | none | 150 | 2.0 % | - | 44.5 m | 35 % |
| | lead brakes hard | 29 | 3.4 % | 0 | 11.3 m | 17 % |
| | lead stops dead | 29 | 0 % | 0 | 6.4 m | 24 % |
| | tailgater behind | 40 | 2.5 % | 0 | 38.9 m | 45 % |
| | pedestrian dart-out | 43 | 4.7 % | 2 | 58.5 m | 42 % |
| learned policy A, no filter | none | 150 | 6.0 % | - | 47.1 m | 22 % |
| | lead brakes hard | 29 | 10.3 % | 2 | 17.3 m | 17 % |
| | lead stops dead | 29 | 6.9 % | 1 | 9.3 m | 14 % |
| | tailgater behind | 40 | 7.5 % | 0 | 41.1 m | 32 % |
| | pedestrian dart-out | 23 | 8.7 % | 2 | 60.0 m | 43 % |

**Result.** The perturbations do what they should to the margins (the gap to a braking or stopping leader falls from ~45 m to 6-17 m) and
produce a few new failures, more for the learned policy (5) than for the rule-based AV (2), which holds its gap behind a stopping car. The
unperturbed runs already crash 2-6 % of the time: those crashes come from the seeding itself (vehicles placed by lane matching, random driver
parameters, guessed junction control; DATA_IMPORT.md shows the simulator predicts these logs poorly), so the control column matters and the
absolute rates do not. The failure scenes are exported for CARLA in `results/tail/counterfactual_scenes/`. The method's limits are its yield
(only a fifth of scenes have a leader on the same imported lane) and the realism of the continuation, not the idea.

---

## 3. A library of different failures: MAP-Elites

**Why.** Searching for the single worst scenario returns one failure many times. MAP-Elites keeps the best scenario for each *kind* of failure.

A scenario is a point in a box of 15 simulator knobs (human risk, red-runner share and duration, drivers who do not look, dart-outs, AV sensor
blackouts and ghosts, breakdowns, tight cut-ins, friction, pedestrian and cyclist volume, close passes, tailgaters). The policy under test is policy A (trained without a
filter, about 0.08 crashes per AV-hour in ordinary traffic). A scenario's *cell* is (which crash type dominates the AV's crashes) x (how severe:
mean impact or closing speed in four buckets).

```
archive = {}                                  # cell -> (genome, fitness)
for i in 1..240:
    genome = random in the box                if i <= 60
           = elite from a random cell, with Gaussian noise on ~40 % of the genes     otherwise
    run policy A in 6 copies of Oakland for 100 s under decode(genome)
    fitness = AV crashes per AV-hour;  cell = (dominant AV crash type, severity bucket)
    if >= 2 AV crashes and (cell empty or fitness > archive[cell].fitness):  archive[cell] = (genome, fitness)
baseline: the same 240 evaluations, all random
verify: re-run each elite on fresh random numbers
```
`avlab/tail_qd.py`, about 30 minutes for both.

| | cells filled | QD score (sum of fitness) | best | elites still crashing on re-run | mean fitness found -> again | same crash type again |
|---|---|---|---|---|---|---|
| **MAP-Elites** | **7** (angle x2, merge, pedestrian x2, rear-end x2) | **3.6** | 0.94 | 5 of 7 | 0.51 -> 0.21 | 2 of 7 |
| random search | 5 (pedestrian x4, rear-end) | 2.3 | 0.56 | 2 of 5 | 0.45 -> 0.16 | 1 of 5 |

**Result.** MAP-Elites finds a broader catalogue than random search (it alone found angle and merge failures), at the same budget. But each
scenario was scored on 6 copies for 100 s, a handful of crashes, and the *elites regress*: on fresh random numbers fitness falls by more than
half and the crash type changes more often than not. This is the winner's curse of picking the best of a noisy evaluation. A usable library needs
each elite re-scored on many more copies (the verified fitness, not the found fitness, is the number to trust). The search also pushed knobs
to the edges of the box (sensor blackouts at their maximum, friction near its minimum), so the catalogue is of extreme conditions.

---

## 4. Mine for the unfamiliar

**Why.** Long-tail data is, by definition, the data unlike the rest. Instead of designing scenarios, run the simulator at scale and keep the
moments that look unfamiliar.

```
describe(AV, 3-s window) = the policy's 28-feature observation + barrier margin + 2 vetoes + 4 scene counts   (35 numbers)
ref  = descriptions of ~340,000 AV-windows from ordinary traffic (humans x1)
mine = descriptions of ~576,000 AV-windows from a stressed world (humans x5, sensor faults, dart-outs, stalled cars, tight cut-ins, tailgaters)
novelty(row) = mean distance to its 10 nearest neighbours in `ref`    (standardised, k-d tree)
label(row)   = that AV crashes within the next 6 s                    (84 crashes in `mine`)
report the crash rate among the most novel 0.1 %, 1 %, 5 % of rows, against the base rate
baselines: rank by barrier margin (small = risky); random
also: within-world rarity (reference = the first half of `mine`, scored on the second half)
```
`avlab/tail_novelty.py`, about 8 minutes.

| ranking | AUC | crashes in the top 1 % of rows (expected by chance 0.8) | crashes in the top 5 % (expected 4.2) |
|---|---|---|---|
| **novelty vs ordinary traffic** | **0.62** | **4** (x4.8) | **13** (x3.1) |
| novelty within the mined world (second half) | 0.62 | 3 (expected 0.6, x4.7) | 11 (expected 3.2, x3.4) |
| barrier margin | 0.59 | 1 (x1.2) | 7 (x1.7) |
| random | 0.48 | 1 | 3 |

**Result.** Unfamiliar situations are about three times more likely to precede an AV crash than average, and a little better than ranking by the
barrier margin, which is the standard safety signal. 13 of the 84 crashes sit in the 5 % most novel rows. The effect is real (13 against 4 expected)
but weak (AUC 0.62), and the top 0.1 % has one crash in it, so its "x12" is not an estimate. Novelty works as a *filter that makes a data set
three times richer in dangerous moments*, not as a predictor.

---

## 5. Go-explore from near-misses

**Why.** The rare-event estimator failed because it had to be unbiased. A generator of crash scenes does not: it only has to find many different
crashes per hour. Go-explore archives the states where trouble was close and restarts from them with new random numbers.

```
archive = {}                                           # cell -> (state, danger, visits)
repeat for 40 windows of 3 s, with 64 copies:
    if the archive has >= 3 cells:
        pick 64 archive entries with probability ~ (1 + danger)^2 / sqrt(1 + visits); clone them with fresh random keys
    run one window;  count the copies in which an AV crashed
    for each copy: danger and cell from its features
        barrier variant:  danger = how far inside the barrier the closest moving AV is;  cell = (margin bucket, AV in a junction, red-runner, pedestrian)
        mistakes variant: danger = a human mistake in progress (red-runner, distracted driver, stalled car, pedestrian crossing) + AV in a junction + closeness
    file the most dangerous copies in the archive (new cell, or danger 5 % above the stored one)
baseline: plain simulation of 64 copies for 40 windows (the same compute)
```
`avlab/tail_explore.py`, 3 runs of each in two worlds, about 18 minutes.

| world (2.1 area-hours per run, 3 runs) | plain: crash windows | go-explore, barrier | go-explore, mistakes |
|---|---|---|---|
| humans x5, rule-based AVs | 1, 1, 2 (4) | 0, 0, 0 (**0**) | 1, 0, 2 (3) |
| humans x1, every AV under the barrier function | 0, 1, 1 (2) | 0, 0, 0 (**0**) | 0, 1, 1 (2) |

**Result: it does not work as built.** Restarting from near-miss states found no more crashes than plain simulation (barrier variant: none at all;
mistakes variant: the same as plain), and it filled only 2-5 archive cells. The same reason as the estimator: how close an AV is to its barrier,
or whether a mistake is in progress somewhere in the city, predicts the next 3 seconds of one AV's life only weakly (AUC 0.6), so concentrating
copies on those states concentrates nothing. Fork-and-explore needs a *local* trigger (this driver, this junction, this red-runner) rather than a
whole-city score.

---

## What the five add up to

* **What worked:** fitting the everyday driver to logs (repairs the statistics), MAP-Elites over scenario knobs (a broader catalogue than random
  search), novelty mining (three times richer data). Counterfactuals on real scenes (2) give believable starting points and a handful of new failures, limited by how
  well real logs import.
* **What did not:** go-explore from near-misses, and, from earlier work, importance splitting and the learned-guide particle estimator
  ([LONG_TAIL.md](LONG_TAIL.md)). All three tried to use a *whole-city* score to steer simulation; that score is a weak predictor because crashes here follow sudden,
  local human events.
* **The pattern.** Methods that *design or filter* scenes (3, 4, 2) worked better than methods that *steer simulation toward danger* (5).
  The simulator's rare events are triggered by local mistakes, not by trends the city-level state shows in advance.
* **What would help most:** a following model fitted to logs with a fat tail (method 1's remaining gap), local triggers for the explorers, and
  re-scoring every catalogue entry on many more copies before it is trusted.

### Honest limits

* Small counts throughout: tens of crashes behind each result. The enrichment in method 4 is statistically clear (13 against 4 expected); the
  differences in methods 3 and 5 are small and I would not rank them on this evidence.
* Everything is Oakland, one simulator, one rule-based AV and one learned policy.
* Method 1 matches the logs on two statistics of four; the real tail has more than the knobs can make.
* None of this has been compared with a real log of a real rare event. It is a tool for producing *candidate* hard scenes, not evidence about
  real-world safety.
