# What you can test in av_world, and how

av_world tests the **decision-making of an AV in city traffic**: when to go, how fast, how close, whom to yield to, how to react when
someone else makes a mistake or the AV's own perception fails. It does not test steering or raw-sensor perception. This page lists what is
testable, the knob or scenario that exercises it, what is measured, and how to run it, then the limits.

## The AV model, in one paragraph

Every vehicle drives along a fixed track: a lane of a road, or a fixed path through a junction. It moves along that track by car following
(IDM) and changes lanes by rule (`xnet/sim.py`, step 0.2 s). An AV policy controls two numbers per AV and step: an **acceleration**
(`u0` in [-1, 1]: up to +3 m/s², down to -8 m/s²) and a **gap-acceptance scale** (`u1`: how big a gap it needs to enter a junction or change
lanes). Routes and the decision to change lanes stay rule-based. Under the policy sits a safety filter, `X.SHIELD`: `"idm"` (an envelope),
`"cbf"` (a control barrier function plus junction and merge vetoes) or `"off"`. How often the filter overrides the policy is itself measured.

## What is testable

| Behaviour | How it is exercised (`X.Behaviour` knobs) | What is measured |
|---|---|---|
| Car following, speed, comfort | ordinary traffic, `demand`, `p_aggressive`, `friction` (wet road, ice) | crashes per AV-hour, `av_hard` (hard brakes), `av_jerk2`, speed vs the rule-based AV |
| Signals and right of way | signals, all-way stops and priority at every junction; `p_red` (humans running reds) | `av_red` (the AV runs a red), `red_runs`, `rlr_crashes`, angle and left-turn crashes |
| Unprotected turns and merges | gap acceptance at junctions and zipper merges; `p_nolook` (humans who do not look) | left-turn, angle, merge crashes; `av_veto_junction/merge` (how often the filter had to stop the AV) |
| Pedestrians and cyclists | `ped_scale`, `p_ped_violate` (jaywalkers), `p_dart` (steps out from behind a bus or van), `p_bike`, `p_close_pass` | pedestrian and bicycle crashes, `tail_masked_ped`, impact speed `imp_ped_speed` |
| Blockages | `breakdown_rate`, `p_van` (double-parking), `p_bus` (stops) | `tail_stalled`, `tail_pileup`, rear-end closing speed |
| The AV's own perception failing | `perception_noise` (gap noise), `p_blind` (leader and stop line hidden for 0.6-2 s: glare, misclassification), `p_ghost` (a phantom obstacle) | `tail_blind`, `tail_phantom`, `blackouts`, `ghosts`, `av_hard` (phantom braking) |
| Humans who target the AV | `adv_dist` (distracted behind an AV), `adv_brk` (breakdown in front of one), `adv_cut` (squeezing in front), `adv_budget` (same number of mistakes, aimed) | AV crash rate, `tail_cutin`, `tail_squeeze` |
| Everyday human texture | `p_tailgate`, `T_med`/`T_sigma` (headways), `accel_noise`, brake pulses | surrogate safety measures (below) |
| Being a good neighbour | mixed traffic at any `p_av` | human speed around AVs, near misses `av_near` |

Every crash is typed (rear-end, angle, left-turn, sideswipe, merge, pedestrian, bicycle), with an AV-involved count for each, and tagged
by cause where one is known (stalled car, pile-up, distracted driver, masked pedestrian, phantom, blind, cut-in, squeeze).

### Ready-made test suites

* **AVBench** (`avrl/bench.py`): 11 scenarios (nominal, rush hour, red runners, inattentive drivers, jaywalkers, blockages, AV majority,
  cyclists, wet road, Squirrel Hill, Downtown) x 8 checks = 88 pass/fail checks. Each check compares the policy with the rule-based AV on
  the same traffic and random numbers: safe, efficient (at least 97 % of its speed), smooth, a good neighbour, all with the filter on; and
  safe, lawful, courteous and self-reliant (the filter intervenes in under 5 % of the time) with the filter off.
  `python -m avrl.bench run`, then `report`.
* **Famous incidents** (`avlab/famous.py`): ten archetypes of publicised AV incidents, each a knob setting with an intensity dial:
  misclassified jaywalker, pedestrian from behind a bus, missed stalled vehicle, phantom braking, sun-glare blackout, aggressive cut-in,
  red-light runner, unprotected left, cyclist close pass, black ice. `python -m avlab.famous evaluate`. See [FAMOUS_EVENTS.md](FAMOUS_EVENTS.md).
* **Surrogate safety measures** on any scene, simulated or a real log (`avlab/incidents.py`, `scene_metrics`): hard brakes, time headway
  under 0.8 s, time-to-collision under 2 s and 1 s, close pedestrian passes. These are what compare the simulator with real Argoverse 2 traffic.
* **Searching for failures** (`av-world tail ...`, [LONG_TAIL_METHODS.md](LONG_TAIL_METHODS.md)): drivers fitted to real logs,
  counterfactuals of real Argoverse 2 scenes (one neighbour brakes, stops, tailgates or a pedestrian steps out), MAP-Elites over 15 knobs,
  novelty mining, and rare-event splitting.
* **Detection quality, offline** (`xnet/perception.py`): a sensor model over recorded scenes (occlusion, range, fog, noise, class confusion,
  dropouts, ghosts, latency) that scores a detector's output against ground truth with tags for occluded, vulnerable and emerging road users.

## Plugging in the AV under test

| Interface | What your stack does | Use |
|---|---|---|
| Python policy | `f(obs, key) -> (u[N, 2], aux)` on a 28-number observation per AV, compiled into the simulator | RL, fast sweeps: `av_world.make(...)`, `env.step(state, policy=f)` |
| Software / hardware in the loop | a separate process over TCP (JSON lines): receives the 28 observations plus decoded speed, lead gap, stop gap, limit and signal; sends `[u0, u1]` | your own controller in any language; lockstep, or real time with a deadline and a fallback (`av-world sil`, [SIL_HIL.md](SIL_HIL.md)) |
| Export | none: the scene goes to another tool | CARLA (`export-carla`), NVIDIA Omniverse and world models (`export-nvidia`, [NVIDIA.md](NVIDIA.md)) for sensor-level testing of the same long-tail scene |

## What it cannot test

* **Steering and lateral control.** Paths are fixed; there is no swerving, lane changes are instantaneous, and the controller cannot choose
  them or the route.
* **Perception from raw sensors in the loop.** The AV sees exact gaps plus noise, blackouts and ghosts. Camera and lidar perception have to
  be tested by exporting the scene to a sensor simulator. Weather is only road friction (and fog in the offline sensor model).
* **Highways and fast roads.** The networks are urban, up to 40 mph, at most two lanes each way.
* **Crash severity and what follows a crash.** Impacts are slower than real ones (pedestrian impacts average 24 mph against 43 mph in the
  Tempe case), and there are no post-crash dynamics.
* **Realism of everyday driving.** The simulated drivers predict real Argoverse 2 vehicles about 2x worse than constant velocity at 5 s,
  and everyday near-misses are 4-6x thinner than in real traffic (40x for tight following). Fitting drivers to real logs closes part of
  this ([INCIDENT_AUDIT.md](INCIDENT_AUDIT.md), [DATA_IMPORT.md](DATA_IMPORT.md)).
* **Real-world safety claims.** Results rank policies and find candidate hard scenes. They are not evidence of how safe an AV is on real
  roads, and the simulator has a residual crash rate of its own.
* **Hard real time.** The SIL link adds about 0.3 ms per exchange over JSON/TCP and has not been run against physical hardware.
