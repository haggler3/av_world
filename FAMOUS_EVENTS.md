# Famous long-tail events: how hard they are, and whether they help training

Ten scenario families, each the simulator's version of a recognisable real-world failure pattern, drawn at random
intensities. They answer three questions, in order: does each scenario actually produce its event, how much harder is it than
ordinary traffic, and does putting it in the training mix make a better driver. The answer to the last one is **no, not at this
scale**; the answer to the second is that they are where the remaining failures are.

These are mechanisms, not reconstructions. The simulator does not know the vehicles, roads or software of any real incident; each
scenario turns on the knobs that carry the mechanism the public reports describe. Code: `avlab/famous.py` (scenarios, difficulty,
evaluation), `avrl/famous_train.py` (the three training runs); tests: `tests/test_famous.py`.

## The scenarios

| scenario | what happens in the simulator | real-world pattern it echoes | knobs (intensity u) | counter that shows it |
|---|---|---|---|---|
| jaywalker_misclassified | a pedestrian crosses outside the signal phase and the AV's perception drops out while it approaches | Tempe, 2018: a pedestrian crossing mid-block at night; the NTSB found the system repeatedly reclassified her | p_ped_violate=0.5, ped_scale=4, p_blind=0.0004 | `pedestrian` |
| bus_dartout | a pedestrian steps out from in front of a stopped bus or van where cars cannot see | the child-darts-out-from-behind-a-bus pattern in school-zone safety guidance | p_dart=0.5, p_bus=0.13, p_van=0.2, ped_scale=3 | `tail_masked_ped` |
| stalled_vehicle_missed | a vehicle is stopped in the lane and the AV's sensor drops it from view while closing | NHTSA's 2021 investigation of driver-assist crashes into stopped first-responder vehicles | breakdown_rate=0.0001, p_blind=0.0003 | `tail_stalled` |
| phantom_braking | the AV brakes hard for an obstacle that is not there while traffic follows | NHTSA's 2022 evaluation of phantom-braking complaints | p_ghost=0.003 | `av_hard` |
| sun_glare_blackout | the AV's sensors are blinded for a second or two at a time | low-sun camera blinding, a standard perception test case | p_blind=0.001 | `tail_blind` |
| aggressive_cut_in | drivers squeeze into a gap far smaller than normal right in front of the AV | short-headway cut-ins, the commonest complaint of cautious AVs | adv_cut=4, p_aggressive=0.6 | `tail_cutin` |
| red_light_runner | many drivers run the red and enter the junction at speed | the T-bone at a signalised junction, the largest single class of urban crashes | p_red=0.55, red_window=3, p_nolook=0.001 | `rlr_crashes` |
| unprotected_left | drivers turn across oncoming traffic without looking | the left-turn-across-path crash, common to every junction type | p_nolook=0.002, p_aggressive=0.5, demand=1.2 | `left-turn` |
| cyclist_close_pass | many cyclists, and aggressive overtakers who clip them | the close-pass pattern behind most urban cyclist injuries | p_bike=0.2, p_close_pass=0.02, p_aggressive=0.5 | `bicycle` |
| black_ice | the road is slippery: braking distances double | winter-weather disengagements and crashes | friction=0.35, demand=0.9 | `rear-end` |

Intensity u is drawn from [0.4, 1.0] for training and [0.8, 1.2] for testing, so the strongest test draws are beyond anything the
policies trained on. Test draws also use random seeds the training never saw. The AV share is 25 % and humans are x1 or x5
riskier than the calibrated rates.

## 1. Does each scenario produce its event, and how hard is it?

Eight copies of Oakland, two intensity draws per scenario, 250 s each after a 100 s warm-up (about 80 AV-hours per cell). Left:
the rule-based AV under the IDM envelope. Right: the 234k-step PPO policy with no safety filter (the starting point of every run
below). "x nominal x5" divides by the start policy's rate in the ordinary x5 world.

| scenario | signature counter | rule-based AV: crashes / AV-h | start policy, no filter: crashes / AV-h | x nominal x5 |
|---|---|---|---|---|
| jaywalker_misclassified | pedestrian = 104 | 0.353 (31) | 1.652 (136) | 3.0 |
| bus_dartout | tail_masked_ped = 5 | 0.036 (2) | 1.179 (63) | 2.2 |
| stalled_vehicle_missed | tail_stalled = 12 | 0.197 (16) | 0.571 (46) | 1.0 |
| phantom_braking | av_hard = 260 | 0.000 (0) | 0.496 (40) | 0.9 |
| sun_glare_blackout | tail_blind = 13 | 0.545 (44) | 0.983 (78) | 1.8 |
| aggressive_cut_in | tail_cutin = 7 | 0.000 (0) | 0.595 (48) | 1.1 |
| red_light_runner | rlr_crashes = 10 | 0.024 (2) | 0.519 (42) | 1.0 |
| unprotected_left | left-turn = 6 | 0.023 (2) | 0.277 (24) | 0.5 |
| cyclist_close_pass | bicycle = 13 | 0.015 (1) | 0.262 (17) | 0.5 |
| black_ice | rear-end = 74 | 0.547 (36) | 1.755 (109) | 3.2 |
| nominal x1 | - = 0 | 0.000 (0) | 0.495 (40) | 0.9 |
| nominal x5 | - = 0 | 0.000 (0) | 0.546 (44) | 1.0 |

* Every scenario produces its signature event (the counter column counts all vehicles, not only AVs). Dart-outs, stalled-car
  rear-ends and sensor-blackout crashes appear only because of the knob; red-runner and left-turn crashes also occur at
  calibrated rates.
* **Hardest for the start policy:** black ice (3.2x), the misclassified jaywalker (3.0x), the dart-out (2.2x), sun glare (1.8x).
  The rule-based AV, which is bound to the IDM envelope, is nearly immune except to the jaywalker, glare and ice.
* **Not harder than ordinary traffic:** stalled vehicle, phantom braking, cut-in, red-light runner, unprotected left and
  cyclist close pass (0.5-1.1x). They are real patterns, but this policy is not more fragile to them than to everything else, and
  the left-turn and cyclist scenarios are easier (lighter traffic mix). A scenario that does not raise the crash rate
  carries no training signal for crashes.

## 2. Does training on them help?

Three PPO runs from the start policy, same compute (100 updates, 16 copies, 1,040 simulator steps each), no safety filter, the
barrier function as the teacher (the setup of `avrl/unaided.py`), two seeds each:

* **A nominal**: every copy is the ladder's randomised world.
* **B famous**: half the copies are that world, half are famous-event draws, redrawn every 10 updates.
* **C stress**: every copy has everything bad at once (humans x15, stalled cars, dart-outs, sensor faults, tight cut-ins), the
  control with the stress but without the named structure.

Scored on test draws of all ten scenarios and both nominal worlds, with no filter and a policy-independent set of random numbers
(S is the start policy; two seeds per variant: A, A1 and so on). Crashes per AV-hour, crash count in brackets.

| held-out scenario | S | A | B | C | A1 | B1 | C1 |
|---|---|---|---|---|---|---|---|
| jaywalker_misclassified | 1.482 (135) | 0.305 (29) | 0.306 (29) | 0.261 (25) | 0.390 (37) | 0.484 (46) | 0.282 (27) |
| bus_dartout | 0.832 (55) | 0.102 (7) | 0.117 (8) | 0.086 (6) | 0.072 (5) | 0.087 (6) | 0.101 (7) |
| stalled_vehicle_missed | 0.636 (56) | 0.098 (9) | 0.076 (7) | 0.087 (8) | 0.120 (11) | 0.208 (19) | 0.164 (15) |
| phantom_braking | 0.385 (34) | 0.011 (1) | 0.000 (0) | 0.000 (0) | 0.011 (1) | 0.077 (7) | 0.033 (3) |
| sun_glare_blackout | 0.796 (70) | 0.307 (28) | 0.174 (16) | 0.240 (22) | 0.286 (26) | 0.507 (46) | 0.220 (20) |
| aggressive_cut_in | 0.477 (42) | 0.000 (0) | 0.022 (2) | 0.022 (2) | 0.022 (2) | 0.087 (8) | 0.022 (2) |
| red_light_runner | 0.457 (40) | 0.044 (4) | 0.066 (6) | 0.066 (6) | 0.055 (5) | 0.143 (13) | 0.066 (6) |
| unprotected_left | 0.381 (36) | 0.021 (2) | 0.000 (0) | 0.020 (2) | 0.021 (2) | 0.082 (8) | 0.021 (2) |
| cyclist_close_pass | 0.246 (18) | 0.013 (1) | 0.026 (2) | 0.000 (0) | 0.000 (0) | 0.079 (6) | 0.026 (2) |
| black_ice | 0.701 (50) | 0.081 (6) | 0.121 (9) | 0.120 (9) | 0.094 (7) | 0.108 (8) | 0.095 (7) |
| nominal x1 | 0.408 (36) | 0.022 (2) | 0.022 (2) | 0.000 (0) | 0.022 (2) | 0.088 (8) | 0.044 (4) |
| nominal x5 | 0.385 (34) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.066 (6) | 0.022 (2) |
| **all conditions** | 0.598 | 0.084 | 0.077 | 0.076 | 0.093 | 0.172 | 0.092 |
| speed, nominal x5 (km/h) | 20.8 | 19.0 | 19.0 | 18.6 | 19.1 | 19.1 | 19.1 |


| | S | A | B | C |
|---|---|---|---|---|
| all conditions, seed 0 / seed 1 | 0.598 | 0.084 / 0.093 | 0.077 / **0.172** | 0.076 / 0.092 |

* Every trained variant cuts crashes by about 7x against the start policy (0.60 to 0.08-0.17). Almost all of that is the barrier
  teacher in 100 updates, whatever the mix.
* **The famous mix did not beat nominal training.** Seed 0: 0.077 vs 0.084 (no difference). Seed 1: 0.172 vs 0.093, with the
  mix clearly worse, and it was worse in 11 of 12 conditions, including the ordinary ones. The stress control sits with
  nominal in both seeds. With two seeds the difference between seeds (B: 0.077 vs 0.172) is as large as any difference between
  variants, so the safest reading is "no benefit shown; the mix may make training less stable", the same pattern as the
  mid-training instability of the mutating-cities run.
* Speed is unchanged (18.6-19.1 km/h in the x5 world for every variant).

## 3. What they are good for

* **Testing.** On ordinary x5 traffic the trained policies (A, C, A1) crash 0-0.02 times per AV-hour, which would pass most
  nominal checks. The same policies crash 0.26-0.39 per AV-hour in the jaywalker scenario and 0.22-0.31 under sensor glare:
  failures the ordinary test hides, found with ~80 AV-hours of simulation per cell. That, rather than training data, is where
  these scenarios earn their keep.
* **Ranking policies.** The scenarios separate the start policy from every trained one by 4-30x, and the dart-out, jaywalker
  and glare cells are where trained policies differ most.
* **Not shown: training on them.** If they should help, it would take more than 100 updates, more seeds, a curriculum that
  raises intensity as the policy improves, or scenarios that are actually hard for the policy being trained (the six
  scenarios above that did not raise the start policy's crash rate dilute the mix).

## Caveats

* Two seeds per variant, about 80 AV-hours per cell: per-cell differences under about 2x are noise, and the totals are good to
  roughly +-0.02.
* The test draws share a map (Oakland) and a simulator with the training worlds. Nothing here says how a real fleet would behave.
* Scenario strengths are chosen to produce the event, not calibrated to the real frequency of these events.

Reproduce: `python -m avlab.famous difficulty` (8 min), `python -m avrl.famous_train A|B|C [seed]` (3 min each),
`python -m avlab.famous evaluate S A B C A1 B1 C1`, `python -m avlab.famous report`.
