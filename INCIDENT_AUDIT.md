# Does the long tail that matters come out of the simulator?

A critical look, in two parts, because "the simulator produces rare crashes" and "the simulator produces the rare events that matter" are
different claims. Code: `avlab/incidents.py` (`real`, `sim`, `audit`); data: `results/incidents/*.json`; tests: `tests/test_incidents.py`.

**Short answer.** The simulator produces *mechanisms* of failure on demand (a pedestrian stepping from behind a bus, a sensor blackout, a
stalled car, a pile-up). It does not reproduce the *famous incidents* (the context that made them dangerous is mostly absent: speed, darkness,
road type, vehicle types), and it badly under-produces the continuous tail of ordinary driving that real logs are full of (tailgating, hard
braking, short time-to-collision): about 4-6 times less than real Argoverse 2 traffic on most measures, and 40 times less on tight following. Treat it as a generator of discrete failure mechanisms for
testing policies, not as a source of realistic long-tail data.

## Part A. The ordinary tail: simulator against real logs

The same function (`scene_metrics`) runs on 400 real Argoverse 2 scenarios (11 s each at 0.2 s after thinning, 9.3 moving
vehicle-hours) and on recorded simulator scenes of Oakland (4 runs x 90 s per row, about 20 moving vehicle-hours). Events: **hard brake**, a speed
drop of more than 4 m/s within a second from above 3 m/s; **tight following**, a time headway under 0.8 s to the vehicle ahead in the same lane
(moving above 3 m/s); **time to collision** under 2 s and under 1 s with the vehicle ahead; **pedestrian close**, seconds per hour spent moving
above 3 m/s within 2.5 m of a pedestrian.

| | hard brakes / moving veh-h | tight following | TTC < 2 s | TTC < 1 s | pedestrian close, s/h | moving veh-h |
|---|---|---|---|---|---|---|
| **real, Argoverse 2 (400 scenes)** | **43.2** | **1.85 %** | **1.05 %** | **0.35 %** | **1.73** | 9 |
| nominal x1 | 11.6 | 0.04 % | 0.19 % | 0.01 % | 0.35 | 20 |
| nominal x5 | 11.4 | 0.04 % | 0.19 % | 0.01 % | 0.35 | 20 |
| jaywalker_misclassified | 6.7 | 0.04 % | 0.24 % | 0.01 % | 1.94 | 15 |
| bus_dartout | 8.1 | 0.06 % | 0.23 % | 0.00 % | 1.12 | 17 |
| stalled_vehicle_missed | 12.1 | 0.03 % | 0.19 % | 0.01 % | 0.34 | 19 |
| phantom_braking | 12.7 | 0.04 % | 0.19 % | 0.01 % | 0.38 | 20 |
| sun_glare_blackout | 11.6 | 0.04 % | 0.19 % | 0.01 % | 0.35 | 20 |
| aggressive_cut_in | 11.3 | 0.04 % | 0.19 % | 0.01 % | 0.31 | 20 |
| red_light_runner | 12.1 | 0.04 % | 0.19 % | 0.00 % | 0.34 | 20 |
| unprotected_left | 12.1 | 0.05 % | 0.21 % | 0.01 % | 0.32 | 21 |
| cyclist_close_pass | 9.8 | 0.08 % | 0.14 % | 0.01 % | 0.30 | 20 |
| black_ice | 2.9 | 0.04 % | 0.17 % | 0.01 % | 0.51 | 16 |

What this shows:

* **The continuous tail is missing.** Nominal simulator traffic has a quarter of the real hard-braking rate, about a fifth of the real
  short-TTC share, and almost no tight following (0.04 % against 1.85 %). The drivers are IDM followers with a headway of 1.0-1.8 s, so they
  never follow closer than 0.8 s at speed; real drivers do, for 1.9 % of their moving time.
* **The human-risk dial does not touch it.** Humans x1 and x5 riskier give the same numbers: the risk knobs change rare discrete mistakes (a
  red-runner, a driver who does not look), not the texture of everyday following.
* **The tail scenarios mostly do not touch it either**, except where they should: the jaywalker and dart-out scenarios raise the pedestrian
  close-pass time to 1.94 and 1.12 s/h (real: 1.73), which brackets the real logs; black ice lowers hard braking (slow
  driving). The cut-in scenario changes the number of crashes (see FAMOUS_EVENTS.md) but not these near-miss statistics, probably because the
  cut-ins are rare among all the following that happens and the followers re-establish their gap afterwards (I did not isolate the cause).
* **A tailgating knob moves it part of the way.** `Behaviour.p_tailgate` makes a share of humans keep a 0.35-0.7 s headway. At 50 % the TTC < 2 s
  share goes from 0.19 % to 0.69 % (real 1.05 %) but tight following only to 0.12 %: the IDM's equilibrium gap includes a minimum gap and a
  speed-dependent term that headway alone does not remove. Reaching the real distribution needs a different following model, fitted to logs.
* **Measurement caveats, in both directions.** Argoverse 2 speeds are estimated from tracks (noise inflates hard braking and TTC events); the
  logs come from six US cities in dense traffic, the simulator from one neighbourhood in Pittsburgh; 400 short scenes are 9 vehicle-hours.
  The direction of the gaps is solid; their size is uncertain by a factor of two or so.

## Part B. The famous incidents, feature by feature

For each incident: the features that defined it, whether the simulator can represent each (**present**, **approximate**, **absent**), and, where
there is a scenario, the mechanism measured (32 copies of Oakland, 300 s each after a 100 s warm-up, 2.7 simulated area-hours; rule-based AVs
under the IDM envelope). *Fidelity* is (present + half the approximate) over the number of features: a rough count, not a measurement.

The network's fastest speed limit is 40 mph and no link has more than 2 lanes in a direction.

### Tempe, Arizona, 18 March 2018  (fidelity 0.21: 0 present, 3 approximate, 4 absent)

| feature | level | note |
|---|---|---|
| pedestrian crosses outside any crosswalk | absent | pedestrians only cross at the crosswalks of junctions |
| wide arterial at ~45 mph | approx | at most two lanes per direction; the speed limits come from the map |
| darkness | absent | no lighting model |
| object seen early but classified inconsistently, so no braking | approx | a blackout hides the pedestrian; there is no classification or prediction error |
| impact at about 43 mph | approx | measured below |
| emergency braking off, backup driver inattentive | absent | not modelled |
| an unusual object (a person pushing a loaded bicycle) | absent | pedestrians are one class |

Measured in the jaywalker scenario: 25 pedestrian impacts in 2.7 area-hours (13 by AVs); mean impact speed **24 mph** against the reported 43 mph. No impact came near the reported speed: the fastest road allows 40 mph and crossings are at junctions, where vehicles slow.

### Driver-assist crashes into stopped emergency vehicles (NHTSA investigation opened August 2021)  (fidelity 0.40: 1 present, 2 approximate, 2 absent)

| feature | level | note |
|---|---|---|
| stationary vehicle in or beside the lane | present | breakdowns and delivery stops |
| flashing lights, cones, a scene around the vehicle | absent | a stopped car is just a stopped car |
| highway speeds | absent | an urban network: free-flow speeds are the limits (measured below) |
| stationary object filtered out or missed by the sensors | approx | blackouts hide it for 0.6-2 s |
| the car ahead changes lane and reveals the stopped one | approx | lane changes exist; not set up as a scenario |

Measured in the stalled-vehicle scenario: 4 rear-ends into a stalled car in 2.7 area-hours, mean closing speed **4 mph**. Real cases happen at highway speed; in this network a stalled car is seen from a distance and passed by changing lane, and the few collisions are slow.

### San Francisco, 2 October 2023  (fidelity 0.25: 0 present, 2 approximate, 2 absent)

| feature | level | note |
|---|---|---|
| a pedestrian hit by another vehicle lands in the AV's lane | absent | no pedestrian dynamics after an impact |
| the AV brakes but cannot stop in time | approx | pile-ups into wrecks exist (tail_pileup), not with pedestrians |
| the AV moves on after the impact | absent | wrecks stay put |
| multi-lane urban street at night | approx | urban, two lanes at most, no night |

No scenario: the defining feature (a pedestrian thrown into the AV's lane by another vehicle) cannot be represented.

### Phantom braking (NHTSA preliminary evaluation PE22-002, February 2022)  (fidelity 0.62: 2 present, 1 approximate, 1 absent)

| feature | level | note |
|---|---|---|
| hard braking without an obstacle | present | p_ghost: a phantom obstacle 12 m ahead |
| a following vehicle collides | present | followers are real drivers with reaction times; measured below |
| highway speeds | absent | urban speeds |
| braking from a false detection of an overpass, shadow or sign | approx | ghosts appear at random, not at overpasses |

Measured with phantom obstacles: 2251 phantom-braking events, **0 followers rear-ended an AV that was braking for one**. The followers keep their headway; the real complaints involve highway speeds and short gaps (Part A: such gaps are rare in the simulator).

### Williston, Florida, May 2016  (fidelity 0.25: 0 present, 2 approximate, 2 absent)

| feature | level | note |
|---|---|---|
| a large vehicle turns across the path of the car | approx | cars, vans and buses turn across traffic; no tractor-trailers |
| camera fails to separate the trailer from the sky | approx | blackouts hide any object |
| under-ride of a high trailer | absent | no vehicle heights |
| divided highway at high speed | absent | urban network |

Measured in the unprotected-left scenario: 2 left-turn crashes in 2.7 area-hours, 0 with an AV. The mechanism (turning across the path) exists; the vehicle that matters (a high trailer at highway speed) does not.

## Verdict

* **Mechanisms, yes; incidents, no.** Each incident has one or two mechanisms the simulator can produce (a pedestrian appearing from behind a
  bus, a sensor dropout, a stalled car, a phantom obstacle, a turn across the path) and several it cannot (darkness, highway speed, road
  type, vehicle type, post-crash dynamics). Fidelity by count is 0.2-0.6. A policy that survives these scenarios has been tested on the
  mechanism, not on the incident.
* **Severity is not reproduced.** The simulator is an urban 25-40 mph world. The incidents that made the news were severe because of speed or
  vulnerability; the simulator's pedestrian impacts in the jaywalker scenario average 24 mph and its rear-ends into stalled cars 5 mph.
* **The simulator's tail is thin where real traffic's is fat.** Ordinary driving in real logs has several times more hard braking,
  tailgating and short-TTC episodes (Part A). Those are the raw material of most real near-misses, and a policy trained or tested only here
  has seen very few of them.
* **Where it does help:** exposing policies to events that are absent from ordinary exposure (the FAMOUS_EVENTS.md results: trained policies
  that crash about never in ordinary traffic crash 0.2-0.4 times per AV-hour in the jaywalker and glare scenarios), and giving a controllable
  rate of mechanism-level failures.

## What it would take to close the gap

1. Fit the car-following model to real logs (the Argoverse 2 importer gives the data and `scene_metrics` gives the target); a model with a
   fat tail of short headways and hard braking. `p_tailgate` is a stop-gap.
2. A highway network and higher speeds (a 65 mph road, ramps), trucks with trailers, darkness as a sensor-quality state, mid-block pedestrian
   crossings, and post-crash dynamics for pedestrians.
3. Calibrate scenario frequency to real exposure (these scenarios are tuned to produce the event, not to occur at the real rate).
4. Check each scenario against a log of the real event where one exists, rather than against a description of it.
