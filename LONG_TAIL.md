# Long tails, other cities, CARLA, rare events

Overview of the long-tail tooling; the detailed write-ups are linked. New in this round: [LONG_TAIL_METHODS.md](LONG_TAIL_METHODS.md) (five ways to tease novel long-tail data out of the simulator, with pseudocode).

## Long tails, other cities, CARLA, rare events

`pip install .` (or `pip install git+https://github.com/haggler3/av_world`) installs everything with the Pittsburgh areas and the
start policy bundled; `av-world --help` lists the commands. `import av_world; env = av_world.make("oakland", n_envs=8)` gives a
functional environment (`reset`, `step(state, policy)`, `rollout`).

**Any city.** `av-world import-osm map.osm city.npz` turns an OpenStreetMap extract into the road network (roads, one-ways,
speed limits; lanes, signals and demand inferred as for Pittsburgh) and `av_world.make("map.osm")` runs on it
(`xnet/osm.py`, tested on a generated street grid; I could not fetch a real extract from the build container).

**Into CARLA.** `av-world export-carla scene --seconds 60` writes `scene.xodr` (OpenDRIVE: every link a road, every move a
connecting road), `scene.json` (every vehicle's trajectory, AVs and crashes flagged, pedestrian crossings) and
`scene_replay.py`, which plays the scene in CARLA with physics off. Not run against a real CARLA (none was available); the
files are checked structurally, the Oakland map parses in pyxodr, and the replay runs against a stand-in API
(`tests/test_carla_export.py`). No traffic lights, signs or elevation are written.

**Long-tail knobs** (all off by default, results unchanged when off): AV sensor blackouts and phantom obstacles
(`p_blind`, `p_ghost`), pedestrians stepping out from behind a stopped bus or van (`p_dart`), aimed human mistakes
(`adv_dist`, `adv_brk`, `adv_cut`), and cause tags on crashes (`xnet.sim.TAILS`). `xnet/perception.py` adds a sensor model on
recorded scenes (occlusion by other vehicles, range and fog, dropouts, ghosts, latency, and tags for pedestrians and cyclists
that were hidden and reappeared) for scoring a perception stack on the hard cases.

Per 1,000 simulated hours of Oakland (rule-based AVs, 5.3 area-hours per column, so small counts are noisy):

| event | calibrated humans | humans x5 | + AV sensor faults | + aimed cut-ins | stress |
|---|---|---|---|---|---|
| AV crashes | 750 | 1,880 | 15,000 | 2,800 | 20,000 |
| rear-end into a stalled car | 0 | 0 | 380 | 380 | 560 |
| pile-up into a wreck | 190 | 0 | 0 | 0 | 940 |
| pedestrian struck stepping out from behind a bus/van | 0 | 0 | 0 | 0 | 5,600 |
| AV crash during a sensor blackout | 0 | 0 | 9,800 | 0 | 11,600 |
| rear-end after a cut-in | 1,300 | 0 | 380 | 560 | 0 |
| phantom braking rear-ended | 0 | 0 | 0 | 0 | 0 |

(stress: humans x15, breakdowns x50, more vans and buses, dart-outs, sensor faults). What arises on its own at calibrated
rates: red-runner and left-turn crashes, cut-in rear-ends, an occasional pile-up. What needs a knob: stalled-car rear-ends,
dart-outs, sensor-blackout crashes. Phantom braking never gets an AV rear-ended here (2,300 ghosts, no crash): the following
drivers keep their headway. Not representable: wrong-way drivers, emergency vehicles, debris, animals.

**Real logs, intersections, and an honest audit.** [docs/DATA_IMPORT.md](DATA_IMPORT.md): Argoverse 2 scenarios (tested on 150 real ones, public
bucket) and Waymo Open Motion scenarios (adapter tested on the documented layout only) become scenes, road networks and simulator states
(`av-world fetch-av2`, `import-av2`, `validate-av2`); the simulator's drivers predict the logged vehicles about twice as badly as constant
velocity (17 m against 9 m at 5 s), so they are not calibrated to real traffic. [docs/JUNCTIONS.md](JUNCTIONS.md): the intersection
detector for imported maps (roundabouts, divided roads, overpasses, signal tags; 110,000 nodes in 2.3 s) with pseudocode and its limits.
[docs/INCIDENT_AUDIT.md](INCIDENT_AUDIT.md): whether the long tail that matters comes out of the simulator. It produces failure
*mechanisms* on demand, not the famous incidents (fidelity 0.2-0.6 by feature count; pedestrian impacts 24 mph against Tempe's 43), and its
everyday near-miss tail is 4-6x thinner than real Argoverse 2 traffic.

**Famous events as scenarios** ([docs/FAMOUS_EVENTS.md](FAMOUS_EVENTS.md)): ten mechanisms from public reports (a misclassified
jaywalker, dart-out from behind a bus, a stalled vehicle the sensor misses, phantom braking, sun glare, tight cut-ins, red-light
runners, unprotected lefts, close passes, black ice). They are where trained policies still fail (0.25-0.4 crashes per AV-hour in
the jaywalker and glare scenarios against ~0 in ordinary traffic), but mixing them into training did not help: two seeds, 0.077/0.172
for the mix against 0.084/0.093 for nominal training.

**Adversarial humans** (`avlab/adversary.py`, optional). Evolution strategies over three multipliers, scored on the AV crash
rate of the unfiltered start policy (common random numbers), with caps (mistakes aimed up to 1000x, a cut-in accepting a gap
down to a quarter). Held-out seeds: 0.424 -> 0.523 crashes per AV-hour (123 -> 151 crashes, 290 AV-hours each). The cut-in
lever alone gives 0.502 and 29 cut-in rear-ends against 2; aiming distraction or breakdowns at AVs does nothing at calibrated
rates (0.435, 0.428). The same-mistakes budget (`adv_budget`) does not touch the cut-in lever, which is a change of behaviour,
not a count of mistakes. Without the cap on cut-ins the search reaches 0.66 (x1.56) by letting humans squeeze in with
almost no gap, which is why the cap exists.

**Rare crash probabilities: a negative result, again** (`avlab/rare.py`). The estimator is a Feynman-Kac particle system with
weights exp(lambda (V_new - V_old)) from a learned per-AV committor V; it is unbiased for any V
(`tests/test_rare.py`, a random walk with a known answer, where a good V also cuts the spread). In the city the learned V is weak
(held-out AUC 0.70 per AV, 0.62 per copy), and at equal cost the particles were no better than plain Monte Carlo:
12 replicates of 128 copies x 60 s, same warm states: means 0.0119 vs 0.0117, spread 0.0118 vs 0.0091 (variance ratio 0.59); with
every AV under the barrier function 0.0110 vs 0.0065 (ratio 0.23). The crashes here follow sudden human events (a runner appears,
a car cuts in) that 15 s of aggregate features do not predict, so "1 in 10^6 in minutes" is not reached. What would change it:
a guide with the precursor itself in its features, or conditioning on the precursor events and forking from them.
