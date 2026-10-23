# Metrics: bars the rule-based AV can fail, and numbers to optimise

AVBench used to ask one kind of question: is the learned policy at least as good as the rule-based AV on the same traffic? That makes
the rule-based AV pass all 88 checks by construction, so the benchmark could never say where *it* falls short, and a policy that
matched it had nowhere further to go. This page adds two things (`avrl/scorecard.py`, `avrl/bench.py`):

1. **An absolute tier**: eight pass/fail checks with fixed bars that come from outside the simulator, over the 11 core scenarios plus
   7 long-tail ones. The rule-based AV fails these too.
2. **A scorecard**: continuous metrics, some the field's usual ones and some only this simulator can compute, plus one number to
   optimise, the *av_world score*.

The relative tier stays as it was, since "better than the hand-written driver on identical traffic" is still a useful question.

```bash
python -m avrl.bench run glare dartout ...    # long-tail scenarios (~4 min each, every policy)
python -m avrl.bench run-wm                   # the latent world-model planner v1 (3 scenarios x 2 modes, ~10 min each on CPU)
python -m avrl.bench run-wm2 nominal ...      # planner v2, both modes from one warm state (~3 min each on an L4)
python -m avrl.bench report                   # -> results/bench/leaderboard.md: absolute tier, scorecard, relative tier
```
```python
from avrl import bench, scorecard
bench.evaluate("results/unaided/phase3/policy.npz", ["nominal", "glare"])["av_score"]   # an objective for model selection
scorecard.scorecard(raw=m_off, deployed=m_envelope, cbf=m_cbf)                          # from any run's summed counters
```

## The absolute tier

| check | bar | where the bar comes from |
|---|---|---|
| safer than people | AV crashes per AV-hour (with the envelope) at most **half** the humans' rate in the same run | operators report 55-90 % fewer crashes than human benchmarks in the same cities (Waymo: 55 % fewer police-reported crashes over its first 7 million rider-only miles, 82 % fewer injury crashes by late 2025) |
| comfortable | at least 95 % of 20 s windows free of braking harder than -4 m/s² | nuPlan's comfort bound on longitudinal deceleration, -4.05 m/s², scored per ~20 s scenario |
| keeps up | AV mean speed at least the humans' | not holding up traffic |
| light touch | the safety envelope cuts the policy's command in at most 2 % of AV time | the policy, not the filter, should be driving |
| safer unaided | with no filter, at most half the humans' crash rate | as above, for what the policy learned by itself |
| lawful | no red-light runs at all | zero tolerance, as in every driving test |
| right of way | no junction entries ahead of a car with priority | a right-of-way violation is an infraction, not a cost |
| inside the barrier | unaided, at most 0.2 % of moving time outside the control barrier function's safe set | the barrier encodes braking distance; leaving its safe set means a crash is possible if the car ahead brakes |

The long-tail scenarios are the famous-incident archetypes of `avlab/famous.py` at intensity 1, on top of nominal Oakland traffic: sun
glare (sensor blackouts), phantom braking, a stalled car the sensor drops, a jaywalker the perception misses, a pedestrian stepping out
from behind a bus, aggressive cut-ins, black ice.

## The scorecard

**Popular metrics**, as the field uses them:

| metric | what it is | from |
|---|---|---|
| `crash_per_h`, `crash_per_1000km` | AV crashes per AV-hour and per 1,000 km, with the envelope (`crash_per_h_unaided` without) | every closed-loop benchmark |
| `crash_ratio_vs_human` | AV crash rate over the humans' in the same run | operators' safety reports |
| `progress` | AV mean speed over the humans' | nuPlan's ego progress, CARLA's route completion |
| `comfort_pass` | share of 20 s windows with no braking harder than -4 m/s² | nuPlan comfort |
| `hard_brakes_per_h`, `jerk_rms` | comfort rates | common |
| `red_runs_per_100km` | traffic-light violations | CARLA infraction |
| `near_miss_per_h` | entering a junction just ahead of a car with priority (the simulator's time-to-collision proxy) | TTC-style surrogate safety |
| `driving_score` | CARLA Leaderboard 2.0 style: 100 x progress x the product of penalty^count per 10 km (pedestrian or cyclist 0.5, vehicle 0.6, red light 0.7, right of way 0.8) | CARLA |

**Metrics unique to av_world**, which follow from how it works:

| metric | what it is | why only here |
|---|---|---|
| `harm_per_h` (and `_unaided`) | crashes weighted by who gets hurt: pedestrian and cyclist x10, angle and left-turn x3, the rest x1 | every crash is typed; the weights are an assumption (`scorecard.HARM`) |
| `crash_ucb95_per_h` | the 95 % upper confidence bound on the crash rate given the AV-hours driven: what was *demonstrated* | exposure is counted exactly; zero crashes in 10 AV-hours only demonstrates < 0.3 per hour |
| `filter_share` | share of AV time the safety envelope cut the command | the filter's interventions are counted |
| `filter_dependence` | crashes per hour unaided minus with the envelope | the same traffic can be driven with and without the filter |
| `barrier_exit_share` | unaided, share of moving time outside the control barrier function's safe set | the barrier is evaluated every step |
| `barrier_veto_s_per_h` | under the CBF, seconds per AV-hour it blocks the AV from entering a junction or merge | low means the policy already drives as the barrier would |
| `human_crash_per_h`, `human_speed_kmh` | what the AVs do to the humans around them | humans and AVs share one run |

**The av_world score** (0-100, higher is better) is the product of five factors in [0, 1]:

```
efficiency     min(progress, 1)
safety         1 / (1 + harm_per_h_unaided / 0.1)
comfort        comfort_pass
conduct        1 / (1 + (red runs + near misses) per hour unaided / 0.1)
self-reliance  1 - filter_share
```

Safety and conduct use CARLA Leaderboard 2.1's additive penalty form, 1 / (1 + rate / scale), rather than an exponential, so a policy
with some crashes still has a gradient to climb. The scales are exchange rates, and they are choices: 0.1 harm per hour halves the
safety factor. `scorecard.SCALE` changes them.

## Results

All 18 scenarios (11 core + 7 long-tail), every policy, from one L4 GPU session (`scripts/gpu_session.sh`, 2026-10-22); the full
tables are in `results/bench/leaderboard.md` and the chart in `docs/img/avbench.png`.

**Absolute tier** (8 checks x 18 scenarios = 144). No policy clears every scenario, and the rule-based AV clears none: it fails
"comfortable" in 16 of 18, and the long tail costs it most (35 of 56 there).

| policy | checks passed (of 144) | core (of 88) | long-tail (of 56) | scenarios cleared | av_world score: core | long-tail | worst long-tail |
|---|---|---|---|---|---|---|---|
| PPO + barrier teacher, then no filter (phase 2) | **123** | 77 | **46** | 4 | 50.3 | 30.5 | 2.6 |
| PPO + barrier teacher, speed + jaywalkers (phase 3) | 120 | 75 | 45 | **5** | 56.8 | **40.9** | 4.5 |
| PPO + barrier teacher (phase 1) | 115 | 73 | 42 | 3 | 42.9 | 22.1 | 2.7 |
| rule-based AV | 104 | 69 | 35 | 0 | **77.2** | 39.0 | 2.8 |
| PPO, full world, 10x envelope penalty | 104 | 66 | 38 | 0 | 49.2 | 25.9 | **5.5** |
| PPO, 234k steps (xnet3) | 76 | 48 | 28 | 0 | 6.1 | 5.4 | 0.8 |

* **The best learned policies beat the rule-based AV on the absolute tier but not on the score.** Phase 2 passes 19 more checks,
  mostly on comfort, yet the rule-based AV keeps the highest core av_world score (77.2): the score multiplies in unaided harm per
  hour, and the barrier-teacher policies still crash 1.3-1.6x as often as it does with no filter (0.10-0.12 vs 0.08 per AV-hour).
  "safer unaided" is their most failed check (12-13 of 18).
* **The long tail is where the learned policies catch up.** Phase 3, trained with jaywalkers and a progress bonus, edges out the
  rule-based AV on the long-tail score (40.9 vs 39.0). Every policy's worst long-tail scenario scores under 6: the perception
  faults (missed stalled car, misclassified jaywalker) are unsolved.
* **Retraining is noisy.** Phases 1-3 were retrained on the GPU with the same recipe (`results/unaided/gpu/`, listed as "GPU
  retrain"). They pass 107-112 checks against the originals' 115-123, and score 7-10 points lower on the core scenarios, but higher
  on the long tail for phases 1 and 2 (22.6 and 37.3 vs 22.1 and 30.5). A one-run comparison between policies within about 10
  points of av_world score, or about 10 absolute checks, is within the spread of one recipe; the originals stay the reference
  checkpoints.

### Latent world-model planners

They replan every AV every 0.2 s, so they were benchmarked on the 3 junction-heavy scenarios (nominal, red runners,
inattentive), unaided and with the envelope:

| driver | absolute checks (of 24) | av_world score | crashes /AV-h unaided | harm /AV-h unaided | red runs /100 km | progress |
|---|---|---|---|---|---|---|
| rule-based AV | 18 | **78.2** | **0.008** | **0.008** | 0 | 1.06 |
| PPO + barrier teacher, then no filter (phase 2) | **23** | 77.9 | 0.033 | 0.050 | 0 | 1.02 |
| PPO + barrier teacher, speed + jaywalkers (phase 3) | 22 | 63.8 | 0.025 | 0.099 | 0 | 1.01 |
| latent planner v2 (ensemble model, phase 3 as prior) | 17 | 46.0 | 0.039 | 0.211 | 0 | 0.95 |
| PPO, 234k steps (xnet3) | 13 | 7.9 | 0.640 | 1.092 | 0 | 1.07 |
| latent planner v1 (xnet3 as prior) | 8 | 0.8 | 0.471 | 0.829 | 4.5-5.4 | 1.05 |

* **v1 ran red lights** that its prior never runs, so its conduct factor was 0.08. Its model also predicted the stop-line gap
  worse than holding it (36 m vs 8.6 m at 2 s).
* **v2 fixed that** (no red runs, 93.0 on nominal) with 4 s plans, masked distance targets, an ensemble and a cautious-only trust
  region around the phase 3 policy. Its caution costs about 5 % speed, so it still trails that policy.
* **v3 trained the policy inside the model instead.** An actor-critic in imagination exploited the model (30-40 crashes per
  AV-hour in the real simulator); one-step improvement on counterfactual data only matched the prior. Neither has an 18-scenario
  row; the screens are in `results/wmplan/v3/screen*.json`.
* **Crash aversion**, a test of the model where the policy fails. The simulator is forked 2-4 s before each of the phase 3
  policy's crashes and the same seconds replayed with a fix. All 91 crashes reproduced exactly. Braking harder averts 75 % (81 %
  with the hazard visible, 72 % with perception faults); the world model's converged plan averts 18 % (16 % and 18 %). Both are
  told when to act. Details and GIFs: [wmplan/README.md](../wmplan/README.md).

## Limits

* **Few crashes.** About 40 AV-hours per scenario gives 0-2 crashes in most core scenarios, so per-scenario crash checks are
  noisy. `crash_ucb95_per_h` shows how little a crash-free run proves: about 0.075 per hour at 40 AV-hours, above the humans' own
  rate. Demonstrating "half the human rate" with confidence would take roughly 10x more driving per scenario.
* **The driving score saturates.** Infractions per 10 km are rare here, so CARLA-style scores sit near 99 for almost every policy. It
  is reported because it is standard, not because it separates policies.
* **The bars are borrowed.** The safety bar compares with simulated humans, whose crash rates are calibrated to published statistics
  but whose everyday driving is not realistic (see [INCIDENT_AUDIT.md](INCIDENT_AUDIT.md)). The comfort bar is nuPlan's. The harm
  weights and score scales are assumptions, stated so they can be changed.
* **The latent planners ran on 3 of 18 scenarios**, unaided and with the envelope only: v1 takes about 10 minutes per run on CPU
  and under 2 on an L4, v2 about 3 on an L4.
* **One training run per policy.** The GPU retrains show the spread of one recipe is about as large as most gaps between
  policies; ranking them properly needs several seeds each.
