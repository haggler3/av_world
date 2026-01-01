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
python -m avrl.bench run-wm                   # the latent world-model planner (3 scenarios x 2 modes, ~10 min each)
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
* **The latent world-model planner (wm-mpc) is not competitive yet.** On its 3 scenarios it passes 8 of 24 checks with an av_world
  score of 0.8. With no filter it crashes 0.47 times per AV-hour, fewer than the PPO policy it uses as its proposal prior (0.64),
  so planning in the latent space helps, but it is 15-60x the barrier-teacher policies' rate. The retrained model ranks crashes well
  (held-out AUROC 0.958) but predicts the stop-line gap worse than holding it constant (2 s error 36 m vs 8.6 m) and the merge gaps
  no better (`results/wmplan/model_eval.json`), and those are the margins the planner scores plans by.
* **The latent planner v2 fixes that, but does not beat its prior.** v2 uses an ensemble model with 4 s plans and the
  phase-3 policy as prior, overriding it only towards caution ([wmplan/README.md](../wmplan/README.md)). On the same 3
  scenarios it passes 17 of 24 checks with an av_world score of 46.0 (v1: 8 and 0.8), with no red runs and 0.039 unaided crashes
  per AV-hour. Its prior passes 22 with a score of 63.8; on nominal it is 93.0 against 98.0.
* **World model v3 (counterfactual data, policy learning in the model) does not beat it either.** In paired 60 s screens the
  best variant matched the phase 3 policy's crashes at 6 % lower speed. The actor-critic trained in imagination exploited the
  model and crashed 30-40 times per AV-hour. With these 28 features, braking less causes crashes but braking more does not
  prevent them ([wmplan/README.md](../wmplan/README.md)).
* **Crash aversion**, a new metric: fork the simulator 2-4 s before each of the policy's crashes and replay it with a fix.
  Over 91 crashes of the phase 3 policy, all reproduced exactly, naive harder braking averts 75 %. The v3 world model's plan
  averts 18 %. Both are told when to act ([wmplan/README.md](../wmplan/README.md)).

## Limits

* **Few crashes.** About 40 AV-hours per scenario gives 0-2 crashes in most core scenarios, so per-scenario crash checks are
  noisy. `crash_ucb95_per_h` shows how little a crash-free run proves: about 0.075 per hour at 40 AV-hours, above the humans' own
  rate. Demonstrating "half the human rate" with confidence would take roughly 10x more driving per scenario.
* **The driving score saturates.** Infractions per 10 km are rare here, so CARLA-style scores sit near 99 for almost every policy. It
  is reported because it is standard, not because it separates policies.
* **The bars are borrowed.** The safety bar compares with simulated humans, whose crash rates are calibrated to published statistics
  but whose everyday driving is not realistic (see [INCIDENT_AUDIT.md](INCIDENT_AUDIT.md)). The comfort bar is nuPlan's. The harm
  weights and score scales are assumptions, stated so they can be changed.
* **The latent planner ran on 3 of 18 scenarios** (about 10 minutes per run on CPU, under 2 on an L4, with every AV planned), unaided
  and with the envelope only.
* **One training run per policy.** The GPU retrains show the spread of one recipe is about as large as most gaps between
  policies; ranking them properly needs several seeds each.
