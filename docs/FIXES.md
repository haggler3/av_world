# Fix round: every open question and weak result, turned into a fix

The last two rounds ([HYPOTHESES.md](HYPOTHESES.md), [IDEAS.md](IDEAS.md)) ended with a list of things that were underpowered,
unexplained or plainly not good enough. This round takes each one, says what was wrong, what was changed, and what happened. Two
fixes did not work and are reported as such, and two problems turned up while fixing (13, 14). The simulator core (`xnet/`) is unchanged. One opt-in hook was added to the PPO trainer
(`avrl/xppo.make_fns(bonus=...)`); it is off by default, so every existing run is identical.

| # | problem | fix | result |
|---|---|---|---|
| 1 | Crash rates flip direction with the exposure (H2: per hour vs per km) | `easy` summaries report crashes per 1,000 vehicle-hours, per million vehicle-km and per million junction entries | every summary and sweep now shows all three |
| 2 | H3 wet roads underpowered (could only detect 2x) | 4x the copies (192 + 192) | $RR = 1.88\ [1.47, 2.42]$, up from $1.44\ [0.87, 2.42]$. Still *undecided* by the rule fixed in advance (the lower end, 1.47, sits just under 1.5), but the data clearly favour it |
| 3 | H4 AV-share curve shape undecided (5 points) | 9 shares x 80 copies; a pre-registered "survives" rule | still undecided: curvature $0.48\ [-1.66, 2.63]$. Settling it needs ~18x the data (a GPU job). The "crashes halve at" point moved from 55 % to 33 % between two sweeps: single sweeps are noisy |
| 4 | Sweeps over different share grids silently reused random seeds | `av_share_sweep.main(seed=...)` seeds by share value | new sweeps are independent; old results still reproduce |
| 5 | H11 fog undecided (12 egos) | 60 egos over 5 runs, cached | difference $-2\ [-4, +1]$ points: fog hurts pedestrians and vehicles about equally. The claim has no support |
| 6 | Attribution: an unexplained *negative* Shapley value for distraction | per-copy counters saved; Shapley values of the crash rate per vehicle-km and of network speed | not an exposure effect (distraction leaves speed unchanged, +0.06 ± 0.11 km/h, and is negative per km too); but the mechanism cannot carry it: its ~80 episodes took part in **3** crashes, the value implies **59**. With a robust (sandwich) variance (crashes cluster) and a Holm correction it is no longer significant (p_Holm 0.08): a fluke, now caught by the tool itself |
| 7 | Green wave: crashes went 11 → 17 on validation | 64-copy validation; an offsets-only search that cannot change any green's length | the crash rise was noise: at 64 copies crashes are **29 → 28** (×0.98 [0.56, 1.70]) for **+3.1 % speed** (56 of 64 copies faster). Offsets only: **+1.1 %** (41 of 64), crashes 29 → 24 (×0.83 [0.46, 1.48]) |
| 8 | Hotspots: empirical Bayes over-corrected (−16 %); 20 splits too noisy; a "rate" ranking identical to the count by construction | an SPF with conflict-point counts, a ridge fit (no separation), maximum-likelihood dispersion, 200 paired splits, fixed rate ranking | calibration fixed: prediction bias **+2.3 ± 2.9 %** (raw counts +19.7 %, plain EB −11.6 %). Ranking: no EB variant beats the counts (best −1.4 ± 0.7 points) |
| 9 | AV Turing test: the PPO policy is recognisable (AUC 0.88) | (a) reward re-weighting (failed); (b) an adversarial human-likeness reward, a small GAIL, with a no-bonus control | (a) failed: AUC went *up* (0.915, 0.908). (b) worked: AUC **0.875** vs 0.911 for its no-bonus control, nominal score **97.4** vs 17.2, unaided crashes **0** vs 0.167; across the 11 core scenarios mean **29.2** vs 19.8 and fewer unaided crashes in **10 of 11** |
| 10 | IDM fails AVBench's comfort check (0.85) | `easy.drivers.comfortable()`: IDM that brakes only as hard as the situation needs, full IDM in emergencies | nominal score **93.1** (IDM 81.0), comfort 0.953 (passes), 0 crashes unaided. Still misses "keeps up" (0.998) and "light touch" (0.021) by hairs |
| 11 | Black box: showed the light at the *far* end of a link for a car that had just crossed a junction; said nothing about the pedestrian | the light the car came through (with its turn), and whether WALK was shown when the pedestrian started | e.g. "came through junction 217 (right) on red 2.2 s before" (a legal right on red: no red-run flag), "struck a pedestrian who started crossing without the WALK signal" |
| 12 | Notebook 02's plots showed placeholder values (no leader = 150 m, no stop line = 300 m) as data | filtered | re-executed |
| 13 | `tests/test_xnet_rl.py::test_shield_keeps_a_full_throttle_policy_safe` failed (an AV under the safety envelope entered on red) | bisected to commit `7cf1394`, traced to the dilemma zone, fixed in the envelope (`xnet/sim.py`, two lines) | the test passes; AVBench numbers unchanged where checked (rule-based nominal, red runners, rush hour; the human-like policy's nominal; phase 3's blockages and wet road) |
| 14 | Re-scoring the rule-based AV on rush hour gave 92.0, not the leaderboard's 68.0 | `examples/leaderboard_drift.py`: re-score leaderboard rows on today's code | **the leaderboard does not reproduce on a CPU**: per-scenario drift up to ±75 points, core means 77.2 → 68.9 (rule-based) and 56.8 → 58.9 (phase 3). Same code with and without fix 13: identical. The board was computed on an L4 GPU; floating-point differences flip rare events, and per-scenario scores are nearly bimodal |

---

## The fixes in pseudocode

### 6. Is distraction's negative Shapley value real, or an exposure effect?

```
per coalition S of mechanisms: keep every copy's counters (not just the sum)
three outcomes, the same exact Shapley weights:
    crashes per 1,000 vehicle-hours      v_h(S)  = k_S / t_S             Var = k_S / t_S^2
    crashes per million vehicle-km       v_km(S) = k_S / km_S            Var = k_S / km_S^2
    network mean speed                   v_sp(S) = mean over copies      Var = sample variance / copies
a mechanism that lowers v_h but not v_km, and lowers speed, is slowing traffic (fewer crashes per hour, not per km)
```

| mechanism (every mistake x10) | crashes / 1,000 veh-h (95 %, robust) | Holm p | crashes / M veh-km | network speed, km/h |
|---|---|---|---|---|
| pedestrians crossing against the light | **+13.6 ± 4.2** | < 0.001 | +843 ± 224 | **−1.68 ± 0.09** |
| aggressive drivers | +2.3 ± 4.0 | 0.54 | +111 ± 219 | +0.24 ± 0.10 |
| not looking | −1.4 ± 4.1 | 0.54 | −84 ± 223 | +0.01 ± 0.10 |
| red-light running | −3.9 ± 4.4 | 0.24 | −236 ± 224 | +0.18 ± 0.10 |
| distraction | −5.1 ± 4.3 | 0.08 | −294 ± 220 | +0.06 ± 0.11 |

The exposure explanation was wrong: distraction does not slow traffic, and its per-km value is as negative as its per-hour one. What
settles it is a plausibility budget the tool now prints. Distraction was on for 11,600 vehicle-hours; at its rate that is ~80 episodes
of 1.5-3 s of coasting, and 3 crashes involved a distracted driver. A mechanism that touched 3 crashes cannot have prevented 59. With
the copies' own spread as the variance (crash counts are overdispersed: clusters and jams) and Holm's correction over the five
mechanisms, nothing but the pedestrians survives. Two tools, not one result, were missing: `shapley` now reports robust standard
errors and Holm p-values, and `--reanalyse` recomputes everything from the saved counters.

### 7. A green wave that cannot buy speed with clearance

```
offsets-only search: the same cross-entropy method with the split dimensions frozen (sd = 0):
    every green, yellow and red lasts exactly as long as today; only when each light starts its cycle changes
validation: 64 new copies (2x), paired by seed; speed difference and an exact crash rate ratio
```

| timing | speed on new traffic (64 copies x 10 min) | copies faster | trips | crashes (today → new) | crash rate ratio [95 %] |
|---|---|---|---|---|---|
| offsets + splits | **+3.1 %** [+0.34, +0.54 km/h] | 56 / 64 | +2.2 % | 29 → 28 | 0.98 [0.56, 1.70] |
| offsets only | +1.1 % [+0.06, +0.26 km/h] | 41 / 64 | +0.6 % | 29 → 24 | 0.83 [0.46, 1.48] |

With twice the validation copies the crash difference of the first run (11 → 17) disappears: it was noise. Both timings are faster
on new traffic. The offsets-only search, safer by construction, buys a third of the speed. Which to use depends on how much one trusts
the simulator's clearance-interval behaviour; the tool now offers both (`--offsets-only`).

### 8. Hotspots: a calibrated empirical Bayes

```
site structure Z = [log(1 + angle conflicts), log(1 + left-turn conflicts), log(1 + merge conflicts), signalised, stop sign, max approach speed]
SPF:  log mu = a + b log(exposure) + c'Z, fitted by Poisson IRLS with a ridge (lam = 1) on b, c
      (unpenalised, site types that never crash drive coefficients to +-30: separation)
alpha: argmax of the negative-binomial log-likelihood given mu (1-D bounded search), not the method of moments
EB   = w mu + (1 - w) y,  w = 1 / (1 + alpha mu)
judge on 200 random half-splits, every method on the same split (paired): captured share - the count's, and the prediction bias
```

| method | top-10 hold of next-period crashes | vs the crash count (paired) | prediction bias for its top 10 |
|---|---|---|---|
| crash count | 58.7 % | | **+19.7 ± 3.1 %** (regression to the mean) |
| crash rate | 32.1 % | −26.6 ± 1.1 | +50.5 ± 6.3 % |
| EB, exposure only | 56.0 % | −2.7 ± 1.2 | −11.6 ± 3.1 % |
| EB + signalised, movements | 58.7 % | −0.0 ± 0.8 | −9.6 ± 2.7 % |
| **EB + conflict points (ridge, ML)** | 57.3 % | −1.4 ± 0.7 | **+2.3 ± 2.9 %** |

(± are 95 % intervals over the 200 splits.) The fix works for what EB is for: an unbiased estimate of each site's expected crashes, so
a treatment's "before vs after" is not inflated by regression to the mean. It does not pick a better top 10 than the raw count in
this city, where the dangerous sites are dangerous by structure, and the table says so.

### 9. Driving like people: the AV Turing test, attempted twice

```
attempt (a), reward re-weighting: fine-tune the bundled policy 40 updates with jerk, comfort -> 0 (and 3x the hard-brake penalty)
attempt (b), adversarial human-likeness (GAIL-lite), from (a)'s safer policy:
    repeat 4 rounds:
        record 4 copies x 2 min; per step: speed v, acceleration a, jerk j of humans and of the AVs
        D = logistic regression on (v, a, j, a^2, j^2, v a): human vs AV
        10 PPO updates with reward += 0.03 * clip(logit P(human | v, a, j), -3, 3) on every AV step
    control: the identical run with the bonus weight 0
evaluate both: the Turing test (trip-level, 8 features, cross-validated AUC) and AVBench
```

| policy | Turing AUC [95 %] | AVBench nominal | crashes / AV-h, no filter | share of time braking > 2 m/s² (humans 1.6 %) |
|---|---|---|---|---|
| bundled PPO | 0.88 [0.87, 0.89] | 7.6 | 0.49 | 2.7 % |
| (a) no smoothness reward | 0.915 [0.902, 0.928] | 15.9 | 0.32 | |
| (a) + 3x hard-brake penalty (start of b) | 0.908 [0.895, 0.920] | 53.2 | 0.072 | 5.4 % |
| (b) + 40 updates, no bonus (control) | 0.911 [0.898, 0.923] | 17.2 | 0.167 | |
| **(b) + human-likeness reward** | **0.875 [0.860, 0.889]** | **97.4** | **0.000** | **1.6 %** |

Reward re-weighting made the policy *more* recognisable, so attempt (a) failed. The adversarial reward made it less recognisable
(0.875 vs 0.911, the intervals do not overlap), matched the humans' hard-braking share exactly (1.6 %), and the control shows the
bonus, not the extra updates, did it. The surprise is safety: a policy rewarded for moving like people crashed less. That held beyond
the nominal scenario:

| scenario | human-like: score / unaided crashes per AV-h | control: score / unaided crashes per AV-h |
|---|---|---|
| nominal | **97.4** / 0.000 | 17.2 / 0.167 |
| rush hour | 21.4 / 0.068 | 18.0 / 0.273 |
| red runners | 16.2 / 0.120 | 12.6 / 0.215 |
| inattentive | 57.0 / 0.071 | 11.1 / 0.212 |
| jaywalkers | 2.9 / 0.242 | 3.6 / 0.219 |
| blockages | 12.8 / 0.095 | 12.8 / 0.443 |
| AV majority | 32.9 / 0.032 | 20.4 / 0.235 |
| cyclists | 24.4 / 0.082 | 41.5 / 0.163 |
| wet road | 28.6 / 0.024 | 18.1 / 0.168 |
| Squirrel Hill | 15.8 / 0.156 | 5.1 / 0.230 |
| Downtown | 11.2 / 0.181 | 57.9 / 0.120 |
| **mean** | **29.2** | 19.8 |

Unaided crashes are lower in 10 of 11 scenarios (sign test p = 0.012). Both arms were scored on the same CPU, so the platform drift of item 14 does not enter the comparison, but its noise does: single-scenario differences (e.g. Downtown) mean little; the count of scenarios and the means carry the result. Limits: one training seed per arm (H7 showed retrains vary
by tens of points per scenario), the core mean 29.2 is still below the best leaderboard entry (56.8) and the rule-based AV (77.2), and
the trip-level AUC is still 0.875: a per-step discriminator (its own AUC stayed ~0.53) cannot see trip-level statistics such as
smoothness (jerk 1.97 vs the humans' 2.81). The next step is a trip-level discriminator.

### 10. A comfortable IDM

```
need = max(braking needed for the stop line, braking needed behind the leader)        # from the simulator's observation
if need < 3.5 m/s^2:  a = max(a_IDM, -1.5 * need - 0.2)     # brake as hard as needed (with a margin), not as hard as IDM wants
else:                 a = a_IDM                             # an emergency: full IDM
```

Margins tried on AVBench nominal: 1.3 (43.9, crashes unaided), **1.5 (93.1)**, 1.6 (92.4, comfort 0.948); faster variants (a_max 2)
keep up with humans but lose comfort and safety (73.7).

### 11. The black box's light and pedestrian

```
on a link: look back up to 15 s for the last junction movement of the same trip;
           light = the signal of that movement's approach at the step it entered; report the junction, the turn, the light, when
pedestrian crash: legs of the struck vehicle's junction with someone crossing; walk back to the step the crossing started;
           was WALK shown then? (xnet.sim.leg_walk) -> "started without the WALK signal" or "crossing with the WALK signal"
```

### 13. The dilemma-zone bug in the safety envelope

```
found: a full-throttle policy under the envelope entered on red twice in 2 x 10 min (a test that had failed since commit 7cf1394)
black-box replay: the AV crossed a junction at 13.8 m/s onto a 31 m link just as its light turned yellow
    yellow decision at onset (correct): cannot stop at 3 m/s^2 (needs 31.6 m, has 27.1 m) and can clear in 2.0 s < 2.6 s -> GO
    but the envelope's speed-limit term braked it at -2 m/s^2 toward the 11.2 m/s limit -> the time it counted on was lost
    the light turned red 2.3 m short of the line; the GO decision is cleared on red -> it had to stop, could not, and entered on red
fix: while committed to a yellow (on the link, light yellow, decision GO), the speed-limit term may hold speed but not brake:
    v_term = max(1.5 (v0 - v), 0) if committed else max(1.5 (v0 - v), -2)          (the barrier function: the same clamp)
cost: one elementwise select per step
```

### 14. The leaderboard does not reproduce across hardware

```
for each row in {rule-based AV, phase 3} and each core scenario: score today on this CPU, compare with the stored (L4 GPU) value
then: the same with fix 13 reverted (identical) -> the drift is the platform, not the code
```

| | stored core mean (GPU) | today (CPU) | scenarios off by > 1 point | largest per-scenario drift |
|---|---|---|---|---|
| rule-based AV | 77.2 | 68.9 | 7 of 11 | −63.4 (wet road) |
| phase 3 | 56.8 | 58.9 | 10 of 11 | +75.4 (blockages) |

A scenario's av_world score turns on whether a rare event (a crash, a near miss) happens in its run, so it is nearly bimodal (~25 or
~95), and any numerical difference (GPU vs CPU, a library version) re-rolls it. This is H7's finding from another side. What it means
for every leaderboard number in this repository: **per-scenario scores are not evidence; means over the 11 core scenarios move by up
to ~8 points**. The ranking at the top survives (rule-based first on both platforms). The fix that would settle it is several seeds
per scenario with an interval on every score (a GPU job); until then `examples/leaderboard_drift.py` measures the drift, and
METRICS.md and the README say so next to the numbers.
