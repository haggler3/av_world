# 2026-11-20: Timed Elastic Band as an AV planning rule (written before the runs)

Rule: `avlab/teb.py`, a 1-D Timed Elastic Band (Rösmann et al. 2012/2013): N = 8 intervals, 4 s horizon, speeds and
durations optimised by 10 Levenberg-Marquardt steps from an IDM warm start, minimising squared time plus soft speed / acceleration / leader /
stop-target / smoothness penalties; the first acceleration is applied. It runs in the policy slot (`rl=`), under the
simulator's usual envelope, with the same features as the baseline `careful_idm` (examples/my_policy.py). No xnet change.

E1 (confirmatory, `avlab/tebbench.py`): Oakland, calibrated humans, p_av 0.5, friction 1.0, 24 copies per policy, copy i on the
same key in both arms (stream 31), warm 120 s, 360 s scored. Baseline: careful_idm in the same slot.

- H1 (primary, "plans better progress"): AV mean speed (km/h) TEB > IDM. Paired t-test over the 24 copies, bar two-sided
  p < 0.05 and a positive mean difference. Expectation: TEB minimises time, so +3 to +10 %. With a per-copy SD of the
  difference around 3 % of speed, 24 pairs give > 0.9 power for +3 %.
- H2 (co-primary, safety, non-inferiority): AV crash rate ratio TEB / IDM, exact conditional Poisson interval, upper 95 %
  bound below 2. With an expected 15-30 AV crashes per arm this margin is loose; I will say so.
- Success = H1 and H2 both hold.
- Secondary (Holm, m = 2): S1, AV hard brakes per AV-hour ratio < 1 (exact Poisson); S2, AV jerk RMS lower (paired t).

E2 (descriptive): the AVBench scorecard (avrl.bench runner, the leaderboard's warm states and keys, 8 copies, 180 s,
envelope off and on) for TEB and careful_idm on nominal, rush_hour, red_runners, jaywalkers, wet_road, cut_in; the
leaderboard's rule-based AV row for reference. No test (8 copies per scenario).

Development note (before the confirmatory runs): a first version optimised with 30 Adam steps; a 2-copy quick run and a
single-car check showed it did not converge (a car from rest stalled at 1.7 m/s). The solver was replaced by
Levenberg-Marquardt on squared residuals (TEB's own g2o formulation), checked on a single car only (reaches 13 m/s, stops
1.0 m before a red line and 2.5 m behind a stopped car). The confirmatory stream 31 was not used in development.
AV capacity per copy: 256 (max seen 219 at p_av 0.5); rows beyond it would fall back to IDM.
