# 2026-11-16 predictions (written before the full runs)

## Idea 1: gap caution, confirmatory replication (avlab.gapconfirm, fresh random stream seed 7)
Single pre-registered test (no Holm needed; bar fixed in advance): at friction 0.4, AV-involved crashes per AV-hour,
`gap` (gap scale 1.495) vs `blind` (1.15), 40 copies each, exact conditional Poisson.
* C1 (confirmatory): rate ratio < 1 with two-sided p < 0.05.
* C2: merge-crash rate ratio <= 0.5 (point estimate).
* C3 (cost, not tested): at friction 0.6 and 1.0, maximum gap caution costs < 5 % of AV speed and < 8 % of AV trips.
* C4 (exploratory): at friction 1.0 AV crash rate ratio gap/blind has interval including 1 (too few crashes to tell).

## Idea 2: where is the knee, and is it a knee? (avlab.kneeprofile, no simulation, four pooled streams)
* K1: the profile deviance of the hinge position is flat: the set of hinges within 3.84 deviance of the best spans > 0.2 in friction.
* K2: a hinge (best position) does not beat a smooth quadratic in log friction by a significant margin
  (parametric bootstrap under the quadratic, p > 0.05) -> "convex curve", not a kink.
* K3: the copy-resampling bootstrap of the best hinge has a 95 % interval wider than 0.6-0.8.

Note: K1 was half-known before writing: a 0.4 s probe of the deviance profile on a 0.025 grid showed drops 21-23.6 between 0.55 and 0.85. K2 and K3 were not computed before writing.
