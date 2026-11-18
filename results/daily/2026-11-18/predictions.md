# 2026-11-18 predictions (written before the full run of `python -m avlab.patchyice --run`)

Capability: per-street friction (`xnet.sim.patch_links`, `Behaviour.patch_friction`, optional `net["link_patch"]` mask).

Design: humans-only Oakland, calibrated drivers, random stream seed 9 (no earlier run used it), warm 120 s, 600 s scored,
40 copies per cell, one compiled program. Cells: U1.0, U0.7, U0.4 (uniform friction) and P (friction 1.0, x 0.4 on random links
covering 50 % of the network's length, mask seed 0; length-weighted mean friction 0.7).

**H1 (the one primary test).** The crash rate ratio RR = lambda_P / lambda_U0.7 is above 1. Exact conditional Poisson test
(k_P ~ Binomial(k_P + k_U0.7, t_P / (t_P + t_U0.7)) under RR = 1), bar: two-sided p < 0.05 with RR > 1.

Why: earlier runs give roughly 8 (friction 1.0), 10 (0.7) and 71 (0.4, results/cliffcause/cliff.json) crashes per 1000
vehicle-hours. Crash rate is convex in friction, so by Jensen's inequality the mix should be worse than its mean: expected
lambda_P ~ 0.5 x 71 + 0.5 x 8 ~ 40, RR ~ 4.

Sample size and power: about 75 vehicle-hours per copy -> ~3000 h per cell -> ~30 crashes in U0.7. For RR = 2 (half the expected
effect), k_P ~ 60, se(log RR) ~ sqrt(1/30 + 1/60) = 0.22, z = 3.1, power ~ 0.88. The rule k >= 2 (2.8)^2 / (log r)^2 needs
k >= 33 per arm at r = 2. Not powered for RR < 1.6.

**Secondary (Holm, m = 2).**
- S1: P against the mixture of uniform cells, E = t_P (w r_0.4 + (1 - w) r_1.0) with w = P's share of vehicle-steps on patched links.
  Prediction: O / E is within 0.75-1.33 (drivers here do not adapt to friction, so a transition from dry to ice should add little).
  Test: delta method on log(O/E), two-sided.
- S2: lambda_P / lambda_U0.4 < 1 (half the ice should be safer than all ice).
