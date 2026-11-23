# 2026-11-23 pre-registration: limited AV sight distance and a phantom obstacle at the edge of sight

Written before the full run, after a 2-copy, 80 s probe (sight 8 m unaided: 3 AV crashes in 2.45 AV-hours; with the phantom: 0
in 2.54 AV-hours at 7.9 km/h instead of 19.4).

Setup: `avlab.visibility`, Oakland, 16 copies per cell, keys 44000+i (stream 44, new), half AVs, friction 1.0, envelope off,
IDM AV policy, 120 s warm-up then 600 s. Only the AV's view of its leader is limited; human drivers see as before.

Cells: clear; blind (no phantom) at sight 6, 8, 10, 14 m; phantom at 8 and 14 m.

**H1.** At sight 8 m, the phantom lowers the AV crash rate: RR = lambda(phantom 8) / lambda(blind 8) < 1.
Test: exact conditional Poisson test (k1 ~ Binomial(n, t1 / (t1 + t2)) under RR = 1), two-sided p < 0.05, and RR < 1.
Power: the probe implies ~1.2 AV crashes per AV-hour unaided, ~700 in ~580 AV-hours; an RR of 0.5 needs ~33 per arm. Power > 0.99.

Secondary (Holm, m = 2):
- S1: blind 8 m vs clear, RR > 1 expected (rejected).
- S2: phantom 8 m vs clear (expected: not rejected; the phantom should bring crashes back to the clear-day level).

Descriptive (no test): the AV crash rate against sight distance (6, 8, 10, 14 m, clear), and the AV speed cost of the phantom.
