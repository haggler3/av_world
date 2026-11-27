# 2026-11-27 pre-registration: a short-range, zero-delay reflex sensor fused with a delayed long-range view

Written before any full run. Tool: `avlab.reflex` (policy slot, `xnet/` untouched).

Setup: same as the saved 2026-11-22 cells (Oakland, 16 copies, keys 43000+i, half AVs, friction 1.0, envelope off,
IDM policy, 120 s warm-up then 600 s). Long-range view: `avlab.latency` with delay 2.0 s and constant-velocity
prediction (cell `comp2.0`, saved: 272 AV crashes). Reflex: if the true leader is within R metres, the AV sees the
true gap and leader speed whenever that gap is smaller than the gap the delayed view shows (or the delayed view shows
no leader). Fused gap = min(delayed-predicted gap, true gap if true gap <= R).

lambda = AV crashes per AV-hour.

- **H1**: with R = 10 m, RR = lambda_reflex10 / lambda_comp2.0 < 1. Exact conditional Poisson test, two-sided p < 0.05.
  Expected RR about 0.4 (the 2026-11-26 attribution put 38 % of crashes in over-estimated-gap steps, most of them at
  short range).
- Power: the reference arm has 272 crashes. For RR = 0.5 we need k >= 2 (1.96+0.84)^2 / (log 0.5)^2 = 33 crashes per
  arm; an expected 100+ in the reflex arm gives power above 0.95. Even RR = 0.7 (k >= 123) is covered.
- Secondary (Holm, m = 2): S1 R = 20 m vs R = 10 m, RR < 1 (more range helps); S2 R = 10 m vs clean sensors
  (saved `clean`, 5 crashes), RR > 1 (the reflex does not make up for the whole delay).
- Check: R = 0 must reproduce the saved `comp2.0` counters exactly (272 AV crashes), run as cell `rf0`.
- Speed cost reported, not tested: AV speed in km/h for each cell.
