# 2026-11-28 pre-registration: an imperfect reflex sensor (avlab.reflexfault)

Written before any full run. Setup as 2026-11-27 (Oakland, 16 copies, keys 43000+i, half AVs, friction 1.0, envelope off,
IDM policy, 2.0 s delayed long-range view with constant-velocity prediction, 120 s warm-up, 600 s measured), so the saved
cells `rf10` (perfect 10 m reflex, 7 AV crashes / 587.1 AV-h), `comp2.0` (no reflex, 272 / 576.7) and `clean` are the references.
lambda = AV crashes per AV-hour.

New cells, all with a 10 m reflex:
- fb: reflex misses 15 % of steps in bursts of mean 1.0 s (Markov chain of avlab.tracker)
- fi: reflex misses 15 % of steps independently (burst 0.2 s)
- fn: no misses, relative range noise sigma = 0.2

**H1**: RR = lambda_fb / lambda_rf10 > 1. Exact conditional Poisson test, two-sided p < 0.05. Expected RR about 3
(a 1 s outage is half the delay, so a cut-in during a burst is seen only by the delayed view).
Power: the reference arm is fixed at 7 crashes; with RR 3 (about 21 crashes) SE(log RR) ~ sqrt(1/7 + 1/21) = 0.44,
z ~ 2.5, power ~ 0.7. Below the 0.8 target because the saved reference arm is small; RR >= 3.5 gives power >= 0.8.
Disclosed here rather than rerunning the reference.

Secondary, Holm over m = 3:
- S1: lambda_fi / lambda_rf10, expected near 1 (a single 0.2 s miss is bridged by the next step).
- S2: lambda_fn / lambda_rf10, expected near 1 (fusion keeps the smaller gap; over-reads are discarded, under-reads only brake).
- S3: lambda_fb / lambda_comp2.0 < 1, expected about 0.08 (a bursty reflex still removes most long-delay crashes).

Success: H1 meets its bar as written. Check: with p_miss = 0 and sigma = 0 the module equals avlab.reflex (unit test).
