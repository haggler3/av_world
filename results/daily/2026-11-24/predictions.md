# 2026-11-24 pre-registration (written before any full run)

Capability: `avlab.latency.Latency(accel=True)`, a constant-acceleration forward prediction for a delayed leader. The leader's
last measured acceleration a = (v_l(t-D) - v_l(t-D-1)) / 0.2 s, clipped to [-9, 3] m/s^2 (0 when either step had no
leader), is extrapolated over the delay: leader speed u_k = max(v_l(t-D) + a k 0.2, 0), gap
g_hat = g(t-D) - 0.2 sum_k (v(t-D+k) - u_k), shown leader speed max(v_l(t-D) + a D 0.2, 0). `xnet/` untouched; with
accel=False the code path is the 2026-11-22 one.

Question (from 2026-11-22 "Next"): is the residual crash rate of the constant-velocity predictor at long delays (2.0 s: 272
AV crashes, 55x clean) caused by leaders that change speed during the delay?

Probe (4 copies, 40 s warm-up + 120 s, seed 43, not part of the test): constant velocity 2.0 s 0 crashes, 19.20 km/h,
305 hard brakes; constant acceleration 0 crashes, 19.22 km/h, 327 hard brakes (smoke test only; too few AV-hours).

Setup (as 2026-11-22, same keys 43000+i so the saved comp / clean cells are the comparison): Oakland, 16 copies, half AVs,
friction 1.0, envelope off, IDM AV policy; 120 s warm-up, then 600 s. New cells: ca1.4, ca2.0.

**H1.** At 2.0 s delay, constant-acceleration prediction lowers the AV crash rate relative to constant velocity:
RR = lambda(ca 2.0) / lambda(comp 2.0) < 1, lambda = AV crashes per AV-hour. Exact conditional Poisson test, two-sided p < 0.05.
Power: comp 2.0 had 272 crashes in 577 AV-hours. With ~270 in the reference arm, an RR of 0.7 is detected with power ~0.95
(k >= 2 (2.8)^2 / (log 0.7)^2 = 123 per arm). Expected: RR about 0.6. Prior that H1 holds: 0.55 (the acceleration estimate
from one 0.2 s difference is noisy and may overshoot).

**Secondary (Holm, m = 2).**
S1: at 1.4 s, RR = lambda(ca 1.4) / lambda(comp 1.4) (73 crashes in the reference), two-sided test of RR = 1.
S2: at 2.0 s the constant-acceleration AV still crashes more than clean sensors: RR = lambda(ca 2.0) / lambda(clean) > 1.
Exploratory: AV speed and hard brakes per cell.

Success: H1 meets its bar as written; the results file reproduces the write-up; tests/test_latency.py passes.
