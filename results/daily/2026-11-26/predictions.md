# 2026-11-26 pre-registration: where do delayed-perception crashes come from?

Written before the full runs. Tool: `avlab.latencyattr` (per-step state of the AV's delayed view vs the truth).
Setup as 2026-11-22/24: Oakland, 16 copies, keys 43000+i, half AVs, friction 1.0, envelope off, IDM policy,
constant-velocity prediction (`Latency(delay_s, compensate=True)`), 120 s warm-up then 600 s.

States per AV step (TOL = 2 m on the gap error e = shown gap - true gap):
none, ok (|e| <= 2), new (true leader, view empty), over (e > 2), other.
lambda_s = AV crashes in state s / AV-hours in state s.

- **H1** (2.0 s delay): lambda_new / lambda_ok > 1. Exact conditional Poisson test, two-sided p < 0.05.
  Expected RR about 10. Power: the 2026-11-22 cell had 272 AV crashes; if even 30 fall in `new` and 30 in `ok`,
  an RR of 3 is detected with power > 0.95.
- Secondary (Holm, m = 3):
  - S1 (2.0 s): among crashes in `new` or `over`, the share in `new` differs from 0.5 (exact binomial). Expected > 0.5:
    leaders that arrive during the delay, not leaders that brake, carry the residual (consistent with the 2026-11-24 null).
  - S2 (2.0 s): lambda_over / lambda_ok > 1.
  - S3 (1.4 s delay): lambda_new / lambda_ok > 1.
- Tool check (must hold, else the day fails): per copy, the sum of per-state crashes equals the simulator's `av_crash`.
- Success: H1 meets its bar and the tool check holds.
