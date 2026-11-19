# 2026-11-19 (second run): the barrier's braking assumption, formal result then test (written before the run)

Formal result (Lean 4 + Mathlib, `formal/AvFormal/Barrier.lean`, all proofs check, no sorry): the AV barrier of
`xnet.sim.cbf_bound` assumes braking b = 6 m/s^2; the road gives 9 mu. With tau = 0.5 the acceleration cap is feasible
on every state with h >= 0 iff mu >= 2/3. Below, on the barrier's edge with a stopped obstacle, the filter asks for more
braking than the road gives at every speed above v_feas(mu) = 9 mu b tau / (b - 9 mu): 27 m/s at 0.6, 9 at 0.5, 4.5 at 0.4.
Fix suggested by the proof: b_eff = min(6, 9 mu), which makes `bound_feasible` apply at every friction.

Simulator: half-AV Oakland (calibrated humans, p_av 0.5), `X.SHIELD = "cbf"` (every AV filtered), no xnet change; the
fix is set through the module dict `X.CBF["b"]` before tracing.

E1 (dose-response, descriptive + one test). Default barrier (b = 6), friction 1.0, 0.8, 0.7, 0.6, 0.5, 0.4; 12 copies each,
random stream 21, warm 60 s, 300 s measured. Measure per friction: share of moving AV steps (v > 2 m/s) inside the barrier
(h < 0), and share of moving AV steps faster than v_feas(mu).
  P1: the inside-barrier share is about flat from 1.0 to 0.7 (ratio 0.7/1.0 within 0.67-1.5) and rises below 2/3, with the
      0.4 share at least 2x the 0.7 share. Test for "at least 2x": two-proportion comparison of step shares is not valid
      (steps are correlated), so the bar is a copy-level bootstrap 95 % interval for share(0.4)/share(0.7) above 1.

E2 (the fix; the one confirmatory test). Friction 0.4, b = 6 (default) vs b = 3.6 (= 9 mu), 40 copies each, random stream 22,
warm 120 s, 480 s scored, two compiled programs (same keys).
  H1: AV-involved crash rate ratio (b 3.6 / b 6) < 1. Exact conditional Poisson test on AV crashes per AV-hour,
      bar two-sided p < 0.05 and RR < 1. Power: with ~35 AV crashes per arm, SE(log RR) ~ sqrt(2/35) = 0.24, so 80 % power
      for RR <= 0.51; a smaller real effect will be missed and I will say so.
  Secondary: S2, AV speed cost under 15 % (one secondary, no correction needed).
  (Amended before the run: an earlier S1 compared the inside-barrier share across arms, but h is computed with each arm's
  own b, so the two shares measure different things. Dropped.)
