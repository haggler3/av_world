# 2026-11-13 predictions (written after the 2-copy --quick probe, before the full run)

Run: `python -m avlab.stopline` (16 copies x 6 frictions, seed 3, a fresh random stream, warm 120 s, count 480 s).

* P1 (stop-line channel makes the knee): friction-induced overruns (cars told to stop that cannot stop at mu but could on a dry road),
  per moving car-step, bend at 0.7: log-log slope below 0.7 is steeper than above by at least 1.
* P2: crash-involved human cars were in a friction-induced overrun within 2 s far more often than moving cars
  (pooled mu <= 0.6: enrichment lower 95 % bound > 1, and share >= 10 %).
* P3 (code check): the commitment channel (`can_stop`, min(4, 9 mu)) adds no blind entries for mu >= 0.5 (exactly 0), only at 0.4.
* P4: across the six cells, Spearman(friction-overrun rate, crash rate) >= 0.8.
* P5 (friction tomography): with exact braking, the max-braking estimator is within 0.05 of mu for >= 95 % of copies after 10 s
  of one city, at every mu. With 0.3 m/s^2 sensor noise, the spike (difference-of-Gaussians) estimator reaches >= 90 % within 0.05
  by 80 s at every mu.
