# Predictions, 2026-11-12 (written before the full run of avlab.brakedemand; after a 2-copy, 60 s probe at mu 1.0/0.7/0.4)

Run: humans-only Oakland, 10 copies per friction (1.0 0.85 0.7 0.6 0.5 0.4), seed 2 (fresh stream), warm 120 s, count 480 s.
"Saturated" = a moving human car whose applied acceleration is at the road's cap -9 mu m/s^2.

* P1 (the knee is mechanical): the saturated-braking share changes by less than 1.5x from mu 1.0 to 0.7, and at least doubles from 0.7 to 0.5.
* P2 (drivers do not adapt): the dry-road braking histogram predicts the saturated share at every lower mu within a factor 2
  (share of dry braking steps at or beyond 9 mu).
* P3: cars involved in crashes braked at the cap within the 1 s before far more often than moving cars in general: pooled
  enrichment over mu <= 0.6 with a 95 % lower bound > 1.
* P4: across the six cells, crash rate and saturated share rank together (Spearman rho >= 0.8).

The probe showed 0.09 % saturated at mu 1.0, 0.15 % at 0.7, 0.97 % at 0.4 (2 copies, 60 s): P1's first half may fail.
