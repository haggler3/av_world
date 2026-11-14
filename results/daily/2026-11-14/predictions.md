# Predictions, 2026-11-14 (written before the full runs)

## Idea 1: friction-aware AV (avlab.frictionaware; half-AV Oakland, 24 copies per cell, seed 5, 480 s scored)
* P1: at friction 0.4, `headway` (T = 1.5 s / mu) cuts the AV-involved crash rate per AV-hour vs `blind` by >= 30 %, with the 95 % upper bound of the rate ratio below 1 (Holm over the 4 tests).
* P2: `slow` (also v0 x sqrt(mu)) cuts it more than `headway` at 0.4 and 0.6.
* P3: cost: `headway` lowers AV mean speed by < 10 % at 0.4; `slow` by 25-40 % (sqrt(0.4) = 0.63 bounds it).
* P4: city-wide crash rate (humans included) falls with `slow` at 0.4 (rate ratio < 1), but by less than the AV rate.

## Idea 2: pool the third random stream into the knee test (avlab.knee, no new simulation)
* P5: with 2026-11-13's seed-3 run (stopline.json, 16 copies x 6 frictions incl. 0.4) added, the fixed 0.7 hinge on the
  pooled data keeps an extra slope > 0 with one-sided p < 0.05; stream 3 alone, restricted to mu >= 0.5, has extra slope > 0 (point estimate).
