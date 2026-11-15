# Predictions for 2026-11-15, written before the full runs

## Idea 1: the merge channel (avlab.frictionmerge, friction 0.4, 40 copies x 4 cells, seed 6, p_av 0.5)

* P1: `gap` (gap-acceptance scale 1.15 -> 1.495) cuts AV-involved **merge** crashes vs `blind` by >= 50 %.
* P2: `headway+gap` vs `blind`: AV-involved crash rate ratio <= 0.6 with the 95 % upper bound below 1 (Holm over 4 tests p < 0.05).
* P3: `headway+gap` vs `headway`: rate ratio < 1 (the gap channel adds to headway).
* P4: `gap` costs < 10 % of AV speed (km per AV-hour) vs `blind`.

## Idea 2: the friction knee on a fourth random stream (avlab.knee4, humans-only, 32 copies at 0.85 / 0.7 / 0.6, seed 4)

* P5: stream 4's 0.7 cell rate lies below its 0.6 cell (stream 3's 0.7 > 0.6 was a fluke).
* P6: pooled over four streams, the fixed 0.7 hinge keeps c > 0 with one-sided p < 0.001.
* P7: the streams' hinge slopes do not differ (heterogeneity LR p > 0.05).
