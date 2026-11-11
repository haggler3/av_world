# Predictions, 2026-11-11 (written before the replication run and before the bootstrap tests)

Replication run: humans-only Oakland, calibrated drivers, NEW random stream (seed 1), 36 copies at friction 1.0 / 0.85 / 0.7 / 0.6 / 0.5,
warm 120 s, count 600 s. Hinge FIXED at 0.7 (from the 2026-11-10 pooled fit), no search.

* P1 (knee replicates): in the new run the extra slope below 0.7 is positive (steeper as friction falls) with one-sided p < 0.05.
* P2 (flat above the knee): in the new run the slope above 0.7 (log rate on log friction, from 1.0 to 0.7) has a 95 % interval that includes 0.
* P3 (search correction): on the pooled 2026-11-10 data, the search-corrected (parametric bootstrap, sup over the 6-point grid) p-value
  for any hinge is still < 0.05 (nominal F p was 0.003, Bonferroni 0.017).
* P4 (speed limit): the pooled human-only speed-limit runs (32 x 5 scales + 12 x 3) show NO knee: search-corrected p > 0.05; a
  plain power law (elasticity near 4.5) is enough.
Holm correction over the three primary tests (P1, P3, P4's p-value).
