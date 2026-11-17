# 2026-11-17 predictions (written before the full runs)

## Idea 1: powered gap-caution confirmation (avlab.gappowered, seed 8, half-AV Oakland, friction 0.4, 120 copies per cell)

* G1 (the ONE confirmatory test): gap (gsc 1.495) vs blind (gsc 1.15), AV crashes per AV-hour, exact conditional Poisson:
  rate ratio < 1 with two-sided p < 0.05. Power at a true ratio of 0.6 is roughly 0.9 at ~100 blind crashes.
* G2 (descriptive): merge-crash rate ratio <= 0.5.
* G3 (descriptive): AV speed cost < 5 %.
* Post hoc (labelled as such): fixed-effect pooling of seeds 6, 7, 8.

## Idea 2: fill the friction gaps 0.65 and 0.75 (avlab.knee5, seed 5, humans-only Oakland, 48 copies per cell, 600 s scored)

Predicted rate ratio 0.65 / 0.75 from the four-stream fits: hinge 0.64 -> 1.16; hinge 0.7 -> 1.33; hinge 0.8 -> 1.59; power law -> 1.49.

* K1: the observed stream-5 ratio 0.65/0.75 is > 1 (point estimate). Not expected to be significant alone (~35 crashes per cell).
* K2: with five streams the 3.84 likelihood set for the hinge stays wider than 0.15 in friction (today 0.56-0.84).
* K3: the kink-vs-smooth-quadratic p-value stays below 0.1 (today 0.037).
