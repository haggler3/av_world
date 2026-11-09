# Predictions (written before the full run), 2026-11-09

Idea 1 - friction-aware AVs (avlab/frictionav.py), half-AV Oakland, friction 0.45, 16 copies per cell.
The AV planner reads friction only through the braking cap. Cells: base, headway (T >= 1.3/mu = 2.9 s), speed (v0 factor <= sqrt(mu) = 0.67), both.

* P1: the AV-involved crash rate per AV-hour falls by at least 25 % with both changes (ratio <= 0.75, Holm-adjusted p < 0.05).
* P2: headway alone does little (ratio > 0.8) because slick-road crashes are merge / pedestrian / bicycle, not rear-end (2026-11-08).
* P3: the price is speed: AV mean speed falls by more than 15 % in the speed cell.
