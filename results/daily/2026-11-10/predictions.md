# Predictions, written before the full run (2026-11-10)

Idea 1 (avlab/whocrashes.py): half-AV Oakland, 48 copies per cell, cells = (friction, speed-limit scale) = (1.0,1), (0.6,1), (0.4,1), (1.0,1.3).
Pedestrian + bicycle strikes are single-vehicle events; compare AV and human drivers per vehicle-hour in the same city.

* P1: at friction 1.0 and the normal limit AVs strike fewer road users per vehicle-hour than humans: ratio < 1 with the upper 95 % bound < 1.
* P2: the ratio moves toward 1 as the road gets slick (the AV planner does not read friction: grep of xnet/sim.py shows friction only caps braking at 9 x mu), so at friction 0.4 the interval includes 1.
* P3: at limit x1.3 the AV advantage persists (ratio < 1), because AVs react in 0.3 s against 0.7-1.5 s for humans.

Idea 2 (avlab/frictionpool.py): pool the two humans-only friction runs (2026-11-06: 16 copies x 5 frictions; 2026-11-08: 12 copies x 6), no new simulation.
* P4: pooled over 140 copies, a cliff is not supported: the quadratic term and the best hinge each fail an F-test against a plain power law at 0.05, and the power law's leave-one-friction-out deviance is within 10 % of the best shape's.
* P5: the two runs' elasticities agree within 2 combined standard errors.
