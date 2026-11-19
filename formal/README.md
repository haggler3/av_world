# Formal results (Lean 4 + Mathlib)

Machine-checked statements about the simulator's code. Today: when is the AV braking barrier sound?

| theorem | statement (all over the reals) |
|---|---|
| `bound_feasible` | if the road allows braking B >= b, the cap of `xnet.sim.cbf_bound` is above -B on every state with h >= 0 (obstacle braking no harder than b) |
| `bound_infeasible` | if B < b, every speed v > B b tau / (b - B) has a state with h = 0 (obstacle stopped) whose cap is below -B |
| `stop_gap_sound` | AV braking B_f >= b, leader B_l <= b, h >= 0: the gap after both stop is at least d0 |
| `stop_gap_unsound` | B_f < b: some state with h = 0 ends in contact |
| `friction_threshold` | with the simulator's b = 6, tau = 1/2 and B = 9 mu: feasible on every h >= 0 state iff mu >= 2/3 |
| `vfeas_06/05/04` | the critical speeds 27, 9, 4.5 m/s at mu = 0.6, 0.5, 0.4 |

`tests/test_barrier_formal.py` checks the Lean definitions against the code (`cbf_bound` on random states) so they cannot
drift apart. The simulation that tests the consequences is `avlab/barrierfriction.py` (see docs/DAILY.md, 2026-11-19).

Build (about 10 s after the Mathlib cache, ~4 GB in `.lake`, which is git-ignored):

```
# Lean 4.25.0 from https://releases.lean-lang.org/lean4/v4.25.0/ on PATH
cd formal && lake exe cache get && lake build
```

Only the standard axioms are used (`#print axioms AvFormal.friction_threshold`: propext, Classical.choice, Quot.sound).
