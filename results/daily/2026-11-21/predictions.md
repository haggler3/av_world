# 2026-11-21 pre-registration (written before any full run)

Capability: `avlab.tracker`, stateful AV perception in the policy slot (memory passed through `aux`): Markov (bursty)
missed detections and a hold-last-detection tracker with constant-velocity gap prediction. `xnet/` untouched.

Setup (every cell): Oakland, 16 copies, keys 43000+i, half AVs, friction 1.0, envelope off (`X.SHIELD = "off"`), AV policy =
IDM baseline (`avlab.teb.idm_policy_masked`) behind the perception model; 120 s warm-up, then 600 s.

Cells: clean; iid (p_miss 0.15, burst 0.2 s); iid+hold (hold 1 s); burst (p_miss 0.15, mean burst 1 s); burst+hold (hold 1 s);
burst+hold2 (hold 2 s, exploratory, not tested).

**H1.** With independent misses (p_miss 0.15), the 1 s hold tracker lowers the AV crash rate:
RR = lambda(iid+hold) / lambda(iid) < 1. Exact conditional Poisson test, two-sided p < 0.05.
Power: the 2026-11-20 miss-only cell had 1603 AV crashes in ~509 AV-hours; even an RR of 0.8 needs only
k >= 2 (2.8)^2 / (log 0.8)^2 = 315 crashes per arm, so 16 copies are ample if the iid arm reproduces hundreds of crashes.
Expected: RR below 0.1.

**Secondary (Holm, m = 2).**
S1: bursty misses at the same miss share crash more than independent ones: RR = lambda(burst) / lambda(iid) > 1 (two-sided).
S2: the 1 s hold also lowers crashes with bursty misses: RR = lambda(burst+hold) / lambda(burst) < 1 (two-sided).
Expected: S1 RR > 1 (a 1 s blind spell at standstill moves the car ~1.5 m; a 0.2 s one ~0.06 m); S2 RR < 1 but weaker
than H1 (about a third of 1 s-mean bursts outlast the 1 s hold).

Success: H1 meets its bar, the results file reproduces the write-up, the clean-model identity test passes.
