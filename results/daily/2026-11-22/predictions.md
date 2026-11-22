# 2026-11-22 pre-registration (written before any full run)

Capability: `avlab.latency`, perception latency on the AV's leader in the policy slot (a ring of the last D = delay / 0.2 s
leader observations carried through `aux`), with optional constant-velocity forward prediction of the delayed gap using the
AV's own speed history. Built on a generalised `avlab.tracker.run(perceive_fn=, init_fn=)`. `xnet/` untouched.

Probe (4 copies, 40 s warm-up + 80 s, seed 43, not part of the test): AV crashes raw / compensated at delay 0.6 s 0 / 0,
1.0 s 9 / 0, 1.4 s 40 / 0, 2.0 s 52 / 0; clean 0.

Setup (every cell): Oakland, 16 copies, keys 43000+i, half AVs, friction 1.0, envelope off, IDM AV policy behind the latency
model; 120 s warm-up, then 600 s. Cells: clean; raw and compensated at delay 0.6, 1.0, 1.4, 2.0 s (9 cells).

**H1.** At 1.0 s delay, forward prediction lowers the AV crash rate: RR = lambda(comp 1.0) / lambda(raw 1.0) < 1.
Exact conditional Poisson test, two-sided p < 0.05.
Power: the probe's raw 1.0 s rate is 9 / 5.04 = 1.8 per AV-hour; a full cell has ~500 AV-hours, so ~900 crashes in the raw
arm. An RR of 0.8 needs k >= 2 (2.8)^2 / (log 0.8)^2 = 315 per arm; an RR of 0.1 needs ~3 in the smaller arm. Ample.
Expected: RR below 0.1.

**Secondary (Holm, m = 2).**
S1: raw 1.0 s delay crashes more than clean sensors: RR = lambda(raw 1.0) / lambda(clean) > 1 (two-sided).
S2: with 2.0 s delay, the compensated AV crashes at the clean rate: RR = lambda(comp 2.0) / lambda(clean), two-sided test
of RR = 1. Expected: not rejected (p > 0.05); a rejection means prediction does not fully cover a 2 s delay.
Exploratory, no test: the crash rate against delay, raw vs compensated (dose response), and AV speed.

Success: H1 meets its bar, the results file reproduces the write-up, the delay-0 identity tests pass.
