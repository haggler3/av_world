# 2026-11-20 (second run): AV sensor noise (written before the confirmatory runs)

Tool: `avlab/sensornoise.py`, a wrapper around any AV policy that corrupts the leader features before the policy sees them:
relative range error $\sigma_r$ (log-normal), leader-speed error $\sigma_v$ (m/s), missed detection with probability $p_m$ per
0.2 s step. The envelope inside `xnet.sim.step` still sees the true state. No `xnet/` change.

Levels: clean (0, 0, 0), mild (0.05, 0.25, 0.02), moderate (0.10, 0.5, 0.05), severe (0.20, 1.0, 0.15); and three single-channel
cells at the severe value of one channel only: range, speed, miss.

Probe (8 copies, seed 77, 160 s scored, not used below): envelope off, clean 0 AV crashes, moderate 0, severe 10 (31 AV-hours);
envelope on, severe 0.

Experiment (`avlab/sensorbench.py`): Oakland, calibrated humans, p_av 0.5, friction 1.0, AV policy careful_idm
(`avlab.teb.idm_policy_masked`), 16 copies per cell, copy i on key PRNGKey(41000 + i) in every cell (fresh stream 41), warm 120 s,
600 s scored. Cells: envelope off x {clean, mild, moderate, severe, range, speed, miss}, envelope on x {clean, severe}.
Expected exposure ~63 AV-hours per cell.

- **H1 (primary): the envelope absorbs severe sensor noise.** With severe noise, AV crash rate ratio
  $\mathrm{RR} = \lambda_\text{on} / \lambda_\text{off} < 1$. Exact conditional Poisson test, two-sided $p < 0.05$ and RR < 1.
  Power: the probe rate unaided is ~0.3 per AV-hour, so ~20 crashes are expected in the off arm and ~0-2 in the on arm; with
  $n = 20$, $k_\text{on} \le 5$ gives $p < 0.05$, so power is > 0.9 if the probe rate holds.
- Secondary (Holm, m = 3, envelope off, each vs clean, RR > 1): S1 severe; S2 miss only; S3 range only.
- Descriptive: crash rate per level (dose response), speed-only cell, envelope cut share, hard brakes, AV speed.

Success = H1 meets its bar.
