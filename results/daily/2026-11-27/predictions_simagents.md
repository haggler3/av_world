# 2026-11-27 (second change) pre-registration: learned sim agents for a whole city

Written after a 2-copy smoke test (300 training steps; its numbers are not evidence) and before the full run.
Tool: `dwm/simagents.py` (reads the simulator; `xnet/` untouched).

Setup: Oakland, calibrated humans. Training data: 8 copies x 600 s (seed 11), every 5th step, every live unwrecked
vehicle: 28 observation features -> acceleration token (32 bins on [-8, 3] m/s^2). Model: MLP 28-256-256-32, 6000 Adam
steps. Ground truth for scoring: 8 fresh copies (seed 12). Closed-loop cells (seed 13, 8 copies x 600 s after 120 s
warm-up, envelope off), every vehicle driven by: `idm` (the hand-written IDM rule of avlab.teb), `argmax` (most likely
token), `sample` (sampled token + jitter inside the bin); `gt2` = the real rule-based humans again (noise ceiling).

Realism: per feature f in {speed, accel, jerk, gap, headway}, L_f = exp(mean log q_f(x)) over ground-truth samples x,
q_f the cell's smoothed 50-60-bin histogram; meta = mean_f L_f. Per-copy meta (each copy's histogram vs pooled ground
truth) gives 8 values per cell.

- **H1**: meta(sample) > meta(idm). Difference of per-copy means, bootstrap 95 % interval, two-sided permutation p < 0.05.
  Prior 0.5: the smoke test pointed the other way with a tiny model.
- Secondary (Holm, m = 2): S1 meta(sample) > meta(argmax) (sampling keeps the spread of human behaviour); S2 meta(gt2) >
  meta(sample) (the learned model is below the noise ceiling).
- Reported, not tested: crashes per 1000 vehicle-hours, mean speed, trips finished per vehicle-hour, held-out token NLL
  against the marginal-token baseline.
