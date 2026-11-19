# 2026-11-19 predictions (written before the run)

Capability: `avlab.crashlog`, a per-crash event log (time, place, impact speed, role, AV flag, link) with vehicle-seconds per link.

Experiment (`python -m avlab.icemap --run`): humans-only Oakland, calibrated drivers, random stream seed 11 (not used by the
2026-11-18 run, which used seed 9), patch mask `patch_links(net, 0.5, seed=0)` (as 2026-11-18). Three cells, 32 copies each,
warm-up 120 s, 600 s logged:

- P: friction 1.0, patch_friction 0.4 (ice on the masked links)
- D: friction 1.0 everywhere (patch_friction 1)
- I: friction 0.4 everywhere

Crash events are rows grouped by copy, time step and a 25 m distance chain. An event is "icy" if any of its vehicles was on a
masked link. Exposure is vehicle-hours on masked and unmasked links from the log.

**H1.** Inside the patchy city P, the crash rate on icy links over the rate on dry links,
RR = (k_ice / t_ice) / (k_dry / t_dry), is above 1. Test: exact conditional Poisson,
k_ice ~ Bin(k_ice + k_dry, t_ice / (t_ice + t_dry)) under RR = 1; bar two-sided p < 0.05 and RR > 1.
Expected RR about 11 (64.0 / 5.5 per 1000 veh-h in the uniform 0.4 and 1.0 cells of 2026-11-18). With 32 copies we expect
about 80 events in P, about 70 on icy links and 7 on dry. For r = 3 (a conservative effect) we need
k >= 2 (1.96 + 0.84)^2 / (log 3)^2 = 13 per arm; the dry arm may hold fewer, but at r = 11 the need is 2.7 per arm, so power > 0.9.

**Secondary (Holm, m = 2), predictions:**
- S1, ice is local: rate on masked links in P over the rate on the same masked links in I. Predict the 95 % interval contains 1.
  (A rate below 1 would mean drivers in P arrive at ice in a different state, e.g. fewer queues.)
- S2, spillover: rate on unmasked links in P over the rate on the same unmasked links in D. Predict above 1 is possible
  (queues and wrecks backing out of icy links) but the test has low power (about 7 events per arm).

**Checks of the tool (must hold, or the day fails):** the number of events per copy equals the simulator's crash counters
summed over types, for every copy; log exposure equals `veh_s`. Descriptive only: impact speed by role and surface.
