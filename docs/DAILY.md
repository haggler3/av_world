# Daily research log

One entry per day, newest first. Each entry: the ideas kept, the pseudocode, the result, the figures, and what to try next. The
search behind each day's choice (candidates, probes, scores) is in `results/daily/<date>/search.json`.

## What to try next (running list)

1. **Check which random stream the L4 leaderboard used.** The CPU run here used JAX's default generator; the repo's scripts set `JAX_DEFAULT_PRNG_IMPL=unsafe_rbg`, whose numbers differ between backends. Re-score the rule-based AV on the same CPU with `unsafe_rbg` and compare with the stored GPU row: if the gaps are the size seen below (~15 points), the drift is the stream.
2. **Seeds with intervals on every leaderboard row** (FIXES.md #14's fix): `avlab.twins.bench` with `restream=1..N` already produces the scores; 8 streams x 11 scenarios x 2 rows is ~45 min on 4 cores (over a daily budget, so split over days).
3. **Why does one kick in sixteen grow?** The 3 m/s kick that decorrelated one twin: find the first vehicle whose position diverges and the event (a gap acceptance? a signal phase? a pedestrian?) that amplified it. A black-box style trace on the pair.
4. Settle H4's curve shape and H3's wet-road threshold (both need GPU-sized data).
5. Add a speed-limit knob outside `xnet/` (a post-hoc scale on `link_v` in the net dict) and test the power model of crash rate against speed.

---

## 2026-11-03 - Twin cities: is the simulator chaotic, and what moved the leaderboard?

**Ideas kept** (from 16 candidates; [search](../results/daily/2026-11-03/search.json)): (1) twin-run test of sensitive dependence on initial
conditions; (2) its child: AVBench scores of the rule-based AV under the same nudges. Both test one explanation from
[FIXES.md](FIXES.md#14-the-leaderboard-does-not-reproduce-across-hardware): "floating-point differences flip rare events".
Prior art: sensitive dependence is the definition of chaos (Lorenz 1963); traffic models with car following are known to be
stable or unstable depending on parameters (string stability of IDM, e.g. Treiber and Kesting, *Traffic Flow Dynamics*, 2013); the
non-determinism of GPU numerics is documented for deep learning (e.g. Nagarajan et al., *Deterministic implementations for reproducibility in deep reinforcement learning*, 2018). (No web search was run today: the search tools were not used; these are cited from memory and should be checked before quoting.)

### Pseudocode (what `avlab/twins.py` does)

```
A. for each of 16 seeds: warm Oakland (calibrated humans) 120 s; twin = a copy of the state
   nudge the twin: one car +1 mm/s | one car +3 m/s (the fastest car) | every car +1 mm/s | the simulator's random key re-keyed (fold_in)
   run both 600 s in 20 s steps (same code, same everything else)
   displaced(t) = share of vehicles alive in both whose arc position differs by > 1 m (or on another track)
   at the end: same crash count? same trips finished?
B. rule-based AV on AVBench scenarios nominal, rush hour, red runners, wet road (8 copies, 300 s warm, 180 s scored), from the
   leaderboard's warm state, under: as published | every car +1 mm/s | every car +1 m/s | the state's random key re-keyed
   (avrl.bench.runner got `kick`, `restream`, `seed`, `key_seed`; defaults reproduce the leaderboard run exactly)
```

### Predictions, written before the full runs (after a 4-copy probe, so P1 was already half-known)

* P1: nudges of 1 mm/s (one car or every car) displace < 1 % of vehicles at 10 min, and >= 90 % of twins keep the same crash count and trips; the same nudges change no AVBench score by more than 1 point.
* P2: a new random stream decorrelates the city (> 50 % displaced within 1 min, the same crash count in < 60 % of twins) and moves AVBench scores by >= 10 points on average.

### Result

| nudge | displaced at 1 / 5 / 10 min | twins with any displacement | same crash count | same trips |
|---|---|---|---|---|
| one car + 1 mm/s | 0 / 0 / 0 % | 0 of 16 | 16 / 16 | 16 / 16 |
| every car + 1 mm/s | 0.02 / 0 / 0 % | 0 of 16 | 16 / 16 | 16 / 16 |
| one car + 3 m/s | 0.1 / 0.7 / 4.9 % | **1 of 16** (79 % of its vehicles) | 16 / 16 | 15 / 16 |
| new random stream | **71 / 99 / 99.9 %** | 16 of 16 | 7 / 16 | 0 / 16 |

![twins](img/twins.png)

AVBench, rule-based AV, one run each from the leaderboard's warm state (`results/twins/bench.json`):

| scenario | as published | every car +1 mm/s | every car +1 m/s | new random stream |
|---|---|---|---|---|
| nominal | 72.5 | 72.5 | 60.1 | 90.3 |
| rush hour | 92.0 | 92.0 | 91.9 | 78.8 |
| red runners | 91.0 | 91.0 | 90.4 | 61.2 |
| wet road | 26.2 | 26.2 | 26.1 | 26.5 |
| mean absolute change | | 0.01 | 3.3 | **15.3** |

![bench](img/twins_bench.png)

![twin cities GIF](img/twins.gif)

(The GIF shows seed 0, where the +3 m/s twin stays in step: the 1-in-16 divergence is not that one.)

**Both predictions held.** The simulator is **not chaotic at rounding scale**: a nudge of a thousandth of a metre per second to
every car leaves no vehicle displaced by a metre in ten minutes and no AVBench score changed (0 of 16 twins diverged: 95 % upper
bound 21 %). A large kick to one car does cascade sometimes (1 of 16, 95 % interval 1-28 %), so sensitive dependence exists, but
rarely. A different random stream changes everything, and moves a rule-based AV's scenario score by 15 points on average
(+17.9, −13.2, −29.7, +0.3), the same size as the CPU-versus-GPU drift in FIXES.md #14 (core means moved 8; single scenarios up to 75).
A 1 m/s kick to every car, far above rounding, moved one scenario (nominal) by 12 points and the others by under 1.

**What it means.** Per-scenario scores are draws from a wide distribution, not constants, and a hardware change matters through
whatever re-rolls the random stream, not through rounding. That is consistent with the drift but does not prove its cause: the
GPU run's generator was not recorded (next step 1). The practical rule is the one already in the README: read means over scenarios
and rankings, and put an interval on any single score.

**What failed.** The first version of the "new random stream" condition changed `key_seed`, which only reaches a *learned* policy;
the rule-based AV and the humans draw from the simulator state's own key, so the four scores came back identical and the condition
did nothing. It was caught by the null result, fixed with `restream` (re-keying the state), and rerun; only the fixed numbers are
reported above. Limits: 16 twin pairs and one AVBench run per cell (a single scenario score is itself a draw), one city, one nudge
location rule (the fastest car), and "displaced" uses a 1 m threshold, so sub-metre differences are not counted.

Commands: `python -m avlab.twins` (~3 min), `python -m avlab.twins --bench` (~10 min), `python -m avlab.twins --plot`; test: `tests/test_twins.py`.
