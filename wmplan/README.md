# wmplan: driving by imagining in a latent world model

An AV that plans merges and junctions by unrolling learned latent representations into the future, in the whole-area
Pittsburgh simulator (`xnet`), among risky human drivers, pedestrians and cyclists.

![latent planning at a junction](../docs/img/wm_planner_junction.gif)

*Left: the AV (white outline) and, along its route, where each of the 32 candidate plans of the last search round puts it
over the next 2.4 s, coloured by score (red worst, green best); cyan is the plan it executes, dashed white is what really
happened. Right: the same plans decoded into speed, the junction time-to-conflict margin, and crash risk.*

![latent planning in a merge](../docs/img/wm_planner_merge.gif)

## How it works

```
obs_t (28 features: own motion, leader, stop target, signal, right of way, conflicting traffic, merge gaps)
  │ enc
  ▼
 z_t ──dyn(z, u_t)──▶ z_t+1 ──dyn(z, u_t+1)──▶ z_t+2 ──▶ …  z_t+12        (2.4 s, in latent space only)
  │                    │                        │
  ▼ decoders           ▼                        ▼
 margins, obs        reward, P(crash), P(near miss), P(red-light run), margins
```

1. **Data** (`data.py`, 25 s of simulation for 800 k decisions). The 234k-step PPO driver, plus exploration. Every
   vehicle slot gets its own persistent style, from timid to reckless, in both acceleration and gap acceptance. No
   safety filter, so the logs hold the consequences: 666 crashes and 101 red-light runs. Each decision is logged
   with five safety margins, which are the barrier values: stop target, leader, junction time-to-conflict, and the
   merge gaps behind and ahead.
2. **World model** (`model.py`, ~5 min on CPU). An encoder to a 48-d latent, residual latent dynamics conditioned on the
   2-d action (acceleration, gap acceptance), and decoders for the scene, the margins, the reward and three events.
   Trained by unrolling 15 steps (3 s) from each encoded start. Every prediction is compared with what happened, plus a
   consistency loss toward the encoding of the real next scene. Windows that end in a crash still teach.
3. **Planner** (`planner.py`). Model-predictive control by the cross-entropy method. Every 0.2 s, for every AV at once:
   - propose 32 smooth action sequences (3 knots per dimension) around a prior from the PPO policy;
   - unroll each in latent space;
   - score them: reward − 30·P(crash) − 3·P(near miss) − 20·P(red run) − margin violations − jerk;
   - refit to the best 6 and repeat twice;
   - execute the first action.

   The second action dimension is the AV's gap acceptance at the next junction and in a merge. So *whether to go* is
   chosen by imagining what the junction margin and the crash risk do a few seconds after going.

## Results

![results](../docs/img/wm_planner.png)

**The model** (a held-out copy of the area, 85 k windows):

| prediction | latent model | holding the scene still |
|---|---|---|
| AV speed, 1 s / 2 s / 3 s ahead (RMS) | **0.46 / 0.57 / 0.68 m/s** | 1.02 / 1.52 / 1.88 m/s |
| stop-line margin at 2 s (RMS, units of 20 m) | **0.40** | 0.61 |
| leader margin at 2 s | **0.27** | 0.32 |
| junction time-to-conflict margin at 2 s (units of 5 s) | **0.23** | 0.27 |
| crash in the next step: AUROC (300 held-out crashes) | **0.954** | |
| red-light run: AUROC (75 held-out) | **0.9999** | |

**Driving** (`evaluate.py`). Oakland with humans 5× riskier than calibrated, demand ×1.2, 1.5× pedestrians, 5 %
cyclists, 15 % AVs. 4 copies × 5 min per driver, about 23 AV-hours each, all from the same warm state:

| driver | crashes /AV-h, no filter (AVs crashed) | crash events by type | with the barrier function | AV km/h | hard brakes /h |
|---|---|---|---|---|---|
| rule-based AV | 0 | | | 15.5 | 17.0 |
| PPO policy (234k steps) | 0.57 (13) | 10 rear-end, 2 pedestrian | 0 | 15.7 | 2.5 |
| **latent planner, PPO prior** | **0.26 (6)** | 3 rear-end, 1 angle, 1 left-turn | 0 | 15.2 | 4.8 |

- **Planning in the latent model halves the unaided crash rate of the policy it starts from**, at 3 % lower speed. The
  crashes it still has are spread across junction types, because it now makes the junction decisions itself.
- **Under the barrier function every driver has 0 crashes.** Without it, the planner and PPO both spend about 1 % of AV
  time outside the barrier's safe set; with it, 0.1 %.
- **Cost.** With the planner driving every AV, 4 copies of the area run at about 3× real time on 4 CPU cores, against
  about 75× with the policy alone.

```bash
python -m wmplan.data 300       # 800 k decisions (~1 min incl. warm-up)
python -m wmplan.model          # train the world model (~5 min)
python -m wmplan.evaluate       # the five drivers (~15 min; the two planner runs ~7 min each)
python -m wmplan.plot           # docs/img/wm_planner.png
python -m wmplan.viz            # the two GIFs
```

## Next

- **Amortise the planner into the policy** (Dreamer / TD-MPC style). Train the PPO actor on imagined rollouts with
  the planner's targets, so the deployed driver keeps the planner's foresight at the policy's speed.
- **Scene-level latent.** Encode the AV's neighbourhood as a set of agents, not 28 hand-built features, so the model
  can imagine the other drivers' choices: the merge partner who does not yield, the left turner who goes.
- **Plan against the barrier.** Put the barrier's constraint inside the search (reject plans whose decoded margins go
  negative), so the filter almost never has to step in.
