# What's next

Moved from the README.

## What's next

1. **Take mutating cities further.** It was the one addition that clearly helped, so push it: stronger mutations on a
   curriculum, hidden per-driver styles and hidden road friction that the policy must infer, and junction microworlds for
   more junction decisions per second. Fork and shaping stay available for longer runs where the plateau is lower.
2. **Pass AVBench's absolute tier unaided.** The best policy passes 123 of 144 checks. With no filter it still crashes 1.3-1.6x
   as often as the rule-based AV, and every policy scores under 6 on the perception-fault incidents. Train several seeds:
   one recipe's spread is about 10 av_world points.
3. **Give the world model what the policy lacks.** Planning (v2) and training the policy inside the model (v3) both stalled
   at the barrier-taught policy, because the model sees the same 28 features. Next: other road users as agent tokens, and
   counterfactual data forked at real near-crash moments ([wmplan/README.md](../wmplan/README.md)).
4. **Close the operating-domain gaps the lab found:** low friction (make the barrier's braking assumption adapt to
   the road), sensor noise (train with it), and Downtown's grid (train across neighbourhoods).
5. **Scale the simulation.** The power laws are fitted up to 62 k simulator steps (RL) and 18 k (world-model data).
   Next: 4-16× longer runs on a GPU. A single L4 already halves PPO time ([SPEED.md](SPEED.md)); the trainer's sharded GPU
   preset is still unmeasured.
6. **Close the loop with Rust.** Port the 28-feature observation to `small_av_world`, so the city-area policy runs in
   the Rust AV stack the way the corridor driver does.
