# What's next

Moved from the README.

## What's next

1. **Take mutating cities further.** It was the one addition that clearly helped, so push it: stronger mutations on a
   curriculum, hidden per-driver styles and hidden road friction that the policy must infer, and junction microworlds for
   more junction decisions per second. Fork and shaping stay available for longer runs where the plateau is lower.
2. **Finish passing AVBench unaided.** The barrier-taught policy is 2 % short on speed and has one or two unaided
   crashes where the rule-based AV has none. Next: distil the latent planner's junction timing into it, and train on
   the scenarios it still fails. Target: all 88 checks.
3. **Amortise the planner.** Train the policy on the planner's choices in imagined rollouts (Dreamer / TD-MPC style),
   so the deployed driver keeps the planner's foresight at the policy's speed.
4. **Close the operating-domain gaps the lab found:** low friction (make the barrier's braking assumption adapt to
   the road), sensor noise (train with it), and Downtown's grid (train across neighbourhoods).
5. **Scale the simulation.** The power laws are fitted up to 62 k simulator steps (RL) and 18 k (world-model data).
   Next: 4-16× longer runs on a GPU, where JAX should push `xnet` well past 380× real time.
6. **Close the loop with Rust.** Port the 28-feature observation to `small_av_world`, so the city-area policy runs in
   the Rust AV stack the way the corridor driver does.
