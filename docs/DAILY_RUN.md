# Daily research run: the prompt (paste into the scheduled task)

The routine's stored prompt cannot be edited from a session. This is the revised text. It replaces steps 1-3 of the old prompt; steps 0 and 4-6 (set up, clean up, publish, artifact) stay as they were.

**Focus.** Every day's work must advance one of three tracks. Anything else is dropped, however neat.

1. **World-model simulation features.** Make the simulator and its learned world models (`xnet/`, `dwm/`, `wmplan/`) more capable or more faithful: new simulated phenomena, better scene fidelity, learned dynamics, agents as tokens, counterfactual data, rollout quality, imagination-based planning and training.
2. **Frontier-paper reproduction and extension.** Read a recent paper (last ~12 months: world models, driving simulation, generative scene models, planning in latent space, e.g. DreamerV3-style, GAIA, Cosmos, Waymax/GPUDrive-class simulators), implement its core idea on av_world, and compare against the repo's existing baseline.
3. **Tools that make 1 and 2 faster**: only if a feature or paper needs them that day.

**Not allowed as a main idea:** audits of the simulator's own statistics or reproducibility, leaderboard bookkeeping, doc and README polish, one-off analyses with no new capability. (The 2026-11-02 and 2026-11-03 twin-city and seed-noise work is the kind of thing to stop doing.)

**Step 1, propose (12 ideas).** At least 8 from track 1 or 2. Each idea names the paper or the capability, the artefact (a GIF of the model's rollout, a figure against a baseline), the compute in minutes, and a prior.

**Step 2, search and filter.** Same PUCT-style search, but score "relevance" as: how much it adds to the world-model capability or a paper comparison. Do 2-3 real web searches for papers (not one) before choosing; write the paper citations into search.json. Keep the top 1-2 ideas. One substantial idea done well beats three small ones.

**Step 3, build.** Implement in `xnet/`, `dwm/`, `wmplan/` or a new module. New simulator behaviour is allowed when it is opt-in and off by default (existing runs stay identical), with a test showing that. Falsifiable prediction first; compare against the existing planner or world model, not against nothing; honest numbers with intervals. Budget: up to ~25 minutes of compute per job, 2 jobs at once, CPU only.

**Always:** check `docs/DAILY.md` and unmerged `claude/*` branches so a day's idea is not repeated; end by writing "what to try next" as concrete next features or papers.
