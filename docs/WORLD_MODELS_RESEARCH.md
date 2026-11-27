# World models for driving: a research map (2026-11-27)

This page maps world-model research that bears on av_world and says where the repo's own models sit in it. It does not
repeat the experiments: those live in [`WORLD_MODELS.md`](WORLD_MODELS.md) and [`../wmplan/README.md`](../wmplan/README.md).
The simulator comparison (Waymax, GPUDrive, Gigaflow) is in [`RESEARCH_CONTEXT.md`](RESEARCH_CONTEXT.md).

A *world model* here means a learned model that predicts what comes next given what has happened and what the agent does.
Three families matter for this repo:

* **Agent world models**: learn the environment so an agent can plan or train inside it (Dreamer, MuZero, TD-MPC2).
* **Generative sensor models**: produce camera video or 3D occupancy of a driving scene (GAIA, Vista, Cosmos).
* **Sim agents**: generate the joint motion of every road user, run in closed loop (TrafficSim, SMART, WOSAC entries).

av_world works on object states, not pixels, so the third family is the closest match and the first gives the planning ideas.
Every citation below was checked against its arXiv page or the lab's official page on 2026-11-27.

## 1. Foundations

| paper | year | one-line idea | why it matters here |
|---|---|---|---|
| [World Models](https://arxiv.org/abs/1803.10122) (Ha & Schmidhuber) | 2018 | A VAE plus an RNN learns a car-racing game; a tiny controller trains inside the "dream". | The template for `avrl/worldmodel.py`: learn the env, then train in it. |
| [PlaNet](https://arxiv.org/abs/1811.04551) (Hafner et al.) | 2018 | A latent model with deterministic and stochastic parts; plans with the cross-entropy method in latent space. | `wmplan` v1 and v2 plan exactly this way: CEM over latent rollouts. |
| [Dreamer](https://arxiv.org/abs/1912.01603) (Hafner et al.) | 2019 | Learn the policy from imagined latent rollouts by backpropagating values. | `wmplan/v3.py` trains the actor inside the model, after Dreamer. |
| [DreamerV2](https://arxiv.org/abs/2010.02193) (Hafner et al.) | 2020 | Discrete (categorical) latents; human-level Atari from imagination alone. | Discrete latents are one option for the multimodal "go / wait" choice at junctions. |
| [DreamerV3](https://arxiv.org/abs/2301.04104) (Hafner et al.) | 2023 | One set of hyperparameters across 150+ tasks; larger models learn faster. | Its normalisation tricks (symlog, return scaling) are what a fixed-config driving agent needs. |
| [MuZero](https://arxiv.org/abs/1911.08265) (Schrittwieser et al.) | 2019 | Learn only what planning needs: reward, value and policy; search with MCTS. | `wmplan` decodes reward and event risks rather than full scenes; MuZero is the extreme of that. |
| [IRIS](https://arxiv.org/abs/2209.00588) (Micheli et al.) | 2022 | A discrete autoencoder plus an autoregressive transformer as the world model. | Tokens plus a transformer is the same recipe the sim-agent models in section 3 use. |
| [TD-MPC2](https://arxiv.org/abs/2310.16828) (Hansen et al.) | 2023 | Decoder-free latent model, MPPI planning, SimNorm latents; one 317M agent for 80 tasks. | `wmplan/v2.py` already uses a SimNorm latent and a consistency loss from this line of work. |
| [DIAMOND](https://arxiv.org/abs/2405.12399) (Alonso et al.) | 2024 | An RL agent trained inside a diffusion world model; visual detail matters. | The pixel-space twin of `dwm/`, which runs diffusion on traffic states. |
| [GameNGen](https://arxiv.org/abs/2408.14837) (Valevski et al., Google) | 2024 | A diffusion model runs DOOM in real time from past frames and actions. | Shows action-conditioned diffusion can stay stable over long closed-loop rollouts. |
| [JEPA position paper](https://openreview.net/forum?id=BZ5a1r-kVsf) (LeCun) | 2022 | Predict in representation space and ignore unpredictable detail. | Argues for latent targets, as `wmplan`'s consistency loss does. |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) (Assran et al., Meta FAIR) | 2025 | Self-supervised video pretraining, then a small action-conditioned latent model plans robot actions. | Evidence that latent prediction alone is enough to plan; no decoder needed. |
| [Genie](https://arxiv.org/abs/2402.15391) (Bruce et al., DeepMind) | 2024 | Learns a playable world and latent actions from unlabelled video. | Latent actions could label human driving logs that lack controls. |
| [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) (DeepMind blog) | 2024 | Autoregressive latent diffusion; playable 3D worlds from one image, up to a minute. | Scale reference only; far beyond a CPU budget. |
| [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) (DeepMind blog) | 2025 | Real-time interactive worlds at 24 fps for minutes; text-promptable world events. | "Promptable events" match av_world's scripted hazards (red runners, breakdowns). |

## 2. Driving world models (video, sensors, occupancy)

| paper | year | one-line idea | why it matters here |
|---|---|---|---|
| [GAIA-1](https://arxiv.org/abs/2309.17080) (Hu et al., Wayve) | 2023 | Video, text and action tokens into an autoregressive model; a diffusion decoder makes video. | First large driving world model; frames world modelling as next-token prediction. |
| [GAIA-2](https://arxiv.org/abs/2503.20523) (Russell et al., Wayve) | 2025 | Latent diffusion over five camera streams, conditioned on ego motion, agents and weather. | Conditioning on structured agent states is the bridge from av_world's states to sensors. |
| [GAIA-3](https://wayve.ai/thinking/gaia-3/) (Wayve) | 2025 | 15B latent diffusion model aimed at offline evaluation and safety-critical scenario generation. | Industry is moving world models from data generation to evaluation, as `wmplan/converge.py` does. |
| [DriveDreamer](https://arxiv.org/abs/2309.09777) (Wang et al.) | 2023 | Diffusion world model trained on real nuScenes driving; predicts video and actions. | Two-stage training (structure first, then future) mirrors `dwm`'s physics residual idea. |
| [Vista](https://arxiv.org/abs/2405.17398) (Gao et al., OpenDriveLab) | 2024 | High-resolution driving video; controllable by commands, goal points or trajectories; also a reward. | Using the world model as a reward function is open for `dwm`. |
| [OccWorld](https://arxiv.org/abs/2311.16038) (Zheng et al.) | 2023 | Tokenise 3D semantic occupancy; a GPT-style model predicts future scenes and the ego path. | Occupancy is the middle ground between av_world's object lists and pixels. |
| [Cosmos WFM platform](https://arxiv.org/abs/2501.03575) (NVIDIA) | 2025 | Open-weight diffusion and autoregressive world foundation models plus tokenizers. | The open model to fine-tune if av_world ever renders its scenes to video. |
| [Cosmos-Drive-Dreams](https://arxiv.org/abs/2506.09042) (Ren et al., NVIDIA) | 2025 | Cosmos specialised to driving; turns HD-map and box labels into multi-view synthetic video. | av_world already exports maps and boxes (`xnet/nvidia.py`, `bev/`), the inputs this needs. |

## 3. Multi-agent traffic simulation (closest to av_world)

The task: given a short history and the map, roll out every agent for several seconds, step by step, so the joint
distribution of futures matches real traffic. av_world's human drivers are hand-written and calibrated; this line of work
learns them.

| paper | year | one-line idea | why it matters here |
|---|---|---|---|
| [TrafficSim](https://arxiv.org/abs/2101.06557) (Suo et al., Uber ATG) | 2021 | Implicit latent-variable model plans all actors jointly; trained through the simulator. | First learned multi-agent simulator trained with closed-loop rollouts. |
| [WOSAC](https://arxiv.org/abs/2305.12032) (Montali et al., Waymo) | 2023 | Public benchmark: 32 joint 8 s rollouts per scene, scored by a realism meta-metric. | The yardstick `dwm/simagents.py` copies. |
| [MotionLM](https://arxiv.org/abs/2309.16534) (Seff et al., Waymo) | 2023 | Discretise motion into tokens; joint multi-agent forecasting as language modelling. | Same token idea as the new sim-agents model, which tokenises acceleration. |
| [SMART](https://arxiv.org/abs/2405.15677) (Wu et al.) | 2024 | Map and trajectory tokens, decoder-only transformer, next-token prediction; under 15 ms per step. | Showed next-token sim agents lead WOSAC and transfer zero-shot from nuPlan. |
| [BehaviorGPT](https://arxiv.org/abs/2405.17372) (Zhou et al.) | 2024 | Fully autoregressive transformer predicting patches of steps; 3M parameters. | Won WOSAC 2024 (0.7473) at a size a CPU can train. |
| [CAT-K](https://arxiv.org/abs/2412.05334) (Zhang et al., NVIDIA) | 2024 | Closed-loop fine-tuning: roll out, pick the top-K token closest to the log, supervise. | Cheap fix for covariate shift; a 7M model beat a 102M one. Fits a CPU. |
| [TrajTok](https://arxiv.org/abs/2506.21618) (Zhang et al., SJTU) | 2025 | A better trajectory tokenizer than k-disks for SMART; realism 0.7852 on WOSAC 2025. | The tokenizer choice moves the score; relevant to the acceleration bins in `simagents.py`. |
| [SMART-R1](https://arxiv.org/abs/2509.23993) (Pei et al., HKUST) | 2025 | R1-style RL fine-tuning of SMART with a realism reward; 0.7858 on WOSAC 2025. | Next step after CAT-K: reward the metric directly. ICLR 2026. |

WOSAC 2025 is the latest formal round; Waymo is not hosting challenges in 2026 but the leaderboards stay open
([challenge page](https://waymo.com/open/challenges/2025/sim-agents)). The 2025 metrics add a traffic-light violation term and
filter the time-to-collision term to vehicles. The leaders above report their own ranks; check the leaderboard before quoting a
winner.

## 4. Evaluating world models

| idea | source | what it says | in av_world |
|---|---|---|---|
| Proper scoring, CRPS | [Gneiting & Raftery, JASA 2007](https://apps.dtic.mil/sti/html/tr/ADA459827/index.html) | A strictly proper score is minimised only by the true distribution; CRPS is one for real-valued outcomes. | `dwm` reports CRPS skill vs physics and 90 % band coverage. |
| Realism meta-metric | [WOSAC](https://arxiv.org/abs/2305.12032) | Fit a histogram per scene from the 32 rollouts, score the log's likelihood, weight kinematic, interactive and map-based terms. | `dwm/simagents.py` (added 2026-11-27), kinematic and gap terms so far. |
| Collision and offroad rates | [WOSAC](https://arxiv.org/abs/2305.12032) | Report the share of agents that collide or leave the road; a realistic model must not be safer or riskier than the log. | The simulator counts crashes by type; `avlab/split.py` estimates rare crash rates. |
| Closed loop over open loop | [Dauner et al., CoRL 2023](https://arxiv.org/abs/2306.07962) | Open- and closed-loop scores disagree; an IDM-based planner beat learned ones on nuPlan. | `dwm/benchmark.py` is open loop; `wmplan/evaluate.py` and `converge.py` are closed loop. |
| Shortcut check | [Li et al., CVPR 2024](https://arxiv.org/abs/2312.03031) | On nuScenes, ego status alone scores well in open loop, so open-loop metrics can be gamed. | Compare every world model with the kinematic baseline, as `dwm` already does. |
| Scaling and closed loop | [Baniodeh et al., Waymo 2025](https://arxiv.org/abs/2506.08228) | Loss falls as a power law in compute; closed-loop metrics improve with scale too. | `dwm/scaling.py` fits the same power laws on `xnet` data. |

Three rules follow for this repo. Score distributions with a proper rule, never a best-of-K distance alone. Report
calibration (coverage) next to accuracy. Judge any model that will drive or be driven against in closed loop.

## 5. Where av_world stands

| line of work | what exists | files | gap |
|---|---|---|---|
| Dream env (Ha & Schmidhuber) | One-step 3-member ensemble MLP in the policy's 22 features; works as a sim-to-world gap meter. | `avrl/worldmodel.py`, `avrl/viz_wm.py` | No history, no multi-step loss; too inaccurate at 3 s to train in. |
| Diffusion / flow world models (DIAMOND, GameNGen, GAIA-2) | Action-conditioned 3 s futures of the AV and its leader; physics residual; DCT and control-point bases; flow matching. | `dwm/model.py`, `dwm/train.py`, `dwm/basis.py`, `dwm/explore.py`, `dwm/gifs2d.py` | Single agent, open loop; over-confident at short budgets; fails on a new driving policy. |
| Scaling laws (Waymo 2025, DreamerV3) | Error vs compute and vs data, four architectures, on `xnet`. | `dwm/scaling.py`, `dwm/scaling_plot.py` | No closed-loop metric in the fits. |
| Latent planning (PlaNet, TD-MPC2) | Encoder, latent dynamics, reward and event heads; CEM planning; ensemble and SimNorm in v2. | `wmplan/model.py`, `wmplan/planner.py`, `wmplan/v2.py` | Sees the same 28 features as the policy, so lookahead is its only edge. |
| Learning in imagination (Dreamer) | Actor-critic trained inside the v2 model, with counterfactual forks and a Lagrangian cost. | `wmplan/v3.py` | The actor exploited the model. |
| Value-only use (MuZero) | The crash head as a shaping potential; fork-and-replay crash aversion. | `wmplan/risk.py`, `wmplan/converge.py` | Not yet tested as a planner value. |
| Latent prediction (JEPA) | Consistency loss toward the encoding of the real next scene. | `wmplan/model.py` | Decoders still carry most of the loss. |
| Sim agents (SMART, MotionLM, WOSAC) | **Added 2026-11-27:** next-token model (MLP over 28 features, 32 acceleration bins) for every human vehicle, run closed loop inside the simulator; WOSAC-style meta-metric over speed, acceleration, jerk, gap and headway; compared with IDM and with a second seed of the calibrated humans (the ceiling). | `dwm/simagents.py` | Trained open loop only; meta-metric has kinematic and gap terms but no collision, TTC or red-light terms; no CAT-K or RL fine-tuning yet. |
| Sensor world models (GAIA, Vista, Cosmos, OccWorld) | None. The repo renders camera images for perception, not for prediction. | `bev/`, `xnet/scene3d.py` | Out of CPU scope; the export path to Cosmos-style inputs exists. |
| Evaluation | CRPS skill, coverage, roughness; closed-loop AVBench; crash rates by type; splitting estimator. | `dwm/benchmark.py`, `wmplan/evaluate.py`, `avlab/split.py` | No single realism score for the hand-written humans to compare with the learned ones. |

The repo's position is unusual. Most sim-agent work learns from logs and has no ground-truth simulator to check against.
av_world has a calibrated simulator, so it can measure a learned model's realism against a known truth, and it can measure
the hand-written humans with the same meta-metric a learned model gets.

## 6. Roadmap: five concrete next steps

Each step runs on 4 CPU cores in under an hour, uses existing data paths, and has a pass mark set in advance.

1. **CAT-K closed-loop fine-tuning of the sim agents** (follows [CAT-K](https://arxiv.org/abs/2412.05334)).
   Roll `dwm/simagents.py` out in the simulator; at each step keep the top 5 acceleration tokens and take the one whose next
   state is closest to the simulator's real human. Fine-tune on those rollouts for 15 minutes. Pass: the realism meta-metric
   rises and the collision rate moves toward the simulator's, on held-out seeds.

2. **Add the interactive and map-based terms to the meta-metric** (follows [WOSAC](https://arxiv.org/abs/2305.12032) and its
   2025 update). `dwm/simagents.py` scores only kinematic and gap features. Add per-vehicle collision, time-to-collision and
   red-light-violation likelihoods, weighted as WOSAC weights its groups, and rescore the existing `gt2`, `idm` and `sample`
   cells. No retraining; under 15 minutes. Pass: the `gt2` ceiling stays near 1 and the IDM and learned cells separate more
   clearly than on kinematics alone.

3. **Recalibrate the flow-matching DCT model** (follows [Gneiting & Raftery](https://apps.dtic.mil/sti/html/tr/ADA459827/index.html)).
   Its 90 % band covers 74 %. Fit one noise-scale temperature on a validation split to maximise CRPS, then report the PIT
   histogram and coverage on the test split. Pass: coverage within 88 to 92 % with no loss in CRPS skill.

4. **Train `dwm` on several drivers' data** (follows [Baniodeh et al.](https://arxiv.org/abs/2506.08228), which tests data
   from other agents). Mix learned-driver and planner-stack windows in `dwm.data`, retrain the DCT flow model for 4,000 steps,
   and rerun `python -m dwm.benchmark`. Pass: skill below 1.0 (beats physics) on the planner-stack trips, where every purely
   learned model now scores 1.9 to 2.8.

5. **Next-patch prediction for the sim agents** (follows [BehaviorGPT](https://arxiv.org/abs/2405.17372)).
   Predict 5-step patches of acceleration tokens instead of one token per step. Compare with the one-token model at equal
   training time on the meta-metric and on simulator steps per second. Pass: equal or better realism at no more than half the
   rollout cost.

After these five, the larger options are RL fine-tuning on the meta-metric ([SMART-R1](https://arxiv.org/abs/2509.23993)) and
giving `wmplan` the other road users as objects, which the `wmplan` README names as the step that can change its results.
