# Notebooks

Short guided notebooks, committed executed: every number and figure was produced by the cells. Each runs in a few minutes on a
laptop CPU. In Colab the first cell clones the repository and installs it, so they also run with no setup.

| notebook | what you do | CPU time | |
|---|---|---|---|
| [01 Start here](01_start_here.ipynb) | a tour: many copies of a city in one call, the map, an AV-share miniature, your own driver on AVBench | ~1 min | [Colab](https://colab.research.google.com/github/haggler3/av_world/blob/main/notebooks/01_start_here.ipynb) |
| [02 The RL environment](02_rl_environment.ipynb) | the 28 observation features, actions, the transition dict, three policies compared, custom reward weights | ~1 min | [Colab](https://colab.research.google.com/github/haggler3/av_world/blob/main/notebooks/02_rl_environment.ipynb) |
| [03 Training with PPO](03_ppo_training.ipynb) | a short `avrl.xppo` run, learning curves against the full run, a fine-tune with your own reward, AVBench scores | ~3 min | [Colab](https://colab.research.google.com/github/haggler3/av_world/blob/main/notebooks/03_ppo_training.ipynb) |
| [04 Perception export](04_perception_export.ipynb) | six cameras, 3-D boxes and a sensor model's detections for a famous-incident scenario; a dataset class | ~2 min | [Colab](https://colab.research.google.com/github/haggler3/av_world/blob/main/notebooks/04_perception_export.ipynb) |
| [05 AVBench and metrics](05_avbench_and_metrics.ipynb) | the leaderboard and the eight absolute checks, a pass/fail map, tuning a driver's headway | ~1 min | [Colab](https://colab.research.google.com/github/haggler3/av_world/blob/main/notebooks/05_avbench_and_metrics.ipynb) |
| [06 Easy mode](06_easy_mode.ipynb) | `av_world.easy` (simulate, write a driver in plain units, compare, sweep, train, save) and `av_world.compat` (Gymnasium + SB3, PettingZoo, PyTorch, ONNX, pandas) | ~5 min | [Colab](https://colab.research.google.com/github/haggler3/av_world/blob/main/notebooks/06_easy_mode.ipynb) |

Run locally: `pip install -e . jupyter` (and `pip install pyvista` for 04), then `jupyter lab notebooks/`. Outputs they write go
to `results/notebooks/`.
