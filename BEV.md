# Camera BEV perception: a CPU-sized BEVFormer on av_world, traced front to back

`bev/` is a new, separate section: rendered multi-camera images from av_world's 3-D cities go in, 3-D boxes in a bird's-eye view
come out, with BEVFormer's encoder in the middle. Nothing in the simulator imports it, and it changes no existing behaviour (the one
touch outside `bev/` is `scene3d.Renderer` gaining `set_agents` / `shoot`, checked to render pixel-identical frames for every
existing caller). This page is the pipeline traced end to end, before any real training.

```bash
python -m bev.data new_york 0          # render + label: data/bev/new_york_s0.npz (needs pyvista + OSMesa, ~1.5 min per 150 samples)
python -m bev.backbone fetch           # optional: BEVFormer-tiny's ResNet-50 stem from the released checkpoint (400 MB download, 6 MB kept)
python -m bev.trace data/bev/new_york_s0.npz    # every stage of one sample: results/bev/trace.json, docs/img/bev_trace_*.png
python -m bev.train --smoke            # overfit one batch: proves the gradients reach every stage (not training)
```

## 1. How good is BEVFormer already

The reference is [BEVFormer](https://github.com/fundamentalvision/BEVFormer) (Li et al., ECCV 2022, Apache-2.0), camera-only 3-D
detection on nuScenes. Its released models (validation set, 24 epochs):

| model | backbone | NDS | mAP | GPU memory |
|---|---|---|---|---|
| BEVFormer-tiny | ResNet-50 | 35.4 | 25.2 | 6.5 GB |
| BEVFormer-small | ResNet-101-DCN | 47.9 | 37.0 | 10.5 GB |
| BEVFormer-base | ResNet-101-DCN | 51.7 | 41.6 | 28.5 GB |
| BEVFormerV2 (t8, 24 ep.) | ResNet-50 | 55.3 | 46.0 | 40 GB |

I downloaded BEVFormer-tiny's checkpoint and read it without PyTorch (`bev/torch_ckpt.py`: a pickle reader that maps torch's tensor
rebuilders to inert descriptors and refuses to run anything else from the file). Its real inventory
(`results/bev/bevformer_tiny_inventory.json`): **643 tensors, 33.6 M parameters**:

| part | parameters | |
|---|---|---|
| ResNet-50 image backbone | 23.6 M | 70 % |
| FPN neck | 1.1 M | |
| BEV encoder: 3 layers of temporal self-attention + spatial cross-attention + FFN | 2.0 M | 6 % |
| deformable-DETR decoder: 6 layers, 900 object queries | 4.1 M | |
| BEV queries (50 x 50 x 256) and object queries | 1.1 M | |
| class and box heads | 1.6 M | |

Its config: 800 x 450 inputs from 6 cameras, a 50 x 50 BEV over +-51.2 m, 256 channels, 4 pillar heights, 8 sampling points per
camera, a queue of 3 frames, 10 nuScenes classes. The lesson for a CPU version: **the BEV part is small; the image backbone and the
DETR decoder are the cost.**

## 2. What was built, and how it maps to BEVFormer

| | BEVFormer-tiny | `bev` MiniBEVFormer |
|---|---|---|
| input | 6 cameras, 800 x 450, nuScenes | 6 cameras in nuScenes' layout, 176 x 96, rendered from av_world (`bev/rig.py`) |
| backbone | ResNet-50 (+FPN), fine-tuned | **scratch**: 4-layer CNN, 49 k params; or **bevformer_r50**: BEVFormer-tiny's own stem + layer1 + layer2, frozen (1.44 M) |
| BEV grid | 50 x 50, 2.05 m cells, 256 ch | 32 x 32, 2 m cells (+-32 m), 64 ch |
| encoder | 3 x (TSA, SCA, FFN), 8 heads | 2 x (TSA, SCA, FFN), 4 heads, same mechanism |
| detection | DETR decoder, 900 queries, Hungarian matching | dense CenterPoint-style head on the BEV map |
| classes | 10 | 5: car, truck, bus, cyclist, pedestrian |
| parameters | 33.6 M | 0.32 M (scratch) / 0.33 M trainable + 1.44 M frozen |
| metric | nuScenes NDS / mAP | the same definitions (`bev/metrics.py`), with four TP errors instead of five (no attributes) |

**The ported backbone is exact.** `bev/backbone.py` folds BEVFormer-tiny's frozen batch norms into its convolutions and runs them
in JAX; against a PyTorch rebuild of the same layers loaded from the same checkpoint, the outputs agree to a relative error of
4.5e-6 (float rounding).

## 3. The pipeline, front to back

Every number below is from `python -m bev.trace` on one New York frame (`results/bev/trace.json`), on this 4-core CPU.

| stage | what happens | tensor | time |
|---|---|---|---|
| 1 sample | the simulator runs; a moving car in traffic carries the rig; 6 views rendered from the open-data city with the agents (the carrier hidden) | images [6, 96, 176, 3] | 65 ms render |
| 2 labels | every object within 36 m: box in the ego frame (x, y, z, l, w, h, yaw, vx, vy), class, *occluded* (another vehicle blocks the line of sight) | [20, 9] here: 18 cars, 2 cyclists, 13 occluded | |
| 3 calibration | ego2img = K [R \| t] per camera: BEVFormer's `lidar2img` | [6, 4, 4] | |
| 4 image features | backbone at stride 8 | [6, 12, 22, 64] scratch / [6, 12, 22, 512] R50 | 4 ms / 28 ms |
| 5 queries + pillars | one learned query per BEV cell; each cell lifted to points at 0.4, 1.0, 1.6, 2.6 m | [1024, 4, 3] | |
| 6 projection | pillar points through ego2img into every camera; a camera that receives any of a cell's points is one of its *hit views* | [6, 1024, 4, 2] | |
| 7 temporal alignment | the previous BEV warped by the ego motion into the current frame | [1024, 64] | |
| 8 encoder | 2 x (temporal self-attention over [queries, aligned previous BEV] -> spatial cross-attention into the hit views -> FFN) | BEV [1024, 64] | 36 ms |
| 9 head | per cell: 5 class logits, 10 box numbers | [32, 32, 5], [32, 32, 10] | |
| 10 targets + loss | Gaussian heatmaps at object centres (focal loss) + L1 on the box numbers at the centres | | |
| 11 decode + metrics | 3x3 peak picking -> boxes -> nuScenes-style mAP / NDS | | |
| 12 smoke fit | 80 steps on one batch of 4 frame pairs | | 0.5 s / step |

**Stages 1-3, 6** (`docs/img/bev_trace_cameras.png`): the labelled boxes, projected through the calibration, sit on the rendered
vehicles in every camera, so labels, geometry and pixels agree. The coloured dots are three BEV cells' pillar points: each lands in
the camera that faces it, stacked by height.

![cameras](img/bev_trace_cameras.png)

**Stages 2, 5, 6, 10** (`docs/img/bev_trace_bev.png`): the labels in the ego frame (x forward = up) with the cameras' fields of view;
how many cameras each cell is seen by; the heatmap targets. 72 % of cells are seen by one camera only and 27 % by two, so most of
the scene rests on a single view (nuScenes' overlaps are similarly narrow).

![bev](img/bev_trace_bev.png)

**Stage 4** (`docs/img/bev_trace_features.png`): the untrained scratch CNN, and BEVFormer-tiny's ResNet on these renders (top three
principal components of its 512 channels). Trained on real nuScenes photos, it still separates sky, facades and road here, so the
port carries some useful prior across the domain gap.

![features](img/bev_trace_features.png)

**Stages 7, 9, 12** (`docs/img/bev_trace_temporal_output.png`): warping the previous frame's labels by the ego motion raises their
correlation with the current labels from 0.72 to 0.90 (on a faster frame, 0.03 to 0.72); the rest is the other vehicles' own motion,
which temporal attention has to learn. Then the untrained output, and the output after the 80-step smoke fit on a batch containing
this frame: it has learned *where cars are likely* (the lane) before *where they are*, and reaches mAP 0.13 on that batch. That is
the plumbing working, not a result.

![temporal and output](img/bev_trace_temporal_output.png)

Checks that must hold and do (`tests/test_bev.py`): a point ahead projects to the image centre and every direction is seen; the ego
motion maps previous-frame points exactly onto current ones; labels sit in the ego frame with the drawn sizes; labels -> targets ->
decoding is exact (NDS 1.00); with no motion the warp is the identity; the metrics penalise a 1.5 m shift as they should; a few
steps lower the loss.

## 4. What it costs on a CPU

| step | measured |
|---|---|
| render 6 views | 65 ms per sample (176 x 96) |
| a data run (simulate, warm up, render, label) | 84 s for 150 samples, of which 10 s is rendering; the rest is the 200 s simulator warm-up, compilation and labelling, mostly paid once per run |
| BEVFormer R50 features (cached once) | 45 ms per sample |
| encoder + head, inference | ~36 ms per sample |
| a training step (batch 4, temporal: two encoder passes) | 0.50 s scratch, 0.44 s with cached R50 features |

So a first real experiment fits in an hour or two of CPU: about 6,000 samples from several cities and scenarios (roughly an hour of
rendering, run in ~10-minute jobs), then ~3,000 steps per backbone (about 25 minutes each).

## 5. What the numbers will and will not mean

* **Not comparable to nuScenes NDS 35.4.** The renders are untextured, the agents are proxy shapes, there is no sensor noise, blur
  or night, the range is 32 m instead of 51 m, and there are 5 classes. The *definitions* match, so the numbers are the same kind,
  but synthetic and real are different tests.
* **Class balance comes from the simulator.** One New York run gave 1,156 cars, 138 cyclists, 5 pedestrians and no buses; the
  dataset has to mix cities and scenarios (bus dart-outs, jaywalkers) for the rare classes, and report them separately.
* **Occlusion is common and tagged.** 13 of 20 objects in the traced frame are occluded; evaluation can be split by the tag, which
  is exactly where camera BEV models struggle.
* **The occlusion tag is approximate.** It is the sensor model's 2-D line-of-sight test against vehicle footprints, so a car
  directly behind a cyclist can be tagged occluded while visible over it.

## 6. Proposed training plan (waiting for the go-ahead)

1. Data: New York, Chicago and Oakland, everyday traffic plus the bus dart-out and jaywalker scenarios, ~6,000 samples; Boston held
   out entirely as the test city.
2. Train both backbones, 3,000 steps each, with and without temporal attention (4 runs).
3. Report nuScenes-style mAP / NDS on Boston, split by class, distance, and occluded vs visible, with the training curves.
4. If the frozen BEVFormer stem beats the scratch CNN, try more of BEVFormer: its layer3 and FPN at a higher resolution, which is
   where a GPU would start to pay.
