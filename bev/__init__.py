"""bev: a minimal BEVFormer-style camera perception pipeline on av_world's 3-D scenes (rendered multi-camera images ->
bird's-eye-view features -> 3-D boxes), small enough to train on a CPU. See docs/BEV.md for the front-to-back trace.

  rig.py        the 6-camera rig: intrinsics, extrinsics, ego<->camera<->image projections (nuScenes-style conventions)
  data.py       rendered samples from the simulator: images, ego poses, 3-D box labels, BEV targets
  backbone.py   image features: a small CNN from scratch, or BEVFormer-tiny's own ResNet-50 stem (weights read without torch)
  model.py      the BEV encoder (spatial cross-attention + temporal self-attention), the dense box head, loss and decoding
  metrics.py    nuScenes-style detection metrics (centre-distance AP, translation / scale / orientation errors)
  trace.py      one sample through every stage, with shapes, timings and figures
Nothing here is imported by the simulator; it needs pyvista + OSMesa only to render new data.
"""
