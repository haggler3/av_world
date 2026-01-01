| run | trainable params | s / step | val mAP | val NDS | blind mAP (NDS) | train-city mAP | 0-16 m mAP | 16-32 m mAP | recall@2m visible / occluded |
|---|---|---|---|---|---|---|---|---|---|
| bevformer_r50 | 273,423 | 0.47 | **0.144** | 0.179 | 0.000 (0.084) | 0.360 | 0.312 | 0.080 | 0.56 / 0.27 |
| scratch | 322,623 | 0.51 | **0.134** | 0.174 | 0.001 (0.161) | 0.267 | 0.242 | 0.090 | 0.54 / 0.25 |

Per class (val AP, mean over 0.5 / 1 / 2 / 4 m; n = ground-truth objects):

| run | car | truck | bus | cyclist | pedestrian | mATE | mASE | mAOE | mAVE |
|---|---|---|---|---|---|---|---|---|---|
| bevformer_r50 | 0.376 (n=4244) | 0.167 (n=624) | 0.166 (n=235) | 0.004 (n=514) | 0.008 (n=251) | 1.01 | 0.33 | 0.77 | 1.28 |
| scratch | 0.400 (n=4244) | 0.118 (n=624) | 0.150 (n=235) | 0.000 (n=514) | 0.000 (n=251) | 1.05 | 0.32 | 0.78 | 1.11 |
