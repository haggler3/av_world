"""Idea 5 in the simulator: do human drivers that follow cars like the learned model reproduce the everyday tail of real traffic?

Four driver populations are run in the Oakland area and scored with the statistics `avlab.incidents.scene_metrics` measures on Argoverse 2:
  default      IDM humans (headway 1.0-1.8 s, no noise)
  fitted       IDM + the four tail knobs fitted by cross-entropy to those statistics (avlab.tail_calibrate)
  learned      car-following taken from the agent-token model (wm2, leader variant, trained on 136k real samples), nothing fitted to the statistics
  learned+fit  both
The learned model was trained to predict real accelerations, not to match these four rates, so it is the unfitted comparison.

    python -m avlab.learned_humans            # ~10 min on 4 cores -> results/extras/learned_humans.json
"""
import json
import os
import pickle
import sys

from xnet import sim as X

from . import adversary as AD
from . import tail_calibrate as TC

OUT = "results/extras"


def main(reps=3, params="results/wm2/params_leader.pkl"):
    os.makedirs(OUT, exist_ok=True)
    X.set_learned_follower(pickle.load(open(params, "rb")))
    fitted = json.load(open("results/tail/calibrated_drivers.json"))["knobs"]
    real = json.load(open(TC.REAL))
    rec = TC.Recorder()
    worlds = {"default": AD.world(1.0), "fitted": AD.world(1.0, **fitted),
              "learned": AD.world(1.0)._replace(learned=1.0), "learned+fit": AD.world(1.0, **fitted)._replace(learned=1.0)}
    res = {"real": {k: real[k] for k in TC.STATS}, "worlds": {}}
    for name, beh in worlds.items():
        st = rec.stats(beh, reps=reps)
        res["worlds"][name] = {**{k: float(st[k]) for k in TC.STATS}, "loss": TC.loss(st, real)}
        print(name, {k: round(v, 5) for k, v in res["worlds"][name].items()}, flush=True)
    with open(os.path.join(OUT, "learned_humans.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:2]))
