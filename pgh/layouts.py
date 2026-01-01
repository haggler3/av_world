"""Turn mined Pittsburgh intersections (data/pgh_intersections.json) into xworld geometries and layouts."""
import json

import numpy as np

from xworld import geometry as G
from xworld import sim as S

MPH = 0.44704


def load(path="data/pgh_intersections.json"):
    with open(path) as f:
        return json.load(f)["intersections"]


def build(it):
    spec = {"legs": [dict(slot=l["slot"], bearing=l["bearing"], lanes_in=l["lanes_in"], lanes_out=l["lanes_out"])
                     for l in it["legs"]], "control": it["control"]}
    geo = G.Geometry(spec)
    demand = np.zeros(4)
    vlim = np.full(4, 25 * MPH)
    for l in it["legs"]:
        demand[l["slot"]] = l["demand_vph"] / S.Behaviour().demand  # multiplier on Behaviour.demand (700 veh/h)
        vlim[l["slot"]] = l["speed_mph"] * MPH
    green = None
    if it["control"] == "signal":  # split 60 s of green between the two roads by demand
        d = {ax: demand[list(ax)].sum() for ax in ((0, 2), (1, 3))}
        tot = sum(d.values()) or 1.0
        green = {ax: float(np.clip(60 * v / tot, 15, 45)) for ax, v in d.items()}
    control = "signal" if it["control"] == "signal" else "stop"
    lay = S.make_layout(geo, control, tuple(it["major"]), green, demand, vlim)
    return geo, lay
