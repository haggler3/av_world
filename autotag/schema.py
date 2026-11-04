"""Tags: claims about a scene, one of its road users, or an interval of time, with a confidence and where they came from.

A tag is the unit everything else works on (search, fusion, calibration, review), so it carries its provenance: which tagger produced
it, which version of that tagger, and how confident it is. The store is a flat table (pandas DataFrame, saved as parquet, or JSON
lines if pyarrow is missing): one row per tag, so a million tags stay one file and every query is a vectorised filter.
"""
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

VERSION = "autotag-1"
COLUMNS = ("scene_id", "source", "name", "level", "agent", "t_start", "t_end", "confidence", "tagger", "version", "status")


@dataclass
class Tag:
    scene_id: str
    source: str                  # "av2" | "sim"
    name: str                    # from ONTOLOGY
    level: str                   # "scene" | "agent" | "interval"
    agent: int = -1              # vehicle index in the scene (-1: the scene)
    t_start: float = 0.0         # seconds from the scene start
    t_end: float = 0.0
    confidence: float = 1.0      # calibrated probability for learned tags; 1 for rules
    tagger: str = ""
    version: str = VERSION
    status: str = "auto"         # "auto" (emitted), "review" (abstained: goes to a person), "rejected", "confirmed"


ONTOLOGY = {
    # kinematic: computed from trajectories alone (rules.py)
    "hard_brake": ("interval", "kinematic", "speed drops by more than 4 m/s within 1 s from above 3 m/s"),
    "tailgating": ("interval", "kinematic", "time headway to the vehicle ahead under 0.8 s for at least 1 s, above 3 m/s"),
    "low_ttc": ("interval", "kinematic", "time to collision with the vehicle ahead under 1.5 s (closing faster than 0.5 m/s, gap < 25 m)"),
    "cut_in": ("interval", "kinematic", "a vehicle from the next lane becomes this vehicle's leader within 15 m"),
    "lane_change": ("interval", "kinematic", "lateral move of more than 2.5 m within 3 s while heading stays within 25 degrees"),
    "turning": ("interval", "kinematic", "heading changes by more than 60 degrees within 4 s"),
    "queued": ("interval", "kinematic", "stopped for at least 2 s behind a stopped vehicle within 10 m"),
    "ped_close": ("interval", "kinematic", "moving faster than 3 m/s within 4 m of a pedestrian"),
    "dense_traffic": ("scene", "kinematic", "more than 8 vehicles within 30 m of the ego on average"),
    # hidden: learned from simulator truth (learned.py)
    "distracted": ("interval", "hidden", "the driver is not reacting (a distraction episode), inferred from 5 s of driving"),
    "tailgater": ("agent", "hidden", "the driver's preferred headway is under 1.25 s (a driving style), inferred from 5 s of driving"),
    "crash_soon": ("interval", "risk", "this vehicle crashes within the next 3 s"),
    "hidden_ahead": ("interval", "hidden", "a road user hidden from the ego within 25 m ahead in its lane (occlusion net)"),
    # counterfactual and fused
    "action_sensitive": ("interval", "risk", "the AV's outcome in the next 4 s depends on its action (from exact forks; sim only)"),
    "risky_moment": ("interval", "risk", "fused: hard brake, low TTC, tailgating, distraction and crash risk combined by the label model"),
    # domain
    "sim_gap": ("scene", "domain", "the scene lies outside what the simulator produces: sim-trained tags are not to be trusted here"),
}


class TagStore:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def add(self, tag):
        self.rows.append(asdict(tag) if isinstance(tag, Tag) else dict(tag))

    def extend(self, tags):
        for t in tags:
            self.add(t)

    def frame(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=COLUMNS)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = self.frame()
        try:
            df.to_parquet(path)
        except Exception:
            path = os.path.splitext(path)[0] + ".jsonl"
            with open(path, "w") as f:
                for r in self.rows:
                    f.write(json.dumps(r, default=float) + "\n")
        return path

    @classmethod
    def load(cls, path):
        import pandas as pd
        if path.endswith(".jsonl"):
            return cls([json.loads(l) for l in open(path)])
        return cls(pd.read_parquet(path).to_dict("records"))

    def counts(self):
        df = self.frame()
        return df[df.status == "auto"].groupby(["source", "name"]).size().unstack(0).fillna(0).astype(int) if len(df) else df


def intervals(mask, dt, min_len=1):
    """Boolean [T] -> list of (t_start, t_end) seconds for runs of at least min_len frames."""
    m = np.asarray(mask, bool)
    out, start = [], None
    for t, on in enumerate(np.r_[m, False]):
        if on and start is None:
            start = t
        if not on and start is not None:
            if t - start >= min_len:
                out.append((start * dt, t * dt))
            start = None
    return out
