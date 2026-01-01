"""The domain check: is this scene like anything the simulator makes? If not, the sim-trained tags on it are not to be trusted.

Every scene becomes a fixed-length feature vector f (rules.scene_features: density, speeds, acceleration spread and tails, pedestrians,
rates of each kinematic tag). Standardise with the simulated scenes' mean and SD, and score a scene by its distance to its k nearest
simulated scenes:
  d(f) = mean_{j in kNN(f)} || z(f) - z(f_j) ||,     z = (f - mu_sim) / sd_sim
The threshold is the 99th percentile of d over held-out simulated scenes (scored against the training ones), so about 1 % of
simulated scenes would be flagged: a scene above it is "sim_gap". Per-feature z-scores say which statistics put it there.
"""
import numpy as np

from .schema import Tag


class Domain:
    def __init__(self, k=5, quantile=0.99):
        self.k, self.q = k, quantile

    def fit(self, ref, held):
        self.mu, self.sd = ref.mean(0), ref.std(0) + 1e-6
        self.ref = (ref - self.mu) / self.sd
        self.threshold = float(np.quantile(self.score(held), self.q))
        return self

    def score(self, f):
        z = (np.asarray(f, float) - self.mu) / self.sd
        d = np.sqrt(((z[:, None, :] - self.ref[None]) ** 2).sum(-1))
        return np.sort(d, 1)[:, :self.k].mean(1)

    def explain(self, f, names):
        """The three statistics furthest from the simulated range, per scene."""
        z = (np.asarray(f, float) - self.mu) / self.sd
        return [[(names[j], float(z[i, j])) for j in np.argsort(-np.abs(z[i]))[:3]] for i in range(len(z))]

    def tags(self, scenes, f):
        s = self.score(f)
        return [Tag(sc["id"], sc["source"], "sim_gap", "scene", -1, 0.0, sc["T"] * sc["dt"], float(min(si / self.threshold, 9.99) / 10), "domain-knn")
                for sc, si in zip(scenes, s) if si > self.threshold], s


FEATURE_NAMES = ("vehicles within 30 m", "vehicles / 50", "median speed", "90th pct speed", "share stopped", "acceleration SD", "2nd pct accel",
                 "98th pct accel", "pedestrians / 10", "hard_brake rate", "tailgating rate", "low_ttc rate", "cut_in rate", "lane_change rate",
                 "turning rate", "queued rate", "ped_close rate")
