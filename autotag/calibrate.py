"""From scores to tags you can trust: isotonic calibration, a precision-guaranteeing threshold, and an abstain band.

A network's sigmoid output is a score, not a frequency. Isotonic regression (pool-adjacent-violators) finds the monotone map
  g = argmin_{g non-decreasing} sum_i (y_i - g(s_i))^2
on a held-out calibration set, so that among items given g(s) = 0.8, about 80 % are positive.

Threshold for a target precision: emit the tag when s >= t*, the smallest t whose one-sided 95 % lower confidence bound on
precision, computed on the calibration set (Wilson score), is at least the target. Then on new data from the same distribution
precision >= target with ~95 % confidence. Items with t_review <= s < t* (precision bound >= 0.5) go to a person; below that, no tag.
"""
import numpy as np


def isotonic_fit(s, y):
    """Pool-adjacent-violators. -> (knots_x, knots_y) for np.interp."""
    o = np.argsort(s)
    s, y = np.asarray(s, float)[o], np.asarray(y, float)[o]
    vals, wts, xs = [], [], []
    for si, yi in zip(s, y):
        vals.append(yi), wts.append(1.0), xs.append(si)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals[-2:], wts[-2:], xs[-2:] = [v], [w], [xs[-1]]
    # knots at block right ends
    return np.asarray(xs), np.asarray(vals)


def isotonic_apply(knots, s):
    kx, ky = knots
    return np.interp(np.asarray(s, float), kx, ky, left=ky[0], right=ky[-1])


def wilson_lower(k, n, z=1.645):
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (c - r) / den


def precision_threshold(s, y, target=0.9, min_n=10):
    """Smallest threshold whose Wilson lower bound on precision >= target. Returns np.inf if no threshold reaches it."""
    s, y = np.asarray(s, float), np.asarray(y, bool)
    o = np.argsort(-s)
    s, y = s[o], y[o]
    k = np.cumsum(y)
    best = np.inf
    for n in range(min_n, len(s) + 1):
        if n < len(s) and s[n - 1] == s[n]:
            continue                                   # thresholds sit between distinct scores
        if wilson_lower(k[n - 1], n) >= target:
            best = s[n - 1]
    return best


def ece(p, y, bins=10):
    """Expected calibration error: mean |frequency - confidence| over equal-width bins, weighted by bin size."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))


class Calibrator:
    def __init__(self, target=0.9, review=0.5):
        self.target, self.review = target, review

    def fit(self, s, y):
        self.knots = isotonic_fit(s, y)
        p = isotonic_apply(self.knots, s)
        self.t_emit = precision_threshold(p, y, self.target)
        self.t_review = precision_threshold(p, y, self.review)
        return self

    def __call__(self, s):
        return isotonic_apply(self.knots, s)

    def decide(self, s):
        """-> (calibrated probability, status: 'auto' | 'review' | '' (no tag))."""
        p = self(s)
        st = np.where(p >= self.t_emit, "auto", np.where(p >= self.t_review, "review", ""))
        return p, st

    def state(self):
        return {"knots_x": self.knots[0].tolist(), "knots_y": self.knots[1].tolist(), "t_emit": float(self.t_emit), "t_review": float(self.t_review),
                "target": self.target}

    @classmethod
    def from_state(cls, d):
        c = cls(d["target"])
        c.knots = (np.asarray(d["knots_x"]), np.asarray(d["knots_y"]))
        c.t_emit, c.t_review = d["t_emit"], d["t_review"]
        return c
