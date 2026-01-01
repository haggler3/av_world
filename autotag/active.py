"""Active learning: spend a reviewer's time where it teaches the tagger most.

The simulator plays the reviewer (it knows the truth), so strategies can be compared before a person is involved. Each round the
tagger's head (a logistic regression on the network's logit and 12 summary features of the window) is refitted on the items reviewed so
far, and `batch` more are chosen from the pool:
  random        uniform
  uncertainty   the items whose predicted probability is closest to the decision threshold (here 0.5 after calibration)
  unc+div       the most uncertain 5 x batch, then k-centre greedy in feature space for diversity (no two near-duplicates)
Reported: held-out AUROC against the number of reviewed items.
"""
import numpy as np

from nets import common as CM


def summary(X_):
    return np.concatenate([X_.mean(1), X_.min(1)[:, :2], X_.max(1)[:, :2]], 1)


def logistic(Z, y, iters=300, lr=0.5, l2=1e-3):
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def kcenter(Z, k, rng):
    sel = [int(rng.integers(len(Z)))]
    d = np.linalg.norm(Z - Z[sel[0]], axis=1)
    while len(sel) < k:
        j = int(np.argmax(d))
        sel.append(j)
        d = np.minimum(d, np.linalg.norm(Z - Z[j], axis=1))
    return np.array(sel)


def run(X_, logit, y, X_test, logit_test, y_test, rounds=10, batch=100, seed=0):
    feats = lambda X, lg: np.concatenate([lg[:, None], summary(X)], 1)
    Zp, Zt = feats(X_, logit), feats(X_test, logit_test)
    mu, sd = Zp.mean(0), Zp.std(0) + 1e-6
    Zp, Zt = (Zp - mu) / sd, (Zt - mu) / sd
    out = {}
    for strat in ("random", "uncertainty", "unc+div"):
        rng = np.random.default_rng(seed)
        lab = list(rng.choice(len(Zp), batch, replace=False))
        curve = []
        for r in range(rounds):
            yl = y[lab]
            if yl.min() == yl.max():
                curve.append((len(lab), 0.5))
            else:
                w, b = logistic(Zp[lab], yl)
                curve.append((len(lab), CM.auroc(Zt @ w + b, y_test)))
            rest = np.setdiff1d(np.arange(len(Zp)), lab)
            if strat == "random" or yl.min() == yl.max():
                new = rng.choice(rest, batch, replace=False)
            else:
                p = 1 / (1 + np.exp(-(Zp[rest] @ w + b)))
                order = rest[np.argsort(np.abs(p - 0.5))]
                if strat == "uncertainty":
                    new = order[:batch]
                else:
                    cand = order[:5 * batch]
                    new = cand[kcenter(Zp[cand], batch, rng)]
            lab += list(new)
        out[strat] = curve
    return out
