"""Weak supervision: fuse several noisy taggers into one probabilistic tag without labels (Dawid-Skene, binary).

Each labelling function j votes v_ij in {1, 0} or abstains (-1) on item i. Assume votes are independent given the true label y_i and
that tagger j has a sensitivity a_j = P(v=1 | y=1) and specificity b_j = P(v=0 | y=0). EM:
  E   q_i = P(y_i = 1 | votes) = pi prod_j a_j^[v=1] (1-a_j)^[v=0]  /  (that + (1-pi) prod_j (1-b_j)^[v=1] b_j^[v=0])
  M   a_j = sum_i q_i [v_ij=1] / sum_i q_i [v_ij != -1],   b_j = sum_i (1-q_i) [v_ij=0] / sum_i (1-q_i) [v_ij != -1],   pi = mean q
Agreement between taggers is what identifies their accuracies: no ground truth is used. The simulator then checks the result.
"""
import numpy as np


def fit(votes, iters=100):
    """EM started from the majority vote (the standard initialisation: start from 'every tagger is better than chance')."""
    V = np.asarray(votes)
    q = majority(V)
    for _ in range(iters):
        on = V != -1
        a = np.clip(((V == 1) * q[:, None]).sum(0) / np.maximum((on * q[:, None]).sum(0), 1e-9), 0.02, 0.98)
        b = np.clip(((V == 0) * (1 - q)[:, None]).sum(0) / np.maximum((on * (1 - q)[:, None]).sum(0), 1e-9), 0.02, 0.98)
        pi = float(np.clip(q.mean(), 1e-4, 0.5))
        l1 = np.log(pi) + np.where(V == 1, np.log(a), 0).sum(1) + np.where(V == 0, np.log(1 - a), 0).sum(1)
        l0 = np.log(1 - pi) + np.where(V == 1, np.log(1 - b), 0).sum(1) + np.where(V == 0, np.log(b), 0).sum(1)
        q = 1 / (1 + np.exp(np.clip(l0 - l1, -50, 50)))
    return {"sensitivity": a, "specificity": b, "prior": pi}, q


def majority(votes):
    V = np.asarray(votes)
    on = (V != -1).sum(1)
    return np.where(on > 0, (V == 1).sum(1) / np.maximum(on, 1), 0.0)
