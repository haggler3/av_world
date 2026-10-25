"""nuScenes-style detection metrics (Caesar et al., 2020), so numbers here are the same kind as BEVFormer's (not the same difficulty).

  AP            per class and centre-distance threshold (0.5, 1, 2, 4 m): greedy matching by score to the nearest unmatched ground
                truth in the ground plane; area under the precision-recall curve above 10 % recall and 10 % precision
  mAP           the mean over classes and thresholds
  TP errors     at the 2 m threshold, over matched pairs: ATE (centre distance, m), ASE (1 - IoU of the aligned sizes), AOE (yaw
                error, rad), AVE (velocity error, m/s)
  NDS           (5 mAP + sum over the TP errors of (1 - min(1, error))) / 9. nuScenes also has AAE (attributes, which do not exist
                here), so its NDS divides by 10; ours uses the four TP errors that do.
"""
import numpy as np

DISTS = (0.5, 1.0, 2.0, 4.0)


def _match(preds, gts, c, dist):
    """-> (scores of all predictions of class c, tp flags, matched (pred, gt) pairs, number of gts)."""
    rows, n_gt = [], 0
    for (pb, ps, pc), (gb, gc) in zip(preds, gts):
        g = gb[gc == c]
        n_gt += len(g)
        used = np.zeros(len(g), bool)
        for i in np.argsort(-ps[pc == c]):
            b, s = pb[pc == c][i], ps[pc == c][i]
            if len(g):
                d = np.hypot(g[:, 0] - b[0], g[:, 1] - b[1])
                d[used] = np.inf
                j = int(np.argmin(d))
                if d[j] < dist:
                    used[j] = True
                    rows.append((s, True, b, g[j]))
                    continue
            rows.append((s, False, b, None))
    rows.sort(key=lambda r: -r[0])
    return rows, n_gt


def _ap(rows, n_gt, min_recall=0.1, min_precision=0.1):
    if n_gt == 0:
        return np.nan
    if not rows:
        return 0.0
    tp = np.cumsum([r[1] for r in rows])
    prec = tp / np.arange(1, len(rows) + 1)
    rec = tp / n_gt
    rs = np.linspace(0, 1, 101)
    p_at = np.array([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in rs])
    p_at = p_at[rs >= min_recall] - min_precision
    return float(np.clip(p_at, 0, None).mean() / (1 - min_precision))


def _tp_errors(rows):
    m = [(b, g) for _, ok, b, g in rows if ok]
    if not m:
        return dict(ATE=np.nan, ASE=np.nan, AOE=np.nan, AVE=np.nan)
    b, g = np.array([x[0] for x in m]), np.array([x[1] for x in m])
    ate = np.hypot(b[:, 0] - g[:, 0], b[:, 1] - g[:, 1]).mean()
    inter = np.prod(np.minimum(b[:, 3:6], g[:, 3:6]), 1)
    ase = (1 - inter / (np.prod(b[:, 3:6], 1) + np.prod(g[:, 3:6], 1) - inter)).mean()
    aoe = np.abs(np.remainder(b[:, 6] - g[:, 6] + np.pi, 2 * np.pi) - np.pi).mean()
    ave = np.hypot(b[:, 7] - g[:, 7], b[:, 8] - g[:, 8]).mean()
    return dict(ATE=float(ate), ASE=float(ase), AOE=float(aoe), AVE=float(ave))


def evaluate(preds, gts, classes):
    """preds: per sample (boxes [n, 9], scores [n], classes [n]); gts: per sample (boxes [m, 9], classes [m]) -> metrics dict."""
    per = {}
    for c, name in enumerate(classes):
        aps = []
        for d in DISTS:
            rows, n = _match(preds, gts, c, d)
            aps.append(_ap(rows, n))
            if d == 2.0:
                tpe = _tp_errors(rows)
        if not np.isnan(aps[0]):
            per[name] = {"AP": float(np.mean(aps)), **{f"AP@{d}": a for d, a in zip(DISTS, aps)}, **tpe, "n_gt": n}
    if not per:
        return {"mAP": np.nan, "NDS": np.nan, "per_class": {}}
    mAP = float(np.mean([v["AP"] for v in per.values()]))
    errs = {k: float(np.nanmean([v[k] for v in per.values()])) if any(not np.isnan(v[k]) for v in per.values()) else np.nan
            for k in ("ATE", "ASE", "AOE", "AVE")}
    nds = (5 * mAP + sum(1 - min(1.0, e) if not np.isnan(e) else 0.0 for e in errs.values())) / 9
    return {"mAP": mAP, "NDS": float(nds), **{"m" + k: v for k, v in errs.items()}, "per_class": per}
