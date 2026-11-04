"""The auto-tagging pipeline as a DAG of cached stages (autotag.dag). `run()` builds everything; `explain()` shows what is fresh.

  sim_scenes ──┬─ rules_sim ──────────────┬─ domain ──────────┐
               ├─ windows_sim ─ learned_model ─ calibration ──┤
               │                                │             ├─ tags ─ store ─ qa ─ report
  av2_scenes ──┼─ rules_av2 ──────────────┐     │             │
               ├─ windows_av2 ────────────┴─ learned_scores ──┤
               └─ hidden_ahead ───────────────────────────────┤
  forks (nets) ── counterfactual ─────────────────────────────┤
  learned_scores + rules ── label_model ──────────────────────┤
  windows_sim + learned_model ── active ──────────────────────┘
"""
import json
import os

import numpy as np

from nets import common as CM

from . import calibrate as CA
from . import domain as DO
from . import labelmodel as LM
from . import learned as LE
from . import rules as RU
from . import scenes as SC
from .dag import stage
from .schema import ONTOLOGY, Tag, TagStore, intervals

OUT = "results/autotag"
SPLIT = {"train": (0, 5), "calib": (5, 6), "test": (6, 8)}             # copies of the simulated recording


def part(env, name):
    a, b = SPLIT[name]
    return (env >= a) & (env < b)


# ---------------------------------------------------------------- sources
@stage("sim_scenes", config={"minutes": 8, "n_envs": 8, "seed": 0})
def sim_scenes(minutes, n_envs, seed):
    """The simulated golden set: 10 s scenes around AVs, non-overlapping, with every hidden truth (tagging, QA, domain)."""
    rec, w = SC.record_sim(minutes=minutes, n_envs=n_envs, seed=seed)
    return SC.sim_scenes(rec, w, seed=seed)


@stage("sim_train_scenes", config={"minutes": 8, "n_envs": 8, "seed": 1, "per_window": 12, "stride": 25})
def sim_train_scenes(minutes, n_envs, seed, per_window, stride):
    """A separate, denser recording to train and calibrate the learned taggers on (overlapping scenes, more AVs per window), so the
    golden set above stays independent of training."""
    rec, w = SC.record_sim(minutes=minutes, n_envs=n_envs, seed=seed)
    return SC.sim_scenes(rec, w, per_window=per_window, stride=stride, seed=seed)


@stage("av2_scenes", config={"limit": 400})
def av2_scenes(limit):
    return SC.av2_scenes(limit=limit)


# ---------------------------------------------------------------- rules
@stage("rule_tuning", deps=("sim_scenes",))
def rule_tuning(scenes):
    """Tune the lane-change and cut-in thresholds on the golden set's tuning copies (0-5) by F1 against simulator truth; QA reports
    copies 6-7, which the search never saw."""
    env = np.array([s["truth"]["env"] for s in scenes])
    tune = [s for s, e in zip(scenes, env) if e < SPLIT["test"][0]]
    best, log = dict(RU.DEFAULTS), []
    f1 = lambda r: 2 * r["precision"] * r["recall"] / max(r["precision"] + r["recall"], 1e-9)
    for hd in (25.0, 12.0, 8.0, 5.0):
        for curve in (90.0, 30.0, 15.0):
            for lat in (2.5, 3.0):
                P = dict(best, lc_heading_deg=hd, lc_curve_deg=curve, lc_lateral=lat)
                r = _rule_qa(tune, [RU.tag_scene(s, params=P) for s in tune], "lane_change", lambda s: _truth_events(s, "lane_change"))
                log.append(("lane_change", P, r, f1(r)))
    lc_best = max((l for l in log if l[0] == "lane_change"), key=lambda l: l[3])[1]
    best.update({k: lc_best[k] for k in ("lc_heading_deg", "lc_curve_deg", "lc_lateral")})
    for hd in (90.0, 20.0, 10.0):
        for straight in (90.0, 15.0, 8.0):
            P = dict(best, ci_heading_deg=hd, ci_straight_deg=straight)
            r = _rule_qa(tune, [RU.tag_scene(s, params=P) for s in tune], "cut_in", _truth_cut_in)
            log.append(("cut_in", P, r, f1(r)))
    ci_best = max((l for l in log if l[0] == "cut_in"), key=lambda l: l[3])[1]
    best.update({k: ci_best[k] for k in ("ci_heading_deg", "ci_straight_deg")})
    return {"params": best, "log": [(n, {k: P[k] for k in P}, r, f) for n, P, r, f in log]}


@stage("rules_sim", deps=("sim_scenes", "rule_tuning"))
def rules_sim(scenes, tuning):
    return [RU.tag_scene(s, params=tuning["params"]) for s in scenes]


@stage("rules_av2", deps=("av2_scenes", "rule_tuning"))
def rules_av2(scenes, tuning):
    return [RU.tag_scene(s, params=tuning["params"]) for s in scenes]


# ---------------------------------------------------------------- learned
def _windows(scenes, labels):
    xs, ys, meta = [], [], []
    for k, s in enumerate(scenes):
        X_, Y_, m = LE.windows(s, labels)
        if len(X_):
            xs.append(X_)
            ys.append(Y_ if Y_ is not None else np.zeros((len(X_), len(LE.TAGS)), np.float32))
            env = s["truth"]["env"] if s.get("truth") else -1
            meta += [(k, i, t, env) for i, t in m]
    return {"X": np.concatenate(xs), "Y": np.concatenate(ys), "meta": np.asarray(meta)}


@stage("windows_sim", deps=("sim_scenes",))
def windows_sim(scenes):
    return _windows(scenes, True)


@stage("windows_train", deps=("sim_train_scenes",))
def windows_train(scenes):
    return _windows(scenes, True)


@stage("windows_av2", deps=("av2_scenes",))
def windows_av2(scenes):
    return _windows(scenes, False)


@stage("learned_model", deps=("windows_train",), config={"steps": 4000})
def learned_model(w, steps):
    tr = w["meta"][:, 3] < 6                                                # copies 6-7 of the training recording calibrate
    p, curve = LE.train(w["X"][tr], w["Y"][tr], steps=steps)
    import jax
    return {"params": jax.tree.map(np.asarray, p), "curve": curve[::30].tolist()}


@stage("learned_scores", deps=("learned_model", "windows_sim", "windows_av2", "windows_train"))
def learned_scores(model, ws, wa, wt):
    import jax
    p = jax.tree.map(jax.numpy.asarray, model["params"])
    return {"sim": LE.scores(p, ws["X"]), "av2": LE.scores(p, wa["X"]), "train": LE.scores(p, wt["X"])}


@stage("calibration", deps=("learned_scores", "windows_train", "hidden_ahead"), config={"target": 0.9})
def calibration(sc, wt, ha, target):
    cal = wt["meta"][:, 3] >= 6
    out = {}
    for j, name in enumerate(LE.TAGS):
        out[name] = CA.Calibrator(target).fit(sc["train"][cal, j], wt["Y"][cal, j]).state()
    c = ha["sim_env"] == SPLIT["calib"][0]
    out["hidden_ahead"] = CA.Calibrator(target).fit(ha["sim_p"][c], ha["sim_truth"][c]).state()
    return out


@stage("hidden_ahead", deps=("sim_scenes", "av2_scenes"))
def hidden_ahead(sim, av2):
    from nets import use
    occ = use.OcclusionNet("results/nets")
    out = {"sim_p": [], "sim_truth": [], "sim_env": [], "sim_ref": [], "av2_p": [], "av2_ref": []}
    for k, s in enumerate(sim):
        for t, p in LE.hidden_ahead_scores(s, occ):
            out["sim_p"].append(p), out["sim_truth"].append(LE.hidden_ahead_truth(s, t)), out["sim_env"].append(s["truth"]["env"]), out["sim_ref"].append((k, t))
    for k, s in enumerate(av2):
        for t, p in LE.hidden_ahead_scores(s, occ):
            out["av2_p"].append(p), out["av2_ref"].append((k, t))
    return {k: np.asarray(v) for k, v in out.items()}


# ---------------------------------------------------------------- domain, fusion, counterfactual, active learning
@stage("domain", deps=("sim_scenes", "av2_scenes", "rules_sim", "rules_av2"))
def domain(sim, av2, rs, ra):
    fs = np.stack([RU.scene_features(s, t) for s, t in zip(sim, rs)])
    fa = np.stack([RU.scene_features(s, t) for s, t in zip(av2, ra)])
    env = np.array([s["truth"]["env"] for s in sim])
    d = DO.Domain().fit(fs[part(env, "train")], fs[part(env, "test")])
    tags_a, score_a = d.tags(av2, fa)
    tags_s, score_s = d.tags(sim, fs)
    return {"fs": fs, "fa": fa, "score_sim": score_s, "score_av2": score_a, "threshold": d.threshold, "tags_av2": tags_a, "tags_sim": tags_s,
            "env": env, "explain_av2": d.explain(fa, DO.FEATURE_NAMES), "z_av2": (fa - d.mu) / d.sd}


def _votes(meta, scenes, rules, scores, cals):
    """Labelling functions per window: rule tags of that vehicle in the last 2 s, and the learned tags above their review threshold."""
    by = {}
    for k, tags in enumerate(rules):
        for t in tags:
            by.setdefault((k, t.agent, t.name), []).append((t.t_start, t.t_end))
    V = np.full((len(meta), 5), 0, np.int8)
    for r, (k, i, t, _) in enumerate(meta):
        end = t * SC.DT
        for j, n in enumerate(("hard_brake", "low_ttc", "tailgating")):
            V[r, j] = any(a <= end and b >= end - 2.0 for a, b in by.get((k, i, n), []))
    for j, n in ((3, "distracted"), (4, "crash_soon")):
        V[:, j] = (scores[:, LE.TAGS.index(n)] >= cals[n + "_lf"]).astype(np.int8)      # fires on the top 2 % of simulated windows
    return V


LF_NAMES = ("hard_brake (rule)", "low_ttc (rule)", "tailgating (rule)", "distracted (learned)", "crash_soon (learned)")


@stage("label_model", deps=("windows_sim", "windows_av2", "sim_scenes", "av2_scenes", "rules_sim", "rules_av2", "learned_scores", "calibration"))
def label_model(ws, wa, sim, av2, rs, ra, sc, cals):
    cals = dict(cals, **{n + "_lf": float(np.quantile(sc["train"][:, LE.TAGS.index(n)], 0.98)) for n in ("distracted", "crash_soon")})
    Vs = _votes(ws["meta"], sim, rs, sc["sim"], cals)
    Va = _votes(wa["meta"], av2, ra, sc["av2"], cals)
    params, q_all = LM.fit(np.concatenate([Vs, Va]))
    qs, qa = q_all[:len(Vs)], q_all[len(Vs):]
    te = np.ones(len(Vs), bool)                                  # the golden set is independent of training: all of it is test
    y = ws["Y"][te, LE.TAGS.index("crash_soon")]
    risky = (y > 0.5) | (ws["Y"][te, LE.TAGS.index("distracted")] > 0.5)   # a crash within 3 s, or a driver not reacting now
    res = {"label_model_auroc": CM.auroc(qs[te], y), "majority_auroc": CM.auroc(LM.majority(Vs[te]), y),
           "best_single": max(((n, CM.auroc(Vs[te, j], y)) for j, n in enumerate(LF_NAMES)), key=lambda z: z[1]),
           "single": {n: CM.auroc(Vs[te, j], y) for j, n in enumerate(LF_NAMES)}, "sensitivity": dict(zip(LF_NAMES, params["sensitivity"].tolist())),
           "specificity": dict(zip(LF_NAMES, params["specificity"].tolist())), "prior": params["prior"],
           "label_model_auroc_risky": CM.auroc(qs[te], risky), "majority_auroc_risky": CM.auroc(LM.majority(Vs[te]), risky)}
    return {"q_sim": qs, "q_av2": qa, "votes_sim": Vs, "votes_av2": Va, "eval": res}


@stage("counterfactual")
def counterfactual():
    from . import counterfactual as CF
    return CF.evaluate()


@stage("active", deps=("windows_train", "windows_sim", "learned_scores"), config={"tag": "tailgater", "rounds": 10, "batch": 100})
def active(wt, ws, sc, tag, rounds, batch):
    """The simulator plays the reviewer: pool = the calibration copies of the training recording, test = the golden set."""
    from . import active as AL
    j = LE.TAGS.index(tag)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    rng = np.random.default_rng(0)
    pi = np.nonzero(wt["meta"][:, 3] >= 6)[0]
    pi = rng.choice(pi, min(len(pi), 20000), replace=False)
    ti = rng.choice(len(ws["X"]), min(len(ws["X"]), 20000), replace=False)
    return AL.run(wt["X"][pi], lg(sc["train"][pi, j]), wt["Y"][pi, j], ws["X"][ti], lg(sc["sim"][ti, j]), ws["Y"][ti, j], rounds=rounds, batch=batch)


# ---------------------------------------------------------------- tags, store, QA
def _learned_tags(scenes, w, scores, cals, src):
    out = []
    meta = w["meta"]
    for j, name in enumerate(LE.TAGS):
        c = CA.Calibrator.from_state(cals[name])
        p, st = c.decide(scores[:, j])
        if name == "tailgater":                                   # a style: one tag per driver, from the mean over its windows
            agg = {}
            for r, (k, i, t, _) in enumerate(meta):
                agg.setdefault((k, i), []).append(p[r])
            for (k, i), ps in agg.items():
                m = float(np.mean(ps))
                stt = "auto" if m >= c.t_emit else ("review" if m >= c.t_review else "")
                if stt:
                    out.append(Tag(scenes[k]["id"], src, name, "agent", int(i), 0.0, scenes[k]["T"] * SC.DT, m, "learned-conv", status=stt))
            continue
        for r in np.nonzero(st != "")[0]:
            k, i, t, _ = meta[r]
            a, b = (t * SC.DT - 1.0, t * SC.DT) if name == "distracted" else (t * SC.DT, t * SC.DT + 3.0)
            out.append(Tag(scenes[k]["id"], src, name, "interval", int(i), a, b, float(p[r]), "learned-conv", status=str(st[r])))
    return out


@stage("tags", deps=("sim_scenes", "av2_scenes", "rules_sim", "rules_av2", "windows_sim", "windows_av2", "learned_scores", "calibration",
                     "hidden_ahead", "domain", "label_model"))
def tags(sim, av2, rs, ra, ws, wa, sc, cals, ha, dom, lm):
    stores = {}
    for src, scenes, rules, w, s, q in (("sim", sim, rs, ws, sc["sim"], lm["q_sim"]), ("av2", av2, ra, wa, sc["av2"], lm["q_av2"])):
        st = TagStore()
        for t in rules:
            st.extend(t)
        st.extend(_learned_tags(scenes, w, s, cals, src))
        c = CA.Calibrator.from_state(cals["hidden_ahead"])
        p_, refs = ha[f"{src}_p"], ha[f"{src}_ref"]
        if len(p_):
            pc, stat = c.decide(p_)
            for (k, t), pp, ss in zip(refs, pc, stat):
                if ss:
                    st.add(Tag(scenes[k]["id"], src, "hidden_ahead", "interval", int(scenes[k]["ego"]), t * SC.DT, t * SC.DT + 1.0, float(pp), "occlusion-net", status=str(ss)))
        for r in np.nonzero(q >= 0.5)[0]:
            k, i, t, _ = w["meta"][r]
            st.add(Tag(scenes[k]["id"], src, "risky_moment", "interval", int(i), t * SC.DT - 2.0, t * SC.DT, float(q[r]), "label-model"))
        st.extend(dom[f"tags_{src}"])
        stores[src] = st.rows
    return stores


@stage("store", deps=("tags", "domain", "sim_scenes", "av2_scenes"))
def store(tg, dom, sim, av2):
    os.makedirs(OUT, exist_ok=True)
    paths = {src: TagStore(rows).save(f"{OUT}/tags_{src}.parquet") for src, rows in tg.items()}
    from . import search as SE
    vocab = sorted(ONTOLOGY)
    ids, vecs = [], []
    for src, scenes, F_ in (("sim", sim, dom["fs"]), ("av2", av2, dom["fa"])):
        names = {}
        for r in tg[src]:
            if r["status"] == "auto":
                names.setdefault(r["scene_id"], []).append(r["name"])
        for s, f in zip(scenes, F_):
            ids.append(s["id"])
            vecs.append(SE.embedding(f, None, names.get(s["id"], []), vocab))
    np.savez_compressed(f"{OUT}/index.npz", ids=np.array(ids), vectors=np.array(vecs), vocab=np.array(vocab))
    review = [r for src in tg for r in tg[src] if r["status"] == "review"]
    rq = TagStore(sorted(review, key=lambda r: abs(r["confidence"] - 0.5))).save(f"{OUT}/review_queue.parquet")
    return {"paths": paths, "index": f"{OUT}/index.npz", "review_queue": rq, "n_review": len(review)}


def _rule_qa(scenes, rules, name, truth_fn, tol=1.0):
    """Event-level precision / recall of one rule tagger against simulator truth (interval overlap with tolerance)."""
    tp_tag = n_tag = tp_truth = n_truth = 0
    for s, tg in zip(scenes, rules):
        ev = truth_fn(s)                                         # list of (agent, t_start, t_end) in seconds
        mine = [(t.agent, t.t_start, t.t_end) for t in tg if t.name == name]
        hit = lambda a, b: a[0] == b[0] and a[1] - tol <= b[2] and b[1] - tol <= a[2]
        n_tag += len(mine)
        tp_tag += sum(any(hit(m, e) for e in ev) for m in mine)
        n_truth += len(ev)
        tp_truth += sum(any(hit(m, e) for m in mine) for e in ev)
    return {"precision": tp_tag / max(n_tag, 1), "recall": tp_truth / max(n_truth, 1), "n_tags": n_tag, "n_truth": n_truth}


def _truth_events(s, key):
    out = []
    for i in range(s["x"].shape[0]):
        for a, b in intervals(s["truth"][key][i], SC.DT):
            out.append((i, a, b))
    return out


def _truth_cut_in(s):
    """j changes lane (simulator truth) and within 2 s becomes the leader of some vehicle within 15 m."""
    lead, gap = RU.leaders(s)
    out = []
    for j in range(s["x"].shape[0]):
        for t in np.nonzero(s["truth"]["lane_change"][j])[0]:
            w = slice(t, min(t + 10, s["T"]))
            if ((lead[:, w] == j) & (gap[:, w] < 15.0)).any():
                out.append((j, t * SC.DT - 2.0, t * SC.DT + 2.0))
    return out


@stage("qa", deps=("sim_scenes", "rules_sim", "windows_sim", "learned_scores", "calibration", "hidden_ahead", "domain", "label_model",
                   "counterfactual", "tags", "active"))
def qa(sim, rs, ws, sc, cals, ha, dom, lm, cf, tg, al):
    env = np.array([s["truth"]["env"] for s in sim])
    te = part(env, "test")
    S_, R_ = [s for s, m in zip(sim, te) if m], [r for r, m in zip(rs, te) if m]
    R0 = [RU.tag_scene(s) for s in S_]                               # the untuned defaults, on the same held-out copies
    res = {"rules_untuned": {"lane_change": _rule_qa(S_, R0, "lane_change", lambda s: _truth_events(s, "lane_change")),
                             "cut_in": _rule_qa(S_, R0, "cut_in", _truth_cut_in)},
           "rules": {"lane_change": _rule_qa(S_, R_, "lane_change", lambda s: _truth_events(s, "lane_change")),
                     "turning": _rule_qa(S_, R_, "turning", lambda s: _truth_events(s, "turning"), tol=2.0),
                     "cut_in": _rule_qa(S_, R_, "cut_in", _truth_cut_in)}}
    wt = np.ones(len(ws["meta"]), bool)                           # the whole golden set: an independent recording from training
    res["learned"] = {}
    for j, name in enumerate(LE.TAGS):
        c = CA.Calibrator.from_state(cals[name])
        s_, y_ = sc["sim"][wt, j], ws["Y"][wt, j].astype(bool)
        p_, st = c.decide(s_)
        emit = st == "auto"
        res["learned"][name] = {"auroc": CM.auroc(s_, y_), "base_rate": float(y_.mean()), "ece_raw": CA.ece(s_, y_), "ece_calibrated": CA.ece(p_, y_),
                                "precision_at_threshold": float(y_[emit].mean()) if emit.any() else float("nan"),
                                "recall_at_threshold": float(emit[y_].mean()) if y_.any() else float("nan"),
                                "review_share": float((st == "review").mean()), "threshold": c.t_emit}
    hte = ha["sim_env"] >= SPLIT["test"][0]
    c = CA.Calibrator.from_state(cals["hidden_ahead"])
    p_, st = c.decide(ha["sim_p"][hte])
    y_ = ha["sim_truth"][hte].astype(bool)
    res["learned"]["hidden_ahead"] = {"auroc": CM.auroc(ha["sim_p"][hte], y_), "base_rate": float(y_.mean()), "ece_raw": CA.ece(ha["sim_p"][hte], y_),
                                      "ece_calibrated": CA.ece(p_, y_), "precision_at_threshold": float(y_[st == "auto"].mean()) if (st == "auto").any() else float("nan"),
                                      "recall_at_threshold": float((st == "auto")[y_].mean()) if y_.any() else float("nan"), "review_share": float((st == "review").mean())}
    res["label_model"] = lm["eval"]
    res["rule_params"] = None
    res["counterfactual"] = None if cf is None else {k: v for k, v in cf.items() if not isinstance(v, np.ndarray)}
    res["domain"] = {"threshold": dom["threshold"], "av2_share_sim_gap": float((dom["score_av2"] > dom["threshold"]).mean()),
                     "sim_test_share_sim_gap": float((dom["score_sim"][te] > dom["threshold"]).mean())}
    # drift: tag rates per scene, real against simulated
    rates = {}
    for src, rows in tg.items():
        n_sc = len({r["scene_id"] for r in rows}) or 1
        cnt = {}
        for r in rows:
            if r["status"] == "auto":
                cnt[r["name"]] = cnt.get(r["name"], 0) + 1
        rates[src] = {k: v / (len(dom["fa"]) if src == "av2" else len(sim)) for k, v in cnt.items()}
    res["tag_rates_per_scene"] = rates
    res["active_learning"] = al
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/eval.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return res


def run(target="qa", force=()):
    from . import dag
    return dag.run(target, force=force)
