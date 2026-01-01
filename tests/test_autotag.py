import os

import numpy as np
import pytest

from autotag import calibrate as CA
from autotag import labelmodel as LM
from autotag import rules as RU
from autotag import search as SE
from autotag.schema import ONTOLOGY, Tag, TagStore, intervals


def scene(xs, ys, yaws, vs, kinds=None, px=None, py=None):
    """A hand-built scene: arrays [V, T]."""
    x = np.asarray(xs, float)
    V, T = x.shape
    return {"id": "test:0", "source": "sim", "dt": 0.2, "T": T, "x": x, "y": np.asarray(ys, float), "yaw": np.asarray(yaws, float),
            "v": np.asarray(vs, float), "alive": np.isfinite(x), "kind": np.array(kinds or ["car"] * V), "av": np.zeros(V, bool),
            "length": np.full(V, 4.5), "ego": 0, "px": np.zeros((0, T)) if px is None else np.asarray(px, float),
            "py": np.zeros((0, T)) if py is None else np.asarray(py, float), "truth": None}


def straight(T, x0, v, y=0.0):
    t = np.arange(T) * 0.2
    return x0 + v * t, np.full(T, y), np.zeros(T), np.full(T, float(v))


def test_intervals():
    assert intervals([0, 1, 1, 0, 1, 1, 1], 0.2) == [(0.2, 0.6000000000000001), (0.8, 1.4000000000000001)]
    assert intervals([0, 1, 0, 1, 1], 0.2, min_len=2) == [(0.6000000000000001, 1.0)]


def test_tag_store_round_trip(tmp_path):
    st = TagStore()
    st.add(Tag("a", "sim", "cut_in", "interval", 3, 1.0, 2.0, 0.9, "rules"))
    st.add(Tag("b", "av2", "sim_gap", "scene", -1, 0, 11, 0.5, "domain"))
    p = st.save(str(tmp_path / "t.parquet"))
    back = TagStore.load(p).frame()
    assert list(back.name) == ["cut_in", "sim_gap"] and back.agent.tolist() == [3, -1]
    assert all(n in ONTOLOGY for n in back.name)


def test_hard_brake_and_tailgating():
    T = 50
    x0, y0, h0, v0 = straight(T, 0.0, 10.0)
    v = v0.copy()
    v[20:] = np.maximum(10.0 - 6.0 * (np.arange(30) * 0.2), 2.0)               # brakes at 6 m/s^2 from t = 4 s
    x = np.r_[0.0, np.cumsum(v[:-1] * 0.2)]
    lead = x + 4.5 + 6.0                                                        # a leader 6 m ahead (bumper gap ~6 m at 10 m/s: headway 0.6 s)
    sc = scene([x, lead], [y0, y0], [h0, h0], [v, v])
    tags = RU.tag_scene(sc)
    names = {(t.name, t.agent) for t in tags}
    assert ("hard_brake", 0) in names and ("tailgating", 0) in names
    hb = [t for t in tags if t.name == "hard_brake" and t.agent == 0][0]
    assert 3.0 <= hb.t_start <= 4.6


def test_cut_in_is_detected_and_a_parallel_car_is_not():
    T = 50
    xe, ye, he, ve = straight(T, 0.0, 10.0)
    # the cutter drives 10 m ahead in the next lane (y = 3.5) and moves into ours between t = 4 s and 6 s
    xc = xe + 14.0
    yc = np.clip(3.5 - (np.arange(T) * 0.2 - 4.0) * 1.75, 0.0, 3.5)
    xp, yp, hp, vp = straight(T, 14.0, 10.0, y=3.5)                              # stays in the next lane
    sc = scene([xe, xc, xp], [ye, yc, yp], [he, he, hp], [ve, ve, vp])
    cut = [t for t in RU.tag_scene(sc) if t.name == "cut_in"]
    assert [t.agent for t in cut] == [1]


def test_low_ttc_and_turning():
    T = 50
    xe, ye, he, ve = straight(T, 0.0, 12.0)
    xs, ys, hs, vs = np.full(T, 30.0), np.zeros(T), np.zeros(T), np.zeros(T)    # a stopped car ahead
    xs[:] = 30.0
    xe = np.minimum(xe, 30.0 - 4.5 - 0.5)
    ve = np.r_[np.diff(xe) / 0.2, 0.0]
    th = np.clip(np.linspace(0, 2.5 * np.pi / 2, T), 0, np.pi / 2)               # a third car turns 90 degrees within 4 s
    sc = scene([xe, xs, -40 + 5 * np.cos(th)], [ye, ys, 40 + 5 * np.sin(th)], [he, hs, th + np.pi / 2], [ve, vs, np.full(T, 5.0)])
    names = {(t.name, t.agent) for t in RU.tag_scene(sc)}
    assert ("low_ttc", 0) in names and ("turning", 2) in names


def test_ped_close():
    T = 30
    x, y, h, v = straight(T, 0.0, 8.0)
    px = np.full((1, T), 20.0)
    py = np.full((1, T), 2.0)
    sc = scene([x], [y], [h], [v], px=px, py=py)
    assert any(t.name == "ped_close" for t in RU.tag_scene(sc))


def test_scene_features_fixed_length():
    T = 30
    x, y, h, v = straight(T, 0.0, 8.0)
    assert RU.scene_features(scene([x, x + 20], [y, y], [h, h], [v, v])).shape == (17,)


def test_isotonic_is_monotone_and_calibrates():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=5000)
    y = rng.uniform(size=5000) < s ** 3                                      # the true probability is s^3
    c = CA.Calibrator(0.9).fit(s, y)
    grid = np.linspace(0, 1, 50)
    p = c(grid)
    assert (np.diff(p) >= -1e-12).all()
    assert CA.ece(c(s), y) < CA.ece(s, y)


def test_precision_threshold_reaches_its_target_on_fresh_data():
    rng = np.random.default_rng(1)
    def draw(n):
        s = rng.uniform(size=n)
        return s, rng.uniform(size=n) < s
    s, y = draw(20000)
    t = CA.precision_threshold(s, y, 0.9)
    s2, y2 = draw(20000)
    assert y2[s2 >= t].mean() >= 0.88 and t > 0.75
    assert CA.precision_threshold(np.zeros(100), np.zeros(100, bool), 0.9) == np.inf


def test_query_parser_and_overlap():
    import pandas as pd
    rows = [dict(scene_id="s1", source="av2", name="cut_in", level="interval", agent=1, t_start=1.0, t_end=3.0, confidence=1, tagger="r", version="v", status="auto"),
            dict(scene_id="s1", source="av2", name="hidden_ahead", level="interval", agent=0, t_start=2.0, t_end=4.0, confidence=1, tagger="r", version="v", status="auto"),
            dict(scene_id="s2", source="av2", name="cut_in", level="interval", agent=1, t_start=1.0, t_end=2.0, confidence=1, tagger="r", version="v", status="auto"),
            dict(scene_id="s2", source="av2", name="hidden_ahead", level="interval", agent=0, t_start=5.0, t_end=6.0, confidence=1, tagger="r", version="v", status="auto"),
            dict(scene_id="s3", source="av2", name="sim_gap", level="scene", agent=-1, t_start=0, t_end=11, confidence=1, tagger="d", version="v", status="auto")]
    df = pd.DataFrame(rows)
    assert [s for s, _ in SE.query(df, "cut_in & hidden_ahead")] == ["s1", "s2"]
    hits = SE.query(df, "cut_in & hidden_ahead", overlap=True)
    assert [s for s, _ in hits] == ["s1"] and hits[0][1] == [(2.0, 3.0)]
    assert [s for s, _ in SE.query(df, "!cut_in")] == ["s3"]
    assert [s for s, _ in SE.query(df, "(cut_in | sim_gap) & !hidden_ahead")] == ["s3"]
    with pytest.raises(ValueError):
        SE.parse("cut_in &")


def test_similarity_returns_the_scene_itself_first():
    rng = np.random.default_rng(0)
    idx = SE.Index([f"s{i}" for i in range(20)], rng.normal(size=(20, 6)))
    top = idx.similar("s7", 3)
    assert top[0][0] == "s7" and abs(top[0][1] - 1.0) < 1e-9 and top[1][1] <= top[0][1]


def test_label_model_recovers_tagger_accuracies():
    rng = np.random.default_rng(0)
    n = 20000
    y = rng.uniform(size=n) < 0.2
    sens, spec = np.array([0.9, 0.7, 0.6]), np.array([0.95, 0.9, 0.8])
    V = np.stack([np.where(y, rng.uniform(size=n) < a, rng.uniform(size=n) > b) for a, b in zip(sens, spec)], 1).astype(np.int8)
    params, q = LM.fit(V)
    assert np.allclose(params["sensitivity"], sens, atol=0.05) and np.allclose(params["specificity"], spec, atol=0.05)
    assert abs(params["prior"] - 0.2) < 0.03
    from nets import common as CM
    assert CM.auroc(q, y) >= CM.auroc(LM.majority(V), y) - 1e-3


def test_dag_caches_and_reruns_only_stale_stages(tmp_path, monkeypatch):
    from autotag import dag
    monkeypatch.setattr(dag, "CACHE", str(tmp_path))
    saved = dict(dag.STAGES)
    dag.STAGES.clear()
    calls = []
    try:
        @dag.stage("a", config={"n": 2})
        def a(n):
            calls.append("a")
            return n

        @dag.stage("b", deps=("a",))
        def b(x):
            calls.append("b")
            return x * 10
        assert dag.run("b", log=lambda *_: None) == 20 and calls == ["a", "b"]
        assert dag.run("b", log=lambda *_: None) == 20 and calls == ["a", "b"]       # all cached
        dag.STAGES["a"].config = {"n": 3}
        assert dag.run("b", log=lambda *_: None) == 30 and calls == ["a", "b", "a", "b"]   # a changed, so b is stale too
        assert "a --> b" in dag.mermaid("b")
    finally:
        dag.STAGES.clear()
        dag.STAGES.update(saved)


def test_active_learning_strategies_run():
    from autotag import active as AL
    rng = np.random.default_rng(0)
    X_ = rng.normal(size=(3000, 25, 8)).astype(np.float32)
    y = (X_[:, :, 0].mean(1) > 0.2).astype(float)
    lg = X_[:, :, 0].mean(1) * 3 + rng.normal(size=3000)
    out = AL.run(X_[:2000], lg[:2000], y[:2000], X_[2000:], lg[2000:], y[2000:], rounds=3, batch=50)
    assert set(out) == {"random", "uncertainty", "unc+div"} and all(len(v) == 3 for v in out.values())


needs_av2 = pytest.mark.skipif(not os.path.isdir(os.environ.get("AV2_ROOT", "data/av2")), reason="needs Argoverse 2 scenarios")


@needs_av2
def test_real_scene_has_the_shared_format():
    from autotag import scenes as SC
    s = SC.av2_scenes(limit=1)[0]
    for k in ("x", "y", "yaw", "v", "alive", "kind", "av", "length", "ego", "px", "py"):
        assert k in s
    assert s["x"].shape == s["v"].shape and s["dt"] == 0.2
    RU.tag_scene(s)
    from autotag import learned as LE
    X_, _, _ = LE.windows(s, labels=False)
    assert X_.shape[1:] == (LE.W, LE.F)
