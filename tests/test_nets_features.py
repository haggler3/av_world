import importlib
import os
import re

import numpy as np
import pytest

from nets import api
from nets import registry as RG


def slug(heading):
    """GitHub's heading anchor: lower case, punctuation other than hyphens and spaces dropped, spaces to hyphens."""
    h = heading.strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(path):
    return {slug(l.lstrip("#")) for l in open(path, encoding="utf-8") if l.startswith("#")}


def test_every_feature_is_complete_and_runnable():
    assert len(RG.FEATURES) == 13 and len(RG.BY_NAME) == 13
    for f in RG.FEATURES:
        assert f.kind in ("network", "upgrade") and f.question.endswith("?") and f.result and f.use
        q, full = f.minutes
        assert 0 < q <= full
        mod, fn = f.runner.split(":")
        assert callable(getattr(importlib.import_module(mod), fn)), f.runner
        assert f.outputs and all(o.startswith("results/") for o in f.outputs)
        for n in f.needs:
            assert n in ("city", "av2", "logs") or n in RG.BY_NAME


def test_every_doc_link_points_at_a_real_heading():
    for f in RG.FEATURES:
        path, anchor = f.doc.split("#")
        assert os.path.exists(path), path
        assert anchor in anchors(path), (f.name, anchor)


def test_the_training_guide_covers_every_feature():
    text = open("docs/TRAINING_NETS.md", encoding="utf-8").read()
    for f in RG.FEATURES:
        assert f"`{f.name}`" in text, f.name


def test_network_mains_take_an_output_folder_and_quick_mode():
    import inspect
    for name in ("forks", "counterfactual", "response", "occlusion", "mindreader", "blame", "sbi"):
        sig = inspect.signature(importlib.import_module(f"nets.{name}").main)
        assert "out_dir" in sig.parameters and "quick" in sig.parameters, name


def test_cli_list_and_info(capsys):
    from av_world import cli
    assert cli.main(["nets", "list"]) == 0
    out = capsys.readouterr().out
    for f in RG.FEATURES:
        assert f.name in out
    assert cli.main(["nets", "info", "critic"]) == 0
    assert "av-world nets train critic" in capsys.readouterr().out


def test_unknown_feature_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown feature"):
        RG.get("nope")


def test_easy_nets_namespace():
    from av_world import easy
    assert "list()" in repr(easy.nets)
    table = easy.nets.list()
    names = list(table.index) if hasattr(table, "index") else [r["name"] for r in table]
    assert names == [f.name for f in RG.FEATURES]


def test_quick_runs_do_not_touch_the_full_results(tmp_path, monkeypatch):
    """The redirect used for the upgrades points a module's OUT at the quick folder and restores it."""
    import types
    m = types.SimpleNamespace(OUT="results/crn")
    with RG._redirect(m, str(tmp_path), True, "results/crn") as out:
        assert out.startswith(str(tmp_path)) and m.OUT == out
    assert m.OUT == "results/crn"


@pytest.mark.parametrize("name", ["critic", "response", "occlusion", "mindreader", "blame", "sbi"])
def test_trained_networks_load_and_answer(name):
    f = RG.get(name)
    if not RG.trained(f):
        pytest.skip(f"{name} is not trained here (av-world nets train {name})")
    net = api.load(name)
    if name == "critic":
        r = net.risk(np.zeros((3, 28)), np.zeros((3, 2)))
        assert r.shape == (3, 5) and ((r >= 0) & (r <= 1)).all()
        assert net.best_action(np.zeros((3, 28)), np.zeros((3, 2))).shape == (3,)
    elif name == "response":
        assert net.follower_effect(np.zeros((2, 28)), np.zeros((2, 6)), np.zeros((2, 2)), 1).shape == (2, 19)
    elif name == "occlusion":
        x = np.array([0.0, 10.0, 20.0, 5.0]); y = np.array([0.0, 0.0, 0.0, 8.0])
        tok, vis = net.from_scene(0, x, y, np.zeros(4), np.full(4, 4.5), np.full(4, 8.0), np.zeros(4))
        assert vis.tolist() == [False, True, False, True]               # the car at 20 m hides behind the one at 10 m; ego is not "visible"
        g, a = net.belief(tok)
        assert g.shape == (1, 8, 8) and a.shape == (1,)
    elif name == "mindreader":
        w = np.tile(np.array([8, 0, 30, 8, 60, 2, 13, 0], np.float32), (2, 25, 1))
        r = net.read(w)
        assert 0.3 < float(r["T"][0]) < 3.0 and r["runner"].shape == (2,)
    elif name == "blame":
        s = net.rank(np.zeros((2, 8, 18)), np.r_[np.ones(5), np.zeros(3)].astype(bool)[None].repeat(2, 0))
        assert s.shape == (2, 8) and (s[:, 5:] == 0).all()
    elif name == "sbi":
        k = net.real(n=50)
        assert set(k) >= {"p_red", "T_med"} and len(k["T_med"]) == 50


def test_sigreg_trains_in_quick_mode(tmp_path):
    if not os.path.exists("data/env.bin"):
        pytest.skip("needs data/env.bin")
    RG._run_sigreg(quick=True, out_dir=str(tmp_path))
    assert os.path.exists(tmp_path / "extras" / "sigreg.json")
