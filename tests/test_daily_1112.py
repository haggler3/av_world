"""2026-11-12: braking-demand summary maths (avlab.brakedemand) and the results index (avlab.resultsindex). Fast, no simulation."""
import os
import subprocess

import numpy as np

from avlab import brakedemand as BD
from avlab import resultsindex as RI


def test_exceed_counts_bins_at_or_beyond_the_cap():
    bins = np.arange(0.0, 10.01, 0.25)
    h = np.zeros(len(bins) - 1)
    h[0], h[24], h[36] = 90, 6, 4  # 0 m/s^2, 6.0 m/s^2, 9.0 m/s^2
    assert BD.exceed(h, bins, 9.0) == 0.04
    assert BD.exceed(h, bins, 6.0) == 0.10
    assert BD.exceed(h, bins, 6.3) == 0.04


def test_ratio_ci_brackets_the_ratio_and_detects_enrichment():
    r, lo, hi = BD.ratio_ci(30, 100, 100, 10000)
    assert abs(r - 30.0) < 1e-9 and lo < r < hi and lo > 10
    r, lo, hi = BD.ratio_ci(1, 100, 100, 10000)
    assert lo < 1 < hi


def _cell(mu, sat, moving, inv, inv_sat, mov_sat, crashes, hist):
    return {"mu": mu, "cap": 9 * mu, "hist": hist, "moving_steps": moving, "sat_steps": sat, "per_copy_sat": [sat / 2, sat / 2],
            "per_copy_moving": [moving / 2, moving / 2], "involved": inv, "involved_recent_sat": inv_sat, "moving_recent_sat": mov_sat,
            "crashes": crashes, "hours": 100.0}


def test_summarise_rows_and_pooled_enrichment():
    bins = np.arange(0.0, 10.01, 0.25).tolist()
    h = [0.0] * 40
    h[0], h[36] = 990.0, 10.0
    res = {"bins": bins, "cells": {"1": _cell(1.0, 10, 1000, 0, 0, 20, 2, h), "0.5": _cell(0.5, 100, 1000, 20, 10, 150, 20, h)}}
    s = BD.summarise(res)
    assert [r["mu"] for r in s["rows"]] == [1.0, 0.5]
    assert s["rows"][0]["enrichment"] is None and s["rows"][1]["enrichment"][0] > 1
    assert abs(s["rows"][0]["dry_exceed_cap"] - 0.01) < 1e-12
    assert s["pooled_enrichment"]["mu<=0.6"]["involved"] == 20


def test_results_index_finds_citations_orphans_and_broken_links(tmp_path):
    def w(p, t):
        os.makedirs(os.path.dirname(tmp_path / p) or tmp_path, exist_ok=True)
        (tmp_path / p).write_text(t)
    w("results/a/x.json", "{}")
    w("results/a/y.json", "{}")
    w("docs/A.md", "see [x](../results/a/x.json) and [gone](../results/a/gone.json) and results/tmp/out.json")
    w("tool.py", "open('results/a/y.json')")
    w(".gitignore", "results/tmp/\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    ix = RI.build(str(tmp_path))
    assert ix["cited"]["results/a/x.json"] == ["docs/A.md"] and ix["used"]["results/a/y.json"] == ["tool.py"]
    assert ix["broken"] == [("docs/A.md", "results/a/gone.json")]  # the ignored output is not reported
    md = RI.render(ix)
    assert "2 tracked files in 1 directories; 0 orphans; 1 broken" in md
