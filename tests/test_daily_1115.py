"""2026-11-15: the gap-acceptance channel of the friction-aware AV (avlab.frictionmerge) and the four-stream knee (avlab.knee4)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gap_u1_maps_to_xnet_gap_scale():
    from avlab.frictionmerge import GSC0, GSC_MAX, GSC_SPAN, gap_u1, scales
    # xnet/sim.py: gsc = 1.15 + 0.35 tanh(u1)
    gsc = lambda u1: GSC0 + GSC_SPAN * np.tanh(u1)  # noqa: E731
    assert gap_u1(0.4, False) == 0.0
    assert gsc(gap_u1(1.0, True)) == pytest.approx(1.15)          # dry road: no extra caution
    assert gsc(gap_u1(0.9, True)) == pytest.approx(1.15 / 0.9)    # unclipped range
    assert gsc(gap_u1(0.4, True)) == pytest.approx(GSC_MAX)       # 1.15 / 0.4 = 2.9 is beyond the slot's range
    assert scales("blind", 0.4) == (1.0, 0.0)
    assert scales("headway+gap", 0.5)[0] == pytest.approx(2.0)


def test_merge_idm_only_changes_u1():
    jnp = pytest.importorskip("jax.numpy")
    from avlab.frictionaware import aware_idm
    from avlab.frictionmerge import merge_idm
    obs = jnp.asarray(np.random.default_rng(0).uniform(0, 1.5, (32, 28)), jnp.float32)
    u_a, _ = aware_idm(obs, 2.5, 1.0)
    u_m, _ = merge_idm(obs, 2.5, 1.7)
    np.testing.assert_allclose(np.asarray(u_m[:, 0]), np.asarray(u_a[:, 0]))
    assert np.all(np.asarray(u_m[:, 1]) == pytest.approx(1.7))


def test_knee4_analysis_on_synthetic_stream(tmp_path):
    """A fourth stream drawn from the pooled power law with a 0.7 hinge: the loader appends it and the tests run."""
    from avlab import knee4
    rng = np.random.default_rng(3)
    cells = {}
    for mu in knee4.MUS4:
        rate = 0.004 * mu ** -3 * np.exp(3.0 * max(np.log(0.7) - np.log(mu), 0.0))
        hours = rng.uniform(50, 60, 32)
        y = rng.poisson(rate * hours)
        cells[str(mu)] = {"mu": mu, "per_copy_crashes": y.astype(float).tolist(), "per_copy_hours": hours.tolist(),
                          "crashes": float(y.sum()), "hours": float(hours.sum())}
    p = tmp_path / "stream4.json"
    p.write_text(json.dumps({"cells": cells}))
    y, t, mu, run, stream = knee4.load_all4(str(p))
    assert (stream == 4).sum() == 96 and set(np.unique(run)) == {0.0, 1.0, 2.0, 3.0, 4.0}
    out = knee4.analyse(str(p), B=19)
    assert out["pooled4"]["heterogeneity"]["df"] == 3
    assert set(out["holm"]) == {"stream4_0.7_below_0.6", "stream4_fixed", "pooled4_fixed"}
    assert out["pooled4"]["fixed_0.7"]["extra"] > 0
