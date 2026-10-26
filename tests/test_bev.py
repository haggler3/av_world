"""bev: the camera rig's geometry, labels from a scene, targets <-> decoding, the model's shapes and temporal warp, the metrics,
and that a few training steps lower the loss. None of it needs the renderer."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bev import data as D
from bev import metrics as ME
from bev import model as M
from bev.rig import Rig, box_corners, ego_motion, world_to_ego


def test_rig_projects_like_a_pinhole_and_covers_360():
    r = Rig()
    uv, z, ok = r.project(np.array([[20.0, 0.0, r.mount_z], [-20.0, 0.0, r.mount_z]]))
    assert np.allclose(uv[0, 0], [r.width / 2, r.height / 2]) and ok[0, 0] and not ok[0, 1]
    assert ok[3, 1] and np.allclose(uv[3, 1], [r.width / 2, r.height / 2])  # CAM_BACK sees behind, centred
    ring = np.stack([15 * np.cos(np.linspace(0, 2 * np.pi, 72)), 15 * np.sin(np.linspace(0, 2 * np.pi, 72)), np.full(72, 0.5)], 1)
    assert r.project(ring)[2].any(0).all()
    assert np.allclose(r.ego2img()[:, 3], [0, 0, 0, 1])


def test_ego_motion_maps_previous_frame_points_to_current():
    prev, now = (10.0, 5.0, 0.3), (13.0, 6.0, 0.5)
    world = np.array([20.0, 9.0])
    R, t = ego_motion(prev, now)
    assert np.allclose(R @ world_to_ego(world, prev) + t, world_to_ego(world, now), atol=1e-9)


def scene():
    car = dict(id=1, kind="car", av=False, crashed=False, start=0, length=4.5, x=[10.0, 11.0], y=[0.0, 0.0], yaw=[0.0, 0.0], v=[5.0, 5.0])
    ego = dict(id=2, kind="av", av=True, crashed=False, start=0, length=4.6, x=[0.0, 1.0], y=[0.0, 0.0], yaw=[0.0, 0.0], v=[5.0, 5.0])
    far = dict(id=3, kind="bus", av=False, crashed=False, start=0, length=12.0, x=[200.0, 200.0], y=[0.0, 0.0], yaw=[0.0, 0.0], v=[0.0, 0.0])
    ped = dict(t=0.0, duration=10.0, **{"from": [5.0, -6.0], "to": [5.0, 6.0]})
    return {"dt": 0.2, "frames": 2, "vehicles": [car, ego, far], "pedestrians": [ped]}


def test_labels_are_in_the_ego_frame_and_sized_like_the_drawing():
    B, C, O = D.labels(scene(), 0, 2, (0.0, 0.0, 0.0))
    keep = C >= 0
    assert sorted(C[keep].tolist()) == [0, 4]  # the car and the pedestrian; the ego and the far bus are not labelled
    car = B[keep][C[keep] == 0][0]
    assert car[0] == pytest.approx(10.0) and car[3] == pytest.approx(4.5) and 1.3 < car[5] < 1.7 and car[7] == pytest.approx(5.0)
    assert np.allclose(box_corners(car)[:, 2].min(), 0.0)


def test_targets_decode_round_trip_is_exact():
    cfg = M.Config()
    boxes = np.zeros((4, 9), np.float32)
    boxes[:3] = [[10.3, -4.1, 0.75, 4.5, 1.9, 1.5, 0.4, 3.0, 0.5], [-20.7, 13.2, 0.9, 0.6, 0.6, 1.8, -2.0, 0.0, 1.0],
                 [2.2, 25.9, 1.6, 12.0, 2.5, 3.2, 3.0, -1.0, 0.0]]
    cls = np.array([0, 4, 2, -1])
    heat, reg, mask = M.targets(cfg, boxes, cls)
    b, s, c = M.decode(cfg, np.log(heat / (1 - heat + 1e-6) + 1e-9), reg, thresh=0.5)
    assert len(b) == 3 and sorted(c.tolist()) == [0, 2, 4]
    for i in range(3):
        j = int(np.argmin(np.hypot(b[:, 0] - boxes[i, 0], b[:, 1] - boxes[i, 1])))
        assert np.allclose(b[j], boxes[i], atol=1e-4)
    assert ME.evaluate([(b, s, c)], [(boxes[:3], cls[:3])], D.CLASSES)["NDS"] == pytest.approx(1.0)


def test_model_shapes_and_identity_warp():
    cfg = M.Config(bev=16, range_m=16.0, dim=32, layers=1)
    p = M.init(jax.random.PRNGKey(0), cfg)
    imgs = jnp.asarray(np.random.default_rng(0).integers(0, 255, (2, 6, 96, 176, 3)), jnp.uint8)
    e2i = jnp.broadcast_to(jnp.asarray(Rig().ego2img(), jnp.float32), (2, 6, 4, 4))
    f = M.features(p, cfg, imgs)
    assert f.shape == (2, 6, 12, 22, 64)
    prev = M.encode(p, cfg, f, e2i, (96, 176))
    I, z = jnp.broadcast_to(jnp.eye(2), (2, 2, 2)), jnp.zeros((2, 2))
    assert np.allclose(M.align(cfg, prev, I, z), prev, atol=1e-5)  # no motion: the warp is the identity
    h, r = M.head(p, cfg, M.encode(p, cfg, f, e2i, (96, 176), prev, I, z))
    assert h.shape == (2, 16, 16, 5) and r.shape == (2, 16, 16, M.N_REG) and bool(jnp.isfinite(h).all())


def test_metrics_penalise_misplaced_boxes():
    g = np.array([[5, 0, 0.75, 4.5, 1.9, 1.5, 0.3, 2, 0]], float)
    s = g.copy()
    s[:, 0] += 1.5
    r = ME.evaluate([(s, np.array([0.9]), np.array([0]))], [(g, np.array([0]))], ("car",))
    assert r["mAP"] == pytest.approx(0.5) and r["mATE"] == pytest.approx(1.5) and r["NDS"] < 1


def test_a_few_steps_lower_the_loss(tmp_path):
    import optax

    from bev import train as T
    cfg = M.Config(bev=16, range_m=16.0, dim=32, layers=1)
    rng = np.random.default_rng(0)
    boxes = np.zeros((3, 64, 9), np.float32)
    boxes[:, 0] = [6.0, 0.0, 0.75, 4.5, 1.9, 1.5, 0.0, 0.0, 0.0]
    cls = np.full((3, 64), -1)
    cls[:, 0] = 0
    d = {"imgs": rng.integers(0, 255, (3, 6, 96, 176, 3)).astype(np.uint8), "pose": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32),
         "boxes": boxes, "cls": cls, "seq": np.zeros(3, int), "ego2img": Rig().ego2img().astype(np.float32)}
    p = M.init(jax.random.PRNGKey(1), cfg)
    opt = optax.adam(3e-3)
    st = opt.init(p)
    step, _ = T.make_step(cfg, opt, (96, 176))
    b = {k: jnp.asarray(v) for k, v in T.batch_arrays(cfg, d, T.pairs(d)).items()}
    losses = []
    for _ in range(15):
        p, st, loss, _ = step(p, st, b)
        losses.append(float(loss))
    assert losses[-1] < 0.6 * losses[0]


def test_perception_export_writes_a_consistent_folder(tmp_path):
    pytest.importorskip("pyvista")
    import json
    import os

    from bev import export as E
    try:
        info = E.export(str(tmp_path / "p"), area="chicago", scenario="bus_dartout", seconds=10, egos=1, width=64, height=32, warm=30, stride=5)
    except Exception as e:  # pyvista present but no offscreen OpenGL here
        pytest.skip(f"cannot render: {e}")
    out = info["out"]
    samples, cal = json.load(open(f"{out}/samples.json")), json.load(open(f"{out}/calibration.json"))
    assert len(samples) == info["samples"] > 0 and len(cal) == 6
    s = samples[0]
    assert all(os.path.exists(f"{out}/{f}") for f in s["images"].values())
    assert all(len(b["center"]) == 3 and len(b["size"]) == 3 for b in s["boxes"])
    assert np.array(cal["CAM_FRONT"]["ego2img"]).shape == (4, 4)
    for name in ("data.npz", "scene.json", "meta.json"):
        assert os.path.exists(f"{out}/{name}")
    assert not os.path.exists(f"{out}/_tmp")
