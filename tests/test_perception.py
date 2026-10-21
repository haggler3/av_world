"""Sensor model: occlusion by a bus, emerging pedestrians, range, fog, dropouts, ghosts."""
import numpy as np

from xnet import perception as PC

N = 40


def scene():
    def veh(i, kind, x, y, length=4.5):
        return {"id": i, "kind": kind, "av": kind == "av", "crashed": False, "start": 0, "length": length, "x": [x] * N, "y": [y] * N,
                "yaw": [0.0] * N, "v": [0.0] * N}
    return {"dt": 0.2, "frames": N, "vehicles": [veh(0, "av", 0.0, 0.0), veh(1, "bus", 15.0, 0.0, 12.0), veh(2, "car", 30.0, 8.0),
                                                   veh(3, "car", 120.0, 0.0)],
            "pedestrians": [{"t": 0.0, "duration": 8.0, "from": [30.0, -6.0], "to": [30.0, 6.0]}]}


def test_ped_behind_bus_is_occluded_then_emerges():
    d = PC.detect(scene(), 0, PC.Sensor(), seed=0)
    ped = d["cls"] == PC.CLASSES.index("ped")
    y = d["y"][ped]
    occ = d["occluded"][ped]
    assert occ[np.abs(y) < 2.0].all() and not occ[np.abs(y) > 4.0].any()
    assert not d["detected"][ped][occ].any()  # never reported while hidden
    assert d["emerging"][ped].any() and d["vru"][ped].all()


def test_range_and_fog():
    clear = PC.summary(PC.detect(scene(), 0, PC.Sensor(), seed=1))
    fog = PC.summary(PC.detect(scene(), 0, PC.Sensor(fog=1.0), seed=1))
    d = PC.detect(scene(), 0)
    assert not (d["dist"] > 70).any()  # the car 120 m away is out of range
    assert fog["detected"] < clear["detected"]


def test_dropouts_and_ghosts_and_latency():
    s = PC.summary(PC.detect(scene(), 0, PC.Sensor(dropout_rate=1.0, dropout_frames=(5, 6)), seed=0))
    assert s["dropout_share"] > 0.9 and s["detected"] == 0.0
    d = PC.detect(scene(), 0, PC.Sensor(ghost_rate=1.0, latency=2), seed=0)
    assert len(d["g_frame"]) > 10 and (d["frame_reported"] == d["frame"] + 2).all()
    clean = PC.detect(scene(), 0, PC.Sensor(), seed=0)
    assert len(clean["g_frame"]) == 0
    assert np.allclose(clean["x"], PC.detect(scene(), 0, PC.Sensor(), seed=5)["x"])  # ground truth does not depend on the seed


def test_noise_grows_with_range():
    near = scene()
    d = PC.detect(near, 0, PC.Sensor(range_50=500, pos_sigma=0.5), seed=0)
    ok = d["detected"].astype(bool)
    err = np.hypot(d["dx"][ok], d["dy"][ok])
    assert 0.1 < np.median(err) < 3.0
