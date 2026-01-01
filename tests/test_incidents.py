"""Scene metrics on constructed scenes: hard braking, tight following, short time to collision, pedestrian proximity."""
import numpy as np

from avlab import incidents as IN


def scene(vehicles, peds=(), dt=0.2, frames=60):
    return {"dt": dt, "frames": frames, "vehicles": vehicles, "pedestrians": list(peds)}


def veh(i, x0, v, frames=60, dt=0.2, y=0.0, brake_at=None, length=4.5):
    xs, vs, x, s = [], [], x0, v
    for k in range(frames):
        if brake_at is not None and k >= brake_at:
            s = max(s - 8.0 * dt, 0.0)
        xs.append(x)
        vs.append(s)
        x += s * dt
    return {"id": i, "kind": "car", "av": False, "start": 0, "length": length, "x": xs, "y": [y] * frames, "yaw": [0.0] * frames, "v": vs}


def test_a_hard_brake_is_counted_once():
    m = IN.scene_metrics(scene([veh(0, 0, 15.0, brake_at=20)]))
    assert m["hard_brakes"] == 1 and m["moving_veh_s"] > 5


def test_following_at_half_a_second_is_tight_and_at_two_seconds_is_not():
    near = IN.scene_metrics(scene([veh(0, 0, 12.0), veh(1, 12.0 * 0.5 + 4.5, 12.0)]))  # headway 0.5 s
    far = IN.scene_metrics(scene([veh(0, 0, 12.0), veh(1, 12.0 * 2.0 + 4.5, 12.0)]))
    assert near["tight_s"] > 5 and far["tight_s"] == 0


def test_closing_on_a_slow_leader_gives_a_short_ttc():
    m = IN.scene_metrics(scene([veh(0, 0, 15.0), veh(1, 25.0, 8.0)]))  # closing at 7 m/s from a 20 m gap: TTC about 2.9 s, then shorter
    assert m["ttc2_s"] > 0
    none = IN.scene_metrics(scene([veh(0, 0, 8.0), veh(1, 25.0, 15.0)]))  # opening
    assert none["ttc2_s"] == 0


def test_pedestrian_close_to_a_moving_vehicle():
    ped = [{"id": "p", "start": 0, "x": [30.0] * 60, "y": [1.0] * 60}]
    m = IN.scene_metrics(scene([veh(0, 0, 10.0)], ped))
    assert m["ped_close_s"] > 0
    far = IN.scene_metrics(scene([veh(0, 0, 10.0)], [{"id": "p", "start": 0, "x": [30.0] * 60, "y": [20.0] * 60}]))
    assert far["ped_close_s"] == 0


def test_audit_checklists_are_consistent():
    for inc in IN.INCIDENTS:
        assert all(l in IN.LEVEL for _, l, _ in inc["features"])
        assert any(l == "absent" for _, l, _ in inc["features"])  # every incident has something the simulator cannot show
