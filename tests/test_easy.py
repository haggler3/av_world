"""av_world.easy and av_world.compat: the beginner layer and the bridges to other tools, in tiny configurations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from av_world import compat, easy


def test_car_fields_are_in_plain_units():
    obs = np.zeros((2, 28), np.float32)
    obs[:, 0], obs[:, 8], obs[:, 9] = 0.5, 1.0, 0.4  # 7.5 m/s, a leader 20 m ahead
    car = easy.Car(jnp.asarray(obs))
    assert np.allclose(car.speed, 7.5) and np.allclose(car.lead_gap, 20.0) and bool(car.has_lead.all()) and not bool(car.red.any())


def test_driver_maps_acceleration_to_the_command():
    pol = easy.driver(lambda car: jnp.where(car.speed > 5, -4.0, 1.5))
    obs = np.zeros((3, 28), np.float32)
    obs[0, 0] = 1.0  # 15 m/s
    u, aux = pol(jnp.asarray(obs), jax.random.PRNGKey(0))
    assert u.shape == (3, 2) and aux.shape == (3, 2)
    assert np.isclose(float(u[0, 0]), -0.5) and np.isclose(float(u[1, 0]), 0.5)  # -4 / 8 and 1.5 / 3
    assert float(jnp.abs(u[:, 1]).max()) == 0.0


def test_idm_brakes_for_a_close_leader():
    obs = np.zeros((2, 28), np.float32)
    obs[:, 0], obs[:, 3], obs[:, 16] = 10 / 15, 1.0, 1.0
    obs[0, 8], obs[0, 9], obs[0, 10] = 1.0, 5 / 50, 0.0  # a stopped car 5 m ahead
    u, _ = easy.drivers.idm()(jnp.asarray(obs), None)
    assert float(u[0, 0]) < -0.2 < 0 < float(u[1, 0])


def test_simulate_runs_and_summarises():
    run = easy.simulate("oakland", minutes=0.3, copies=1, warm_minutes=0.2, av_share=0.3, driver=easy.drivers.idm())
    s = run.summary()
    assert s["vehicle_hours"] > 0 and s["copies"] == 1 and s["crashes_per_1000_vehicle_hours_95ci"][1] > 0
    assert isinstance(s["mean_speed_kmh"], float)


def test_from_numpy_runs_under_vmap():
    pol = compat.from_numpy(lambda o: np.tile([0.1, 0.0], (len(o), 1)))
    obs = jnp.zeros((3, 5, 28))
    u = jax.vmap(lambda o: pol(o, None)[0])(obs)
    assert u.shape == (3, 5, 2) and np.allclose(np.asarray(u)[..., 0], 0.1)


def test_scene_to_table_has_one_row_per_vehicle_step():
    scene = {"dt": 0.2, "vehicles": [{"id": 0, "kind": "car", "av": False, "start": 3, "length": 4.5, "crashed": True, "crash_frame": 1,
                                      "x": [0, 1], "y": [0, 0], "yaw": [0, 0], "v": [5, 5]},
                                     {"id": 1, "kind": "av", "av": True, "start": 4, "length": 4.6, "crashed": False, "crash_frame": None,
                                      "x": [9], "y": [1], "yaw": [0], "v": [3]}]}
    t = compat.scene_to_table(scene)
    n = len(t) if hasattr(t, "__len__") and not isinstance(t, dict) else len(t["frame"])
    assert n == 3
    crashed = np.asarray(t["crashed_here"])
    assert crashed.sum() == 1


def test_to_torch_matches_the_jax_policy():
    torch = pytest.importorskip("torch")
    from av_world import data_path
    from avrl import xppo as P
    net = compat.to_torch(data_path("policy.npz"))
    x = np.random.default_rng(0).normal(size=(4, 28)).astype(np.float32)
    ref = np.asarray(P.mean(P.load(data_path("policy.npz")), x))
    assert np.abs(net(torch.from_numpy(x)).detach().numpy() - ref).max() < 1e-5


def test_gym_env_steps():
    env = compat.GymEnv("oakland", max_steps=5, warm_s=20.0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (28,) and "ego_slot" in info
    for _ in range(5):
        obs, r, term, trunc, info = env.step(np.zeros(2, np.float32))
        if term or trunc:
            break
    assert trunc or term and np.isfinite(r)


def test_doctor_reports_every_package_and_the_core_is_ready():
    from av_world import doctor
    rows = doctor.check()
    assert {r["group"] for r in rows} >= {"core", "dev", "easy", "torch", "perception", "notebooks", "usd", "web"}
    assert all(r["ok"] for r in rows if r["group"] == "core")
    assert doctor._tuple("0.4.30") < doctor._tuple("0.11.2") and doctor._tuple("2.0") == (2, 0)


def test_pyproject_extras_cover_every_optional_package():
    import re
    import tomllib
    meta = tomllib.load(open("pyproject.toml", "rb"))["project"]
    declared = {re.split(r"[<>=\[]", d)[0].lower().replace("_", "-") for g, ds in meta["optional-dependencies"].items() if g != "all" for d in ds}
    from av_world import doctor
    needed = {pip for _, pip, _, group, _ in doctor.PACKAGES if group not in ("core", "dev")}
    assert needed <= declared | {"jupyterlab"}, needed - declared
