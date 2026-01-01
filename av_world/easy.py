"""av_world for beginners: the common jobs in one line each, with sensible defaults and plain-unit results.

    from av_world import easy

    run = easy.simulate("oakland", minutes=5, av_share=0.3)          # a city with calibrated human drivers and 30 % AVs
    run.summary()                                                     # crashes, speeds, trips, in plain units (a dict)
    run.show()                                                        # a map of every vehicle
    run.gif("city.gif", seconds=30)                                   # an animation (continues the run)
    run.save("trips.csv")                                             # trajectories: .csv .parquet .json (CARLA/OpenUSD-ready scene)

    @easy.driver                                                      # write a driver in metres, seconds and m/s^2
    def two_second_rule(car):
        jnp = easy.jnp
        follow = jnp.where(car.has_lead & (car.lead_gap < 2 * car.speed + 4), -2 * car.brake_needed_for_lead - 1, 3.0)
        stop = jnp.where(car.has_stop & (car.stop_gap < 2 * car.speed + 8), -2 * car.brake_needed_for_stop - 1, 3.0)
        return jnp.minimum(jnp.minimum(follow, stop), jnp.where(car.speed < car.desired_speed, 1.2, -0.5))

    easy.score(two_second_rule)                                       # AVBench: the same traffic as the leaderboard (43.4)
    easy.compare({"mine": two_second_rule, "IDM": easy.drivers.idm(), "rule-based": None})
    policy = easy.train(updates=25)                                   # PPO, every AV driven by one policy
    easy.export_perception("data/bus", scenario="bus_dartout")        # cameras + 3-D boxes + sensor detections

Everything here is a thin layer over the research code (xnet.sim, avrl.xppo, avrl.bench, bev.export), so results are the same
numbers the rest of the repository reports. Drivers are JAX functions: use `easy.jnp` (jax.numpy) inside them. For a numpy, PyTorch,
Stable-Baselines3 or Gymnasium workflow see av_world.compat.
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:  # importing av_world.easy should be cheap to read about; jax is needed the moment anything runs
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = jnp = None

OUT = "results/easy"


# ------------------------------------------------------------------------------------------------- what exists
def cities():
    """Names accepted wherever a city is asked for (also: an .osm file path or a (west, south, east, north) box)."""
    from xnet import cities as CI
    from xnet import network as NW
    return list(NW.AREAS) + list(CI.CITIES)


def scenarios():
    """{name: description}: the famous-incident archetypes (for simulate / export_perception) and the AVBench scenarios (for score)."""
    from avlab import famous as FM
    from avrl import bench as B
    out = {n: v[0] for n, v in FM.SCENARIOS.items()}
    for n, (area, over) in B.SCENARIOS.items():
        out.setdefault(n, f"AVBench scenario in {area}" + (f": {', '.join(f'{k}={v}' for k, v in list(over.items())[:3])}" if over else ""))
    return out


def _build(city):
    from xnet import cities as CI
    from xnet import network as NW
    if isinstance(city, str) and city in CI.CITIES:
        return CI.build(city)
    if isinstance(city, str) and os.path.exists(city):
        from xnet import osm
        return osm.import_osm(city)
    return NW.build(city)


def world(av_share=0.25, scenario=None, intensity=1.0, risk=1.0, **knobs):
    """The traffic: calibrated human drivers (published crash rates by type), `av_share` of new trips are AVs. `scenario`: a
    famous-incident archetype (easy.scenarios()) at `intensity` (0.4-1.2). `risk`: every human mistake rate x this. Any other
    xnet.sim.Behaviour field (demand=1.5, friction=0.6, ped_scale=2, ...) can be given directly."""
    from avlab import adversary as AD
    from avlab import famous as FM
    from avrl import ladder as LD
    from xnet import sim as X
    if scenario:
        if scenario not in FM.SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; one of {list(FM.SCENARIOS)}")
        knobs = {**FM.SCENARIOS[scenario][2](intensity), **knobs}
    b = X.Behaviour(**dict(LD.knobs())) if risk == 1 else AD.world(risk)
    return b._replace(p_av=av_share, **knobs)


# ------------------------------------------------------------------------------------------------- drivers
class Car:
    """What an AV sees, in plain units (each a JAX array with one entry per vehicle). Booleans are True/False arrays."""
    FIELDS = {  # name: (observation index, scale, is boolean)
        "speed": (0, 15, False), "accel": (1, 3, False), "speed_limit": (2, 15, False), "desired_speed": (3, 15, False),
        "lane_end": (4, 100, False), "on_link": (5, 1, True), "has_stop": (6, 1, True), "stop_gap": (7, 100, False),
        "has_lead": (8, 1, True), "lead_gap": (9, 50, False), "lead_speed": (10, 15, False), "closing_speed": (11, 10, False),
        "red": (12, 1, True), "yellow": (13, 1, True), "stop_sign": (14, 1, True), "must_yield": (15, 1, True),
        "junction_speed_cap": (16, 15, False), "gap_ok": (17, 1, True), "room_after_junction": (18, 1, True),
        "crosswalk_busy": (19, 1, True), "time_to_conflict": (20, 10, False), "must_change_lane": (21, 1, True),
        "waiting": (22, 10, False), "may_enter": (23, 1, True), "brake_needed_for_stop": (24, 8, False),
        "brake_needed_for_lead": (25, 8, False), "headway": (26, 3, False),
    }
    UNITS = {"speed": "m/s", "accel": "m/s^2", "speed_limit": "m/s", "desired_speed": "m/s", "lane_end": "m", "stop_gap": "m",
             "lead_gap": "m", "lead_speed": "m/s", "closing_speed": "m/s", "junction_speed_cap": "m/s", "time_to_conflict": "s",
             "waiting": "s", "brake_needed_for_stop": "m/s^2", "brake_needed_for_lead": "m/s^2", "headway": "s"}

    def __init__(self, obs):
        self.obs = obs
        for k, (i, s, b) in self.FIELDS.items():
            setattr(self, k, obs[..., i] > 0.5 if b else obs[..., i] * s)

    @classmethod
    def describe(cls):
        return {k: cls.UNITS.get(k, "True/False" if b else "") for k, (_, _, b) in cls.FIELDS.items()}


def accel_to_command(a):
    """m/s^2 -> the simulator's command u0 in [-1, 1] (+1 = +3 m/s^2, -1 = -8 m/s^2)."""
    a = jnp.clip(a, -8.0, 3.0)
    return jnp.where(a >= 0, a / 3.0, a / 8.0)


def driver(fn: Callable) -> Callable:
    """Turn fn(car) -> acceleration [m/s^2] (or (acceleration, caution)) into a policy the simulator runs. `caution` > 0 waits for
    bigger gaps at junctions (0 = default). The result is a normal policy f(obs, key) -> (u, aux)."""
    def policy(obs, key):
        out = fn(Car(obs))
        a, caution = (out if isinstance(out, tuple) else (out, 0.0))
        a = jnp.broadcast_to(jnp.asarray(a, jnp.float32), obs.shape[:-1])
        c = jnp.broadcast_to(jnp.asarray(caution, jnp.float32), obs.shape[:-1])
        return jnp.stack([accel_to_command(a), c], -1), jnp.zeros(obs.shape[:-1] + (2,))
    policy.__name__ = getattr(fn, "__name__", "driver")
    policy.__doc__ = fn.__doc__
    return policy


class drivers:  # noqa: N801  (a namespace of ready-made drivers)
    """Ready-made drivers: easy.drivers.idm(), .cautious(), .ppo() (the bundled trained policy), .rule_based (None: the
    simulator's own AV), .load(path) (a PPO checkpoint)."""
    rule_based = None

    @staticmethod
    def idm(T=1.5, s0=2.5, a_max=1.5, b=2.0):
        """The Intelligent Driver Model (Treiber 2000): follow the leader, treat a stop line as a stopped leader."""
        @driver
        def idm(car):
            v = car.speed
            v0 = jnp.maximum(jnp.minimum(car.desired_speed, car.junction_speed_cap + 0.5), 0.5)

            def term(gap, dv):
                s_star = s0 + jnp.maximum(v * T + v * dv / (2 * jnp.sqrt(a_max * b)), 0.0)
                return (s_star / jnp.maximum(gap, 0.1)) ** 2
            inter = jnp.maximum(jnp.where(car.has_lead, term(car.lead_gap, v - car.lead_speed), 0.0),
                                jnp.where(car.has_stop, term(car.stop_gap, v), 0.0))
            return a_max * (1 - (v / v0) ** 4 - inter)
        return idm

    @staticmethod
    def cautious():
        """IDM with a 2.5 s headway and extra caution at junctions."""
        base = drivers.idm(T=2.5, a_max=1.2)

        def cautious(obs, key):
            u, aux = base(obs, key)
            return u.at[..., 1].set(1.0), aux
        return cautious

    @staticmethod
    def load(path):
        from avrl import xppo as P
        return P.deterministic(P.load(path))

    @staticmethod
    def ppo():
        from av_world import data_path
        return drivers.load(data_path("policy.npz"))


def _policy(d):
    """None (rule-based), a path, a Policy, or a function."""
    if d is None or callable(d):
        return d.fn if isinstance(d, Policy) else d
    if isinstance(d, str):
        return drivers.load(d)
    raise TypeError(f"a driver is None, a checkpoint path, or a function f(obs, key) -> (u, aux); got {type(d)}")


# ------------------------------------------------------------------------------------------------- simulate
@dataclass
class Run:
    """A finished simulation; the copies keep running if you ask for more (gif, save, more)."""
    city: str
    sim: object
    net: object
    geo: object
    info: object
    m0: np.ndarray
    seconds: float
    settings: dict = field(default_factory=dict)

    def more(self, minutes=1.0):
        """Simulate longer; summary() then covers the whole measured time."""
        self.sim.run(60.0 * minutes)
        self.seconds += 60.0 * minutes
        return self

    def counters(self):
        from xnet import sim as X
        return dict(zip(X.METRICS, (np.asarray(self.sim.st.m) - self.m0).sum(0)))

    def summary(self):
        """Plain-unit results summed over all copies since the warm-up."""
        from xnet import sim as X
        d = {k: float(v) for k, v in self.counters().items()}
        vh, av_h = d["veh_s"] / 3600, d["av_veh_s"] / 3600
        crashes = sum(d[k] for k in X.CRASH_NAMES)
        lo, hi = _poisson_interval(crashes, max(vh, 1e-9) / 1000)
        return {"city": self.city, "copies": int(self.sim.st.m.shape[0]), "simulated_minutes_per_copy": round(self.seconds / 60, 1),
                "vehicle_hours": round(vh, 1), "av_hours": round(av_h, 1), "trips_finished": int(d["finished"]),
                "mean_speed_kmh": round(3.6 * d["veh_m"] / max(d["veh_s"], 1), 1),
                "av_speed_kmh": round(3.6 * d["av_veh_m"] / max(d["av_veh_s"], 1), 1) if av_h else None,
                "crashes": int(crashes), "crashes_per_1000_vehicle_hours": round(1000 * crashes / max(vh, 1e-9), 2),
                "crashes_per_1000_vehicle_hours_95ci": [round(lo, 2), round(hi, 2)],
                "av_crashes": int(d["av_crash"]), "crashes_by_type": {k: int(d[k]) for k in X.CRASH_NAMES if d[k]},
                "red_light_runs": int(d["red_runs"]), "pedestrian_crossings": int(d["peds"]), **self.settings}

    def __repr__(self):
        s = self.summary()
        return (f"Run({s['city']}: {s['copies']} copies x {s['simulated_minutes_per_copy']} min, {s['vehicle_hours']} vehicle-hours, "
                f"{s['crashes']} crashes, {s['mean_speed_kmh']} km/h)")

    def _copy(self, i=0):
        return jax.tree.map(lambda a: np.asarray(a[i]), self.sim.st)

    def show(self, copy=0, ax=None, title=None):
        """A map of the city with every vehicle, coloured by speed. Returns the matplotlib figure."""
        import matplotlib.pyplot as plt

        from xnet import render as R
        if self.geo is None:
            raise ValueError("this city has no drawing geometry")
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=R.BG)
        ax.set_facecolor(R.GROUND)
        R.Map(self.net, self.geo, self.info).draw(ax, detail=False, labels=False)
        R.snapshot_artists(ax, self.net, self.geo, R.Tracks(self.net, self.geo), self._copy(copy), False, R.signal_heads(self.net, self.geo))
        ax.set_aspect("equal")
        ax.axis("off")
        s = self.summary()
        ax.set_title(title or f"{self.city}: {s['crashes']} crashes, {s['mean_speed_kmh']} km/h mean speed (blue fast, red stopped)",
                     color="#e6edf3", fontsize=10, loc="left")
        return fig or ax.figure

    def record(self, seconds=30.0, copy=0, every=1):
        """Continue the run and keep per-step snapshots of one copy (for gif / save)."""
        frames = self.sim.run(seconds, record=(copy, every))
        self.seconds += seconds
        return frames

    def gif(self, path="av_world.gif", seconds=30.0, copy=0, zoom=None, fps=8):
        """Continue the run for `seconds` and write an animation. zoom=((x, y), radius_m) for a close-up."""
        from xnet import render as R
        frames = self.record(seconds, copy, every=2)
        R.animate(self.net, self.geo, self.info, frames, path, zoom=zoom, title=f"av_world: {self.city}", fps=fps)
        return path

    def save(self, path, seconds=30.0, copy=0):
        """Continue the run for `seconds` recording one copy, and write its trajectories: .csv / .parquet (one row per vehicle per
        step), .json (the scene format that av-world export-nvidia and CARLA replay read), or .xodr (the road network as
        OpenDRIVE, plus the .json scene and a CARLA replay script next to it)."""
        from xnet import carla_export as CE
        frames = self.record(seconds, copy)
        ext = os.path.splitext(path)[1].lower()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if ext == ".xodr":
            CE.export_frames(self.net, self.geo, frames, path[:-5])
            return path
        scene = CE.trajectories(self.net, self.geo, frames)
        if ext == ".json":
            json.dump(scene, open(path, "w"), separators=(",", ":"))
            return path
        from av_world import compat
        df = compat.scene_to_table(scene)
        if ext == ".parquet":
            df.to_parquet(path) if hasattr(df, "to_parquet") else compat._write_parquet(df, path)
        else:
            df.to_csv(path, index=False) if hasattr(df, "to_csv") else compat._write_csv(df, path)
        return path


def simulate(city="oakland", minutes=5.0, av_share=0.25, scenario=None, driver=None, copies=4, warm_minutes=1.0, seed=0, **world_kw):
    """Run `copies` independent copies of a city for `minutes` after a `warm_minutes` warm-up. `driver` drives the AVs (None: the
    simulator's rule-based AV; a function, a Policy from easy.train, or a checkpoint path). Returns a Run."""
    from xnet import sim as X
    net, geo, info = _build(city)
    beh = world(av_share, scenario, **world_kw)
    sim = X.Sim(net, n_envs=copies, beh=beh, seed=seed, policy=_policy(driver))
    sim.run(60.0 * warm_minutes)
    m0 = np.asarray(sim.st.m)
    sim.run(60.0 * minutes)
    settings = {"av_share": av_share, "scenario": scenario, "driver": getattr(_policy(driver), "__name__", "rule-based") if driver is not None else "rule-based"}
    return Run(str(city), sim, net, geo, info, m0, 60.0 * minutes, settings)


def sweep(param, values, city="oakland", minutes=5.0, copies=8, **kw):
    """simulate() once per value of one setting, e.g. sweep("av_share", [0, 0.5, 1]) or sweep("demand", [0.5, 1, 2]).
    Returns a list of summaries (a pandas DataFrame if pandas is installed)."""
    rows = []
    for v in values:
        r = simulate(city, minutes=minutes, copies=copies, **{**kw, param: v}).summary()
        rows.append({param: v, **{k: r[k] for k in ("vehicle_hours", "crashes", "crashes_per_1000_vehicle_hours", "mean_speed_kmh",
                                                     "trips_finished", "av_crashes")}})
    return _frame(rows)


# ------------------------------------------------------------------------------------------------- score / compare
@dataclass
class Score:
    name: str
    av_score: float
    scenarios: dict

    def checks(self):
        """{scenario: {check: passed}}: the eight absolute checks of avrl.scorecard."""
        from avrl import scorecard as SC
        return {n: SC.absolute_checks(s["metrics"]) for n, s in self.scenarios.items()}

    def row(self):
        n = len(self.scenarios)
        m = lambda k: float(np.mean([s["metrics"][k] for s in self.scenarios.values()]))  # noqa: E731
        return {"driver": self.name, "av_world_score": round(self.av_score, 1),
                "checks_passed": f"{sum(sum(c.values()) for c in self.checks().values())}/{8 * n}",
                "crashes_per_AV_hour": round(m("crash_per_h"), 3), "crashes_per_AV_hour_no_filter": round(m("crash_per_h_unaided"), 3),
                "comfort": round(m("comfort_pass"), 3), "progress_vs_humans": round(m("progress"), 3),
                "filter_share": round(m("filter_share"), 3)}

    def __repr__(self):
        r = self.row()
        return f"Score({r['driver']}: av_world score {r['av_world_score']}, checks {r['checks_passed']}, crashes/AV-h unaided {r['crashes_per_AV_hour_no_filter']})"


def score(drv=None, scenarios=("nominal",), name=None):
    """Score a driver on AVBench scenarios (~1 min each on a CPU): the same warm states and random numbers as the leaderboard."""
    from avrl import bench as B
    p = _policy(drv)
    target = drv.path if isinstance(drv, Policy) else drv if isinstance(drv, str) else p  # checkpoints: scored like the leaderboard
    r = B.evaluate(target, list(scenarios))
    return Score(name or (getattr(p, "__name__", None) if p is not None else "rule-based") or "driver", r["av_score"], r["scenarios"])


def compare(named, scenarios=("nominal",), leaderboard=True):
    """Score several drivers {name: driver} and print a table; with `leaderboard` also the leaderboard's rule-based AV and best PPO
    on the same scenarios (no re-run). Returns the rows (a DataFrame with pandas)."""
    rows = [score(d, scenarios, name=n).row() for n, d in named.items()]
    if leaderboard:
        from avrl import bench as B
        lb = json.load(open(f"{B.OUT}/leaderboard.json"))["scores"]
        for p in ("rule-based AV", "PPO + barrier teacher, speed + jaywalkers (phase 3)"):
            if p in lb and all(n in lb[p] for n in scenarios):
                rows.append(Score(p + " [leaderboard]", float(np.mean([lb[p][n]["av_score"] for n in scenarios])),
                                  {n: lb[p][n] for n in scenarios}).row())
    _print_table(rows)
    return _frame(rows)


# ------------------------------------------------------------------------------------------------- train
@dataclass
class Policy:
    """A trained driving policy: call it like any driver, save it, plot its training."""
    path: str
    log: list

    @property
    def fn(self):
        return drivers.load(self.path)

    def __call__(self, obs, key):
        return self.fn(obs, key)

    @property
    def __name__(self):
        return os.path.basename(os.path.dirname(self.path)) or "policy"

    def save(self, path):
        import shutil
        shutil.copy(self.path, path)
        return path

    def plot(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
        u = [r["update"] for r in self.log]
        for ax, k, t in zip(axs, ("reward_per_step", "av_speed_kmh", "av_shield_share"),
                            ("reward per AV-step", "AV speed, km/h", "share of steps the safety filter cut in")):
            ax.plot(u, [r[k] for r in self.log])
            ax.set_title(t, loc="left", fontsize=10)
            ax.set_xlabel("PPO update")
        fig.tight_layout()
        return fig


def train(updates=25, city="oakland", start="bundled", reward=None, copies=4, out=None, seed=0, quiet=False):
    """Train a driving policy with PPO (avrl.xppo): every AV in `copies` copies of the city is driven by one shared policy.
    `start`: "bundled" (fine-tune the shipped policy), "scratch" (behaviour cloning of the rule-based AV, then PPO) or a checkpoint
    path. `reward`: overrides of the reward weights, e.g. {"headway": 1.0} (see easy.reward_weights()). ~2.5 s per update on 4
    CPU cores. Returns a Policy."""
    import contextlib
    import io

    from av_world import data_path
    from avrl import xppo as P
    out = out or os.path.join(OUT, f"ppo_{time.strftime('%Y%m%d_%H%M%S')}")
    init = data_path("policy.npz") if start == "bundled" else (None if start == "scratch" else start)
    kw = dict(area=city, n_envs=copies, warm=60.0, reward=reward or {})
    if init is None:
        kw["bc_updates"] = 15
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        P.train(out=out, updates=updates, seed=seed, init=init, **kw)
    return Policy(os.path.join(out, "policy.npz"), json.load(open(os.path.join(out, "train_log.json"))))


def reward_weights():
    """The default reward weights per 0.2 s step (pass overrides to train(reward=...))."""
    from xnet import sim as X
    return X.Reward()._asdict()


# ------------------------------------------------------------------------------------------------- data out
def export_perception(out, scenario=None, city="chicago", seconds=20.0, egos=2, width=176, height=96, seed=0):
    """Camera images, 3-D boxes and sensor-model detections around `egos` AVs, nuScenes-style (needs `pip install pyvista`)."""
    from bev import export as E
    return E.export(out, area=city, scenario=scenario, seconds=seconds, egos=egos, width=width, height=height, seed=seed)


def export_3d(out, scenario=None, city="chicago", seconds=20.0, find_crash=0.0):
    """An OpenUSD stage (NVIDIA Omniverse / Isaac), the scene JSON, a camera control video and a side-by-side clip."""
    from xnet import nvidia
    return nvidia.export(out, area=city, scenario=scenario, seconds=seconds, find_crash=find_crash)


# ------------------------------------------------------------------------------------------------- helpers
def _poisson_interval(k, t, conf=0.95):
    from scipy.stats import chi2
    a = 1 - conf
    lo = chi2.ppf(a / 2, 2 * k) / 2 / t if k > 0 else 0.0
    return float(lo), float(chi2.ppf(1 - a / 2, 2 * k + 2) / 2 / t)


def _frame(rows):
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def _print_table(rows):
    if not rows:
        return
    keys = list(rows[0])
    w = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    print("  ".join(k.ljust(w[k]) for k in keys))
    for r in rows:
        print("  ".join(str(r[k]).ljust(w[k]) for k in keys))
