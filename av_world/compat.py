"""av_world with the tools people already use: Gymnasium, PettingZoo, numpy / PyTorch / Stable-Baselines3 policies, pandas, and
the export formats of other simulators.

    from av_world import compat

    env = compat.GymEnv("oakland")                    # Gymnasium: drive ONE AV (the ego) among the city's traffic; SB3 works on it
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    par = compat.ParallelEnv("oakland")               # PettingZoo parallel API: every AV is an agent ("av_<slot>")

    policy = compat.from_numpy(lambda obs: ...)       # any numpy function obs [N, 28] -> u [N, 2] as a simulator policy
    policy = compat.from_torch(torch_module)          # a PyTorch module obs -> u
    policy = compat.from_sb3(model)                   # a Stable-Baselines3 model trained on GymEnv, back in the full city
    net = compat.to_torch(av_world.data_path("policy.npz"))   # the JAX PPO policy as a torch.nn.Sequential (deploy anywhere)
    df = compat.scene_to_table(scene)                 # trajectories as a pandas DataFrame (CSV / Parquet)

Formats in and out (all in this repository): OpenStreetMap in (xnet.osm), Argoverse 2 and Waymo Open Motion in (xnet.avdata),
OpenDRIVE + CARLA replay out (xnet.carla_export), OpenUSD out (xnet.nvidia), nuScenes-style perception out (bev.export),
CSV / Parquet / JSON trajectories out (here and av_world.easy), SIL/HIL over TCP for any language (av_world.sil).
Gymnasium, PettingZoo, torch and SB3 are optional: nothing here imports them until it is used.
"""
import numpy as np

OBS_DIM, ACT_DIM = 28, 2


# ------------------------------------------------------------------------------------------------- policies from other frameworks
def from_numpy(fn, name=None):
    """A numpy function u = fn(obs) (obs [N, 28] float32 -> u [N, 2]) as a simulator policy f(obs, key) -> (u, aux). It runs on the
    host through jax.pure_callback, so it is slower than a JAX policy (fine for evaluation, SIL-style tests and small runs)."""
    import jax
    import jax.numpy as jnp

    def host(obs):
        o = np.asarray(obs, np.float32)
        u = np.asarray(fn(o.reshape(-1, OBS_DIM)), np.float32).reshape(o.shape[:-1] + (ACT_DIM,))
        return np.clip(u, -1.0, 1.0)

    def policy(obs, key):
        u = jax.pure_callback(host, jax.ShapeDtypeStruct(obs.shape[:-1] + (ACT_DIM,), jnp.float32), obs, vmap_method="broadcast_all")
        return u, jnp.zeros(obs.shape[:-1] + (2,))
    policy.__name__ = name or getattr(fn, "__name__", "numpy_policy")
    return policy


def from_torch(module, device="cpu", name=None):
    """A PyTorch module obs [N, 28] -> u [N, 2] (or more outputs: the first 2 are used) as a simulator policy."""
    import torch
    module = module.to(device).eval()

    def fn(o):
        with torch.no_grad():
            u = module(torch.from_numpy(np.array(o, np.float32)).to(device))
        u = u[0] if isinstance(u, (tuple, list)) else u
        return u[..., :ACT_DIM].cpu().numpy()
    return from_numpy(fn, name or type(module).__name__)


def from_sb3(model, name=None):
    """A Stable-Baselines3 (or any .predict(obs, deterministic=True)) model trained on GymEnv, driving every AV of the city."""
    return from_numpy(lambda o: model.predict(o, deterministic=True)[0], name or type(model).__name__)


def to_torch(path_or_params):
    """The PPO policy's mean action network (avrl.xppo checkpoint or params) as torch.nn.Sequential: Linear-Tanh-Linear-Tanh-Linear."""
    import torch
    from avrl import xppo as P
    params = P.load(path_or_params) if isinstance(path_or_params, str) else path_or_params
    layers = []
    for i, layer in enumerate(params["pi"]):
        w, b = np.asarray(layer["w"]), np.asarray(layer["b"])
        lin = torch.nn.Linear(*w.shape)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(np.array(w.T, np.float32)))
            lin.bias.copy_(torch.from_numpy(np.array(b, np.float32)))
        layers.append(lin)
        if i < len(params["pi"]) - 1:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


def to_onnx(path_or_params, out="policy.onnx"):
    """The PPO policy as ONNX (for C++, ROS, TensorRT ...): input "obs" [N, 28] -> output "u" [N, 2]. Needs torch."""
    import torch
    net = to_torch(path_or_params)
    torch.onnx.export(net, torch.zeros(1, OBS_DIM), out, input_names=["obs"], output_names=["u"], dynamic_axes={"obs": {0: "n"}, "u": {0: "n"}},
                      dynamo=False)  # the classic exporter: no onnxscript needed
    return out


# ------------------------------------------------------------------------------------------------- environments
def _space(low, high, shape):
    try:
        from gymnasium import spaces
        return spaces.Box(low, high, shape, np.float32)
    except ImportError:  # a minimal stand-in with the same attributes
        class Box:
            def __init__(self):
                self.low, self.high, self.shape, self.dtype = np.full(shape, low, np.float32), np.full(shape, high, np.float32), shape, np.float32

            def sample(self):
                return np.random.uniform(np.maximum(self.low, -1), np.minimum(self.high, 1)).astype(np.float32)

            def contains(self, x):
                return np.shape(x) == self.shape
        return Box()


try:
    import gymnasium as _gym
    _GymBase = _gym.Env
except ImportError:  # pragma: no cover
    _GymBase = object


class _Stepper:
    """One copy of a city; the AVs in `controlled` (a mask) take the given commands, every other AV uses `others`."""

    def __init__(self, area, beh, shield, others, n_slots):
        import jax
        import jax.numpy as jnp

        from av_world import env as E
        from xnet import sim as X
        self.X, self.jnp = X, jnp
        self.env = E.Env(area, n_envs=1, beh=beh, shield=shield, n_slots=n_slots)
        net, nh, b = self.env.net, self.env.nh, self.env.beh
        if others == "ppo":
            others = E.load_policy()
        elif others == "idm":
            from av_world import easy
            others = easy.drivers.idm()

        def step(s, u_ext, ctl):
            def pol(obs, key, mask):
                u_def = others(obs, key)[0]
                return jnp.where(ctl[:, None], u_ext, u_def), jnp.zeros(obs.shape[:-1] + (2,))
            return X.step(s, net, b, nh, rl=(pol, s.key, True))
        self.step = jax.jit(step)
        self.n = n_slots

    def reset(self, seed, warm_s):
        import jax
        st = self.env.reset(seed=seed, warm_s=warm_s)
        return jax.tree.map(lambda a: a[0], st)


class GymEnv(_GymBase):
    """Gymnasium single-agent view: you drive one AV (the ego) in a full city; every other AV uses the bundled PPO policy (others="ppo"), the
    Intelligent Driver Model ("idm") or any policy function, humans and pedestrians are the calibrated simulator. observation: the ego's 28 features
    (av_world.sil.OBS_NAMES); action: [acceleration command in [-1, 1], gap acceptance in [-1, 1]]; reward: the simulator's per-step
    reward (xnet.sim.Reward). The episode ends when the ego's trip ends (truncated) or it crashes (terminated), or after
    `max_steps`. The safety envelope is on by default (shield="off" for the raw command)."""
    metadata = {"render_modes": []}

    def __init__(self, area="oakland", av_share=0.25, others="ppo", shield="idm", warm_s=60.0, max_steps=1500, n_slots=1024, beh=None):
        self.area, self.warm_s, self.max_steps = area, warm_s, max_steps
        b = beh or _calibrated(av_share)
        self._s = _Stepper(area, b, shield, others, n_slots)
        self.observation_space = _space(-np.inf, np.inf, (OBS_DIM,))
        self.action_space = _space(-1.0, 1.0, (ACT_DIM,))
        self.state, self.ego, self.k, self._seed = None, None, 0, 0

    def _pick(self, tr):
        valid = np.asarray(tr["valid"])
        ids = np.nonzero(valid)[0]
        return int(ids[0]) if len(ids) else None

    def reset(self, seed=None, options=None):
        if _GymBase is not object:
            super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
        else:
            self._seed += 1
        self.state = self._s.reset(self._seed, self.warm_s)
        zero = np.zeros((self._s.n, ACT_DIM), np.float32)
        none = np.zeros(self._s.n, bool)
        self.ego = None
        for _ in range(500):  # until an AV is on the road
            self.state, tr = self._s.step(self.state, zero, none)
            self.ego = self._pick(tr)
            if self.ego is not None:
                break
        if self.ego is None:
            raise RuntimeError("no AV appeared; raise av_share")
        self.k = 0
        self._obs = np.asarray(tr["obs"][self.ego], np.float32)
        return self._obs, {"ego_slot": self.ego}

    def step(self, action):
        u = np.zeros((self._s.n, ACT_DIM), np.float32)
        u[self.ego] = np.clip(np.asarray(action, np.float32), -1, 1)
        ctl = np.zeros(self._s.n, bool)
        ctl[self.ego] = True
        self.state, tr = self._s.step(self.state, u, ctl)
        e = self.ego
        crash, done = bool(np.asarray(tr["crash"])[e]), bool(np.asarray(tr["done"])[e])
        reward = float(np.asarray(tr["reward"])[e])
        self.k += 1
        obs = np.asarray(tr["obs"][e], np.float32) if not done else self._obs
        self._obs = obs
        info = {"crash": crash, "red_light": bool(np.asarray(tr["red"])[e]), "near_miss": bool(np.asarray(tr["near"])[e]), "t": float(self.state.t)}
        return obs, reward, crash, (done and not crash) or self.k >= self.max_steps, info


try:
    from pettingzoo import ParallelEnv as _PZBase
except ImportError:  # pragma: no cover
    _PZBase = object


class ParallelEnv(_PZBase):
    """PettingZoo parallel API: every AV in one copy of the city is an agent named "av_<slot>"; agents appear when an AV trip starts
    and leave when it ends or crashes. actions: {agent: [u0, u1]}; AVs without an action use the bundled policy."""
    metadata = {"name": "av_world_parallel_v0", "render_modes": []}

    def __init__(self, area="oakland", av_share=0.25, others="ppo", shield="idm", warm_s=60.0, max_steps=1500, n_slots=1024, beh=None):
        self._s = _Stepper(area, beh or _calibrated(av_share), shield, others, n_slots)
        self.warm_s, self.max_steps = warm_s, max_steps
        self.possible_agents = [f"av_{i}" for i in range(n_slots)]
        self.agents, self._obs, self.k = [], {}, 0

    def observation_space(self, agent):
        return _space(-np.inf, np.inf, (OBS_DIM,))

    def action_space(self, agent):
        return _space(-1.0, 1.0, (ACT_DIM,))

    def _read(self, tr):
        valid = np.asarray(tr["valid"])
        obs = np.asarray(tr["obs"], np.float32)
        return {f"av_{i}": obs[i] for i in np.nonzero(valid)[0]}

    def reset(self, seed=None, options=None):
        self.state = self._s.reset(int(seed or 0), self.warm_s)
        self.state, tr = self._s.step(self.state, np.zeros((self._s.n, ACT_DIM), np.float32), np.zeros(self._s.n, bool))
        self._obs = self._read(tr)
        self.agents, self.k = list(self._obs), 0
        return dict(self._obs), {a: {} for a in self.agents}

    def step(self, actions):
        u = np.zeros((self._s.n, ACT_DIM), np.float32)
        ctl = np.zeros(self._s.n, bool)
        for a, act in actions.items():
            i = int(a[3:])
            u[i], ctl[i] = np.clip(np.asarray(act, np.float32), -1, 1), True
        self.state, tr = self._s.step(self.state, u, ctl)
        self.k += 1
        crash, done, rew = np.asarray(tr["crash"]), np.asarray(tr["done"]), np.asarray(tr["reward"])
        obs = self._read(tr)
        rewards, terms, truncs, infos = {}, {}, {}, {}
        for a in self.agents:
            i = int(a[3:])
            rewards[a] = float(rew[i])
            terms[a] = bool(crash[i])
            truncs[a] = bool(done[i] and not crash[i]) or self.k >= self.max_steps
            infos[a] = {}
        new = [a for a in obs if a not in self.agents]  # trips that started this step
        self.agents = [a for a in self.agents if not (terms[a] or truncs[a]) and a in obs] + new
        return {a: obs[a] for a in self.agents}, rewards, terms, truncs, infos


def _calibrated(av_share):
    from avrl import ladder as LD
    from xnet import sim as X
    return X.Behaviour(**{**dict(LD.knobs()), "p_av": av_share})


# ------------------------------------------------------------------------------------------------- data
COLUMNS = ("vehicle_id", "kind", "is_av", "frame", "t", "x", "y", "yaw", "speed", "length", "crashed_here")


def scene_to_table(scene):
    """A scene (xnet.carla_export.trajectories / av-world export-nvidia scene.json) as one row per vehicle per step, columns
    COLUMNS (metres, radians, m/s, seconds): a pandas DataFrame if pandas is installed, else a dict of numpy columns."""
    cols = {c: [] for c in COLUMNS}
    dt = scene.get("dt", 0.2)
    for v in scene["vehicles"]:
        n = len(v["x"])
        f = np.arange(v["start"], v["start"] + n)
        cols["vehicle_id"] += [v["id"]] * n
        cols["kind"] += [v["kind"]] * n
        cols["is_av"] += [v["av"]] * n
        cols["frame"] += f.tolist()
        cols["t"] += (f * dt).round(3).tolist()
        for a, b in (("x", "x"), ("y", "y"), ("yaw", "yaw"), ("speed", "v")):
            cols[a] += v[b]
        cols["length"] += [v["length"]] * n
        cols["crashed_here"] += [v["crashed"] and i == v["crash_frame"] for i in range(n)]
    try:
        import pandas as pd
        return pd.DataFrame(cols).sort_values(["frame", "vehicle_id"], kind="stable").reset_index(drop=True)
    except ImportError:
        return {k: np.asarray(v) for k, v in cols.items()}


def _write_csv(cols, path):
    keys = list(cols)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in zip(*(cols[k] for k in keys)):
            f.write(",".join(str(x) for x in row) + "\n")


def _write_parquet(cols, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({k: np.asarray(v) for k, v in cols.items()}), path)
