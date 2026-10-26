"""av_world: a JAX city-scale traffic and driving simulator for autonomous-vehicle research.

    import av_world
    env = av_world.make("oakland", n_envs=8)             # or an OpenStreetMap file: av_world.make("map.osm")
    state = env.reset(seed=0)
    state, tr = env.step(state, policy=av_world.rule_policy)   # one 0.2 s step of every copy; tr: observations, rewards of the AVs

See README.md; command line: `av-world --help`.
"""
import os

__version__ = "0.4.0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_FALLBACK = {"policy.npz": "results/xnet3/policy.npz", "calibration.json": "results/calibration.json",
             "pgh_areas.parquet": "data/pgh_segments.parquet"}


def data_path(name):
    """Bundled data file (av_world/_data) or, in a source checkout, the file the repository keeps it in."""
    p = os.path.join(_HERE, "_data", name)
    if os.path.exists(p):
        return p
    q = os.path.join(os.path.dirname(_HERE), _FALLBACK.get(name, name))
    if os.path.exists(q):
        return q
    raise FileNotFoundError(f"{name}: not in {_HERE}/_data; run `python -m pgh.bundle` in a source checkout")


def make(area="oakland", **kw):
    """A vectorised environment (see av_world.env.Env)."""
    from .env import Env
    return Env(area, **kw)


def __getattr__(name):  # lazy: importing av_world must not import jax
    if name in ("rule_policy", "Env", "load_policy"):
        from . import env
        return getattr(env, name)
    raise AttributeError(name)
