"""Simulated scenes for the comparison: the nominal city (no stress knobs), a few AVs only to centre 10 s windows on, as an AV2 log is
centred on its AV. A list of behaviours records one per copy of the city in one compiled run (the refit's population)."""
import numpy as np

from xnet import sim as X

NOMINAL = dict(p_av=0.05, crn=1.0)


def behaviour(**kw):
    return X.Behaviour(**dict(NOMINAL, **kw))


def record(behs, minutes=4, copies=1, seed=0, per_window=4, stride=25, bank=None):
    """behs: a Behaviour or a list (each gets `copies` copies of the city). -> list of scenes, scene['truth']['env'] // copies = index
    into behs."""
    from autotag import scenes as AS
    from nets import collect as C
    behs = [behs] if isinstance(behs, X.Behaviour) else list(behs)
    X.set_trait_bank(bank)
    try:
        n = len(behs) * copies
        fields = [np.repeat(np.array([float(getattr(b, f)) for b in behs], np.float32), copies) for f in X.Behaviour._fields]
        import jax.numpy as jnp
        beh = X.Behaviour(*[jnp.asarray(f) for f in fields])
        rec, w = AS.record_sim(minutes=minutes, n_envs=n, seed=seed, beh=beh)
        sc = AS.sim_scenes(rec, w, per_window=per_window, stride=stride, seed=seed)
    finally:
        X.set_trait_bank(None)
    for s in sc:
        s["truth"]["beh"] = int(s["truth"]["env"]) // copies
    return sc
