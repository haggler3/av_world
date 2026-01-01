"""Log replay: seed the simulator with a real scene's vehicles and compare where they go with where the log says they went
(xnet.avdata.compare_future), for the stored drivers and for the patched ones. Constant velocity is the baseline every prediction
has to beat. Seeded drivers get their traits from seed_state; for the patched run, their (T, s0, amax, b) are redrawn from the trait bank.
"""
import contextlib
import glob
import os

import numpy as np

from xnet import avdata as AV
from xnet import sim as X


@contextlib.contextmanager
def banked(bank, share=1.0, seed=0):
    orig = AV.seed_state
    rng = np.random.default_rng(seed)

    def seeded(*a, **k):
        st, info = orig(*a, **k)
        if bank is None:
            return st, info
        import jax.numpy as jnp
        alive, av, kind = np.asarray(st.alive), np.asarray(st.av), np.asarray(st.kind)
        sel = alive & ~av & (kind == X.CAR) & (rng.random(len(alive)) < share)
        rows = bank[rng.integers(0, len(bank), len(alive))]
        upd = {f: jnp.asarray(np.where(sel, rows[:, c], np.asarray(getattr(st, f)))) for f, c in (("T", 0), ("s0", 1), ("amax", 2), ("b", 3))}
        return st._replace(**upd), info
    AV.seed_state = seeded
    try:
        yield
    finally:
        AV.seed_state = orig


def run(root, behs, bank=None, limit=30, frame=25, horizon_s=5.0):
    """behs: {name: (Behaviour, use_bank)}. -> {name: {"1s": {"sim": m, "cv": m}, ...}} weighted by vehicles (moving, same lane)."""
    dirs = sorted(glob.glob(os.path.join(root, "*")))[:limit]
    acc = {n: {} for n in behs}
    done = 0
    for d in dirs:
        try:
            sc = AV.read_av2(d)
            scene = AV.av2_to_scene(sc, 2)
            net, geo, _ = AV.av2_network(sc)
            m = AV.Matcher(net, geo)
        except Exception:
            continue
        ok = True
        res = {}
        for n, (beh, use_bank) in behs.items():
            try:
                with banked(bank if use_bank else None, float(beh.bank) if use_bank else 0.0):
                    X.set_trait_bank(None)
                    r = AV.compare_future(net, geo, scene, frame, horizon_s, beh=beh._replace(demand=0.0), matcher=m, speed_from_log=True)
                res[n] = r["moving_same_lane"]
            except Exception:
                ok = False
                break
        if not ok:
            continue
        done += 1
        for n, ml in res.items():
            for k in range(1, int(horizon_s) + 1):
                if ml["sim_error_m"][k] is not None:
                    acc[n].setdefault(k, []).append((ml["sim_error_m"][k], ml["const_velocity_error_m"][k], ml["n"][k]))
    out = {"scenes": done}
    for n, by in acc.items():
        out[n] = {}
        for k, v in by.items():
            a = np.array(v)
            w = a[:, 2]
            out[n][f"{k}s"] = {"sim_m": float((a[:, 0] * w).sum() / w.sum()), "const_velocity_m": float((a[:, 1] * w).sum() / w.sum()), "vehicles": int(w.sum())}
    return out
