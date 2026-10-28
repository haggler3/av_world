"""`av-world doctor`: are the dependencies installed, new enough, and does the simulator actually run here?

    av-world doctor              # versions against the floors in pyproject.toml, the features each missing package disables
    av-world doctor --smoke      # also: build Oakland, step 4 copies for 20 simulated seconds, check the numbers (~15 s)
    av-world doctor --fix        # pip install what is missing or too old (core + the extras you name with --extras easy,torch,...)

Exit status 0: the core is fine (optional extras only warn). 1: something the simulator needs is missing, too old, or the smoke test failed.
"""
import argparse
import importlib
import importlib.metadata as md
import re
import subprocess
import sys

# (import name, pip name, floor, group, what needs it)
PACKAGES = [
    ("jax", "jax", "0.4.30", "core", "the simulator"), ("numpy", "numpy", "1.26", "core", "everything"),
    ("scipy", "scipy", "1.11", "core", "statistics, calibration"), ("matplotlib", "matplotlib", "3.8", "core", "figures, GIFs"),
    ("PIL", "pillow", "10", "core", "images, GIFs"), ("shapely", "shapely", "2.0", "core", "maps, watch"),
    ("pyarrow", "pyarrow", "14", "core", "map data (parquet)"), ("optax", "optax", "0.2", "core", "PPO training"),
    ("pytest", "pytest", "8", "dev", "the test suite"),
    ("pandas", "pandas", "2.0", "easy", "easy.sweep / compare tables, run.save(.csv)"),
    ("gymnasium", "gymnasium", "0.29", "easy", "compat.GymEnv"), ("pettingzoo", "pettingzoo", "1.24", "easy", "compat.ParallelEnv"),
    ("torch", "torch", "2.2", "torch", "compat.to_torch / from_torch"), ("onnx", "onnx", "1.15", "torch", "compat.to_onnx"),
    ("stable_baselines3", "stable-baselines3", "2.3", "torch", "training with Stable-Baselines3"),
    ("pyvista", "pyvista", "0.43", "perception", "bev.export, 3-D previews"), ("vtk", "vtk", "9.3", "perception", "bev.export"),
    ("nbformat", "nbformat", "5.9", "notebooks", "running the notebooks"), ("nbclient", "nbclient", "0.9", "notebooks", "executing notebooks"),
    ("ipykernel", "ipykernel", "6", "notebooks", "the notebook kernel"),
    ("pxr", "usd-core", "24", "usd", "OpenUSD export (av-world export-nvidia)"),
    ("playwright", "playwright", "1.40", "web", "website screenshots"),
]


def _tuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def check():
    """-> list of dicts: name, group, installed (str or None), floor, ok (installed and new enough), needed_for."""
    rows = []
    for imp, pip, floor, group, why in PACKAGES:
        try:
            importlib.import_module(imp)
            v = md.version(pip)
        except Exception:
            v = None
        ok = v is not None and _tuple(v) >= _tuple(floor)
        rows.append({"import": imp, "pip": pip, "group": group, "installed": v, "floor": floor, "ok": ok, "needed_for": why})
    return rows


def smoke():
    """A tiny end-to-end run: the numbers must be finite, vehicles must exist and move. -> (ok, message)."""
    import time

    import numpy as np
    t0 = time.time()
    from av_world import easy
    run = easy.simulate("oakland", minutes=0.4, copies=2, warm_minutes=0.2, av_share=0.3)
    s = run.summary()
    ok = s["vehicle_hours"] > 0 and np.isfinite(s["mean_speed_kmh"]) and 1 < s["mean_speed_kmh"] < 80 and s["trips_finished"] > 0
    return ok, f"Oakland, 2 copies: {s['vehicle_hours']} vehicle-hours, {s['mean_speed_kmh']} km/h, {s['trips_finished']} trips ({time.time() - t0:.0f} s)"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="av-world doctor")
    ap.add_argument("--smoke", action="store_true", help="also run a 15-second end-to-end simulation")
    ap.add_argument("--fix", action="store_true", help="pip install missing or outdated packages (core, plus --extras)")
    ap.add_argument("--extras", default="", help="comma-separated groups to include with --fix: dev,easy,torch,perception,notebooks,usd,web")
    a = ap.parse_args(argv)
    rows = check()
    want = {"core", "dev"} | {g for g in a.extras.split(",") if g}
    if a.fix:
        todo = [f"{r['pip']}>={r['floor']}" for r in rows if not r["ok"] and r["group"] in want]
        if todo:
            print("pip install", " ".join(todo))
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *todo], check=False)
            rows = check()
    print(f"python {sys.version.split()[0]}")
    bad = False
    last = None
    for r in rows:
        if r["group"] != last:
            print(f"\n[{r['group']}]" + ("  (required)" if r["group"] == "core" else "  (optional: pip install -e \".[%s]\")" % r["group"]))
            last = r["group"]
        mark = "ok  " if r["ok"] else ("OLD " if r["installed"] else "--  ")
        print(f"  {mark} {r['pip']:20s} {r['installed'] or 'not installed':14s} >= {r['floor']:7s} {r['needed_for']}")
        if not r["ok"] and r["group"] == "core":
            bad = True
    try:
        import jax
        print(f"\njax devices: {jax.devices()}")
    except Exception as e:  # pragma: no cover
        print(f"\njax: {e}")
    if a.smoke and not bad:
        ok, msg = smoke()
        print(("smoke test ok: " if ok else "SMOKE TEST FAILED: ") + msg)
        bad = bad or not ok
    missing_opt = sorted({r["group"] for r in rows if not r["ok"] and r["group"] != "core"})
    if missing_opt:
        print("\noptional groups with something missing or old: " + ", ".join(missing_opt) + "  ->  pip install -e \".[all]\" installs them all")
    print("\nREADY" if not bad else "\nNOT READY: fix the core packages above (av-world doctor --fix)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
