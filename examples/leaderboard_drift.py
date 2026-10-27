"""Does the published leaderboard still reproduce on today's code? Re-score some of its rows and compare, scenario by scenario.

    python examples/leaderboard_drift.py                       # rule-based AV + the best learned policy, 11 core scenarios (~12 min)
    python examples/leaderboard_drift.py --rows "rule-based AV" --scenarios nominal rush_hour

Every leaderboard number was computed by some version of the simulator. When the simulator changes (a bug fix, a new behaviour),
stored rows silently stop matching what the code does. This prints old vs new per scenario and the largest drift, and writes
results/bench/drift.json. A drift of more than a point means the leaderboard row should be re-run before it is quoted.
"""
import argparse
import json
import os
import time

try:
    from examples import _common as C  # noqa: F401  (puts the repository on the path)
except ImportError:
    import _common as C  # noqa: F401
from avrl import bench as B  # noqa: E402

CHECKPOINTS = {  # leaderboard row -> how to re-score it (None: the rule-based AV)
    "rule-based AV": None,
    "PPO + barrier teacher, speed + jaywalkers (phase 3)": "results/unaided/phase3/policy.npz",
}


def main(rows=tuple(CHECKPOINTS), scenarios=None):
    lb = json.load(open(f"{B.OUT}/leaderboard.json"))
    scenarios = list(scenarios or lb["scenarios"])
    out, t0 = {}, time.time()
    for row in rows:
        path = CHECKPOINTS.get(row)
        if path is not None and not os.path.exists(path):
            print(f"{row}: checkpoint {path} not found, skipped")
            continue
        out[row] = {}
        for n in scenarios:
            new = B.evaluate(path, [n])["scenarios"][n]["av_score"]
            old = lb["scores"][row][n]["av_score"]
            out[row][n] = {"stored": old, "today": new, "drift": new - old}
            print(f"  {row[:40]:40s} {n:14s} stored {old:6.1f}  today {new:6.1f}  drift {new - old:+6.1f}  ({time.time() - t0:.0f} s)", flush=True)
        json.dump(out, open(f"{B.OUT}/drift.json", "w"), indent=1)
    for row, d in out.items():
        worst = max(d.items(), key=lambda kv: abs(kv[1]["drift"]))
        print(f"{row}: largest drift {worst[1]['drift']:+.1f} ({worst[0]}); scenarios off by > 1 point: "
              f"{sum(abs(v['drift']) > 1 for v in d.values())} of {len(d)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="*", default=list(CHECKPOINTS))
    ap.add_argument("--scenarios", nargs="*", default=None)
    a = ap.parse_args()
    main(a.rows, a.scenarios)
