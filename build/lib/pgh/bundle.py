"""Trim the Pittsburgh extract to the three named areas and copy the small data the package needs into av_world/_data, so a
pip-installed av_world runs without the 16 MB city file or a network fetch.

    python -m pgh.bundle      # needs data/pgh_segments.parquet (python -m pgh.fetch) and results/xnet3/policy.npz
"""
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from xnet import network as NW

OUT = "av_world/_data"


def main():
    rows, seen = [], set()
    for box in NW.AREAS.values():
        for r in NW.load_rows(box):
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    keep = ["id", "names", "class", "subclass", "connectors", "access_restrictions", "speed_limits", "geometry", "bbox"]
    src = pq.read_table("data/pgh_segments.parquet", columns=keep)
    ids = {r["id"] for r in rows}
    mask = pa.array([i in ids for i in src["id"].to_pylist()])
    tab = src.filter(mask)
    pq.write_table(tab, f"{OUT}/pgh_areas.parquet", compression="zstd")
    shutil.copy("results/xnet3/policy.npz", f"{OUT}/policy.npz")
    shutil.copy("results/calibration.json", f"{OUT}/calibration.json")
    print(f"{tab.num_rows} segments -> {OUT}/pgh_areas.parquet")


if __name__ == "__main__":
    main()
