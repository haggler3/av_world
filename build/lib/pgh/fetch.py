"""Fetch Pittsburgh's road network from Overture Maps (an open map built largely from OpenStreetMap), straight from its
public S3 bucket. Only the Parquet row groups whose bounding boxes touch Pittsburgh are read.

    python -m pgh.fetch            # -> data/pgh_segments.parquet, data/pgh_connectors.parquet
"""
import os
import time

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

RELEASE = "2026-09-23.1"
BUCKET = "overturemaps-us-west-2"
BBOX = (-80.095, 40.361, -79.865, 40.501)  # (west, south, east, north): the City of Pittsburgh
COLS = {"segment": ["id", "names", "subtype", "class", "subclass", "connectors", "road_flags", "access_restrictions",
                    "speed_limits", "width_rules", "level_rules", "routes", "sources", "geometry", "bbox"],
        "connector": ["id", "geometry", "bbox"]}


def fetch(kind, out, bbox=BBOX):
    fs = pafs.S3FileSystem(anonymous=True, region="us-west-2")
    d = ds.dataset(f"{BUCKET}/release/{RELEASE}/theme=transportation/type={kind}/", filesystem=fs, format="parquet")
    w, s, e, n = bbox
    f = ((pc.field("bbox", "xmax") >= w) & (pc.field("bbox", "xmin") <= e) & (pc.field("bbox", "ymax") >= s)
         & (pc.field("bbox", "ymin") <= n))
    t0 = time.time()
    tab = d.to_table(columns=COLS[kind], filter=f)
    if kind == "segment":
        tab = tab.filter(pc.equal(tab["subtype"], "road"))
    pq.write_table(tab, out)
    print(f"{kind}: {tab.num_rows} rows in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    fetch("segment", "data/pgh_segments.parquet")
    fetch("connector", "data/pgh_connectors.parquet")
