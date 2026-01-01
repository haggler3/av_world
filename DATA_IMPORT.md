# Importing real driving logs: Argoverse 2 and Waymo

`xnet/avdata.py`. Turns a logged scenario into three things the rest of the repository already understands:

1. a **scene** (trajectories of every vehicle, cyclist and pedestrian, the ego marked as the AV), the format of `xnet.carla_export`, so a real
   log can be replayed in CARLA, run through the sensor model of `xnet.perception`, and scored with `avlab.incidents.scene_metrics`;
2. a **road network** (`xnet` links, junction boxes, moves, conflict points) built from the log's lane-level map, so the simulator can run on it;
3. a **simulator state** holding the logged vehicles at a chosen moment (map-matched onto the network), which the reactive simulator then
   continues: "start from the log, then let the world push back".

| | Argoverse 2 motion forecasting | Waymo Open Motion |
|---|---|---|
| tracks, map, scene | `read_av2`, `av2_to_scene`, `av2_network` | `waymo_to_dict`, `waymo_scene`, `waymo_network` |
| tested on | **150 real scenarios** (downloaded from the public bucket), plus a made-up crossroads | a made-up crossroads in the Waymo field layout only |
| access | public S3 bucket, no account (CC BY-NC-SA 4.0, so not redistributed here): `av-world fetch-av2 DIR` | needs a Waymo account and licence, and `tensorflow` + `waymo-open-dataset` to read the TFRecords; **not possible from the build container** |

Be clear about the Waymo status: the adapter reads the documented `Scenario` protobuf fields and was tested on a dict in that layout, not on
a real file. Argoverse 2 is the one verified on real data.

## How it works

```
# scene
for each track in the log:
    kind = vehicle | bus | cyclist/motorcyclist | pedestrian | (static, background, ... dropped)
    ego track ("AV" in Argoverse 2, sdc_track_index in Waymo) -> the AV
    vehicles: x, y, heading, speed per frame;  pedestrians: their own path
    optionally keep every k-th frame (every=2 gives the simulator's 0.2 s)

# map -> network (Argoverse 2; Waymo first infers the flag below)
lane segments = driving lanes with centre line, predecessors, successors, is_intersection
junction clusters:
    intersection lanes = lanes flagged is_intersection
    sample each at 2 m;  union lanes whose samples are closer than 2.5 m (KD-tree)
    union a lane with its successors that are also intersection lanes
connectors (union-find):
    end of a lane  ~  start of its successor (plain lane to plain lane)
    end of an approach lane  ~  junction cluster  ~  start of an exit lane
rows = one one-way road segment per non-intersection lane, connectors at both ends
net = xnet.network.build(rows)       # the builder makes nodes, boxes, moves, conflicts, routing

# Waymo has no is_intersection flag:
intersection lanes = lanes whose centre line crosses another lane's centre line in its interior
                     (not the one it enters from or exits to)

# state
Matcher: every track of the network sampled every metre, KD-tree
for each logged vehicle at frame f:
    candidates = samples within 3 m whose heading is within 57 deg of the vehicle's; take the nearest
    track, arc length = that sample's;  front bumper = arc + length/2
    put it in a free slot with its logged speed; the ego is the AV; others are humans with random
    IDM parameters; desired speed set from the logged speed; destination random or, with route_from_log,
    the destination nearest to where the log shows the vehicle later
run sim (arrivals off) for the horizon; compare positions with the log each second
```

## What was verified on real Argoverse 2 data

150 random validation scenarios (`av-world validate-av2 DIR`, 14 minutes; `results/avdata/av2_validation.json`):

* 149 of 150 imported without error (one scenario failed to build; I did not investigate it). A median scenario has 2 junction clusters.
* 72 % of logged vehicles could be put on a lane of the imported network (within 3 m and pointing the same way); the median offset is 0.5 m. The
  rest are probably parked at the curb, in lanes the map does not show, or on lanes the builder dropped as disconnected (not broken down).
* **How well do the simulator's drivers predict what the logged vehicles did next?** Seed at 2.5 s, run 5 s, average position error of the
  vehicles that were moving at the start and stayed in their lane (357-447 vehicles):

| horizon | simulator (reactive drivers) | constant velocity |
|---|---|---|
| 1 s | 1.9 m | 0.9 m |
| 2 s | 4.7 m | 2.2 m |
| 3 s | 8.3 m | 4.0 m |
| 5 s | 17.0 m | 9.2 m |

The simulator is about twice as far off as the standard no-model baseline. This says the simulator's car following is **not calibrated to real
traffic**: its drivers have random IDM parameters, the control at junctions is inferred (the log's signals are unknown, so cars that waited
at a red in the log do not in the simulator and vice versa), the destination is a guess, and the lane geometry of the imported network is
coarser than the log. It does not say the simulator is wrong about which events are possible. Fitting the driver parameters to logs (the
natural next step: this module gives the data and the metric) is not done here.

## Using it

```bash
av-world fetch-av2 data/av2 --n 20                 # 20 random scenarios, ~4 MB
av-world import-av2 data/av2/<scenario_id> out     # out.json (scene), out.xodr (OpenDRIVE), out_net.npz (network)
av-world validate-av2 data/av2 --limit 100
```

```python
from xnet import avdata as AV, carla_export as CE
sc = AV.read_av2("data/av2/<id>");  scene = AV.av2_to_scene(sc, every=2)
net, geo, info = AV.av2_network(sc)                  # info["junction_clusters"], info["lanes"] ...
st, m = AV.seed_state(net, geo, scene, frame=25)     # simulator state at 2.5 s, m["matched"] vehicles placed
r = AV.compare_future(net, geo, scene, frame=25, horizon_s=5.0, speed_from_log=True)
# Waymo: d = AV.waymo_to_dict(scenario_proto)  (or  for d in AV.read_waymo("file.tfrecord"))
#        scene = AV.waymo_scene(d, every=2);   net, geo, info = AV.waymo_network(d)
```

## Limits

* Lane-level maps become one-way links of one lane each: two parallel lanes in one direction are two separate links, so lane changes
  between them are lost. Turn permissions inside junctions are re-derived by the builder (every approach connects to every exit it can
  geometrically reach), not read from the map.
* No signals or stop signs in the Argoverse 2 map (Waymo's scenarios carry traffic-light states, which are read into the dict but not yet
  used to drive the simulator's signals). Control at junctions is inferred.
* Pedestrians in a log keep their paths in the scene but the simulator's pedestrians only cross at crosswalks, so seeding a log does not put
  its pedestrians into the simulation.
* Argoverse 2's `static` and `background` objects (parked cars, cones) are dropped.
* The scenarios are 11 s long: they are a source of initial conditions and of ground truth for short horizons, not of whole trips.
