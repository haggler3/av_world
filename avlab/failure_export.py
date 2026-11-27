"""Export canonical failure scenarios (avlab.failurelog, schema av_world.failure/1) into the formats of common AV tools.

    from avlab import failurelog as F, failure_export as FX
    sc = F.load("data/failures/dry/oakland/<id>.json.gz")
    FX.export(sc, "out/", formats=FX.FORMATS)

Formats (each function writes one scenario; the time window is cropped to the format's convention):

| key          | tool / dataset convention                         | files                                     | window                      |
|--------------|---------------------------------------------------|-------------------------------------------|-----------------------------|
| waymo        | Waymo Open Motion `Scenario` proto as proto3 JSON  | <id>.waymo.json (json_format.Parse-able)   | 91 steps, current index 10  |
|              | (Waymax, WOMD tooling; lowerCamelCase fields)      |                                           |                             |
| av2          | Argoverse 2 motion forecasting                     | <id>/scenario_<id>.parquet + map JSON      | 110 steps, 50 observed      |
| scenarionet  | ScenarioNet / MetaDrive ScenarioDescription        | <id>.scenarionet.pkl                       | all 111 steps               |
| gpudrive     | GPUDrive / Nocturne JSON (from WOMD)               | <id>.gpudrive.json                         | 91 steps                    |
| commonroad   | CommonRoad XML 2020a (lanelets, obstacles, planning problem for the ego) | <id>.cr.xml       | all 111 steps               |
| openscenario | ASAM OpenSCENARIO 1.0 + OpenDRIVE 1.4 (esmini, CARLA ScenarioRunner) | <id>.xosc + maps/<city>.xodr | all 111 steps |
| carla        | the xnet.carla_export replay (scene JSON + script) | <id>.carla.json, <id>_replay.py            | all 111 steps               |
| bev          | bird's-eye raster tensor [T, C, H, W] (uint8) + GIF | <id>.bev.npz, <id>.bev.gif                 | 2 Hz, 64 m x 64 m           |
| table        | long-format track table (pandas / Parquet)         | <id>.tracks.parquet                        | all 111 steps               |

Coordinates are the city's local metres (x east, y north) with the origin given in each file's metadata. What each format
loses is listed in docs/FAILURE_LOGS.md.
"""
import json
import math
import os
import pickle
from xml.sax.saxutils import escape

import numpy as np

FORMATS = ("waymo", "av2", "scenarionet", "gpudrive", "commonroad", "openscenario", "carla", "bev", "table")
WAYMO_TYPE = {"vehicle": 1, "bus": 1, "pedestrian": 2, "cyclist": 3}
WAYMO_TYPE_NAME = {1: "TYPE_VEHICLE", 2: "TYPE_PEDESTRIAN", 3: "TYPE_CYCLIST"}
SIG_WAYMO = {0: "LANE_STATE_STOP", 1: "LANE_STATE_CAUTION", 2: "LANE_STATE_GO"}
MPS_TO_MPH = 2.236936


def _crop(sc, n, end_at_last=True):
    """Index range of n steps: the last n steps of the window (the event keeps its 3 s of aftermath)."""
    T = sc["n_steps"]
    lo = T - n if end_at_last else 0
    return lo, lo + n, sc["event_index"] - lo


def _tid(t):
    return "AV" if t["is_ego"] else str(t["id"])


# ---------------------------------------------------------------- Waymo Open Motion (proto3 JSON)
def to_waymo(sc, path):
    lo, hi, ev = _crop(sc, 91)
    tracks, interest = [], []
    for t in sc["tracks"]:
        states = []
        for j in range(lo, hi):
            v = bool(t["valid"][j])
            states.append({"centerX": t["x"][j] if v else 0.0, "centerY": t["y"][j] if v else 0.0, "centerZ": 0.0,
                           "length": t["length"] if v else 0.0, "width": t["width"] if v else 0.0, "height": t["height"] if v else 0.0,
                           "heading": t["heading"][j] if v else 0.0, "velocityX": t["vx"][j] if v else 0.0,
                           "velocityY": t["vy"][j] if v else 0.0, "valid": v})
        tracks.append({"id": int(t["id"]), "objectType": WAYMO_TYPE_NAME[WAYMO_TYPE[t["type"]]], "states": states})
        if t["involved"]:
            interest.append(int(t["id"]))
    feats = []
    for ln in sc["map"]["lanes"]:
        feats.append({"id": int(ln["id"]), "lane": {"speedLimitMph": round(ln["speed_limit_mps"] * MPS_TO_MPH, 2),
                                                     "type": "TYPE_SURFACE_STREET", "interpolating": bool(ln["is_intersection"]),
                                                     "polyline": [{"x": p[0], "y": p[1], "z": 0.0} for p in ln["polyline"]],
                                                     "entryLanes": ln["predecessors"], "exitLanes": ln["successors"]}})
    base = 1_000_000
    for e in sc["map"]["road_edges"]:
        feats.append({"id": base + int(e["id"]), "roadEdge": {"type": "TYPE_ROAD_EDGE_BOUNDARY",
                                                             "polyline": [{"x": p[0], "y": p[1], "z": 0.0} for p in e["polyline"]]}})
    for c in sc["map"]["crosswalks"]:
        feats.append({"id": 2 * base + int(c["id"]), "crosswalk": {"polygon": [{"x": p[0], "y": p[1], "z": 0.0} for p in c["polygon"]]}})
    dms = []
    for j in range(lo, hi):
        dms.append({"laneStates": [{"lane": s["lane"], "state": SIG_WAYMO[s["states"][j]],
                                    "stopPoint": {"x": s["stop_point"][0], "y": s["stop_point"][1], "z": 0.0}} for s in sc["map"]["signals"]]})
    ego = sc["event"]["ego_track"]
    out = {"scenarioId": sc["id"], "timestampsSeconds": [round(k * 0.1, 1) for k in range(hi - lo)], "currentTimeIndex": 10,
           "tracks": tracks, "dynamicMapStates": dms, "mapFeatures": feats, "sdcTrackIndex": ego, "objectsOfInterest": interest,
           "tracksToPredict": [{"trackIndex": i, "difficulty": "LEVEL_1"} for i in interest if i != ego]}
    json.dump(out, open(path, "w"), separators=(",", ":"))
    return path


def waymo_json_to_dict(d):
    """Waymo proto3 JSON -> the plain dict of xnet.avdata.waymo_to_dict (to read it back with waymo_scene / waymo_network)."""
    tracks = [{"id": t["id"], "type": {"TYPE_VEHICLE": 1, "TYPE_PEDESTRIAN": 2, "TYPE_CYCLIST": 3}[t["objectType"]],
               "states": [[s["centerX"], s["centerY"], s["length"], s["width"], s["heading"], s["velocityX"], s["velocityY"], s["valid"]]
                          for s in t["states"]]} for t in d["tracks"]]
    lanes = {f["id"]: {"type": 2, "speed_limit_mph": f["lane"]["speedLimitMph"], "polyline": [[p["x"], p["y"]] for p in f["lane"]["polyline"]],
                       "entry": f["lane"]["entryLanes"], "exit": f["lane"]["exitLanes"]} for f in d["mapFeatures"] if "lane" in f}
    cws = [[[p["x"], p["y"]] for p in f["crosswalk"]["polygon"]] for f in d["mapFeatures"] if "crosswalk" in f]
    return {"id": d["scenarioId"], "dt": 0.1, "current": d["currentTimeIndex"], "sdc": d["sdcTrackIndex"], "tracks": tracks,
            "lanes": lanes, "crosswalks": cws, "stop_signs": [], "signals": []}


# ---------------------------------------------------------------- Argoverse 2 motion forecasting
AV2_TYPE = {"vehicle": "vehicle", "bus": "bus", "pedestrian": "pedestrian", "cyclist": "cyclist"}


def to_av2(sc, outdir):
    import pandas as pd
    lo, hi, ev = _crop(sc, 110)
    sid = sc["id"]
    d = os.path.join(outdir, sid)
    os.makedirs(d, exist_ok=True)
    ego = sc["tracks"][sc["event"]["ego_track"]]
    others = [t for t in sc["tracks"] if t["involved"] and not t["is_ego"]]
    focal = others[0] if others else next((t for t in sc["tracks"] if not t["is_ego"]), ego)
    rows = []
    for t in sc["tracks"]:
        cat = 3 if t is focal else 2 if t["involved"] else 1
        for j in range(lo, hi):
            if t["valid"][j]:
                rows.append(dict(observed=(j - lo) < 50, track_id=_tid(t), object_type=AV2_TYPE[t["type"]], object_category=cat,
                                 timestep=j - lo, position_x=t["x"][j], position_y=t["y"][j], heading=t["heading"][j],
                                 velocity_x=t["vx"][j], velocity_y=t["vy"][j]))
    df = pd.DataFrame(rows)
    t0 = int(round(sc["t0_sim"] * 1e9))
    df["scenario_id"] = sid
    df["start_timestamp"] = t0 + int(lo * 1e8)
    df["end_timestamp"] = t0 + int((hi - 1) * 1e8)
    df["num_timestamps"] = hi - lo
    df["focal_track_id"] = _tid(focal)
    df["city"] = sc["city"]["key"]
    df.to_parquet(os.path.join(d, f"scenario_{sid}.parquet"), index=False)
    pt = lambda p: {"x": p[0], "y": p[1], "z": 0.0}  # noqa: E731
    seg = {str(ln["id"]): {"id": ln["id"], "is_intersection": ln["is_intersection"], "lane_type": "VEHICLE",
                           "left_lane_boundary": [pt(p) for p in ln["left"]], "right_lane_boundary": [pt(p) for p in ln["right"]],
                           "left_lane_mark_type": "NONE", "right_lane_mark_type": "NONE", "left_neighbor_id": None,
                           "right_neighbor_id": None, "predecessors": ln["predecessors"], "successors": ln["successors"],
                           "centerline": [pt(p) for p in ln["polyline"]]} for ln in sc["map"]["lanes"]}
    pc = {str(c["id"]): {"id": c["id"], "edge1": [pt(c["polygon"][0]), pt(c["polygon"][1])],
                         "edge2": [pt(c["polygon"][3]), pt(c["polygon"][2])]} for c in sc["map"]["crosswalks"]}
    json.dump({"lane_segments": seg, "drivable_areas": {}, "pedestrian_crossings": pc},
              open(os.path.join(d, f"log_map_archive_{sid}.json"), "w"), separators=(",", ":"))
    return d


# ---------------------------------------------------------------- ScenarioNet / MetaDrive
SN_TYPE = {"vehicle": "VEHICLE", "bus": "VEHICLE", "pedestrian": "PEDESTRIAN", "cyclist": "CYCLIST"}


def scenarionet_dict(sc):
    T = sc["n_steps"]
    tracks = {}
    for t in sc["tracks"]:
        v = np.asarray(t["valid"], bool)
        tracks[_tid(t)] = {"type": SN_TYPE[t["type"]],
                           "state": {"position": np.stack([t["x"], t["y"], np.zeros(T)], 1).astype(np.float32) * v[:, None],
                                     "heading": np.asarray(t["heading"], np.float32) * v,
                                     "velocity": np.stack([t["vx"], t["vy"]], 1).astype(np.float32) * v[:, None],
                                     "length": np.full(T, t["length"], np.float32) * v, "width": np.full(T, t["width"], np.float32) * v,
                                     "height": np.full(T, t["height"], np.float32) * v, "valid": v},
                           "metadata": {"track_length": T, "type": SN_TYPE[t["type"]], "object_id": _tid(t), "dataset": "av_world",
                                        "is_av": t["is_av"], "involved": t["involved"]}}
    mf = {}
    for ln in sc["map"]["lanes"]:
        mf[str(ln["id"])] = {"type": "LANE_SURFACE_STREET", "polyline": np.c_[np.asarray(ln["polyline"]), np.zeros(len(ln["polyline"]))].astype(np.float32),
                             "entry_lanes": [str(i) for i in ln["predecessors"]], "exit_lanes": [str(i) for i in ln["successors"]],
                             "left_neighbor": [], "right_neighbor": [], "speed_limit_kmh": round(ln["speed_limit_mps"] * 3.6, 1)}
    for e in sc["map"]["road_edges"]:
        mf[f"edge_{e['id']}"] = {"type": "ROAD_EDGE_BOUNDARY", "polyline": np.c_[np.asarray(e["polyline"]), np.zeros(len(e["polyline"]))].astype(np.float32)}
    for c in sc["map"]["crosswalks"]:
        mf[f"cw_{c['id']}"] = {"type": "CROSSWALK", "polygon": np.c_[np.asarray(c["polygon"]), np.zeros(4)].astype(np.float32)}
    dms = {}
    for s in sc["map"]["signals"]:
        dms[f"tl_{s['lane']}"] = {"type": "TRAFFIC_LIGHT", "lane": str(s["lane"]),
                                  "stop_point": np.array([*s["stop_point"], 0.0], np.float32),
                                  "state": {"object_state": [SIG_WAYMO[x] for x in s["states"]]},
                                  "metadata": {"track_length": T, "type": "TRAFFIC_LIGHT", "object_id": f"tl_{s['lane']}", "dataset": "av_world"}}
    ego = _tid(sc["tracks"][sc["event"]["ego_track"]])
    inv = [_tid(t) for t in sc["tracks"] if t["involved"] and not t["is_ego"]]
    meta = {"ts": np.arange(T, dtype=np.float32) * sc["dt"], "metadata_version": "2023-08-14", "coordinate": "waymo", "metadrive_processed": False,
            "dataset": "av_world", "scenario_id": sc["id"], "id": sc["id"], "source_file": sc["id"], "track_length": T,
            "current_time_index": 50, "sdc_id": ego, "objects_of_interest": [ego] + inv,
            "tracks_to_predict": {i: {"track_index": list(tracks).index(i), "track_id": i, "difficulty": 0,
                                      "object_type": tracks[i]["type"]} for i in inv},
            "av_world": {"city": sc["city"], "condition": sc["condition"], "event": sc["event"]}}
    return {"id": sc["id"], "version": "av_world.failure/1", "length": T, "tracks": tracks, "dynamic_map_states": dms,
            "map_features": mf, "metadata": meta}


def to_scenarionet(sc, path):
    pickle.dump(scenarionet_dict(sc), open(path, "wb"))
    return path


# ---------------------------------------------------------------- GPUDrive / Nocturne JSON
# Mirrors GPUDrive's own Waymo converter (data_utils/process_waymo_files.py): invalid states filled with -1e4, map element
# ids from Waymax (lane surface street 2, road edge boundary 15, crosswalk 18), traffic lights per lane as time series, and
# `mark_as_expert` on vehicles whose logged trajectories collide (here: the vehicles in the crash).
GPD_TYPE = {"vehicle": "vehicle", "bus": "vehicle", "pedestrian": "pedestrian", "cyclist": "cyclist"}
GPD_ERR = -1e4
GPD_TL = {0: "stop", 1: "caution", 2: "go"}


def to_gpudrive(sc, path):
    lo, hi, ev = _crop(sc, 91)
    objs = []
    for t in sc["tracks"]:
        js = range(lo, hi)
        ok = [bool(t["valid"][j]) for j in js]
        last = max([j for j in js if t["valid"][j]], default=lo)
        objs.append({"position": [{"x": t["x"][j], "y": t["y"][j], "z": 0.0} if t["valid"][j] else {"x": GPD_ERR, "y": GPD_ERR, "z": GPD_ERR}
                                  for j in js],
                     "width": t["width"], "length": t["length"], "height": t["height"],
                     "heading": [t["heading"][j] if t["valid"][j] else GPD_ERR for j in js],
                     "velocity": [{"x": t["vx"][j], "y": t["vy"][j]} if t["valid"][j] else {"x": GPD_ERR, "y": GPD_ERR} for j in js],
                     "valid": ok, "goalPosition": {"x": t["x"][last], "y": t["y"][last], "z": 0.0},
                     "type": GPD_TYPE[t["type"]], "id": int(t["id"]),
                     "mark_as_expert": bool(t["involved"] and t["type"] != "pedestrian")})
    roads = []
    for ln in sc["map"]["lanes"]:
        roads.append({"geometry": [{"x": p[0], "y": p[1], "z": 0.0} for p in ln["polyline"]], "type": "lane", "map_element_id": 2,
                      "id": int(ln["id"])})
    for e in sc["map"]["road_edges"]:
        roads.append({"geometry": [{"x": p[0], "y": p[1], "z": 0.0} for p in e["polyline"]], "type": "road_edge", "map_element_id": 15,
                      "id": 1_000_000 + int(e["id"])})
    for c in sc["map"]["crosswalks"]:
        roads.append({"geometry": [{"x": p[0], "y": p[1], "z": 0.0} for p in c["polygon"]], "type": "crosswalk", "map_element_id": 18,
                      "id": 2_000_000 + int(c["id"])})
    tl = {str(s["lane"]): {"state": [GPD_TL[x] for x in s["states"][lo:hi]], "x": [s["stop_point"][0]] * (hi - lo),
                           "y": [s["stop_point"][1]] * (hi - lo), "z": [0.0] * (hi - lo), "time_index": list(range(hi - lo))}
          for s in sc["map"]["signals"]}
    inv = [int(t["id"]) for t in sc["tracks"] if t["involved"]]
    ego = sc["event"]["ego_track"]
    json.dump({"name": f"{sc['id']}.json", "scenario_id": sc["id"], "objects": objs, "roads": roads, "tl_states": tl,
               "metadata": {"sdc_track_index": ego, "objects_of_interest": inv,
                            "tracks_to_predict": [{"track_index": i, "difficulty": 1} for i in inv if i != ego],
                            "av_world": {"city": sc["city"]["key"], "event": sc["event"],
                                         "note": "GPUDrive's Waymo converter skips scenes with traffic lights; tl_states is kept here"}}},
              open(path, "w"), separators=(",", ":"))
    return path


# ---------------------------------------------------------------- CommonRoad XML 2020a
CR_TYPE = {"vehicle": "car", "bus": "bus", "pedestrian": "pedestrian", "cyclist": "bicycle"}


def _pts_xml(tag, pts):
    return f"<{tag}>" + "".join(f"<point><x>{p[0]:.2f}</x><y>{p[1]:.2f}</y></point>" for p in pts) + f"</{tag}>"


def _state_xml(tag, t, j, k):
    return (f"<{tag}><position><point><x>{t['x'][j]:.2f}</x><y>{t['y'][j]:.2f}</y></point></position>"
            f"<orientation><exact>{t['heading'][j]:.4f}</exact></orientation><time><exact>{k}</exact></time>"
            f"<velocity><exact>{math.hypot(t['vx'][j], t['vy'][j]):.3f}</exact></velocity>"
            + ("<yawRate><exact>0.0</exact></yawRate><slipAngle><exact>0.0</exact></slipAngle>" if tag == "initialState" else "")
            + f"</{tag}>")


def to_commonroad(sc, path):
    ids = {}

    def nid(key):
        return ids.setdefault(key, len(ids) + 1)
    lanes = sc["map"]["lanes"]
    for ln in lanes:
        nid(("l", ln["id"]))
    known = {ln["id"] for ln in lanes}
    xml = [f'<?xml version="1.0" encoding="UTF-8"?>\n<commonRoad timeStepSize="{sc["dt"]}" commonRoadVersion="2020a" '
           f'author="av_world" affiliation="av_world simulator" source="av_world failure log ({escape(sc["city"]["name"])})" '
           f'benchmarkID="USA_AVW-{len(sc["id"]) % 9 + 1}_1_T-1" date="2026-11-27">',
           f'<location><geoNameId>-999</geoNameId><gpsLatitude>{sc["event"]["lat"]}</gpsLatitude>'
           f'<gpsLongitude>{sc["event"]["lon"]}</gpsLongitude></location>',
           '<scenarioTags><Urban/></scenarioTags>']
    for ln in lanes:
        # CommonRoad lanelets: left bound on the left of the driving direction; both bounds with the same point count
        xml.append(f'<lanelet id="{nid(("l", ln["id"]))}">' + _pts_xml("leftBound", ln["left"]) + _pts_xml("rightBound", ln["right"])
                   + "".join(f'<predecessor ref="{nid(("l", p))}"/>' for p in ln["predecessors"] if p in known)
                   + "".join(f'<successor ref="{nid(("l", s))}"/>' for s in ln["successors"] if s in known)
                   + ("<laneletType>intersection</laneletType>" if ln["is_intersection"] else "<laneletType>urban</laneletType>")
                   + "</lanelet>")
    ego = sc["tracks"][sc["event"]["ego_track"]]
    for t in sc["tracks"]:
        if t is ego:
            continue
        js = [j for j in range(sc["n_steps"]) if t["valid"][j]]
        if len(js) < 2:
            continue
        shape = (f'<circle><radius>{0.35:.2f}</radius></circle>' if t["type"] == "pedestrian" else
                 f'<rectangle><length>{t["length"]:.2f}</length><width>{t["width"]:.2f}</width></rectangle>')
        xml.append(f'<dynamicObstacle id="{nid(("o", t["id"]))}"><type>{CR_TYPE[t["type"]]}</type><shape>{shape}</shape>'
                   + _state_xml("initialState", t, js[0], js[0])
                   + "<trajectory>" + "".join(_state_xml("state", t, j, j) for j in js[1:]) + "</trajectory></dynamicObstacle>")
    js = [j for j in range(sc["n_steps"]) if ego["valid"][j]]
    j0 = js[0]
    xml.append(f'<planningProblem id="{nid(("p", 0))}">' + _state_xml("initialState", ego, j0, j0)
               + f'<goalState><time><intervalStart>{sc["n_steps"] - 1}</intervalStart><intervalEnd>{sc["n_steps"] - 1}</intervalEnd></time>'
               + "</goalState></planningProblem>")
    xml.append("</commonRoad>\n")
    open(path, "w").write("\n".join(xml))
    return path


# ---------------------------------------------------------------- OpenSCENARIO 1.0 + OpenDRIVE (per city)
def city_xodr(city, mapdir):
    os.makedirs(mapdir, exist_ok=True)
    p = os.path.join(mapdir, f"{city}.xodr")
    if not os.path.exists(p):
        from avlab import failurelog as F
        from xnet import carla_export as CE
        net, geo, info, origin, name = F.build_city(city)
        open(p, "w").write(CE.to_opendrive(net, geo, city))
    return p


XOSC_CAT = {"vehicle": "car", "bus": "bus", "cyclist": "bicycle"}


def to_openscenario(sc, path, mapdir):
    xodr = city_xodr(sc["city"]["key"], mapdir)
    rel = os.path.relpath(xodr, os.path.dirname(os.path.abspath(path)))
    ents, inits, mgs = [], [], []
    for t in sc["tracks"]:
        js = [j for j in range(sc["n_steps"]) if t["valid"][j]]
        if len(js) < 2:
            continue
        name = f"{'ego' if t['is_ego'] else t['type']}_{t['id']}"
        bb = (f'<BoundingBox><Center x="{t["length"] / 2 - 0.9:.2f}" y="0" z="{t["height"] / 2:.2f}"/>'
              f'<Dimensions width="{t["width"]}" length="{t["length"]}" height="{t["height"]}"/></BoundingBox>')
        if t["type"] == "pedestrian":
            ents.append(f'<ScenarioObject name="{name}"><Pedestrian name="{name}" model="pedestrian" mass="70" '
                        f'pedestrianCategory="pedestrian"><ParameterDeclarations/>{bb}<Properties/></Pedestrian></ScenarioObject>')
        else:
            ents.append(f'<ScenarioObject name="{name}"><Vehicle name="{name}" vehicleCategory="{XOSC_CAT[t["type"]]}">'
                        f'<ParameterDeclarations/>{bb}<Performance maxSpeed="40" maxAcceleration="10" maxDeceleration="10"/>'
                        '<Axles><FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.6" positionX="2.8" positionZ="0.3"/>'
                        '<RearAxle maxSteering="0" wheelDiameter="0.6" trackWidth="1.6" positionX="0" positionZ="0.3"/></Axles>'
                        '<Properties/></Vehicle></ScenarioObject>')
        j0 = js[0]
        inits.append(f'<Private entityRef="{name}"><PrivateAction><TeleportAction><Position><WorldPosition x="{t["x"][j0]:.2f}" '
                     f'y="{t["y"][j0]:.2f}" z="0" h="{t["heading"][j0]:.4f}"/></Position></TeleportAction></PrivateAction></Private>')
        verts = "".join(f'<Vertex time="{j * sc["dt"]:.2f}"><Position><WorldPosition x="{t["x"][j]:.2f}" y="{t["y"][j]:.2f}" z="0" '
                        f'h="{t["heading"][j]:.4f}"/></Position></Vertex>' for j in js)
        mgs.append(f'<ManeuverGroup maximumExecutionCount="1" name="mg_{name}"><Actors selectTriggeringEntities="false">'
                   f'<EntityRef entityRef="{name}"/></Actors><Maneuver name="m_{name}"><Event name="e_{name}" priority="overwrite">'
                   f'<Action name="a_{name}"><PrivateAction><RoutingAction><FollowTrajectoryAction>'
                   f'<Trajectory name="tr_{name}" closed="false"><ParameterDeclarations/><Shape><Polyline>{verts}</Polyline></Shape>'
                   '</Trajectory><TimeReference><Timing domainAbsoluteRelative="absolute" scale="1" offset="0"/></TimeReference>'
                   '<TrajectoryFollowingMode followingMode="position"/></FollowTrajectoryAction></RoutingAction></PrivateAction></Action>'
                   '<StartTrigger><ConditionGroup><Condition name="start" delay="0" conditionEdge="none"><ByValueCondition>'
                   '<SimulationTimeCondition value="0" rule="greaterThan"/></ByValueCondition></Condition></ConditionGroup></StartTrigger>'
                   '</Event></Maneuver></ManeuverGroup>')
    end = (sc["n_steps"] - 1) * sc["dt"]
    xml = (f'<?xml version="1.0" encoding="UTF-8"?>\n<OpenSCENARIO><FileHeader revMajor="1" revMinor="0" date="2026-11-27T00:00:00" '
           f'description="av_world failure {escape(sc["id"])}: {sc["event"]["type"]} at t={sc["event_index"] * sc["dt"]:.1f} s" author="av_world"/>'
           '<ParameterDeclarations/><CatalogLocations/>'
           f'<RoadNetwork><LogicFile filepath="{escape(rel)}"/><SceneGraphFile filepath=""/></RoadNetwork>'
           f'<Entities>{"".join(ents)}</Entities>'
           f'<Storyboard><Init><Actions>{"".join(inits)}</Actions></Init>'
           f'<Story name="failure"><Act name="replay">{"".join(mgs)}'
           '<StartTrigger><ConditionGroup><Condition name="act_start" delay="0" conditionEdge="none"><ByValueCondition>'
           '<SimulationTimeCondition value="0" rule="greaterThan"/></ByValueCondition></Condition></ConditionGroup></StartTrigger>'
           '</Act></Story><StopTrigger><ConditionGroup><Condition name="end" delay="0" conditionEdge="rising"><ByValueCondition>'
           f'<SimulationTimeCondition value="{end:.1f}" rule="greaterThan"/></ByValueCondition></Condition></ConditionGroup></StopTrigger>'
           '</Storyboard></OpenSCENARIO>\n')
    open(path, "w").write(xml)
    return path


# ---------------------------------------------------------------- CARLA replay (xnet.carla_export scene format)
def to_carla(sc, path_base, mapdir):
    from xnet import carla_export as CE
    xodr = city_xodr(sc["city"]["key"], mapdir)
    veh, ped = [], []
    k = sc["event_index"]
    for t in sc["tracks"]:
        js = [j for j in range(sc["n_steps"]) if t["valid"][j]]
        if not js:
            continue
        if t["type"] == "pedestrian":
            ped.append({"t": round(js[0] * sc["dt"], 2), "duration": round((js[-1] - js[0]) * sc["dt"], 1),
                        "from": [t["x"][js[0]], t["y"][js[0]]], "to": [t["x"][js[-1]], t["y"][js[-1]]]})
            continue
        kind = "av" if t["is_av"] else {"vehicle": "car", "bus": "bus", "cyclist": "bike"}[t["type"]]
        veh.append({"id": t["id"], "kind": kind, "av": t["is_av"], "crashed": t["involved"] and k in js,
                    "crash_frame": (js.index(k) if t["involved"] and k in js else None), "wrecked": t["involved"], "start": js[0],
                    "length": t["length"], "x": [t["x"][j] for j in js], "y": [t["y"][j] for j in js],
                    "yaw": [t["heading"][j] for j in js], "v": [round(math.hypot(t["vx"][j], t["vy"][j]), 2) for j in js]})
    scene = {"dt": sc["dt"], "frames": sc["n_steps"], "vehicles": veh, "pedestrians": ped, "source": f"av_world:{sc['id']}",
             "xodr": os.path.relpath(xodr, os.path.dirname(os.path.abspath(path_base)))}
    json.dump(scene, open(f"{path_base}.carla.json", "w"), separators=(",", ":"))
    name = os.path.basename(path_base)
    open(f"{path_base}_replay.py", "w").write(CE.REPLAY.format(name=name + ".carla", here=os.path.dirname(os.path.abspath(path_base)) + os.sep,
                                                               blueprints=CE.BLUEPRINT))
    return f"{path_base}.carla.json"


# ---------------------------------------------------------------- BEV raster tensor + GIF
BEV_CH = ("lanes", "intersections", "crosswalks", "red_stop_lines", "vehicles", "ego", "pedestrians", "speed")


def bev(sc, size=128, res=0.5, every=5):
    """[T, C, size, size] uint8 centred on the crash point (x right = east, row 0 = north), one frame per `every` steps."""
    from PIL import Image, ImageDraw
    ex, ey = sc["event"]["x"], sc["event"]["y"]
    half = size * res / 2

    def px(p):
        return [((x - ex + half) / res, (half - (y - ey)) / res) for x, y in p]
    static = [Image.new("L", (size, size)) for _ in range(3)]
    for ln in sc["map"]["lanes"]:
        ImageDraw.Draw(static[1 if ln["is_intersection"] else 0]).line(px(ln["polyline"]), fill=255, width=max(1, int(3.6 / res)))
    for c in sc["map"]["crosswalks"]:
        ImageDraw.Draw(static[2]).polygon(px(c["polygon"]), fill=255)
    frames = []
    for j in range(0, sc["n_steps"], every):
        ch = [np.asarray(s) for s in static]
        red = Image.new("L", (size, size))
        for s in sc["map"]["signals"]:
            if s["states"][j] == 0:
                x, y = s["stop_point"]
                ImageDraw.Draw(red).ellipse([*(v - 2 for v in px([(x, y)])[0]), *(v + 2 for v in px([(x, y)])[0])], fill=255)
        veh, ego, ped, spd = (Image.new("L", (size, size)) for _ in range(4))
        for t in sc["tracks"]:
            if not t["valid"][j]:
                continue
            x, y, h = t["x"][j], t["y"][j], t["heading"][j]
            if t["type"] == "pedestrian":
                (cx, cy), = px([(x, y)])
                ImageDraw.Draw(ped).ellipse([cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5], fill=255)
                continue
            c, s_ = math.cos(h), math.sin(h)
            L2, W2 = t["length"] / 2, t["width"] / 2
            corners = [(x + c * a - s_ * b, y + s_ * a + c * b) for a, b in ((L2, W2), (L2, -W2), (-L2, -W2), (-L2, W2))]
            poly = px(corners)
            ImageDraw.Draw(ego if t["is_ego"] else veh).polygon(poly, fill=255)
            ImageDraw.Draw(spd).polygon(poly, fill=int(min(math.hypot(t["vx"][j], t["vy"][j]) / 20.0, 1.0) * 255))
        frames.append(np.stack(ch + [np.asarray(red), np.asarray(veh), np.asarray(ego), np.asarray(ped), np.asarray(spd)]))
    return np.stack(frames).astype(np.uint8)


def to_bev(sc, path_base, gif=True):
    arr = bev(sc)
    np.savez_compressed(f"{path_base}.bev.npz", bev=arr, channels=np.array(BEV_CH), resolution_m=0.5, fps=2.0,
                        center=np.array([sc["event"]["x"], sc["event"]["y"]]), event_frame=sc["event_index"] // 5)
    if gif:
        from PIL import Image
        cols = {"lanes": (48, 54, 61), "intersections": (68, 76, 86), "crosswalks": (110, 118, 129), "red_stop_lines": (255, 77, 77),
                "vehicles": (86, 180, 233), "ego": (230, 159, 0), "pedestrians": (0, 158, 115)}
        ims = []
        for k, f in enumerate(arr):
            img = np.zeros((*f.shape[1:], 3), np.float32)
            img[:] = (13, 17, 23)
            for c, name in enumerate(BEV_CH[:-1]):
                m = f[c][..., None] / 255.0
                img = img * (1 - m) + np.array(cols[name]) * m
            im = Image.fromarray(img.astype(np.uint8)).resize((256, 256), Image.NEAREST)
            if k == sc["event_index"] // 5:
                from PIL import ImageDraw
                ImageDraw.Draw(im).rectangle([1, 1, 254, 254], outline=(255, 77, 77), width=3)
            ims.append(im)
        ims[0].save(f"{path_base}.bev.gif", save_all=True, append_images=ims[1:], duration=250, loop=0)
    return f"{path_base}.bev.npz"


# ---------------------------------------------------------------- table
def tracks_table(sc):
    import pandas as pd
    rows = []
    for t in sc["tracks"]:
        for j in range(sc["n_steps"]):
            if t["valid"][j]:
                rows.append(dict(scenario_id=sc["id"], city=sc["city"]["key"], condition=sc["condition"]["name"], step=j,
                                 t=round(j * sc["dt"], 1), rel_event_s=round((j - sc["event_index"]) * sc["dt"], 1), track=t["id"],
                                 type=t["type"], is_av=t["is_av"], is_ego=t["is_ego"], involved=t["involved"], x=t["x"][j], y=t["y"][j],
                                 heading=t["heading"][j], vx=t["vx"][j], vy=t["vy"][j], length=t["length"], width=t["width"]))
    return pd.DataFrame(rows)


def to_table(sc, path):
    tracks_table(sc).to_parquet(path, index=False)
    return path


def export(sc, out, formats=FORMATS, mapdir=None):
    """Write `sc` in every format of `formats` under out/<format>/. Returns {format: path}."""
    mapdir = mapdir or os.path.join(out, "maps")
    res = {}
    for f in formats:
        d = os.path.join(out, f)
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, sc["id"])
        res[f] = {"waymo": lambda: to_waymo(sc, base + ".waymo.json"), "av2": lambda: to_av2(sc, d),
                  "scenarionet": lambda: to_scenarionet(sc, base + ".scenarionet.pkl"),
                  "gpudrive": lambda: to_gpudrive(sc, base + ".gpudrive.json"),
                  "commonroad": lambda: to_commonroad(sc, base + ".cr.xml"),
                  "openscenario": lambda: to_openscenario(sc, base + ".xosc", mapdir),
                  "carla": lambda: to_carla(sc, base, mapdir), "bev": lambda: to_bev(sc, base),
                  "table": lambda: to_table(sc, base + ".tracks.parquet")}[f]()
    return res


# ---------------------------------------------------------------- Waymo TFRecord shards (needs a compiled scenario_pb2)
def _crc32c_table():
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        t.append(c)
    return t


_CRC = _crc32c_table()


def crc32c(data):
    c = 0xFFFFFFFF
    for b in data:
        c = _CRC[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def _masked(c):
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def write_tfrecord(path, records):
    """TFRecord file of byte strings (the format tf.data.TFRecordDataset reads), without TensorFlow."""
    import struct
    with open(path, "wb") as f:
        for r in records:
            h = struct.pack("<Q", len(r))
            f.write(h + struct.pack("<I", _masked(crc32c(h))) + r + struct.pack("<I", _masked(crc32c(r))))
    return path


def read_tfrecord(path):
    import struct
    out = []
    with open(path, "rb") as f:
        while True:
            h = f.read(8)
            if not h:
                return out
            n = struct.unpack("<Q", h)[0]
            assert struct.unpack("<I", f.read(4))[0] == _masked(crc32c(h))
            r = f.read(n)
            assert struct.unpack("<I", f.read(4))[0] == _masked(crc32c(r))
            out.append(r)


def scenario_pb2_module():
    """Waymo's compiled Scenario message: from `waymo-open-dataset` if installed, else None."""
    try:
        from waymo_open_dataset.protos import scenario_pb2
        return scenario_pb2
    except ImportError:
        return None


def to_waymo_tfrecord(scenarios, path, scenario_pb2):
    from google.protobuf import json_format
    recs = []
    for sc in scenarios:
        tmp = path + ".tmp.json"
        to_waymo(sc, tmp)
        recs.append(json_format.Parse(open(tmp).read(), scenario_pb2.Scenario()).SerializeToString())
        os.remove(tmp)
    return write_tfrecord(path, recs)
