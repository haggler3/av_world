"""`av-world`: command line for the simulator.

  av-world info                              versions, devices, areas
  av-world bench [--area oakland]            simulated seconds per wall-clock second on this machine
  av-world import-osm map.osm city.npz       OpenStreetMap extract -> road network (then av_world.make("city.npz"))
  av-world export-carla OUT [--area oakland] [--seconds 60] [--osm map.osm]    OpenDRIVE map + trajectories + replay script for CARLA
  av-world fetch-av2 DIR [--n 20]            download Argoverse 2 scenarios (public bucket, no account)
  av-world import-av2 SCENARIO_DIR OUT       a logged scenario -> OUT.json (scene), OUT.xodr (OpenDRIVE map), OUT_net.npz (xnet road network)
  av-world validate-av2 DIR [--limit 100]    how well the simulator's drivers predict the logs (xnet.avdata.validate)
  av-world train [gpu_train options]         PPO on every device (python -m avrl.gpu_train)
  av-world survey | adversary | rare ...     the long-tail tools (python -m avlab.tails / adversary / rare)
"""
import argparse
import runpy
import sys
import time


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "info":
        import jax

        import av_world
        from xnet import network as NW
        print(f"av_world {av_world.__version__}; jax {jax.__version__}; devices {jax.devices()}; areas {list(NW.AREAS)}")
        return 0
    if cmd == "bench":
        ap = argparse.ArgumentParser()
        ap.add_argument("--area", default="oakland")
        ap.add_argument("--envs", type=int, default=8)
        ap.add_argument("--seconds", type=float, default=60.0)
        a = ap.parse_args(rest)
        import av_world
        env = av_world.make(a.area, n_envs=a.envs, beh=dict(p_av=0.25))
        st = env.reset(warm_s=20.0)
        t0 = time.time()
        st, _ = env.rollout(st, None, int(a.seconds / 0.2))
        st.t.block_until_ready()
        dt = time.time() - t0
        print(f"{a.envs} copies of {a.area}: {a.envs * a.seconds / dt:,.0f} simulated s per wall-clock s (includes compile)")
        return 0
    if cmd == "import-osm":
        sys.argv = ["xnet.osm"] + rest
        runpy.run_module("xnet.osm", run_name="__main__")
        return 0
    if cmd == "export-carla":
        ap = argparse.ArgumentParser()
        ap.add_argument("out")
        ap.add_argument("--area", default="oakland")
        ap.add_argument("--osm")
        ap.add_argument("--seconds", type=float, default=60.0)
        ap.add_argument("--warm", type=float, default=120.0)
        ap.add_argument("--av-share", type=float, default=0.3)
        a = ap.parse_args(rest)
        from xnet import carla_export as CE
        from xnet import network as NW
        from xnet import osm
        from xnet import sim as X
        net, geo, _ = osm.import_osm(a.osm) if a.osm else NW.build(a.area)
        sim = X.Sim(net, n_envs=1, seed=0, reroute_s=0, beh=X.Behaviour(p_av=a.av_share))
        sim.run(a.warm)
        scene = CE.export(sim, 0, a.seconds, a.out, net=net, geo=geo)
        print(f"{len(scene['vehicles'])} vehicles, {len(scene['pedestrians'])} crossings -> {a.out}.xodr, {a.out}.json, {a.out}_replay.py")
        return 0
    if cmd == "fetch-av2":
        ap = argparse.ArgumentParser()
        ap.add_argument("dir")
        ap.add_argument("--n", type=int, default=20)
        ap.add_argument("--split", default="val")
        a = ap.parse_args(rest)
        from xnet import avdata as AV
        print(f"{len(AV.fetch_av2(a.dir, a.n, a.split))} scenarios -> {a.dir}")
        return 0
    if cmd == "import-av2":
        ap = argparse.ArgumentParser()
        ap.add_argument("scenario")
        ap.add_argument("out")
        ap.add_argument("--every", type=int, default=2)
        a = ap.parse_args(rest)
        import json

        from xnet import avdata as AV
        from xnet import carla_export as CE
        from xnet import osm
        sc = AV.read_av2(a.scenario)
        scene = AV.av2_to_scene(sc, a.every)
        net, geo, info = AV.av2_network(sc)
        json.dump(scene, open(f"{a.out}.json", "w"), separators=(",", ":"))
        open(f"{a.out}.xodr", "w").write(CE.to_opendrive(net, geo, sc["id"]))
        osm.save(net, f"{a.out}_net.npz")
        print(f"{sc['city']}: {len(scene['vehicles'])} vehicles, {info['junction_clusters']} junctions, {net['link_len'].shape[0]} links -> {a.out}.json/.xodr/_net.npz")
        return 0
    if cmd == "validate-av2":
        ap = argparse.ArgumentParser()
        ap.add_argument("dir")
        ap.add_argument("--limit", type=int, default=100)
        a = ap.parse_args(rest)
        from xnet import avdata as AV
        AV.validate(a.dir, a.limit)
        return 0
    mods = {"train": "avrl.gpu_train", "survey": "avlab.tails", "adversary": "avlab.adversary", "rare": "avlab.rare"}
    if cmd in mods:
        sys.argv = [mods[cmd]] + (["survey"] if cmd == "survey" and not rest else rest)
        runpy.run_module(mods[cmd], run_name="__main__")
        return 0
    print(f"unknown command {cmd!r}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
