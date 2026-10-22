"""Optional long-tail features: AV sensor faults, adversarial humans, cause tags. All are off by default and then change nothing."""
import numpy as np

from tests.test_osm import grid_osm
from xnet import osm
from xnet import sim as X

IDX = {k: i for i, k in enumerate(X.METRICS)}


def run(beh, seconds=240, seed=2, n=2):
    net, _, _ = osm.import_osm(grid_osm())
    X.SHIELD = "idm"
    sim = X.Sim(net, n_envs=n, beh=beh, seed=seed, reroute_s=0)
    sim.run(seconds)
    return {k: float(np.asarray(sim.st.m)[:, i].sum()) for k, i in IDX.items()}, sim


BASE = dict(p_av=0.5, demand=10.0)


def test_off_by_default():
    m, sim = run(X.Behaviour(**BASE))
    assert m["blackouts"] == 0 and m["ghosts"] == 0 and m["tail_phantom"] == 0 and m["tail_blind"] == 0
    assert not np.asarray(sim.st.blind_t).any()
    assert m["av_veh_s"] > 100


def test_sensor_blackouts_and_ghosts_are_counted_and_hurt():
    base, _ = run(X.Behaviour(**BASE))
    blind, _ = run(X.Behaviour(**BASE, p_blind=0.02))
    ghost, _ = run(X.Behaviour(**BASE, p_ghost=0.02))
    assert blind["blackouts"] > 20 and blind["ghosts"] == 0
    assert ghost["ghosts"] > 20 and ghost["blackouts"] == 0
    assert blind["av_crash"] >= base["av_crash"]  # a blind AV does not crash less
    assert ghost["av_hard"] > base["av_hard"]  # phantom obstacles make AVs brake hard


def test_adversary_budget_keeps_the_number_of_mistakes():
    beh = dict(BASE, breakdown_rate=4e-3)
    base, _ = run(X.Behaviour(**beh), seconds=600)  # a breakdown parks the AV behind it for minutes: shorter runs saturate
    neutral, _ = run(X.Behaviour(**beh, adv_brk=40.0, adv_budget=1.0), seconds=600)
    extra, _ = run(X.Behaviour(**beh, adv_brk=40.0, adv_budget=0.0), seconds=600)
    assert base["breakdowns"] > 20
    assert 0.6 * base["breakdowns"] < neutral["breakdowns"] < 1.5 * base["breakdowns"]  # same mistakes, aimed
    assert extra["breakdowns"] > 1.5 * base["breakdowns"]  # more mistakes


def test_squeeze_cut_ins_in_front_of_avs():
    base, _ = run(X.Behaviour(**BASE))
    adv, _ = run(X.Behaviour(**BASE, adv_cut=3.0))
    assert adv["tail_squeeze"] >= base["tail_squeeze"]


def test_tail_tags_never_exceed_their_parent_events():
    m, _ = run(X.Behaviour(**BASE, breakdown_rate=2e-3, distract_rate=1e-3), seconds=300)
    assert m["tail_stalled"] + m["tail_pileup"] + m["tail_distracted"] <= m["rear-end"] + m["merge"] + m["bicycle"] + 1e-9


def test_dart_outs_create_the_masked_pedestrian_crash():
    kw = dict(BASE, p_van=0.25, p_bus=0.15, ped_scale=6.0, breakdown_rate=0.0)
    base, _ = run(X.Behaviour(**kw), seconds=300, n=3)
    dart, _ = run(X.Behaviour(**kw, p_dart=1.0), seconds=300, n=3)
    assert base["tail_masked_ped"] == 0
    assert dart["peds"] >= base["peds"] and dart["tail_masked_ped"] >= base["tail_masked_ped"]


def test_tailgaters_keep_a_short_headway():
    _, sim = run(X.Behaviour(**BASE, p_tailgate=1.0), seconds=120)
    st = sim.st
    human = np.asarray(st.alive & ~st.av & (st.kind != X.BUS))  # buses keep their own 1.6 s
    assert human.sum() > 5 and float(np.asarray(st.T)[human].max()) < 0.71
    _, base = run(X.Behaviour(**BASE), seconds=120)
    hb = np.asarray(base.st.alive & ~base.st.av)
    assert float(np.asarray(base.st.T)[hb].min()) >= 1.0
