"""Every network and world-model upgrade as a feature: what it answers, how long it takes, what it writes, how to run it.

The registry is plain data plus one `run(quick, out)` per feature, so the command line (`av-world nets`), the Python API
(`av_world.easy.nets`) and the docs all read the same source. Quick mode shrinks every collection and training to a few minutes and
writes to a separate folder, so it never overwrites the full results.
"""
import importlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, field

FULL_OUT, QUICK_OUT = "results/nets", "results/nets/quick"


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str                 # "network" (nets/) or "upgrade" (simulator / world model)
    title: str
    question: str
    result: str               # the headline number from the full run
    minutes: tuple            # (quick, full) on a 4-core CPU, measured
    outputs: tuple            # files the full run writes (paths relative to the repo)
    doc: str                  # where it is explained
    needs: tuple = ()         # "city" (the Pittsburgh map data), "av2" (Argoverse 2 scenarios), "logs" (data/world.bin, data/env.bin), or another feature
    runner: str = ""          # "module:function" called as function(quick=..., out_dir=...)
    use: str = ""             # one line of Python that uses the trained network
    extra: dict = field(default_factory=dict)


FEATURES = [
    Feature("critic", "network", "Interventional risk critic (N2)", "Which action makes this moment safe?",
            "10 % intervention budget: bad outcomes -71 % (0.186 % -> 0.054 %); trained on the log instead: +69 %",
            (4, 10), ("results/nets/counterfactual.json", "results/nets/critic_params.pkl"), "docs/NEURAL_NETS.md#n2-interventional-risk-critic-netsforkspy-netscounterfactualpy",
            ("city",), "nets.counterfactual:main", 'easy.nets.load("critic").risk(obs, u)'),
    Feature("response", "network", "How the human behind answers (N7)", "What will the driver behind me do if I brake?",
            "effect error when braking 0.11 m/s^2, against 0.21 (assume nothing) and 0.37 (trained on the log)",
            (4, 10), ("results/nets/response.json", "results/nets/response_params.pkl"), "docs/NEURAL_NETS.md#n7-how-the-human-behind-answers-netsresponsepy",
            ("city",), "nets.response:main", 'easy.nets.load("response").follower_effect(obs, follower, u)'),
    Feature("occlusion", "network", "Seeing through the bus (N6)", "Where are the road users the AV cannot see?",
            "AUROC 0.968 per cell (shadow geometry 0.926); someone hidden 25 m ahead 0.978 (0.770)",
            (1, 6), ("results/nets/occlusion.json", "results/nets/occlusion_params.pkl"), "docs/NEURAL_NETS.md#n6-seeing-through-the-bus-netsocclusionpy",
            ("city",), "nets.occlusion:main", 'easy.nets.load("occlusion").belief(tokens)'),
    Feature("mindreader", "network", "The mind reader (N1)", "What kind of driver is that?",
            "headway R^2 0.41, acceleration 0.44, desired speed 0.37 (linear ~0); red-light runner and aggressive: chance",
            (3, 6), ("results/nets/mindreader.json", "results/nets/mindreader_params.pkl"), "docs/NEURAL_NETS.md#n1-the-mind-reader-netsmindreaderpy",
            ("city",), "nets.mindreader:main", 'easy.nets.load("mindreader").read(windows)'),
    Feature("blame", "network", "But-for blame (N5)", "Who caused this crash, and who could have avoided it?",
            "a mistake is a but-for cause in 1.6 % of crashes, a careful driver avoids 46.6 %; ranking AUROC 0.94",
            (3, 14), ("results/nets/blame.json", "results/nets/blame_careful_params.pkl"), "docs/NEURAL_NETS.md#n5-but-for-blame-netsblamepy",
            ("city",), "nets.blame:main", 'easy.nets.load("blame").rank(candidates, mask)'),
    Feature("sbi", "network", "Which settings explain real traffic? (N4)", "Which simulator knobs explain real Argoverse 2 traffic?",
            "posterior draws 2.24 SD from the real statistics (point fit 2.60, prior 3.50); three real statistics outside every simulated world",
            (4, 25), ("results/nets/sbi.json", "results/nets/sbi_params.pkl"), "docs/NEURAL_NETS.md#n4-amortised-inference-of-real-driving-netssbipy",
            ("city", "av2"), "nets.sbi:main", 'easy.nets.load("sbi").real(n=1000)'),
    Feature("crn", "upgrade", "Identity-keyed common random numbers", "How much less noisy is a paired comparison of two worlds?",
            "coupled worlds stay 0.85 nested after 600 s (0.42 before); variance ratio 1.3-3.4",
            (2, 10), ("results/crn/paired.json", "results/crn/decay.json"), "docs/WORLD_MODEL_UPGRADES.md#1-identity-keyed-common-random-numbers",
            ("city",), "nets.registry:_run_crn", "X.Behaviour(crn=1.0)  # then avlab.paired.compare(net, beh_a, beh_b)"),
    Feature("conservative", "upgrade", "Conservative imagination", "How do you stop a planner exploiting the model's mistakes?",
            "exploitation gap 1.54 -> 0.19-0.31 with a distance-from-data penalty; ensemble disagreement does not help",
            (1, 2), ("results/conservative/results.json",), "docs/WORLD_MODEL_UPGRADES.md#2-conservative-imagination-price-the-models-mistakes",
            (), "nets.registry:_run_conservative", "wmplan.conservative.cem(model, norm, o0, key, j_mopo, {'lam': 0.25}, support=s, signal='support')"),
    Feature("multistep", "upgrade", "History + multi-step rollout loss", "Does training on its own rollouts fix a world model's drift?",
            "3 s speed error 1.22 -> 0.33 km/h; 3 s gap error 5.6 -> 7.1 m (worse)",
            (2, 6), ("results/wm_multistep/results.json",), "docs/WORLD_MODEL_UPGRADES.md#4-a-world-model-that-looks-back-and-trains-on-its-own-rollouts",
            ("logs",), "nets.registry:_run_multistep", "avrl.wm_multistep.fit(rows, hist=3, K=10, key=jax.random.PRNGKey(0))"),
    Feature("agent_tokens", "upgrade", "Agent-token world model on real logs", "Does attending to nearby road users predict real driving better?",
            "3 s closed-loop error -0.90 m vs constant velocity with a leader token; 8 neighbours -0.65 m",
            (4, 8), ("results/wm2/results.json", "results/wm2/params_leader.pkl"), "docs/WORLD_MODEL_UPGRADES.md#3-and-5-an-agent-token-world-model-and-human-drivers-learned-from-real-logs",
            ("av2",), "nets.registry:_run_agent_tokens", "wm2.model.apply(params, S, NB, visible)"),
    Feature("learned_humans", "upgrade", "Human drivers learned from real logs", "Are learned car-followers more realistic than IDM?",
            "tail loss vs real traffic 9.83 -> 8.15; close approaches (TTC < 1 s) x8; hard braking unchanged",
            (3, 10), ("results/extras/learned_humans.json",), "docs/WORLD_MODEL_UPGRADES.md#3-and-5-an-agent-token-world-model-and-human-drivers-learned-from-real-logs",
            ("city", "agent_tokens"), "nets.registry:_run_learned_humans", "X.set_learned_follower(params); X.Behaviour(learned=1.0)"),
    Feature("sigreg", "upgrade", "Gaussian-regularised latent (SIGReg)", "How do you stop a joint-embedding model collapsing?",
            "embedding spread 0.0001 -> 0.99; linear probes mixed",
            (1, 1), ("results/extras/sigreg.json",), "docs/WORLD_MODEL_UPGRADES.md#7-a-gaussian-regulariser-instead-of-tricks-sigreg",
            ("logs",), "nets.registry:_run_sigreg", "loss = l_pred + lam * wmplan.sigreg.sigreg(z, key)"),
    Feature("terminal_value", "upgrade", "Learned terminal value for the planner", "Does a value function let a short planner see further?",
            "no measurable gain (h = 5: -0.003; h = 10: +0.15 [-0.31, 0.82])",
            (1, 3), ("results/extras/terminal_value.json",), "docs/WORLD_MODEL_UPGRADES.md#8-a-terminal-value-for-a-short-planner",
            (), "nets.registry:_run_terminal_value", "wmplan.terminal_value.cem_h(model, norm, value, o0, key, h=5, use_v=True)"),
    Feature("autotag", "system", "Scene auto-tagger", "Which scenes contain cut-ins, distracted drivers, hidden road users, risky moments?",
            "17 tags on real Argoverse 2 and simulated scenes; learned tags calibrated to a precision target; see docs/AUTOTAG.md",
            (3, 25), ("results/autotag/eval.json", "results/autotag/tags_av2.parquet"), "docs/AUTOTAG.md#the-ten-methods",
            ("city", "av2"), "nets.registry:_run_autotag", 'easy.tag.query("cut_in & hidden_ahead", source="av2")'),
    Feature("realism", "system", "Sim-to-real gap", "Where does simulated traffic differ from real logs, and what patch closes it?",
            "RESULT", (8, 60), ("results/realism/scoreboard.json", "results/realism/preset.json"), "docs/SIM2REAL.md#the-ten-methods",
            ("city", "av2"), "nets.registry:_run_realism", "beh = easy.realism.preset()   # patched human drivers for xnet"),
]
BY_NAME = {f.name: f for f in FEATURES}


def get(name):
    if name not in BY_NAME:
        raise KeyError(f"unknown feature {name!r}; one of: {', '.join(BY_NAME)}")
    return BY_NAME[name]


def trained(f, out_dir=None):
    """Has the full run (or the run in out_dir) written its outputs?"""
    if out_dir is None:
        return all(os.path.exists(p) for p in f.outputs)
    return all(os.path.exists(os.path.join(out_dir, os.path.basename(p))) for p in f.outputs)


def missing_data(f):
    need = []
    if "city" in f.needs and not os.path.exists("data/pgh_segments.parquet"):
        need.append("the Pittsburgh map data (python -m pgh.fetch)")
    if "av2" in f.needs and not os.path.isdir(os.environ.get("AV2_ROOT", "data/av2")):
        need.append("Argoverse 2 scenarios in data/av2 (av-world fetch-av2 data/av2 --n 400) or AV2_ROOT")
    if "logs" in f.needs and not (os.path.exists("data/world.bin") and os.path.exists("data/env.bin")):
        need.append("the AV decision logs data/world.bin and data/env.bin (see docs/WORLD_MODELS.md)")
    for n in f.needs:
        if n in BY_NAME and not trained(BY_NAME[n]):
            need.append(f"the {n!r} feature trained first (av-world nets train {n})")
    return need


def train(name, quick=False, out_dir=None):
    """Run one feature end to end. Networks: full results in results/nets, quick ones in results/nets/quick."""
    f = get(name)
    out_dir = out_dir or (QUICK_OUT if quick else FULL_OUT)
    os.makedirs(out_dir, exist_ok=True)
    if f.name in ("critic", "response") and not os.path.exists(os.path.join(out_dir, "forks.npz")):
        importlib.import_module("nets.forks").main(out_dir=out_dir, quick=quick)       # the shared interventional data
    mod, fn = f.runner.split(":")
    return getattr(importlib.import_module(mod), fn)(quick=quick, out_dir=out_dir)


# ---------------------------------------------------------------- the upgrades, behind the same signature
@contextmanager
def _redirect(module, out_dir, quick, default):
    """Point a research module's OUT at out_dir for a quick run, so quick results never overwrite the full ones."""
    old = getattr(module, "OUT", None)
    if quick:
        module.OUT = os.path.join(out_dir, os.path.basename(default.rstrip("/")))
    try:
        yield module.OUT if quick else default
    finally:
        if old is not None:
            module.OUT = old


def _run_crn(quick=False, out_dir=FULL_OUT):
    from avlab import crn_decay, paired
    with _redirect(paired, out_dir, quick, "results/crn"):
        paired.main(*((16, 300) if quick else (64, 600)))
    if not quick:
        crn_decay.main()


def _run_conservative(quick=False, out_dir=FULL_OUT):
    from wmplan import conservative as m
    with _redirect(m, out_dir, quick, "results/conservative"):
        m.main(*((64, 4) if quick else (256, 15)))


def _run_multistep(quick=False, out_dir=FULL_OUT):
    from avrl import wm_multistep as m
    with _redirect(m, out_dir, quick, "results/wm_multistep"):
        m.main(*((500, 2) if quick else (2500, 3)))


def _run_agent_tokens(quick=False, out_dir=FULL_OUT):
    from wm2 import run as m
    with _redirect(m, out_dir, quick, "results/wm2"):
        m.main(m.ROOT, 800 if quick else 5000)


def _run_learned_humans(quick=False, out_dir=FULL_OUT):
    from avlab import learned_humans as m
    with _redirect(m, out_dir, quick, "results/extras"):
        m.main(1 if quick else 3)


def _run_sigreg(quick=False, out_dir=FULL_OUT):
    from wmplan import sigreg as m
    m.main(os.path.join(out_dir, "extras") if quick else "results/extras")


def _run_terminal_value(quick=False, out_dir=FULL_OUT):
    from wmplan import terminal_value as m
    m.main(64 if quick else 256, os.path.join(out_dir, "extras") if quick else "results/extras")


def _run_realism(quick=False, out_dir=FULL_OUT):
    from realism import pipeline as PL
    return PL.run(quick=quick)


def _run_autotag(quick=False, out_dir=FULL_OUT):
    from autotag import dag, pipeline as PL
    if quick:
        dag.CACHE, PL.OUT = os.path.join(out_dir, "autotag", "cache"), os.path.join(out_dir, "autotag")
        dag.STAGES["sim_scenes"].config = {"minutes": 2, "n_envs": 8, "seed": 0}
        dag.STAGES["av2_scenes"].config = {"limit": 40}
        dag.STAGES["learned_model"].config = {"steps": 500}
        dag.STAGES["active"].config = {"tag": "distracted", "rounds": 3, "batch": 50}
    dag.run("qa")
    return dag.run("store")

