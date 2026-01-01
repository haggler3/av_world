import jax
import jax.numpy as jnp
import numpy as np

from wm2 import data as D
from wm2 import model as M


def scene():
    """Agent 0 faces +y at the origin, moving 5 m/s. Agent 1: 10 m dead ahead, 3 m/s, same heading. Agent 2: 5 m to its left. Agent 3: 100 m away."""
    A, T = 4, 6
    pos, vel, head = np.zeros((A, T, 2)), np.zeros((A, T, 2)), np.zeros((A, T))
    for t in range(T):
        pos[0, t] = [0, 5.0 * 0.1 * t]
        pos[1, t] = [0, 10 + 3.0 * 0.1 * t]
        pos[2, t] = [-5, 0]
        pos[3, t] = [100, 100]
    vel[0], vel[1] = [0, 5.0], [0, 3.0]
    head[:] = np.pi / 2
    return {"pos": pos, "vel": vel, "head": head, "ok": np.ones((A, T), bool), "kind": np.array([0, 0, 4, 0]), "id": "synthetic"}


def test_neighbour_tokens_are_in_the_agents_own_frame():
    sc = scene()
    S, NB, mask, lead = D.features(sc, 2, np.array([0]))
    assert mask[0].sum() == 2                                          # the agent at 100 m is outside the 30 m radius
    near, far = NB[0, 0], NB[0, 1]                                     # sorted by distance: the parked agent (5 m), then the leader (10 m)
    assert np.allclose(near[:2], [-1.0 / 20.0, 5.0 / 20.0], atol=1e-5)  # at t=2 we have moved 1 m on: it is 1 m behind and 5 m to the left
    assert np.allclose(far[:2], [9.6 / 20.0, 0.0], atol=1e-5)          # the leader is 9.6 m ahead, not to the side
    assert np.isclose(far[2], -2.0 / 10.0, atol=1e-5)                  # it is 2 m/s slower, along our heading
    assert near[6 + 4] == 1.0 and far[6 + 0] == 1.0                    # kind one-hots: static, car
    assert lead[0].tolist()[:2] == [False, True]                       # only the agent ahead in lane is the leader
    assert S[0, 0] == 1.0 and np.isclose(S[0, 4], 0.5)


def test_features_do_not_depend_on_world_orientation():
    sc = scene()
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rot = {**sc, "pos": sc["pos"] @ R.T + 1000.0, "vel": sc["vel"] @ R.T, "head": sc["head"] + th}
    a = D.features(sc, 2, np.array([0]))
    b = D.features(rot, 2, np.array([0]))
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-4)


def test_padding_when_a_scene_has_few_agents():
    sc = scene()
    sc = {k: (v[:2] if k in ("pos", "vel", "head", "ok", "kind") else v) for k, v in sc.items()}
    S, NB, mask, lead = D.features(sc, 2, np.array([0, 1]))
    assert NB.shape == (2, D.K, D.NB_DIM) and mask.sum() == 2 and not NB[~mask].any()


def test_masked_neighbours_cannot_influence_the_output():
    p = M.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    S = jnp.asarray(rng.normal(size=(5, D.SELF_DIM)), jnp.float32)
    NB = jnp.asarray(rng.normal(size=(5, D.K, D.NB_DIM)), jnp.float32)
    vis = jnp.asarray(rng.uniform(size=(5, D.K)) < 0.5)
    base = M.apply(p, S, NB, vis)
    NB2 = jnp.where(vis[..., None], NB, NB + 100.0)                    # change only what the self token may not see
    for x, y in zip(base, M.apply(p, S, NB2, vis)):
        assert np.allclose(np.asarray(x), np.asarray(y), atol=1e-4)
    changed = M.apply(p, S, jnp.where(vis[..., None], NB + 3.0, NB), vis)[0]
    assert not np.allclose(np.asarray(base[0]), np.asarray(changed), atol=1e-4)    # but what it can see does matter


def test_variants_select_the_documented_neighbours():
    mask = np.array([[1, 1, 1, 0]], bool)
    lead = np.array([[0, 1, 0, 0]], bool)
    assert M.visible(mask, lead, "all").tolist() == mask.tolist()
    assert M.visible(mask, lead, "leader").tolist() == lead.tolist()
    assert not M.visible(mask, lead, "none").any()


def test_idm_accelerates_on_a_free_road_and_brakes_when_close():
    free = M.idm(5.0, 1e6, 5.0, T=1.2, a_max=1.5, b=2.0, s0=2.0, v0=15.0)
    close = M.idm(10.0, 6.0, 2.0, T=1.2, a_max=1.5, b=2.0, s0=2.0, v0=15.0)
    assert free > 1.0 and close < -3.0


def test_the_model_learns_a_car_following_rule_from_samples():
    """Synthetic samples where the acceleration is exactly -0.5 * (closing speed): the leader variant must beat predicting the mean."""
    rng = np.random.default_rng(1)
    n = 4000
    S = np.zeros((n, D.SELF_DIM), np.float32)
    NB = np.zeros((n, D.K, D.NB_DIM), np.float32)
    closing = rng.uniform(-3, 3, n).astype(np.float32)
    NB[:, 0, 0], NB[:, 0, 2], NB[:, 0, 6] = 0.5, -closing / 10.0, 1.0
    mask = np.zeros((n, D.K), bool)
    mask[:, 0] = True
    Y = np.stack([-0.5 * closing, np.zeros(n)], 1).astype(np.float32)
    d = {"S": S, "NB": NB, "mask": mask, "lead": mask.copy(), "Y": Y}
    idx = np.arange(n)
    p, curve = M.fit(d, idx, "leader", steps=600, batch=256)
    mu, _ = M.predict(p, d, idx, "leader")
    assert np.abs(mu[:, 0] - Y[:, 0]).mean() < 0.4 * np.abs(Y[:, 0] - Y[:, 0].mean()).mean()
    p0, _ = M.fit(d, idx, "none", steps=600, batch=256)
    mu0, _ = M.predict(p0, d, idx, "none")
    assert np.abs(mu0[:, 0] - Y[:, 0]).mean() > 2 * np.abs(mu[:, 0] - Y[:, 0]).mean()   # without the leader it cannot know
