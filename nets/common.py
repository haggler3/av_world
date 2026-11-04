"""Small training utilities shared by the networks: an MLP ensemble trained in one jitted scan, bootstrap intervals, AUROC."""
import jax
import jax.numpy as jnp
import numpy as np
import optax


def init_mlp(key, sizes):
    ks = jax.random.split(key, len(sizes) - 1)
    return [{"w": jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a), "b": jnp.zeros(b)} for k, a, b in zip(ks, sizes[:-1], sizes[1:])]


def mlp(p, x):
    for l in p[:-1]:
        x = jax.nn.gelu(x @ l["w"] + l["b"])
    return x @ p[-1]["w"] + p[-1]["b"]


def fit(loss_fn, params, data, steps=3000, batch=2048, lr=2e-3, seed=0, wd=1e-4):
    """loss_fn(params, batch_dict) -> scalar. data: dict of equal-length arrays. One jitted scan over `steps` minibatches."""
    data = {k: jnp.asarray(v) for k, v in data.items()}
    n = next(iter(data.values())).shape[0]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=wd))

    @jax.jit
    def run(p, key):
        def body(c, k):
            p, st = c
            i = jax.random.randint(k, (batch,), 0, n)
            l, g = jax.value_and_grad(loss_fn)(p, {a: b[i] for a, b in data.items()})
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), l
        return jax.lax.scan(body, (p, opt.init(p)), jax.random.split(key, steps))

    (p, _), curve = run(params, jax.random.PRNGKey(seed))
    return p, np.asarray(curve)


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(x, n=2000, seed=0, groups=None):
    """Mean and 95 % bootstrap interval; with `groups`, resample whole groups (e.g. copies), not rows."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    else:
        g = np.asarray(groups)
        u = np.unique(g)
        idx = [np.where(g == v)[0] for v in u]
        sums = np.array([x[i].sum() for i in idx])
        cnts = np.array([len(i) for i in idx])
        pick = rng.integers(0, len(u), (n, len(u)))
        bs = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
