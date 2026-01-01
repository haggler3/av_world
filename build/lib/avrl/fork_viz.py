"""One dangerous moment, sixteen futures (docs/img/fork_branches.png).

Run one copy of Oakland (the 234k-step policy, no safety filter, humans x5 riskier) until an AV crashes. Rewind 5 s: that
state is what the trainer saves as a fork. Copy it into 16 copies with different random numbers and replay 10 s. The
AV that crashed in the original run (heavy line) and the same AV in every branch: how fast it drives and how much room the
barrier says it has. Some branches end in the crash, some do not, and PPO learns from the difference.

    python -m avrl.fork_viz            # ~2 min
"""
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from xnet import network as NW  # noqa: E402
from xnet import sim as X  # noqa: E402

from . import ladder as LD  # noqa: E402
from . import xppo as P  # noqa: E402
from .ladder_plot import INK, INK2, SURFACE, style  # noqa: E402

BRANCHES, REWIND, HORIZON, SNAP_EVERY = 16, 25, 50, 5
IDX = {k: i for i, k in enumerate(X.METRICS)}


def main(out="docs/img/fork_branches.png", seed=3):
    net, geo, info = NW.build("oakland")
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = LD.test_world()
    pol = P.deterministic(P.load("results/xnet3/policy.npz"))
    X.SHIELD = "off"

    @jax.jit
    def step_info(s, k):
        o, d = X.step(s, J, beh, nh, rl=(pol, k))
        return o, d["bar_h"]

    sim = X.Sim(net, n_envs=1, beh=beh, seed=seed, reroute_s=0, policy=pol)
    sim.run(240)
    st = jax.tree.map(lambda a: a[0], sim.st)
    key = jax.random.PRNGKey(seed)
    snaps, crash_at, focal = [], None, None
    for i in range(3000):
        if i % SNAP_EVERY == 0:
            snaps.append((i, st))
        new, h = step_info(st, jax.random.fold_in(key, i))
        wreck_new = np.asarray(new.alive & (new.wreck_t > 0) & ~(st.wreck_t > 0)) & np.asarray(new.av)
        if i > 60 and wreck_new.any():
            crash_at, focal = i, int(np.nonzero(wreck_new)[0][0])
            break
        st = new
    if crash_at is None:
        raise SystemExit("no AV crash found; try another seed")
    start_i, start = max((s for s in snaps if s[0] <= crash_at - REWIND), key=lambda s: s[0])
    print(f"crash at step {crash_at}; forking from step {start_i} ({(crash_at - start_i) * X.DT:.1f} s earlier), AV slot {focal}", flush=True)

    def replay(n_branch, salt):
        sts = jax.tree.map(lambda a: jnp.broadcast_to(a, (n_branch,) + a.shape), start)
        sts = sts._replace(key=jax.vmap(jax.random.fold_in)(sts.key, jnp.arange(n_branch) + salt))
        f = jax.jit(lambda s, k: jax.vmap(lambda ss, kk: step_info(ss, kk))(s, jax.random.split(k, n_branch)))
        sp, bh, cr = [], [], []
        for j in range(HORIZON):
            sts, h = f(sts, jax.random.fold_in(key, 10_000 + j + salt))
            sp.append(np.asarray(sts.v[:, focal]))
            bh.append(np.asarray(jnp.minimum(h[:, focal], 30.0)))
            cr.append(np.asarray(sts.alive[:, focal] & (sts.wreck_t[:, focal] > 0)))
        return np.array(sp), np.array(bh), np.array(cr)

    sp, bh, cr = replay(BRANCHES, 0)
    crashed = cr.any(0)
    # the same state, 200 more branches, for the outcome tally (and the original run for reference)
    tally = np.concatenate([replay(BRANCHES, 100 * (k + 1))[2].any(0) for k in range(8)])
    t = (np.arange(HORIZON) + 1) * X.DT
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
    for ax, data, lab, title in ((axs[0], sp, "AV speed (m/s)", "A  The AV's speed in each branch"),
                                 (axs[1], bh, "barrier value h (m; < 0: inside the barrier)", "B  How much room the barrier says it has")):
        style(ax, title)
        for b in range(BRANCHES):
            ax.plot(t, data[:, b], color="#ff6b6b" if crashed[b] else "#3ddc84", lw=1.3, alpha=0.8)
            if crashed[b]:
                k = int(np.argmax(cr[:, b]))
                ax.scatter([t[k]], [data[k, b]], color="#ff6b6b", s=26, zorder=4)
        ax.set_xlabel("seconds after the fork", color=INK2, fontsize=8.5)
        ax.set_ylabel(lab, color=INK2, fontsize=8.5)
    axs[1].axhline(0, color=INK2, lw=0.8)
    ax = axs[2]
    style(ax, "C  What happens from this one state")
    n = len(tally)
    ax.bar([0, 1], [(~tally).sum() / n, tally.sum() / n], color=["#3ddc84", "#ff6b6b"], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no crash", "this AV crashes"], color=INK2, fontsize=9)
    ax.set_ylabel(f"share of {n} branches", color=INK2, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    for x, v in ((0, (~tally).sum() / n), (1, tally.sum() / n)):
        ax.text(x, v + 0.02, f"{100 * v:.0f} %", ha="center", color=INK, fontsize=10)
    ax.text(0.5, 0.9, f"fork {(crash_at - start_i) * X.DT:.0f} s before the original crash", ha="center", color=INK2, fontsize=8,
            transform=ax.transAxes)
    fig.suptitle("Fork at the moment of danger: one saved state, many futures", color=INK, fontsize=12.5, x=0.01, ha="left", y=0.995)
    fig.tight_layout(pad=1.6, rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    X.SHIELD = "idm"
    print("wrote", out, f"({tally.mean():.2f} of branches crash)")


if __name__ == "__main__":
    main()
