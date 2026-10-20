"""The world model's crash head as a reward-shaping potential.

PPO's crash signal is one -30 at the impact. The latent model already knows, from 800 k logged decisions, which scenes
end in a crash (AUROC 0.954 on held-out copies of the area). Its crash head gives the probability of a crash on the
next step from a scene and an action. Evaluated at the *coasting* action (no acceleration, the middle gap acceptance)
it depends on the scene only, so it is a valid potential:

    phi(s) = -LAM * P(crash | s, coast)        r'_t = r_t + gamma * phi(s_t+1) - phi(s_t)

Potential-based shaping leaves the optimal policy unchanged (Ng, Harada & Russell, 1999). What it changes is the
credit: a step that drives the AV into a scene the model rates dangerous is charged when it happens, not seconds
later at the impact.
"""
import jax
import jax.numpy as jnp

from . import model as M
from . import planner as PL

LAM = 10.0
COAST = jnp.array([0.0, 0.0])  # acceleration 0, gap-acceptance scale 1.15 (tanh(0) = 0)


def potential(wm=None, lam=LAM):
    """-> shape(obs [..., 28]) -> phi [...]: minus lam times the model's crash probability under the coasting action."""
    wm = wm if wm is not None else M.load()

    def phi(obs):
        z = M.encode(wm, obs)
        a = jnp.broadcast_to(M.act_in(COAST), z.shape[:-1] + (2,))
        tr = M.run(wm["trans"], jnp.concatenate([z, a], -1))
        return -lam * jax.nn.sigmoid(tr[..., 1] - jnp.log(PL.POS_W))  # undo the x POS_W weight of the positives
    return phi
